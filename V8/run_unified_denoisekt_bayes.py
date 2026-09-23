#!/usr/bin/env python3
"""Run validation-selected Bayesian studies without reading test during HPO."""

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
import yaml


HERE = Path(__file__).resolve().parent
MATRA_WORKER = HERE / 'train_denoisekt_xes.py'
DEFAULT_CONFIG = HERE / 'config_unified_denoisekt_bayes.yaml'
TRITON_CACHE_DIR = Path(
    os.environ.get('TRITON_CACHE_DIR', Path.home() / '.cache' / 'triton')
).expanduser().resolve()
SEQUENCE_LENGTH_MODELS = {
    'fa_kt', 'fluckt', 'mtkt', 'ukt',
}
FORGET_QW_MODELS = {'fa_kt', 'mtkt'}
STANDARD_QW_MODELS = {'fluckt', 'lefokt_akt'}


def load_yaml(path):
    with Path(path).open(encoding='utf-8') as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def parse_integral_sequence_value(value):
    """Accept integer-valued decimals emitted by official pyKT splitters."""
    try:
        return int(value)
    except (TypeError, ValueError):
        parsed = float(value)
        if not parsed.is_integer():
            raise ValueError(
                f'Expected an integer-valued sequence item, got {value!r}'
            )
        return int(parsed)


def install_dkt_forget_input_compatibility(pykt_root):
    """Patch pyKT's process-local time parser without rewriting input CSVs."""
    sys.path.insert(0, str(pykt_root))
    from pykt.datasets import dkt_forget_dataloader

    dkt_forget_dataloader.int = parse_integral_sequence_value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def merge_json_object(path, updates):
    """Merge dataset-keyed updates without discarding prior final summaries."""
    path = Path(path)
    previous = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(previous, dict):
            raise ValueError(f'Expected a JSON object at {path}')
    previous.update(updates)
    write_json(path, previous)
    return previous


def materialize_pykt_data_config(config, datasets, destination):
    """Bind pyKT's runtime config to the dataset paths declared by the campaign."""
    source = Path(config['runtime']['data_config']).resolve()
    with source.open(encoding='utf-8') as handle:
        payload = json.load(handle)
    for dataset_name, dataset in datasets.items():
        pykt_name = dataset['pykt_name']
        if pykt_name not in payload:
            raise KeyError(f'{pykt_name} is missing from {source}')
        payload[pykt_name]['dpath'] = str(Path(dataset['data_dir']).resolve())
    write_json(destination, payload)
    return Path(destination).resolve()


def write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def parameter_signature(params):
    return json.dumps(params, sort_keys=True, separators=(',', ':'))


def raw_complete_trials(study):
    return [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]


def complete_trials(study):
    """Return the best completed trial per unique sampled parameter set."""
    unique = {}
    for trial in raw_complete_trials(study):
        signature = parameter_signature(trial.params)
        previous = unique.get(signature)
        if previous is None or float(trial.value) > float(previous.value):
            unique[signature] = trial
    return list(unique.values())


def earlier_duplicate_trial(study, trial_number, params):
    signature = parameter_signature(params)
    eligible = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.RUNNING,
    }
    for other in study.trials:
        if other.number >= trial_number or other.state not in eligible:
            continue
        if parameter_signature(other.params) == signature:
            return other
    return None


@contextmanager
def exclusive_sample_lock(path):
    """Serialize parameter suggestion so concurrent workers cannot race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stop_reached(study, search):
    complete = complete_trials(study)
    maximum = search.get('maximum_completed_trials')
    if maximum is not None and len(complete) >= int(maximum):
        return True
    if len(complete) <= int(search['minimum_completed_trials']):
        return False
    rounded = [
        round(float(trial.value), int(search['validation_auc_round_decimals']))
        for trial in complete
    ]
    first_best = rounded.index(max(rounded))
    return len(complete) - first_best > int(search['no_improvement_patience'])


def recover_stale_trials(study):
    for trial in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    ):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)


def assert_fixed_fold_history(study, fixed_search_fold):
    """Reject mixed-fold history before it can affect TPE or the stop budget."""
    if fixed_search_fold is None:
        return
    expected = int(fixed_search_fold)
    foreign = [
        trial.number
        for trial in complete_trials(study)
        if int(trial.params['fold']) != expected
    ]
    if foreign:
        preview = ', '.join(str(number) for number in foreign[:8])
        suffix = ' ...' if len(foreign) > 8 else ''
        raise RuntimeError(
            f'Fixed-fold study expects fold {expected}, but contains '
            f'{len(foreign)} completed trial(s) from other folds '
            f'({preview}{suffix}). Start a clean study; mixed-fold history '
            'must not influence TPE or the stopping rule.'
        )


def trial_directory(root, dataset, number):
    return root / dataset / 'trials' / f'trial_{number:04d}'


def sample_params(trial, model, dataset, config):
    params = {
        'fold': trial.suggest_categorical('fold', dataset['folds']),
    }
    model_space = model.get('search_parameters', {})
    for name in model['sampled_parameters']:
        values = model_space.get(name, config['search']['parameters'][name])
        params[name] = trial.suggest_categorical(
            name, values
        )
    return params


def pykt_evaluation_flags(phase):
    """Keep pyKT's native sequence-level test routes disabled.

    The formal result is evaluated separately on the shared question-window
    test route.  That avoids treating pyKT's raw sequence metric as a result
    comparable with the no-Bayes campaign.
    """
    if phase not in {'search', 'final'}:
        raise ValueError(f'Unsupported trial phase: {phase}')
    return 0, 0


def pykt_run_directory(result):
    """Resolve a saved pyKT run directory for a live or resumed final run."""
    if result.get('run_dir'):
        run_dir = Path(result['run_dir'])
        if (run_dir / 'config.json').is_file():
            return run_dir

    directory = Path(result['directory'])
    emb_type = result['resolved_parameters']['emb_type']
    checkpoints = list((directory / 'checkpoint').glob(
        f'**/{emb_type}_model.ckpt'
    ))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f'Expected exactly one {emb_type}_model.ckpt below {directory}, '
            f'found {len(checkpoints)}'
        )
    return checkpoints[0].parent


def evaluate_pykt_question_window(result, pykt_root):
    """Evaluate one saved pyKT checkpoint using the common QW metric only."""
    sys.path.insert(0, str(pykt_root))
    import torch
    from torch.utils.data import DataLoader
    from pykt.datasets.data_loader import KTDataset
    from pykt.datasets.dkt_forget_dataloader import DktForgetDataset
    from pykt.datasets.que_data_loader import KTQueDataset
    from pykt.datasets.ukt_dataloader import UKTDataset
    from pykt.models import evaluate, evaluate_question, init_model

    run_dir = pykt_run_directory(result)
    saved = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    params = saved['params']
    model_name = params['model_name']
    emb_type = params['emb_type']
    data_config = copy.deepcopy(saved['data_config'])
    data_config['dataset_name'] = params['dataset_name']
    model_config = copy.deepcopy(saved['model_config'])
    for key in ('use_wandb', 'learning_rate', 'add_uuid', 'l2'):
        model_config.pop(key, None)
    if model_name in SEQUENCE_LENGTH_MODELS:
        model_config['seq_len'] = int(saved['train_config']['seq_len'])

    net = init_model(model_name, model_config, data_config, emb_type)
    checkpoint = torch.load(
        run_dir / f'{emb_type}_model.ckpt',
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
    )
    net.load_state_dict(checkpoint)
    batch_size = int(saved['train_config']['batch_size'])

    # DenoiseKT predicts questions directly; its QW test file is therefore
    # evaluated without concept-fusion.  FA-KT uses QW late-mean fusion.
    if model_name == 'denoisekt':
        test_path = Path(data_config['dpath']) / data_config[
            'test_window_file_quelevel'
        ]
        test_dataset = KTQueDataset(
            str(test_path),
            input_type=data_config['input_type'],
            folds={-1},
            concept_num=data_config['num_c'],
            max_concepts=data_config['max_concepts'],
        )
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        auc, acc = evaluate(net, loader, model_name)
        aggregation = 'direct_question_window_prediction'
    elif model_name in FORGET_QW_MODELS:
        test_path = Path(data_config['dpath']) / data_config[
            'test_question_window_file'
        ]
        test_dataset = DktForgetDataset(
            str(test_path), data_config['input_type'], {-1}, True
        )
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        aucs, accs = evaluate_question(
            net,
            loader,
            model_name,
            # The protocol reports late_mean only.  Computing early fusion
            # materializes an unused hidden-state tensor for every question
            # window and cannot alter the late-fusion result.
            fusion_type=['late_fusion'],
            # Do not retain per-row test predictions for formal reporting.
            save_path='',
        )
        if 'late_mean' not in aucs or 'late_mean' not in accs:
            raise RuntimeError(
                f'{model_name}/{params["dataset_name"]} did not produce '
                'question-window late_mean metrics'
            )
        auc, acc = aucs['late_mean'], accs['late_mean']
        aggregation = 'question_window_late_mean'
    elif model_name in STANDARD_QW_MODELS or model_name == 'ukt':
        test_path = Path(data_config['dpath']) / data_config[
            'test_question_window_file'
        ]
        if model_name == 'ukt':
            test_dataset = UKTDataset(
                str(test_path), data_config['input_type'], {-1},
                qtest=True, need_aug=False,
            )
        else:
            test_dataset = KTDataset(
                str(test_path), data_config['input_type'], {-1}, True
            )
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        aucs, accs = evaluate_question(
            net,
            loader,
            model_name,
            fusion_type=['late_fusion'],
            save_path='',
        )
        if 'late_mean' not in aucs or 'late_mean' not in accs:
            raise RuntimeError(
                f'{model_name}/{params["dataset_name"]} did not produce '
                'question-window late_mean metrics'
            )
        auc, acc = aucs['late_mean'], accs['late_mean']
        aggregation = 'question_window_late_mean'
    else:
        raise ValueError(f'Unsupported pyKT QW evaluator model: {model_name}')

    return {
        'auc': float(auc),
        'acc': float(acc),
        'input_file': str(test_path),
        'aggregation': aggregation,
        'history': 'question_window_fixed_length_200',
        'run_dir': str(run_dir),
    }


def attach_pykt_question_window_metrics(result, pykt_root):
    """Replace invalid native test fields with the formal QW report metric."""
    metric = evaluate_pykt_question_window(result, pykt_root)
    for key in ('test_auc', 'test_acc', 'window_test_auc', 'window_test_acc'):
        result.pop(key, None)
        result.pop(f'diagnostic_invalid_native_{key}', None)
    result.update({
        'test_auc': metric['auc'],
        'test_acc': metric['acc'],
        'question_window_test_auc': metric['auc'],
        'question_window_test_acc': metric['acc'],
        'run_dir': metric['run_dir'],
        'report_metric': {
            key: value for key, value in metric.items() if key != 'run_dir'
        },
    })
    return result


def remove_invalid_pykt_native_artifacts(run_dir):
    """Discard raw-sequence prediction exports that are not formal outputs."""
    for pattern in ('*_test_predictions.txt', '*_test_window_predictions.txt'):
        for path in Path(run_dir).glob(pattern):
            path.unlink()


def pykt_trial(config, model_name, model, dataset_name, dataset, params,
               directory, phase):
    pykt_root = Path(config['runtime']['pykt_root'])
    install_dkt_forget_input_compatibility(pykt_root)
    sys.path.insert(0, str(pykt_root / 'examples'))
    sys.path.insert(0, str(pykt_root))
    os.chdir(pykt_root / 'examples')
    from wandb_train import main as train_main

    resolved = dict(model.get('fixed_parameters', {}))
    resolved.update({key: value for key, value in params.items() if key != 'fold'})
    evaluate_test, evaluate_window_test = pykt_evaluation_flags(phase)
    resolved.update({
        'model_name': model['pykt_name'],
        'dataset_name': dataset['pykt_name'],
        'emb_type': model['emb_type'],
        'fold': int(params['fold']),
        'seed': int(params['seed']),
        'save_dir': str(directory / 'checkpoint'),
        'batch_size': int(
            model.get('batch_size', config['training']['pykt_batch_size'])
        ),
        'num_epochs': int(config['training']['num_epochs']),
        'use_wandb': 0,
        'add_uuid': 0,
        'save_model': 1,
        # pyKT's native test routes are always disabled.  The selected final
        # checkpoint is evaluated once afterwards on the formal QW route.
        'evaluate_test': evaluate_test,
        'evaluate_window_test': evaluate_window_test,
        'config': str(Path(config['runtime']['model_config']).resolve()),
        'data_config': str(Path(config['runtime']['data_config']).resolve()),
    })
    result = train_main(resolved)
    metrics = {
        'best_epoch': int(result['best_epoch']),
        'valid_auc': float(result['valid_auc']),
        'valid_acc': float(result['valid_acc']),
        'resolved_parameters': resolved,
    }
    if phase == 'final':
        # train_main was intentionally run with all native test evaluation
        # disabled.  The one formal test read is the QW evaluator below.
        metrics['run_dir'] = result['run_dir']
        attach_pykt_question_window_metrics(metrics, pykt_root)
    return metrics


def matra_resolved_config(config, model, dataset_name, params, phase, number):
    base = load_yaml(model['base_config'])
    candidate = base['screen']['candidates'][model['input_variant']]
    resolved = {
        'experiment': {
            **copy.deepcopy(base['experiment']),
            'name': config['experiment']['name'],
            'protocol': config['experiment']['protocol'],
            'input_variant': model['input_variant'],
            'dataset': dataset_name,
            'phase': phase,
            'trial': int(number),
            # The unified Bayes comparison reports the same question-window
            # late-mean metric as the DenoiseKT-style pyKT final evaluation.
            'test_metric': 'question_window_late_mean',
        },
        'model': copy.deepcopy(base['model']),
        'training': copy.deepcopy(base['training']),
        'data': {
            **copy.deepcopy(base['datasets'][dataset_name]),
            'test_question_window_file': 'test_question_window_sequences.csv',
        },
    }
    resolved['model'].update(candidate)
    resolved['model']['d_model'] = int(params['d_model'])
    resolved['model']['dropout'] = float(params['dropout'])
    resolved['model']['n_heads'] = int(params['num_attn_heads'])
    # DenoiseKT has one block count; tie both temporal depths to it rather
    # than introducing an additional architecture search dimension.
    resolved['model']['mamba_layers'] = int(params['n_blocks'])
    resolved['model']['n_layers'] = int(params['n_blocks'])
    resolved['training']['lr'] = float(params['learning_rate'])
    resolved['training']['seed'] = int(params['seed'])
    return resolved


def matra_command(config, directory, fold, phase):
    command = [
        str(Path(config['runtime']['python']).resolve()), '-u', str(MATRA_WORKER),
        '--config', str(directory / 'config.yaml'), '--fold', str(fold),
        '--output_dir', str(directory),
    ]
    if phase == 'search':
        command.append('--validation_only')
    return command


def matra_trial(config, model, dataset_name, params, directory, phase, number, gpu):
    resolved = matra_resolved_config(
        config, model, dataset_name, params, phase, number
    )
    write_yaml(directory / 'config.yaml', resolved)
    fold = int(params['fold'])
    history_path = directory / f'fold{fold}_history.json'
    command = matra_command(config, directory, fold, phase)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    environment['TRITON_CACHE_DIR'] = str(TRITON_CACHE_DIR)
    with (directory / 'train.log').open('w', encoding='utf-8') as handle:
        code = subprocess.run(
            command,
            cwd=HERE.parent,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code != 0 or not history_path.is_file():
        raise RuntimeError(
            f'MaTra4KT trial failed with exit code {code}: '
            f'{directory / "train.log"}'
        )
    history = json.loads(history_path.read_text(encoding='utf-8'))
    index = int(history['best_epoch']) - 1
    metrics = {
        'best_epoch': int(history['best_epoch']),
        'valid_auc': float(history['valid_auc'][index]),
        'valid_acc': float(history['valid_acc'][index]),
        'resolved_parameters': resolved,
    }
    if phase == 'final':
        metrics.update({
            'test_auc': float(history['test_auc'][-1]),
            'test_acc': float(history['test_acc'][-1]),
            'question_window_test_auc': float(history['test_auc'][-1]),
            'question_window_test_acc': float(history['test_acc'][-1]),
            'report_metric': {
                'auc': float(history['test_auc'][-1]),
                'acc': float(history['test_acc'][-1]),
                'input_file': str(
                    Path(resolved['data']['data_dir']) /
                    resolved['data']['test_question_window_file']
                ),
                'aggregation': 'question_window_late_mean',
                'history': 'question_window_fixed_length_200',
            },
        })
    return metrics


def run_trial(config_path, config, model_name, model, dataset_name, dataset,
              params, directory, phase, number, gpu):
    if phase not in {'search', 'final'}:
        raise ValueError(f'Unsupported trial phase: {phase}')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'params.json', {
        'phase': phase,
        'model': model_name,
        'dataset': dataset_name,
        'params': params,
        'selected_by': 'valid_auc',
        'test_evaluation_enabled': phase == 'final',
    })
    if model['backend'] == 'pykt':
        result = pykt_trial(
            config, model_name, model, dataset_name, dataset, params, directory,
            phase,
        )
    elif model['backend'] == 'matra4kt':
        result = matra_trial(
            config, model, dataset_name, params, directory, phase, number, gpu
        )
    else:
        raise ValueError(f'Unsupported backend: {model["backend"]}')
    result.update({
        'model': model_name,
        'dataset': dataset_name,
        'fold': int(params['fold']),
        'phase': phase,
        'directory': str(directory),
    })
    write_json(directory / 'trial_result.json', result)
    if phase == 'search':
        shutil.rmtree(directory / 'checkpoint', ignore_errors=True)
        checkpoint = directory / f'fold{params["fold"]}_best.pt'
        if checkpoint.is_file():
            checkpoint.unlink()
    return result


def study_payload(study, config):
    complete = complete_trials(study)
    raw_complete = raw_complete_trials(study)
    by_fold = defaultdict(int)
    for trial in complete:
        by_fold[str(trial.params['fold'])] += 1
    best = max(complete, key=lambda trial: float(trial.value)) if complete else None
    selection_metric = config['experiment'].get(
        'selection_metric', 'valid_auc'
    )
    return {
        'completed_trials': len(complete),
        'raw_completed_trials': len(raw_complete),
        'duplicate_completed_trials': len(raw_complete) - len(complete),
        'failed_trials': sum(
            trial.state == optuna.trial.TrialState.FAIL for trial in study.trials
        ),
        'trials_by_fold': dict(by_fold),
        'official_stop_reached': stop_reached(study, config['search']),
        'best_trial': ({
            'number': best.number,
            'objective_name': selection_metric,
            'objective_value': best.value,
            'valid_auc': best.value,
            'params': best.params,
        } if best else None),
    }


def final_summary(final_results, selection_metric='valid_auc'):
    auc = np.asarray([
        item['question_window_test_auc'] for item in final_results
    ], dtype=float)
    acc = np.asarray([
        item['question_window_test_acc'] for item in final_results
    ], dtype=float)
    return {
        'completed_folds': len(final_results),
        'test_auc_mean': float(auc.mean()),
        'test_auc_std': float(auc.std(ddof=0)),
        'test_acc_mean': float(acc.mean()),
        'test_acc_std': float(acc.std(ddof=0)),
        'selected_by': selection_metric,
        'test_metric': 'question_window_late_mean_test_auc',
    }


def restore_final_question_window_results(config, model_name, model, datasets):
    """Re-evaluate saved final pyKT checkpoints without any training or HPO."""
    if model['backend'] != 'pykt':
        raise ValueError('--reevaluate-final-qw only applies to pyKT models')
    root = Path(model['output_root']).resolve()
    summaries = {}
    pykt_root = Path(config['runtime']['pykt_root'])
    for dataset_name, dataset in datasets.items():
        final_results = []
        for fold in dataset.get('final_folds', dataset['folds']):
            result_path = (
                root / dataset_name / 'selected_final' / f'fold{fold}' /
                'trial_result.json'
            )
            if not result_path.is_file():
                raise FileNotFoundError(
                    f'No completed selected final checkpoint for '
                    f'{model_name}/{dataset_name}/fold{fold}: {result_path}'
                )
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if result.get('phase') != 'final':
                raise RuntimeError(f'Unexpected non-final result: {result_path}')
            attach_pykt_question_window_metrics(result, pykt_root)
            remove_invalid_pykt_native_artifacts(result['run_dir'])
            write_json(result_path, result)
            final_results.append(result)
        summaries[dataset_name] = final_summary(
            final_results,
            config['experiment'].get('selection_metric', 'valid_auc'),
        )

    merge_json_object(root / 'summary.json', summaries)
    write_json(root / 'question_window_re_evaluation.json', {
        'model': model_name,
        'datasets': summaries,
        'selection_metric': config['experiment'].get(
            'selection_metric', 'valid_auc'
        ),
        'test_metric': config['experiment'].get(
            'report_metric', 'question_window_late_mean_test_auc'
        ),
        'training_or_search_rerun': False,
    })
    return summaries


def select_trials_for_final_folds(study, dataset):
    """Map each final fold to a validation-selected search trial."""
    grouped = defaultdict(list)
    for trial in complete_trials(study):
        grouped[int(trial.params['fold'])].append(trial)

    final_folds = [int(fold) for fold in dataset.get(
        'final_folds', dataset['folds']
    )]
    selection_fold = dataset.get('fixed_search_fold')
    if selection_fold is not None:
        selection_fold = int(selection_fold)
        if not grouped[selection_fold]:
            raise RuntimeError(
                f'No completed search trials for selection fold {selection_fold}'
            )
        best = max(
            grouped[selection_fold], key=lambda trial: float(trial.value)
        )
        return {fold: best for fold in final_folds}

    missing = sorted(set(final_folds) - set(grouped))
    if missing:
        raise RuntimeError(f'Sampled no trials for folds {missing}')
    return {
        fold: max(grouped[fold], key=lambda trial: float(trial.value))
        for fold in final_folds
    }


def finalize_dataset(config_path, config, model_name, model, dataset_name,
                     dataset, study, gpu):
    root = Path(model['output_root']).resolve()
    selected_trials = select_trials_for_final_folds(study, dataset)
    selected = {}
    final_results = []
    for fold, best in selected_trials.items():
        final_params = dict(best.params)
        final_params['fold'] = int(fold)
        final_dir = root / dataset_name / 'selected_final' / f'fold{fold}'
        result_path = final_dir / 'trial_result.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding='utf-8'))
        else:
            result = run_trial(
                config_path, config, model_name, model, dataset_name, dataset,
                final_params, final_dir, 'final', best.number, gpu
            )
        if model['backend'] == 'pykt' and 'question_window_test_auc' not in result:
            attach_pykt_question_window_metrics(
                result, Path(config['runtime']['pykt_root'])
            )
            remove_invalid_pykt_native_artifacts(result['run_dir'])
            write_json(result_path, result)
        elif model['backend'] == 'matra4kt' and 'question_window_test_auc' not in result:
            result['question_window_test_auc'] = result['test_auc']
            result['question_window_test_acc'] = result['test_acc']
            result['report_metric'] = {
                'auc': result['test_auc'],
                'acc': result['test_acc'],
                'aggregation': 'question_window_late_mean',
                'history': 'question_window_fixed_length_200',
            }
            write_json(result_path, result)
        selected[str(fold)] = {
            'trial': best.number,
            'selection_fold': int(best.params['fold']),
            'objective_name': config['experiment'].get(
                'selection_metric', 'valid_auc'
            ),
            'objective_value': float(best.value),
            'valid_auc': float(best.value),
            'params': final_params,
        }
        final_results.append(result)
    return selected, final_summary(
        final_results,
        config['experiment'].get('selection_metric', 'valid_auc'),
    )


def check_inputs(model, dataset):
    missing = [
        name for name in model.get('required_files', [])
        if not (Path(dataset['data_dir']) / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'{dataset["pykt_name"]} missing required files: {missing}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument(
        '--datasets', nargs='+', metavar='DATASET',
        help='Resume only these model datasets; completed trials remain in place.',
    )
    parser.add_argument(
        '--search-only', action='store_true',
        help='Run validation-only search without final test evaluation.',
    )
    parser.add_argument(
        '--concurrent-worker', action='store_true',
        help=(
            'Join an active shared study without recovering RUNNING trials. '
            'Requires --search-only so only one coordinator can run final evaluation.'
        ),
    )
    parser.add_argument(
        '--worker-id', type=int, default=0,
        help='Non-negative worker identifier used to decorrelate concurrent TPE samplers.',
    )
    parser.add_argument(
        '--max-new-trials', type=int,
        help=(
            'Bound new search trials per selected dataset for smoke tests or '
            'scheduled slices. Requires --search-only.'
        ),
    )
    parser.add_argument(
        '--reevaluate-final-qw', action='store_true',
        help=(
            'Re-evaluate completed selected pyKT checkpoints with the formal '
            'question-window metric; does not train or run Optuna.'
        ),
    )
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.concurrent_worker and not args.search_only:
        parser.error('--concurrent-worker requires --search-only')
    if args.max_new_trials is not None:
        if not args.search_only:
            parser.error('--max-new-trials requires --search-only')
        if args.max_new_trials <= 0:
            parser.error('--max-new-trials must be positive')
    if args.worker_id < 0:
        parser.error('--worker-id must be non-negative')
    config = load_yaml(args.config)
    if args.model not in config['models']:
        parser.error(
            f'unknown model {args.model!r}; choose from '
            f'{", ".join(config["models"])}'
        )
    model = config['models'][args.model]
    if model['backend'] == 'pykt':
        # pyKT is imported lazily below, so bind its CUDA device before torch
        # can initialize.  This keeps --gpu from silently falling back to GPU0.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    root = Path(model['output_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_datasets = list(model['datasets'])
    requested = set(args.datasets or model_datasets)
    unknown = requested - set(model_datasets)
    if unknown:
        raise ValueError(
            f'{args.model} does not support requested datasets: {sorted(unknown)}'
        )
    selected_names = [name for name in model_datasets if name in requested]
    all_datasets = {
        name: config['datasets'][name]
        for name in model_datasets
    }
    datasets = {
        name: all_datasets[name]
        for name in selected_names
    }
    for dataset in datasets.values():
        check_inputs(model, dataset)
    if model['backend'] == 'pykt':
        effective_data_config = materialize_pykt_data_config(
            config,
            all_datasets,
            root / 'effective_data_config.json',
        )
        config['runtime']['data_config'] = str(effective_data_config)
    if not args.concurrent_worker:
        fixed_folds = {
            name: dataset.get('fixed_search_fold')
            for name, dataset in datasets.items()
        }
        write_json(root / 'manifest.json', {
            'experiment': config['experiment'],
            'model': args.model,
            'datasets': all_datasets,
            'active_datasets': selected_names,
            'search': config['search'],
            'model_search_parameters': model.get('search_parameters', {}),
            'search_parameter_names': model['sampled_parameters'],
            'fixed_parameters': model.get('fixed_parameters', {}),
            'fold_policy': (
                'fixed_search_fold'
                if all(fold is not None for fold in fixed_folds.values())
                else 'sampled_inside_one_global_study'
            ),
            'fixed_search_folds': fixed_folds,
            'effective_data_config': config['runtime']['data_config'],
        })
    if args.prepare_only:
        print(f'Prepared unified Bayes: {args.model} -> {root}')
        return

    if args.reevaluate_final_qw:
        if args.search_only:
            raise ValueError('--search-only cannot be combined with --reevaluate-final-qw')
        summaries = restore_final_question_window_results(
            config, args.model, model, datasets
        )
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    studies = {}
    for dataset_name, dataset in datasets.items():
        # Keep a dataset's TPE seed stable when work is resharded across GPUs.
        index = model_datasets.index(dataset_name)
        (root / dataset_name).mkdir(parents=True, exist_ok=True)
        storage = f'sqlite:///{root / dataset_name / "study.sqlite3"}'
        base_sampler = optuna.samplers.TPESampler(
            seed=(
                int(config['search']['sampler_seed'])
                + index
                + (1009 * args.worker_id if args.concurrent_worker else 0)
            ),
            n_startup_trials=int(config['search']['startup_trials']),
        )
        fixed_search_fold = dataset.get('fixed_search_fold')
        sampler = (
            optuna.samplers.PartialFixedSampler(
                {'fold': int(fixed_search_fold)}, base_sampler
            )
            if fixed_search_fold is not None else base_sampler
        )
        study = optuna.create_study(
            study_name=f'{config["experiment"]["name"]}__{args.model}__{dataset_name}',
            direction='maximize',
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
        if not args.concurrent_worker:
            recover_stale_trials(study)
        assert_fixed_fold_history(study, dataset.get('fixed_search_fold'))
        studies[dataset_name] = study

    for dataset_name, study in studies.items():
        dataset = datasets[dataset_name]
        fixed_search_fold = dataset.get('fixed_search_fold')
        consecutive_failures = 0
        new_trials = 0
        while (
            not stop_reached(study, config['search'])
            and (
                args.max_new_trials is None
                or new_trials < args.max_new_trials
            )
        ):
            current_trial_number = None

            def objective(trial):
                nonlocal current_trial_number
                current_trial_number = trial.number
                sample_lock = root / dataset_name / 'parameter_sampling.lock'
                with exclusive_sample_lock(sample_lock):
                    params = sample_params(trial, model, dataset, config)
                    duplicate = earlier_duplicate_trial(
                        study, trial.number, params
                    )
                    if duplicate is not None:
                        trial.set_user_attr('duplicate_of', duplicate.number)
                        raise optuna.TrialPruned(
                            f'duplicate parameters from trial {duplicate.number}'
                        )
                directory = trial_directory(root, dataset_name, trial.number)
                result = run_trial(
                    args.config, config, args.model, model, dataset_name, dataset,
                    params, directory, 'search', trial.number, args.gpu
                )
                trial.set_user_attr('fold', int(params['fold']))
                trial.set_user_attr('best_epoch', result['best_epoch'])
                trial.set_user_attr('valid_acc', result['valid_acc'])
                return result['valid_auc']
            if fixed_search_fold is not None:
                if int(fixed_search_fold) not in dataset['folds']:
                    raise ValueError(
                        f'{dataset_name} fixed_search_fold={fixed_search_fold} '
                        f'is not in folds={dataset["folds"]}'
                    )
            study.optimize(objective, n_trials=1, gc_after_trial=True, catch=(Exception,))
            new_trials += 1
            latest = study.trials[current_trial_number]
            consecutive_failures = (
                consecutive_failures + 1
                if latest.state == optuna.trial.TrialState.FAIL else 0
            )
            if consecutive_failures >= 3:
                raise RuntimeError(
                    f'{args.model}/{dataset_name} stopped after '
                    f'{consecutive_failures} consecutive failed trials; '
                    'inspect the latest train.log before resuming.'
                )
            if not args.concurrent_worker:
                write_json(root / 'progress.json', {
                    name: study_payload(item, config)
                    for name, item in studies.items()
                })

        # Publish each dataset as soon as its search completes.  Long campaigns
        # can then yield usable final results without waiting for slower datasets.
        if not args.search_only:
            dataset_selected, dataset_summary = finalize_dataset(
                args.config, config, args.model, model, dataset_name,
                dataset, study, args.gpu
            )
            merge_json_object(
                root / 'selected_validation_configs.json',
                {dataset_name: dataset_selected},
            )
            merge_json_object(
                root / 'summary.json',
                {dataset_name: dataset_summary},
            )

    if args.search_only:
        return


if __name__ == '__main__':
    main()
