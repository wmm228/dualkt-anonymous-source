#!/usr/bin/env python3
"""Run one promoted MaTra4KT architecture on four sealed test sets."""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

V8_DIR = Path(__file__).resolve().parent
REPO_ROOT = V8_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from V8.run_question_semantic_generalization import load_screen


WORKER = V8_DIR / 'train_denoisekt_xes.py'
DEFAULT_CONFIG = V8_DIR / 'config_single_model_4dataset_5fold.yaml'


def load_config(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def materialize_jobs(config):
    screen_path = Path(config['experiment']['screen_config']).resolve()
    screen = load_screen(screen_path)
    candidate = config['experiment']['candidate']
    if candidate not in screen['screen']['candidates']:
        raise ValueError(f'candidate is not registered by screen: {candidate}')
    overrides = screen['screen']['candidates'][candidate]
    selected_model = config.get('model_overrides', {})
    selected_training = config.get('training_overrides', {})
    data_overrides = config.get('data_overrides', {})
    output_root = Path(config['output_root']).resolve()
    config_root = output_root / 'resolved_configs'
    config_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, data in screen['datasets'].items():
        resolved = {
            'experiment': {
                **screen['experiment'],
                'name': config['experiment']['name'],
                'protocol': config['experiment']['protocol'],
                'input_variant': candidate,
                'dataset': dataset,
                'test_metric': config['experiment'].get(
                    'test_metric', 'question_window_late_mean'
                ),
            },
            'model': copy.deepcopy(screen['model']),
            'training': copy.deepcopy(screen['training']),
            'data': {
                **copy.deepcopy(data),
                **data_overrides,
            },
            'output_root': str(output_root / dataset),
        }
        resolved['model'].update(overrides)
        # Bayesian search selects these values on validation only.  Apply them
        # after the registered input variant so this sealed-test runner can
        # retrain the chosen architecture without exposing test metrics to
        # selection.
        resolved['model'].update(selected_model)
        resolved['training'].update(selected_training)
        config_path = config_root / f'{dataset}.yaml'
        with config_path.open('w', encoding='utf-8') as handle:
            yaml.safe_dump(
                resolved, handle, sort_keys=False, allow_unicode=True
            )
        run_folds = config['experiment'].get('run_folds', data['folds'])
        unknown_folds = sorted(set(run_folds) - set(data['folds']))
        if unknown_folds:
            raise ValueError(
                f'{dataset} run_folds are outside the data folds: {unknown_folds}'
            )
        for fold in run_folds:
            jobs.append({
                'dataset': dataset,
                'fold': int(fold),
                'config': config_path,
                'run_dir': output_root / dataset / f'fold{fold}',
            })
    manifest = output_root / 'formal_manifest.json'
    manifest.write_text(
        json.dumps([
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in job.items()
            }
            for job in jobs
        ], indent=2) + '\n',
        encoding='utf-8',
    )
    return screen, jobs, manifest


def history_path(job):
    return job['run_dir'] / f'fold{job["fold"]}_history.json'


def require_screen_acceptance(config, screen):
    if not config['experiment'].get('require_screen_acceptance', True):
        return
    decision_path = (
        Path(screen['output_root']).resolve()
        / 'generalization_decision.json'
    )
    if not decision_path.is_file():
        raise RuntimeError(
            f'validation screen is incomplete: {decision_path}'
        )
    decisions = json.loads(decision_path.read_text(encoding='utf-8'))
    candidate = config['experiment']['candidate']
    fold = int(screen['screen']['fold'])
    baseline = screen['screen'].get(
        'baseline_candidate', 'shared_prefix_baseline'
    )
    missing_evidence = []
    for dataset in screen['datasets']:
        for name in {baseline, candidate}:
            history = (
                Path(screen['output_root']).resolve()
                / dataset
                / name
                / f'fold{fold}_history.json'
            )
            if not history.is_file():
                missing_evidence.append(str(history))
    if missing_evidence:
        raise RuntimeError(
            'validation decision is stale or incomplete; missing histories: '
            + ', '.join(missing_evidence)
        )
    if not decisions.get(candidate, {}).get('accepted', False):
        raise RuntimeError(
            f'candidate did not pass the four-dataset screen: {candidate}'
        )


def run_jobs(config, jobs, gpu_ids):
    pending = [job for job in jobs if not history_path(job).is_file()]
    print(f'skipping {len(jobs) - len(pending)} completed MaTra4KT folds')
    active = {}
    python = str(Path(config['runtime']['python']).resolve())
    while pending or active:
        for gpu in gpu_ids:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            job['run_dir'].mkdir(parents=True, exist_ok=True)
            log_handle = (job['run_dir'] / 'train.log').open(
                'w', encoding='utf-8'
            )
            command = [
                python,
                '-u',
                str(WORKER),
                '--config',
                str(job['config']),
                '--fold',
                str(job['fold']),
                '--output_dir',
                str(job['run_dir']),
            ]
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=V8_DIR.parent,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, job, log_handle)
            print(
                f'gpu={gpu} started {job["dataset"]}/fold{job["fold"]} '
                f'pid={process.pid}'
            )
        if active:
            time.sleep(5)
        failures = []
        for gpu, (process, job, log_handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[gpu]
            if return_code != 0:
                failures.append((job, return_code))
            else:
                print(f'completed {job["dataset"]}/fold{job["fold"]}')
        if failures:
            for process, _, log_handle in active.values():
                process.terminate()
                log_handle.close()
            details = ', '.join(
                f'{job["dataset"]}/fold{job["fold"]}:exit={code}'
                for job, code in failures
            )
            raise RuntimeError(f'MaTra4KT jobs failed: {details}')


def load_local_denoisekt(config):
    path = Path(config['denoisekt_reproduction_summary'])
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(config, jobs):
    output_root = Path(config['output_root']).resolve()
    fold_rows = []
    for job in jobs:
        path = history_path(job)
        if not path.is_file():
            continue
        history = json.loads(path.read_text(encoding='utf-8'))
        best_epoch = int(history['best_epoch'])
        index = best_epoch - 1
        fold_rows.append({
            'dataset': job['dataset'],
            'fold': job['fold'],
            'best_epoch': best_epoch,
            'valid_auc': float(history['valid_auc'][index]),
            'valid_acc': float(history['valid_acc'][index]),
            'test_auc': float(history['test_auc'][-1]),
            'test_acc': float(history['test_acc'][-1]),
        })
    if fold_rows:
        with (output_root / 'fold_results.csv').open(
            'w', encoding='utf-8', newline=''
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fold_rows[0])
            )
            writer.writeheader()
            writer.writerows(fold_rows)

    local = load_local_denoisekt(config)
    minimum = float(config['targets']['minimum_auc_delta_vs_denoisekt'])
    ideal = float(config['targets']['ideal_auc_delta_vs_denoisekt'])
    summary = {}
    datasets = sorted({job['dataset'] for job in jobs})
    for dataset in datasets:
        rows = [row for row in fold_rows if row['dataset'] == dataset]
        if not rows:
            continue
        expected_folds = sum(
            job['dataset'] == dataset for job in jobs
        )
        complete = len(rows) == expected_folds
        auc = np.asarray([row['test_auc'] for row in rows])
        acc = np.asarray([row['test_acc'] for row in rows])
        paper = config.get('paper_reference', {}).get(dataset)
        local_auc = local.get(dataset, {}).get('test_auc_mean')
        local_acc = local.get(dataset, {}).get('test_acc_mean')
        primary_auc = float(local_auc) if local_auc is not None else (
            float(paper['auc']) if paper is not None else None
        )
        primary_acc = float(local_acc) if local_acc is not None else (
            float(paper['acc']) if paper is not None else None
        )
        primary_source = (
            'local_reproduction' if local_auc is not None else
            ('paper' if paper is not None else None)
        )
        delta = (
            float(auc.mean() - primary_auc)
            if primary_auc is not None else None
        )
        acc_delta = (
            float(acc.mean() - primary_acc)
            if primary_acc is not None else None
        )
        summary[dataset] = {
            'completed_folds': len(rows),
            'expected_folds': expected_folds,
            'complete': complete,
            'test_auc_mean': float(auc.mean()),
            'test_auc_std': float(auc.std(ddof=0)),
            'test_acc_mean': float(acc.mean()),
            'test_acc_std': float(acc.std(ddof=0)),
            'denoisekt_primary_source': primary_source,
            'denoisekt_primary_auc': primary_auc,
            'denoisekt_primary_acc': primary_acc,
            'auc_delta_vs_denoisekt': delta,
            'acc_delta_vs_denoisekt': acc_delta,
            'minimum_target_met': (
                complete and delta is not None and delta >= minimum
            ),
            'ideal_target_met': (
                complete and delta is not None and delta >= ideal
            ),
            'paper_auc': float(paper['auc']) if paper is not None else None,
            'paper_acc': float(paper['acc']) if paper is not None else None,
            'auc_delta_vs_paper': (
                float(auc.mean() - paper['auc']) if paper is not None else None
            ),
            'acc_delta_vs_paper': (
                float(acc.mean() - paper['acc']) if paper is not None else None
            ),
            'local_denoisekt_auc': local_auc,
            'local_denoisekt_acc': local_acc,
        }
    (output_root / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--gpus', default='')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    screen, jobs, manifest = materialize_jobs(config)
    print(f'prepared {len(jobs)} MaTra4KT folds: {manifest}')
    if args.summarize_only:
        summarize(config, jobs)
        return
    if args.prepare_only:
        return
    require_screen_acceptance(config, screen)
    gpu_ids = (
        [int(value) for value in args.gpus.split(',') if value.strip()]
        if args.gpus else [int(value) for value in config['runtime']['gpus']]
    )
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError('CUDA is unavailable; no MaTra4KT jobs were started')
    if any(gpu < 0 or gpu >= available for gpu in gpu_ids):
        raise ValueError(f'requested GPUs {gpu_ids}, available count is {available}')
    run_jobs(config, jobs, gpu_ids)
    summarize(config, jobs)


if __name__ == '__main__':
    main()
