#!/usr/bin/env python3
"""Train one MaTra4KT fold with the shared DenoiseKT input contract."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


V8_DIR = Path(__file__).resolve().parent
KT_ROOT = V8_DIR.parent
PYKT_ROOT = Path(
    os.environ.get('PYKT_ROOT', KT_ROOT / 'third_party' / 'pykt-toolkit')
).expanduser().resolve()
TRITON_CACHE_DIR = Path(
    os.environ.get('TRITON_CACHE_DIR', Path.home() / '.cache' / 'triton')
).expanduser().resolve()
TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR', str(TRITON_CACHE_DIR))
for import_root in (KT_ROOT, PYKT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from V7.trainer import KTTrainer  # noqa: E402
from V8.model import ModularEvidenceKT  # noqa: E402
from V8.denoisekt_xes_protocol import (  # noqa: E402
    IndexedDataset,
    QuestionWindowIndexedDataset,
    QuestionLevelCollator,
    evaluate_question_window_late_mean,
    load_question_structure,
    load_question_structure_from_metadata,
)
from pykt.datasets import que_data_loader  # noqa: E402
from pykt.datasets.que_data_loader import KTQueDataset  # noqa: E402


def load_config(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_integral_sequence_value(value):
    """Accept pyKT splitter timestamps serialized as integer-valued decimals."""
    try:
        return int(value)
    except (TypeError, ValueError):
        parsed = float(value)
        if not parsed.is_integer():
            raise ValueError(f'Expected an integer-valued sequence item, got {value!r}')
    return int(parsed)


def disable_pykt_pickle_cache():
    """Avoid multi-gigabyte derived caches in long-sequence stress tests."""
    original_pickle = pd.to_pickle
    original_exists = os.path.exists

    def guarded_to_pickle(obj, path, *args, **kwargs):
        value = str(path)
        if ".csv_" in value and value.endswith(".pkl"):
            return None
        return original_pickle(obj, path, *args, **kwargs)

    def guarded_exists(path):
        value = str(path)
        if ".csv_" in value and value.endswith(".pkl"):
            return False
        return original_exists(path)

    pd.to_pickle = guarded_to_pickle
    os.path.exists = guarded_exists


def build_dataset(config, path, folds, question_window=False):
    # ASSIST2017's official splitter emits values such as "29000.0" in
    # `usetimes`; patch only this worker process instead of rewriting the CSV.
    que_data_loader.int = parse_integral_sequence_value
    dataset = KTQueDataset(
        str(path),
        input_type=['questions', 'concepts'],
        folds=set(folds),
        concept_num=int(config['data']['num_concepts']),
        max_concepts=int(config['data']['max_concepts']),
    )
    if question_window:
        return QuestionWindowIndexedDataset(dataset, path, folds)
    return IndexedDataset(dataset)


def build_model(config, structure):
    model_config = config['model']
    data_config = config['data']
    use_question_context = bool(model_config['use_question_context'])
    model = ModularEvidenceKT(
        n_questions=int(data_config['num_questions']) + 1,
        n_concepts=int(data_config['num_concepts']) + 1,
        d_model=int(model_config['d_model']),
        d_state=int(model_config['d_state']),
        d_conv=int(model_config['d_conv']),
        expand=int(model_config['expand']),
        dropout=float(model_config['dropout']),
        task_mode='concept',
        mamba_version=model_config['mamba_version'],
        mamba_layers=int(model_config['mamba_layers']),
        temporal_backbone=model_config['temporal_backbone'],
        n_heads=int(model_config['n_heads']),
        n_layers=int(model_config['n_layers']),
        short_window=int(model_config['short_window']),
        summary_block_size=int(model_config['summary_block_size']),
        max_seq_len=int(config['training']['max_seq_len']),
        short_memory_mode=model_config['short_memory_mode'],
        branch_fusion=model_config['branch_fusion'],
        short_memory_source=model_config['short_memory_source'],
        mamba_interaction=model_config['mamba_interaction'],
        dynamics_mode=model_config['dynamics_mode'],
        use_difficulty=bool(model_config['use_difficulty']),
        use_student_graph=bool(model_config['use_student_graph']),
        use_population_graph=bool(model_config['use_population_graph']),
        use_mastery=bool(model_config['use_mastery']),
        use_concept_memory=bool(model_config['use_concept_memory']),
        use_transition_graph=bool(model_config['use_transition_graph']),
        transition_aggregation=str(model_config['transition_aggregation']),
        transition_decay=float(model_config['transition_decay']),
        use_target_retrieval=bool(model_config['use_target_retrieval']),
        ability_mode=str(model_config['ability_mode']),
        evidence_placement=str(model_config['evidence_placement']),
        target_conditioned_readout=bool(
            model_config['target_conditioned_readout']
        ),
        use_question_context=use_question_context,
        use_question_rasch=bool(
            model_config.get('use_question_rasch', False)
        ),
        use_bundle_attention_bias=bool(
            model_config.get('use_bundle_attention_bias', False)
        ),
        bundle_attention_strength=float(
            model_config.get('bundle_attention_strength', 0.5)
        ),
        bundle_attention_decay=float(
            model_config.get('bundle_attention_decay', 0.9)
        ),
        question_graph=(
            structure['question_graph'] if use_question_context else None
        ),
        question_concept_incidence=(
            structure['question_concept_incidence']
            if use_question_context else None
        ),
        concept_question_incidence=(
            structure['concept_question_incidence']
            if use_question_context else None
        ),
    )
    mamba_layers = (
        0 if model.sequence_encoder is None
        else len(model.sequence_encoder.layers)
    )
    transformer_layers = (
        0 if model.short_term is None else len(model.short_term.layers)
    )
    expected_mamba_layers = (
        int(model_config['mamba_layers'])
        if model_config['temporal_backbone'] in {'mamba', 'mamba_transformer'}
        else 0
    )
    expected_transformer_layers = (
        int(model_config['n_layers'])
        if model_config['temporal_backbone'] in {'transformer', 'mamba_transformer'}
        else 0
    )
    if mamba_layers != expected_mamba_layers:
        raise RuntimeError('Mamba depth does not match the resolved config')
    if transformer_layers != expected_transformer_layers:
        raise RuntimeError('Transformer depth does not match the resolved config')
    variant = config['experiment']['input_variant']
    expected = {
        'dgmkt_ks_only': (
            False, False, False, False,
            'adaptive_orthogonal_vector',
        ),
        'denoisekt_qid_multikc': (
            True, False, False, False,
            'adaptive_orthogonal_vector',
        ),
        'matra_prefix_transition': (
            True, True, False, False,
            'adaptive_orthogonal_vector',
        ),
        'shared_prefix_baseline': (
            True, True, False, False,
            'adaptive_orthogonal_vector',
        ),
        'shared_prefix_rasch': (
            True, True, True, False,
            'adaptive_orthogonal_vector',
        ),
        'shared_prefix_bundle_attention': (
            True, True, False, True,
            'adaptive_orthogonal_vector',
        ),
        'shared_prefix_rasch_bundle': (
            True, True, True, True,
            'adaptive_orthogonal_vector',
        ),
        'shared_no_transition': (
            True, False, False, False,
            'adaptive_orthogonal_vector',
        ),
        'single_current_fusion': (
            True, True, True, True,
            'adaptive_orthogonal_vector',
        ),
        'single_denoised_orthogonal_fusion': (
            True, True, True, True,
            'denoised_adaptive_orthogonal_vector',
        ),
        'ablation_mamba_only': (
            True, True, True, True,
            'long_only',
        ),
        'ablation_transformer_only': (
            True, True, True, True,
            'short_only',
        ),
        'ablation_mean_fusion': (
            True, True, True, True,
            'mean',
        ),
        'ablation_orthogonal_fusion': (
            True, True, True, True,
            'adaptive_orthogonal_vector',
        ),
        'ablation_no_orthogonal_decomposition': (
            True, True, True, True,
            'denoised_adaptive_raw_short_vector',
        ),
        'ablation_no_orthogonal_fusion': (
            True, True, True, True,
            'fixed_mean_no_orthogonal_fusion',
        ),
        'ablation_concat_linear_fusion': (
            True, True, True, True,
            'concat_linear_no_orthogonal_fusion',
        ),
        'exploratory_normalized_weighted_sum_fusion': (
            True, True, True, True,
            'normalized_channel_weighted_sum',
        ),
        'hard_ablation_mamba_only': (
            True, True, True, False, None,
        ),
        'hard_ablation_transformer_only': (
            True, True, True, True, None,
        ),
        'ablation_no_question_graph_context': (
            False, True, False, True,
            'denoised_adaptive_orthogonal_vector',
        ),
        'ablation_no_transition_graph': (
            True, False, True, True,
            'denoised_adaptive_orthogonal_vector',
        ),
        'ablation_no_question_rasch': (
            True, True, False, True,
            'denoised_adaptive_orthogonal_vector',
        ),
    }
    if variant not in expected:
        raise ValueError(f'unsupported input variant: {variant}')
    actual = (
        model.use_question_context,
        model.use_transition_graph,
        model.use_question_rasch,
        model.use_bundle_attention_bias,
        None if model.branch_fusion is None else model.branch_fusion.mode,
    )
    if actual != expected[variant]:
        raise RuntimeError(
            f'input variant {variant} expects {expected[variant]}, got {actual}'
        )
    return model


def install_finite_forward_hooks(model):
    """Fail at the first module that emits a non-finite tensor."""
    def tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)

    def make_hook(name):
        def check_output(module, inputs, output):
            del inputs
            if any(not torch.isfinite(value).all() for value in tensors(output)):
                raise FloatingPointError(
                    f'non-finite forward output from {name} '
                    f'({type(module).__name__})'
                )
        return check_output

    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(make_hook(name))


def assert_formal_cuda(model):
    if not torch.cuda.is_available():
        raise RuntimeError('formal training requires CUDA')
    if model.sequence_encoder is None:
        return
    missing = [
        index for index, layer in enumerate(model.sequence_encoder.layers)
        if layer.cuda_layer is None
    ]
    if missing:
        raise RuntimeError(
            f'Mamba2 CUDA implementation is unavailable for layers {missing}; '
            'refusing the GRU fallback in a formal run'
        )


def load_recoverable_training(checkpoint_path, progress_path, config, fold):
    """Load a fully trained fold for test-only recovery."""
    checkpoint_path = Path(checkpoint_path)
    progress_path = Path(progress_path)
    if not checkpoint_path.is_file() or not progress_path.is_file():
        raise FileNotFoundError('test-only recovery requires checkpoint and progress files')
    progress = json.loads(progress_path.read_text(encoding='utf-8'))
    completed_epochs = int(progress.get('completed_epochs') or 0)
    early_stop_counter = int(progress.get('early_stop_counter') or 0)
    patience = int(config['training']['patience'])
    epochs = int(config['training']['epochs'])
    if completed_epochs < epochs and early_stop_counter < patience:
        raise RuntimeError(
            'refusing test-only recovery because training did not finish its '
            'epoch budget or early-stopping patience'
        )
    history = progress.get('history')
    if not isinstance(history, dict) or not history.get('best_epoch'):
        raise RuntimeError('recoverable progress is missing its best epoch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    expected = {
        'fold': int(fold),
        'protocol': config['experiment']['protocol'],
        'input_variant': config['experiment']['input_variant'],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f'checkpoint {key} mismatch: {checkpoint.get(key)!r} != {value!r}'
            )
    recovered_history = dict(history)
    recovered_history['test_auc'] = []
    recovered_history['test_acc'] = []
    recovered_history.pop('test_metric', None)
    recovered_history.pop('wall_time_seconds', None)
    recovered_history.pop('peak_gpu_memory_mb', None)
    return checkpoint, progress, recovered_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('--prepare_test_only', action='store_true')
    parser.add_argument('--validation_only', action='store_true')
    parser.add_argument('--recover_test_only', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    if config['training'].get('disable_pykt_pickle_cache', False):
        disable_pykt_pickle_cache()
    data_dir = Path(config['data']['data_dir'])
    train_valid_path = data_dir / config['data']['train_valid_file']
    test_path = data_dir / config['data']['test_file']
    test_metric = config['experiment'].get(
        'test_metric', 'question_window_late_mean'
    )
    if test_metric not in {'pykt_sequence', 'question_window_late_mean'}:
        raise ValueError(
            'experiment.test_metric must be pykt_sequence or '
            'question_window_late_mean'
        )
    question_window_path = data_dir / config['data'].get(
        'test_question_window_file', config['data']['test_file']
    )
    metadata_files = config['data'].get('structure_metadata_files')
    if metadata_files:
        structure = load_question_structure_from_metadata(
            [data_dir / name for name in metadata_files],
            int(config['data']['num_questions']),
            int(config['data']['num_concepts']),
        )
    else:
        structure = load_question_structure(
            data_dir / config['data']['qmatrix_file'],
            int(config['data']['num_questions']),
            int(config['data']['num_concepts']),
        )
    if args.prepare_test_only:
        dataset = build_dataset(config, question_window_path, {-1}, True)
        print(
            f'Prepared question-level test set: {len(dataset)} sequences; '
            f'q-kc edges={structure["question_concept_edges"]:,}; '
            f'q-q edges={structure["question_graph_edges"]:,}; '
            f'KC bundles={structure["num_bundles"]:,}'
        )
        return
    if args.fold not in config['data']['folds']:
        raise ValueError(f'fold must be one of {config["data"]["folds"]}')
    if not args.output_dir:
        raise ValueError('--output_dir is required for training')

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f'fold{args.fold}_history.json'
    progress_path = output_dir / f'fold{args.fold}_progress.json'
    checkpoint_path = output_dir / f'fold{args.fold}_best.pt'
    resolved_config_path = output_dir / f'fold{args.fold}_config.yaml'
    protocol_path = output_dir / f'fold{args.fold}_protocol.json'
    prediction_path = output_dir / f'fold{args.fold}_test_predictions.npz'
    with resolved_config_path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    seed = int(config['training']['seed'])
    set_seed(seed)
    train_folds = set(config['data']['folds']) - {args.fold}
    train_dataset = build_dataset(config, train_valid_path, train_folds)
    valid_dataset = build_dataset(config, train_valid_path, {args.fold})
    if args.validation_only:
        test_dataset = None
    elif test_metric == 'pykt_sequence':
        # This matches pyKT's ordinary test loader: every `smasks` position
        # in test_sequences_quelevel.csv contributes one prediction.
        test_dataset = build_dataset(config, test_path, {-1})
    else:
        test_dataset = build_dataset(config, question_window_path, {-1}, True)
    collator = QuestionLevelCollator(
        structure['question_to_bundle'],
        structure['question_to_concepts'],
    )
    loader_kwargs = {
        'batch_size': int(config['training']['batch_size']),
        'collate_fn': collator,
        'num_workers': int(config['training']['num_workers']),
        'pin_memory': True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = (
        None
        if test_dataset is None else DataLoader(
            test_dataset, shuffle=False, **loader_kwargs
        )
    )

    model = build_model(config, structure)
    if os.environ.get('MATRA4KT_FINITE_HOOKS') == '1':
        install_finite_forward_hooks(model)
    assert_formal_cuda(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    mamba_layers = (
        0 if model.sequence_encoder is None
        else len(model.sequence_encoder.layers)
    )
    transformer_layers = (
        0 if model.short_term is None else len(model.short_term.layers)
    )
    fusion_mode = (
        None if model.branch_fusion is None else model.branch_fusion.mode
    )
    print(
        'Architecture: '
        f'{mamba_layers}xMamba2 + '
        f'{transformer_layers}xTransformer + '
        f'{fusion_mode or "no branch fusion"}'
    )
    print(
        f'fold={args.fold} train={len(train_dataset)} valid={len(valid_dataset)} '
        f'test={len(test_dataset) if test_dataset is not None else "sealed"} '
        f'parameters={parameter_count:,}'
    )
    with protocol_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'input_variant': config['experiment']['input_variant'],
            'temporal_backbone': model.temporal_backbone,
            'branch_fusion': fusion_mode,
            'mamba_layers_instantiated': mamba_layers,
            'transformer_layers_instantiated': transformer_layers,
            'physical_disabled_modules': {
                'mamba': not model.has_mamba,
                'transformer': not model.has_transformer,
            },
            'output': 'one probability per selected next question',
            'question_id_saved': True,
            'multi_concept_ids_saved': True,
            'question_text_used': False,
            'question_text_lookup': config['data'].get('question_text_file'),
            'static_structure_uses_responses': False,
            'prefix_transition_uses_target_response': False,
            'question_concept_edges': structure['question_concept_edges'],
            'question_graph_edges': structure['question_graph_edges'],
            'question_graph_materialized': structure[
                'question_graph_materialized'
            ],
            'unique_concept_bundles': structure['num_bundles'],
            'canonical_question_kc_width': int(
                structure['question_to_concepts'].size(1)
            ),
            'validation_only': args.validation_only,
            'test_input_file': None if args.validation_only else str(
                test_path if test_metric == 'pykt_sequence'
                else question_window_path
            ),
            'test_aggregation': test_metric,
        }, handle, indent=2, ensure_ascii=False)
    trainer = KTTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=(
            test_loader if test_metric == 'pykt_sequence' else None
        ),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay']),
        device='cuda:0',
        patience=int(config['training']['patience']),
        save_path=str(checkpoint_path),
        checkpoint_metadata={
            'protocol': config['experiment']['protocol'],
            'input_variant': config['experiment']['input_variant'],
            'config': config,
            'fold': args.fold,
        },
        sequence_mode=True,
        validation_auxiliary_loss=False,
        scheduler_type=config['training']['scheduler_type'],
        early_stop_metric=config['training']['early_stop_metric'],
        grad_clip_norm=float(config['training']['grad_clip_norm']),
        progress_path=progress_path,
        progress_metadata={
            'model': 'MaTra4KT',
            'dataset': config['experiment']['dataset'],
            'fold': int(args.fold),
            'resolved_config_path': str(resolved_config_path),
            'protocol_path': str(protocol_path),
            'final_history_path': str(history_path),
        },
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    recovered_training_seconds = 0.0
    try:
        if args.recover_test_only:
            checkpoint, progress, history = load_recoverable_training(
                checkpoint_path, progress_path, config, args.fold
            )
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.history = history
            trainer.best_valid_auc = float(progress['best_valid_auc'])
            trainer.best_selection_value = float(progress['best_selection_value'])
            trainer.early_stop_counter = int(progress['early_stop_counter'])
            recovered_training_seconds = float(
                progress.get('elapsed_seconds')
                or sum(history.get('epoch_duration_seconds', []))
            )
            print(
                f'Recovering test only from epoch {checkpoint["best_epoch"]}; '
                'training will not be repeated'
            )
        else:
            history = trainer.fit(epochs=int(config['training']['epochs']))
        if test_loader is not None and test_metric == 'question_window_late_mean':
            question_window_path = (
                output_dir / f'fold{args.fold}_question_window_predictions.npz'
            )
            test_metrics = evaluate_question_window_late_mean(
                trainer.model,
                test_loader,
                trainer.device,
                question_window_path,
            )
            history['test_auc'].append(test_metrics['auc'])
            history['test_acc'].append(test_metrics['acc'])
            history['test_metric'] = test_metrics
            print(
                f'Question-window late_mean AUC: {test_metrics["auc"]:.4f}, '
                f'ACC: {test_metrics["acc"]:.4f}, n={test_metrics["count"]:,}'
            )
            print(f'Saved fused question outputs to {question_window_path}')
    except Exception as error:
        trainer.persist_progress(status='failed')
        failed_path = output_dir / 'FAILED'
        failed_path.write_text(
            f'{type(error).__name__}: {error}\n', encoding='utf-8'
        )
        raise
    torch.cuda.synchronize()
    recovery_seconds = float(time.perf_counter() - started_at)
    history['wall_time_seconds'] = recovered_training_seconds + recovery_seconds
    history['recovered_test_only'] = bool(args.recover_test_only)
    if args.recover_test_only:
        history['recovery_evaluation_seconds'] = recovery_seconds
    history['peak_gpu_memory_mb'] = float(
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )
    trainer.persist_progress(status='completed')
    temporary_history_path = history_path.with_suffix('.json.tmp')
    with temporary_history_path.open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
    temporary_history_path.replace(history_path)
    (output_dir / 'FAILED').unlink(missing_ok=True)
    print(f'History saved to {history_path}')


if __name__ == '__main__':
    main()
