#!/usr/bin/env python3
"""Run one validation-only fold with one shared model/input contract."""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


V8_DIR = Path(__file__).resolve().parent
WORKER = V8_DIR / 'train_denoisekt_xes.py'
EXPORTER = V8_DIR / 'export_question_validation_predictions.py'
ENSEMBLE_ANALYZER = V8_DIR / 'analyze_fixed_validation_ensemble.py'
ALLOWED_CANDIDATE_KEYS = {
    'use_question_rasch',
    'use_bundle_attention_bias',
    'use_transition_graph',
    'branch_fusion',
}
REQUIRED_DATA_KEYS = {
    'dataset_name',
    'dataset_alias',
    'data_dir',
    'train_valid_file',
    'test_file',
    'structure_metadata_files',
    'num_questions',
    'num_concepts',
    'max_concepts',
    'folds',
    'split',
    'evaluation_level',
}
OPTIONAL_DATA_KEYS = {
    'test_question_window_file',
}


def load_screen(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    for name, overrides in config['screen']['candidates'].items():
        unexpected = set(overrides) - ALLOWED_CANDIDATE_KEYS
        if unexpected:
            raise ValueError(
                f'candidate {name} contains forbidden overrides: '
                f'{sorted(unexpected)}'
            )
    for name, data in config['datasets'].items():
        missing = REQUIRED_DATA_KEYS - set(data)
        unexpected = set(data) - REQUIRED_DATA_KEYS - OPTIONAL_DATA_KEYS
        if missing or unexpected:
            raise ValueError(
                f'dataset {name} must contain data metadata only; '
                f'missing={sorted(missing)}, unexpected={sorted(unexpected)}'
            )
    baseline = config['screen'].get(
        'baseline_candidate', 'shared_prefix_baseline'
    )
    if baseline not in config['screen']['candidates']:
        raise ValueError(f'baseline candidate is not registered: {baseline}')
    return config


def materialize_jobs(screen):
    output_root = Path(screen['output_root']).resolve()
    config_root = output_root / 'resolved_configs'
    config_root.mkdir(parents=True, exist_ok=True)
    fold = int(screen['screen']['fold'])
    jobs = []
    for dataset_name, data in screen['datasets'].items():
        for candidate, overrides in screen['screen']['candidates'].items():
            resolved = {
                'experiment': {
                    **screen['experiment'],
                    'input_variant': candidate,
                    'dataset': dataset_name,
                },
                'model': copy.deepcopy(screen['model']),
                'training': copy.deepcopy(screen['training']),
                'data': copy.deepcopy(data),
                'output_root': str(output_root / dataset_name / candidate),
            }
            resolved['model'].update(overrides)
            config_path = config_root / f'{dataset_name}__{candidate}.yaml'
            with config_path.open('w', encoding='utf-8') as handle:
                yaml.safe_dump(
                    resolved, handle, sort_keys=False, allow_unicode=True
                )
            run_dir = output_root / dataset_name / candidate
            jobs.append({
                'dataset': dataset_name,
                'candidate': candidate,
                'config': config_path,
                'run_dir': run_dir,
                'fold': fold,
            })
    manifest_path = output_root / 'screen_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(
            [{key: str(value) if isinstance(value, Path) else value
              for key, value in job.items()} for job in jobs],
            handle,
            indent=2,
        )
    return jobs, manifest_path


def run_jobs(jobs, gpu_ids, runtime_python):
    pending = list(jobs)
    active = {}
    failures = []
    while pending or active:
        for gpu in gpu_ids:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            job['run_dir'].mkdir(parents=True, exist_ok=True)
            log_path = job['run_dir'] / f'fold{job["fold"]}.log'
            log_handle = log_path.open('w', encoding='utf-8')
            command = [
                str(runtime_python),
                '-u',
                str(WORKER),
                '--config',
                str(job['config']),
                '--fold',
                str(job['fold']),
                '--output_dir',
                str(job['run_dir']),
                '--validation_only',
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
                f'gpu={gpu} started {job["dataset"]}/{job["candidate"]} '
                f'pid={process.pid}'
            )
        if active:
            time.sleep(5)
        for gpu, (process, job, log_handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[gpu]
            if return_code != 0:
                failures.append((job, return_code))
                print(
                    f'FAILED {job["dataset"]}/{job["candidate"]} '
                    f'exit={return_code}'
                )
            else:
                print(f'completed {job["dataset"]}/{job["candidate"]}')
        if failures:
            for process, _, log_handle in active.values():
                process.terminate()
                log_handle.close()
            names = ', '.join(
                f'{job["dataset"]}/{job["candidate"]}'
                for job, _ in failures
            )
            raise RuntimeError(f'{len(failures)} jobs failed: {names}')


def export_fixed_pair_predictions(jobs, gpu_ids, runtime_python):
    candidates = {
        'shared_prefix_baseline',
        'shared_prefix_rasch',
        'shared_prefix_rasch_bundle',
        'shared_no_transition',
    }
    pending = []
    for job in jobs:
        if job['candidate'] not in candidates:
            continue
        output = job['run_dir'] / f'fold{job["fold"]}_valid_predictions.npz'
        if output.exists() and output.stat().st_size > 0:
            continue
        checkpoint = job['run_dir'] / f'fold{job["fold"]}_best.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'missing checkpoint: {checkpoint}')
        pending.append((job, checkpoint, output))

    active = {}
    while pending or active:
        for gpu in gpu_ids:
            if gpu in active or not pending:
                continue
            job, checkpoint, output = pending.pop(0)
            log_handle = (
                job['run_dir'] / f'fold{job["fold"]}_validation_export.log'
            ).open('w', encoding='utf-8')
            command = [
                str(runtime_python),
                '-u',
                str(EXPORTER),
                '--config',
                str(job['config']),
                '--fold',
                str(job['fold']),
                '--checkpoint',
                str(checkpoint),
                '--output',
                str(output),
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
                f'gpu={gpu} exporting {job["dataset"]}/'
                f'{job["candidate"]}'
            )
        if active:
            time.sleep(5)
        for gpu, (process, job, log_handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[gpu]
            if return_code != 0:
                for other, _, other_log in active.values():
                    other.terminate()
                    other_log.close()
                raise RuntimeError(
                    f'validation export failed for '
                    f'{job["dataset"]}/{job["candidate"]}'
                )


def summarize_jobs(jobs, screen):
    rows = []
    for job in jobs:
        history_path = job['run_dir'] / f'fold{job["fold"]}_history.json'
        if not history_path.exists():
            continue
        with history_path.open('r', encoding='utf-8') as handle:
            history = json.load(handle)
        best_epoch = int(history['best_epoch'])
        index = best_epoch - 1
        rows.append({
            'dataset': job['dataset'],
            'candidate': job['candidate'],
            'best_epoch': best_epoch,
            'valid_auc': float(history['valid_auc'][index]),
            'valid_acc': float(history['valid_acc'][index]),
        })
    if not rows:
        print('No completed histories are available to summarize')
        return None

    baseline_name = screen['screen'].get(
        'baseline_candidate', 'shared_prefix_baseline'
    )
    baseline = {
        row['dataset']: row for row in rows
        if row['candidate'] == baseline_name
    }
    for row in rows:
        reference = baseline.get(row['dataset'])
        row['delta_auc'] = (
            row['valid_auc'] - reference['valid_auc']
            if reference is not None else float('nan')
        )
        row['delta_acc'] = (
            row['valid_acc'] - reference['valid_acc']
            if reference is not None else float('nan')
        )

    output_root = Path(screen['output_root']).resolve()
    summary_path = output_root / 'validation_summary.csv'
    with summary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    minimum_improved = int(
        screen['screen']['acceptance']['minimum_datasets_improved']
    )
    maximum_regression = float(
        screen['screen']['acceptance'][
            'maximum_single_dataset_auc_regression'
        ]
    )
    decisions = {}
    for candidate in screen['screen']['candidates']:
        if candidate == baseline_name:
            continue
        deltas = [
            row['delta_auc'] for row in rows
            if row['candidate'] == candidate and row['dataset'] in baseline
        ]
        complete = len(deltas) == len(screen['datasets'])
        decisions[candidate] = {
            'complete': complete,
            'mean_delta_auc': sum(deltas) / len(deltas) if deltas else None,
            'minimum_delta_auc': min(deltas) if deltas else None,
            'datasets_improved': sum(delta > 0.0 for delta in deltas),
            'accepted': bool(
                complete
                and sum(delta > 0.0 for delta in deltas) >= minimum_improved
                and min(deltas) >= -maximum_regression
                and sum(deltas) > 0.0
            ),
        }
    decision_path = output_root / 'generalization_decision.json'
    with decision_path.open('w', encoding='utf-8') as handle:
        json.dump(decisions, handle, indent=2)
    print(f'Validation summary: {summary_path}')
    print(f'Generalization decision: {decision_path}')
    return decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=str(V8_DIR / 'config_question_semantic_generalization_screen.yaml'),
    )
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument(
        '--python', default=os.environ.get('DUALKT_PYTHON', sys.executable)
    )
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    parser.add_argument(
        '--skip_ensemble',
        action='store_true',
        help='run and summarize single-model candidates only',
    )
    args = parser.parse_args()

    screen = load_screen(args.config)
    jobs, manifest_path = materialize_jobs(screen)
    print(f'Prepared {len(jobs)} uniform validation jobs: {manifest_path}')
    if args.summarize_only:
        summarize_jobs(jobs, screen)
        return
    if args.prepare_only:
        return
    runtime_python = Path(args.python).resolve()
    if not runtime_python.is_file():
        raise FileNotFoundError(f'training Python not found: {runtime_python}')
    gpu_ids = [int(value) for value in args.gpus.split(',') if value.strip()]
    available = torch.cuda.device_count()
    if not gpu_ids or available == 0:
        raise RuntimeError('CUDA is unavailable; validation jobs were not started')
    if any(gpu < 0 or gpu >= available for gpu in gpu_ids):
        raise ValueError(
            f'requested GPUs {gpu_ids}, but only {available} are visible'
        )
    incomplete = [
        job for job in jobs
        if not (
            job['run_dir'] / f'fold{job["fold"]}_history.json'
        ).exists()
    ]
    print(f'Skipping {len(jobs) - len(incomplete)} completed jobs')
    run_jobs(incomplete, gpu_ids, runtime_python)
    summarize_jobs(jobs, screen)
    if args.skip_ensemble:
        print('Skipped external fixed-probability ensemble by request')
        return
    export_fixed_pair_predictions(jobs, gpu_ids, runtime_python)
    subprocess.run([
        str(runtime_python),
        str(ENSEMBLE_ANALYZER),
        '--config',
        str(Path(args.config).resolve()),
        '--fold',
        str(screen['screen']['fold']),
        '--weight',
        '0.5',
    ], cwd=V8_DIR.parent, check=True)


if __name__ == '__main__':
    main()
