#!/usr/bin/env python3
"""Run the five-model, five-dataset, five-fold history-length matrix."""

import argparse
import copy
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KT_ROOT = HERE.parent
if str(KT_ROOT) not in sys.path:
    sys.path.insert(0, str(KT_ROOT))

from V10.run_sequence_length_study import load_yaml, target_fingerprint, write_yaml


DEFAULT_CONFIG = HERE / "config_sequence_length_matrix.yaml"
PREPARE = HERE / "prepare_aligned_data.py"
BASELINE_RUNNER = HERE / "run_baseline_suite.py"
MATRA_RUNNER = KT_ROOT / "V8" / "run_single_model_4dataset_5fold.py"


def length_root(study, length):
    return Path(study["runtime"]["output_root"]) / f"len{int(length)}"


def config_root(study):
    return Path(study["runtime"]["output_root"]) / "generated_configs"


def baseline_config_path(study, length):
    return config_root(study) / f"baselines_len{int(length)}.yaml"


def matra_config_path(study, length):
    return config_root(study) / f"matra_len{int(length)}.yaml"


def baseline_campaign(study, length):
    root = length_root(study, length)
    runtime = study["runtime"]
    campaign = {
        "experiment": {
            "name": f'{study["experiment"]["name"]}_baselines_len{length}',
            "scope": "fixed_models_real_history_length_5fold",
            "primary_metric": "auc",
            "secondary_metric": "acc",
            "evaluation": study["experiment"]["evaluation"],
            "sequence_length": int(length),
            "test_ratio": 0.2,
            "folds": list(study["experiment"]["folds"]),
            "seed": int(study["experiment"]["seed"]),
        },
        "models_config": study["templates"]["models_config"],
        "runtime": {
            "python": runtime["python"],
            "pykt_root": runtime["pykt_root"],
            "source_data_config": runtime["source_data_config"],
            "gpus": list(runtime["baseline_gpus"]),
            "poll_seconds": int(runtime.get("poll_seconds", 5)),
        },
        "training": {
            "optimizer": "adam",
            "batch_size": int(study["training"]["batch_size"]),
            "num_epochs": int(study["training"]["epochs"]),
            "early_stop_patience": int(study["training"]["patience"]),
            "early_stop_metric": "auc",
            "retain_checkpoints": False,
            "retain_predictions": False,
        },
        "paths": {
            "data_root": str(root / "generated_pykt_data"),
            "output_root": str(root / "baselines"),
            "data_config": str(root / "data_config_pykt.json"),
            "input_manifest": str(root / "input_manifest_pykt.json"),
            "jobs_file": str(root / "baseline_jobs.jsonl"),
            "summary_csv": str(root / "baseline_summary.csv"),
            "summary_json": str(root / "baseline_summary.json"),
        },
        "datasets": copy.deepcopy(study["datasets"]),
    }
    if study.get("model_extensions"):
        campaign["model_extensions"] = copy.deepcopy(study["model_extensions"])
    pilot_root = runtime.get("pilot_data_root")
    if pilot_root:
        campaign["paths"]["reusable_data_root"] = str(
            Path(pilot_root) / f"len{int(length)}" / "generated_pykt_data"
        )
    if int(length) != 200:
        campaign["raw_sources"] = copy.deepcopy(study.get("raw_sources", {}))
    return campaign


def create_matra_config(study, length, data_config):
    screen = load_yaml(study["templates"]["matra_screen"])
    screen["experiment"]["name"] = (
        f'{study["experiment"]["name"]}_matra_len{length}'
    )
    screen["training"].update({
        "max_seq_len": int(length),
        "batch_size": int(study["training"]["batch_size"]),
        "epochs": int(study["training"]["epochs"]),
        "patience": int(study["training"]["patience"]),
        "disable_pykt_pickle_cache": True,
    })
    screen["model"]["short_window"] = int(study["training"]["short_window"])
    screen["datasets"] = {}
    for dataset, metadata in study["datasets"].items():
        entry = data_config[dataset]
        screen["datasets"][dataset] = {
            "dataset_name": metadata["paper_name"],
            "dataset_alias": dataset,
            "data_dir": entry["dpath"],
            "train_valid_file": entry["train_valid_file_quelevel"],
            "test_file": entry["test_file_quelevel"],
            "test_question_window_file": entry["test_question_window_file"],
            "structure_metadata_files": [
                entry["train_valid_file_quelevel"],
                entry["test_file_quelevel"],
            ],
            "num_questions": int(entry["num_q"]),
            "num_concepts": int(entry["num_c"]),
            "max_concepts": int(entry["max_concepts"]),
            "folds": list(study["experiment"]["folds"]),
            "split": "public_5fold_train_valid_plus_sealed_test",
            "evaluation_level": "question",
        }
    screen_path = config_root(study) / f"matra_screen_len{int(length)}.yaml"
    write_yaml(screen_path, screen)
    formal = {
        "experiment": {
            "name": f'{study["experiment"]["name"]}_matra_len{length}',
            "protocol": "fixed_matra_real_history_length_5fold",
            "screen_config": str(screen_path),
            "candidate": "single_denoised_orthogonal_fusion",
            "require_screen_acceptance": False,
            "run_folds": list(study["experiment"]["folds"]),
            "selection": "fixed_architecture_no_search",
            "test_metric": study["experiment"]["evaluation"],
        },
        "runtime": {
            "python": study["runtime"]["python"],
            "gpus": list(study["runtime"]["matra_gpus"]),
        },
        "targets": {
            "minimum_auc_delta_vs_denoisekt": 0.0,
            "ideal_auc_delta_vs_denoisekt": 0.0,
        },
        "paper_reference": {},
        "denoisekt_reproduction_summary": "",
        "data_overrides": {},
        "output_root": str(length_root(study, length) / "matra4kt"),
    }
    path = matra_config_path(study, length)
    write_yaml(path, formal)
    return path


def prepare_length(study, length):
    python = study["runtime"]["python"]
    campaign = baseline_campaign(study, length)
    campaign_path = baseline_config_path(study, length)
    write_yaml(campaign_path, campaign)
    subprocess.run([python, str(PREPARE), "--config", str(campaign_path)], check=True)
    models = list(study["models"]["baselines"])
    folds = [str(value) for value in study["experiment"]["folds"]]
    subprocess.run([
        python, str(BASELINE_RUNNER), "--config", str(campaign_path),
        "jobs", "--models", *models, "--folds", *folds,
    ], check=True)
    data_config = json.loads(
        Path(campaign["paths"]["data_config"]).read_text(encoding="utf-8")
    )
    matra_path = create_matra_config(study, length, data_config)
    subprocess.run([
        python, str(MATRA_RUNNER), "--config", str(matra_path), "--prepare-only",
    ], check=True)
    input_manifest = json.loads(
        Path(campaign["paths"]["input_manifest"]).read_text(encoding="utf-8")
    )
    item = {
        "length": int(length),
        "baseline_config": str(campaign_path),
        "matra_config": str(matra_path),
        "split_fingerprints": {
            dataset: {
                "train_folds": value["standard_train_fold_sha256"],
                "test_students": value["standard_test_fold_sha256"],
            }
            for dataset, value in input_manifest["datasets"].items()
        },
        "target_fingerprints": {
            dataset: target_fingerprint(data_config, dataset)
            for dataset in study["datasets"]
        },
    }
    manifest_path = Path(study["runtime"]["output_root"]) / "study_manifest.json"
    existing = []
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing:
        reference = existing[0]
        for key in ("split_fingerprints", "target_fingerprints"):
            if item[key] != reference[key]:
                raise ValueError(f"length={length} changes cross-length {key}")
    by_length = {int(value["length"]): value for value in existing}
    by_length[int(length)] = item
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([by_length[key] for key in sorted(by_length)], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PREPARED length={length} baseline_jobs={len(models) * 25} matra_jobs=25")


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("RUN " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_length(study, length):
    python = study["runtime"]["python"]
    baseline_gpus = [str(value) for value in study["runtime"]["baseline_gpus"]]
    matra_gpus = ",".join(str(value) for value in study["runtime"]["matra_gpus"])
    baseline = [
        python, "-u", str(BASELINE_RUNNER), "--config",
        str(baseline_config_path(study, length)), "run", "--gpus", *baseline_gpus,
        "--max-parallel", str(len(baseline_gpus)), "--retry-failed",
    ]
    matra = [
        python, "-u", str(MATRA_RUNNER), "--config",
        str(matra_config_path(study, length)), "--gpus", matra_gpus,
    ]
    root = Path(study["runtime"]["output_root"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_command, baseline, root / "baseline_lane.log"),
            executor.submit(run_command, matra, root / "matra_lane.log"),
        ]
        for future in futures:
            future.result()


def collect_rows(study):
    rows = []
    folds = [int(value) for value in study["experiment"]["folds"]]
    baseline_models = list(study["models"]["baselines"])
    for length in study["experiment"]["lengths"]:
        root = length_root(study, length)
        for model in [*baseline_models, study["models"]["matra"]]:
            for dataset in study["datasets"]:
                for fold in folds:
                    if model == study["models"]["matra"]:
                        run_dir = root / "matra4kt" / dataset / f"fold{fold}"
                        result_path = run_dir / f"fold{fold}_history.json"
                        running = run_dir / f"fold{fold}_progress.json"
                    else:
                        slug = model.lower().replace("-", "_")
                        run_dir = root / "baselines" / slug / dataset / f"fold{fold}"
                        result_path = run_dir / "result.json"
                        running = run_dir / "RUNNING"
                    if result_path.is_file():
                        status = "complete"
                        if model != study["models"]["matra"]:
                            result = json.loads(result_path.read_text(encoding="utf-8"))
                            if result.get("status") == "skipped":
                                status = "unsupported"
                    else:
                        status = (
                            "failed" if (run_dir / "FAILED").is_file() else
                            "running" if running.is_file() else "pending"
                        )
                    rows.append({
                        "length": int(length), "model": model, "dataset": dataset,
                        "fold": fold, "status": status,
                    })
    return rows


def summarize(study):
    rows = collect_rows(study)
    root = Path(study["runtime"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "sequence_length_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    payload = {"expected": len(rows), "status": counts}
    (root / "sequence_length_status.json").write_text(
        json.dumps({**payload, "rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("prepare", "run", "execute", "status"))
    parser.add_argument("--lengths", nargs="*", type=int)
    args = parser.parse_args()
    study = load_yaml(args.config)
    lengths = args.lengths or [int(value) for value in study["experiment"]["lengths"]]
    unknown = sorted(set(lengths) - set(study["experiment"]["lengths"]))
    if unknown:
        raise ValueError(f"lengths not registered in study: {unknown}")
    if args.command == "status":
        summarize(study)
        return
    for length in lengths:
        if args.command in {"prepare", "execute"}:
            prepare_length(study, length)
        if args.command in {"run", "execute"}:
            run_length(study, length)
        summarize(study)


if __name__ == "__main__":
    main()
