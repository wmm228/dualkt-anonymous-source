#!/usr/bin/env python3
"""Prepare and run a real-history-length FlucKT versus MaTra4KT study."""

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
KT_ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "config_sequence_length_study.yaml"
PREPARE = HERE / "prepare_aligned_data.py"
BASELINE_RUNNER = HERE / "run_baseline_suite.py"
MATRA_RUNNER = KT_ROOT / "V8" / "run_single_model_4dataset_5fold.py"


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def generated_paths(study, length):
    root = Path(study["runtime"]["output_root"]) / f"len{length}"
    configs = Path(study["runtime"]["output_root"]) / "generated_configs"
    return root, configs


def matra_output_root(study, length):
    root, _ = generated_paths(study, length)
    dirname = study["runtime"].get("matra_output_dirname", "matra4kt")
    return root / dirname


def target_fingerprint(data_config, dataset):
    """Hash the scored question ids and labels, independent of window length."""
    entry = data_config[dataset]
    path = Path(entry["dpath"]) / entry["test_question_window_file"]
    targets = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            masks = row["selectmasks"].split(",")
            qidxs = row["qidxs"].split(",")
            responses = row["responses"].split(",")
            # pyKT predicts shifted positions only, so column position zero is
            # context rather than an evaluated target.
            for mask, qidx, response in zip(
                masks[1:], qidxs[1:], responses[1:]
            ):
                if int(mask) != 1 or int(qidx) < 0:
                    continue
                key = int(qidx)
                value = int(float(response))
                previous = targets.setdefault(key, value)
                if previous != value:
                    raise ValueError(
                        f"{dataset} qidx={key} has inconsistent labels in {path}"
                    )
    digest = hashlib.sha256()
    for qidx, response in sorted(targets.items()):
        digest.update(f"{qidx}\t{response}\n".encode("ascii"))
    return {"count": len(targets), "sha256": digest.hexdigest()}


def baseline_campaign(study, length):
    root, _ = generated_paths(study, length)
    campaign = {
        "experiment": {
            "name": f'{study["experiment"]["name"]}_fluckt_len{length}',
            "scope": "fixed_fluckt_real_history_length_pilot",
            "primary_metric": "auc",
            "secondary_metric": "acc",
            "evaluation": "pykt_question_window_late_mean",
            "sequence_length": int(length),
            "test_ratio": 0.2,
            "folds": list(study["experiment"]["folds"]),
            "seed": int(study["experiment"]["seed"]),
        },
        "models_config": study["templates"]["models_config"],
        "runtime": {
            "python": study["runtime"]["python"],
            "pykt_root": study["runtime"]["pykt_root"],
            "source_data_config": study["runtime"]["source_data_config"],
            "gpus": [int(study["runtime"]["fluckt_gpu"])],
            "poll_seconds": 5,
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
            "output_root": str(root / "fluckt"),
            "data_config": str(root / "data_config_pykt.json"),
            "input_manifest": str(root / "input_manifest_pykt.json"),
            "jobs_file": str(root / "fluckt_jobs.jsonl"),
            "summary_csv": str(root / "fluckt_summary.csv"),
            "summary_json": str(root / "fluckt_summary.json"),
        },
        "datasets": copy.deepcopy(study["datasets"]),
    }
    # Length 200 already exists and should be reused byte-for-byte. Longer
    # ASSIST2009 splits must be regenerated from the corrected raw source.
    if int(length) != 200:
        campaign["raw_sources"] = copy.deepcopy(study["raw_sources"])
    return campaign


def matra_configs(study, length, data_config):
    root, configs = generated_paths(study, length)
    screen = load_yaml(study["templates"]["matra_screen"])
    screen["experiment"]["name"] = f'{study["experiment"]["name"]}_matra_len{length}'
    screen["training"].update({
        "max_seq_len": int(length),
        "batch_size": int(study["training"]["batch_size"]),
        "epochs": int(study["training"]["epochs"]),
        "patience": int(study["training"]["patience"]),
        "disable_pykt_pickle_cache": True,
    })
    screen["model"]["short_window"] = int(study["training"]["short_window"])
    screen["datasets"] = {}
    for name, metadata in study["datasets"].items():
        entry = data_config[name]
        screen["datasets"][name] = {
            "dataset_name": metadata["paper_name"],
            "dataset_alias": name,
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
            "folds": [0, 1, 2, 3, 4],
            "split": "public_5fold_train_valid_plus_sealed_test",
            "evaluation_level": "question",
        }
    screen_path = configs / f"matra_screen_len{length}.yaml"
    write_yaml(screen_path, screen)

    formal = {
        "experiment": {
            "name": f'{study["experiment"]["name"]}_matra_len{length}',
            "protocol": "fixed_matra_real_history_length_pilot",
            "screen_config": str(screen_path),
            "candidate": "single_denoised_orthogonal_fusion",
            "require_screen_acceptance": False,
            "run_folds": list(study["experiment"]["folds"]),
            "selection": "fixed_architecture_no_search",
            "test_metric": "question_window_late_mean",
        },
        "runtime": {
            "python": study["runtime"]["python"],
            "gpus": [int(study["runtime"]["matra_gpu"])],
        },
        "targets": {
            "minimum_auc_delta_vs_denoisekt": 0.0,
            "ideal_auc_delta_vs_denoisekt": 0.0,
        },
        "paper_reference": {},
        "denoisekt_reproduction_summary": "",
        "data_overrides": {},
        "output_root": str(matra_output_root(study, length)),
    }
    formal_path = configs / f"matra_formal_len{length}.yaml"
    write_yaml(formal_path, formal)
    return formal_path


def prepare(study, lengths):
    python = study["runtime"]["python"]
    manifest = Path(study["runtime"]["output_root"]) / "study_manifest.json"
    existing = []
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
    reference = existing[0] if existing else None
    generated = []
    for length in lengths:
        root, configs = generated_paths(study, length)
        campaign = baseline_campaign(study, length)
        campaign_path = configs / f"fluckt_len{length}.yaml"
        write_yaml(campaign_path, campaign)
        subprocess.run(
            [python, str(PREPARE), "--config", str(campaign_path)], check=True
        )
        subprocess.run(
            [python, str(BASELINE_RUNNER), "--config", str(campaign_path),
             "jobs", "--models", "FlucKT", "--folds",
             *[str(value) for value in study["experiment"]["folds"]]],
            check=True,
        )
        data_config = json.loads(Path(campaign["paths"]["data_config"]).read_text())
        input_manifest = json.loads(
            Path(campaign["paths"]["input_manifest"]).read_text(encoding="utf-8")
        )
        split_fingerprints = {
            dataset: {
                "train_folds": input_manifest["datasets"][dataset][
                    "standard_train_fold_sha256"
                ],
                "test_students": input_manifest["datasets"][dataset][
                    "standard_test_fold_sha256"
                ],
            }
            for dataset in study["datasets"]
        }
        target_fingerprints = {
            dataset: target_fingerprint(data_config, dataset)
            for dataset in study["datasets"]
        }
        if reference is not None:
            for field, observed in (
                ("split_fingerprints", split_fingerprints),
                ("target_fingerprints", target_fingerprints),
            ):
                expected = reference.get(field)
                if expected is not None and observed != expected:
                    raise ValueError(
                        f"length={length} changes cross-length {field}"
                    )
        formal_path = matra_configs(study, length, data_config)
        subprocess.run(
            [python, str(MATRA_RUNNER), "--config", str(formal_path),
             "--prepare-only"], check=True,
        )
        generated_item = {
            "length": int(length),
            "fluckt_config": str(campaign_path),
            "matra_config": str(formal_path),
            "split_fingerprints": split_fingerprints,
            "target_fingerprints": target_fingerprints,
        }
        generated.append(generated_item)
        if reference is None or "target_fingerprints" not in reference:
            reference = generated_item
    by_length = {int(item["length"]): item for item in existing}
    by_length.update({int(item["length"]): item for item in generated})
    merged = [by_length[key] for key in sorted(by_length)]
    manifest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(generated)} lengths ({len(merged)} total): {manifest}")


def run_lane(commands, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write("RUN " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(study, lengths):
    python = study["runtime"]["python"]
    _, configs = generated_paths(study, lengths[0])
    fluckt_commands = []
    matra_commands = []
    for length in lengths:
        fluckt_parallel = int(
            study["runtime"].get("fluckt_parallel_per_gpu", 1)
        )
        fluckt_commands.append([
            python, "-u", str(BASELINE_RUNNER), "--config",
            str(configs / f"fluckt_len{length}.yaml"), "run", "--gpus",
            *([str(study["runtime"]["fluckt_gpu"])] * fluckt_parallel),
            "--max-parallel", str(fluckt_parallel),
        ])
        matra_commands.append([
            python, "-u", str(MATRA_RUNNER), "--config",
            str(configs / f"matra_formal_len{length}.yaml"), "--gpus",
            str(study["runtime"]["matra_gpu"]),
        ])
    output_root = Path(study["runtime"]["output_root"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_lane, fluckt_commands, output_root / "fluckt_lane.log"
            ),
            executor.submit(
                run_lane, matra_commands, output_root / "matra_lane.log"
            ),
        ]
        for future in futures:
            future.result()
    summarize(study, lengths)


def summarize(study, lengths):
    rows = []
    folds = [int(value) for value in study["experiment"]["folds"]]
    for length in lengths:
        root, _ = generated_paths(study, length)
        for dataset in study["datasets"]:
            for fold in folds:
                fluckt_dir = root / "fluckt" / "fluckt" / dataset / f"fold{fold}"
                fluckt_result = fluckt_dir / "result.json"
                row = {
                    "length": int(length), "model": "FlucKT",
                    "dataset": dataset, "fold": fold,
                    "status": "complete" if fluckt_result.is_file() else (
                        "failed" if (fluckt_dir / "FAILED").is_file() else
                        "running" if (fluckt_dir / "RUNNING").is_file() else "pending"
                    ),
                    "auc": None, "acc": None, "wall_time_seconds": None,
                    "peak_gpu_memory_mb": None, "batch_size": None,
                }
                if fluckt_result.is_file():
                    result = json.loads(fluckt_result.read_text(encoding="utf-8"))
                    saved_config = Path(result["run_dir"]) / "config.json"
                    actual_batch = None
                    if saved_config.is_file():
                        actual_batch = json.loads(
                            saved_config.read_text(encoding="utf-8")
                        )["train_config"]["batch_size"]
                    row.update({
                        "auc": result.get("test_auc"),
                        "acc": result.get("test_acc"),
                        "wall_time_seconds": result.get("wall_time_seconds"),
                        "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb"),
                        "batch_size": actual_batch,
                    })
                    if actual_batch != int(study["training"]["batch_size"]):
                        row["status"] = "invalid_batch_size"
                rows.append(row)

                matra_dir = matra_output_root(study, length) / dataset / f"fold{fold}"
                history_path = matra_dir / f"fold{fold}_history.json"
                progress_path = matra_dir / f"fold{fold}_progress.json"
                status = "pending"
                if history_path.is_file():
                    status = "complete"
                elif (matra_dir / "FAILED").is_file():
                    status = "failed"
                elif progress_path.is_file():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    status = (
                        "failed"
                        if progress.get("status") == "failed" else "running"
                    )
                row = {
                    "length": int(length), "model": "MaTra4KT",
                    "dataset": dataset, "fold": fold, "status": status,
                    "auc": None, "acc": None, "wall_time_seconds": None,
                    "peak_gpu_memory_mb": None,
                    "batch_size": int(study["training"]["batch_size"]),
                }
                if history_path.is_file():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    row.update({
                        "auc": history.get("test_auc", [None])[-1],
                        "acc": history.get("test_acc", [None])[-1],
                        "wall_time_seconds": history.get("wall_time_seconds"),
                        "peak_gpu_memory_mb": history.get("peak_gpu_memory_mb"),
                    })
                rows.append(row)

    output_root = Path(study["runtime"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "sequence_length_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_root / "sequence_length_results.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"rows": len(rows), "status": counts, "csv": str(csv_path)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "status"):
        item = subparsers.add_parser(command)
        item.add_argument("--lengths", nargs="*", type=int)
    args = parser.parse_args()
    study = load_yaml(args.config)
    lengths = args.lengths or [int(value) for value in study["experiment"]["lengths"]]
    invalid = sorted(set(lengths) - set(study["experiment"]["lengths"]))
    if invalid:
        raise ValueError(f"lengths not registered in study: {invalid}")
    if args.command == "prepare":
        prepare(study, lengths)
    elif args.command == "run":
        run(study, lengths)
    else:
        summarize(study, lengths)


if __name__ == "__main__":
    main()
