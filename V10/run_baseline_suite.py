#!/usr/bin/env python3
"""Generate, execute, inspect, and summarize the 22-baseline campaign."""

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config_baseline_reproduction.yaml"
CONTROL_KEYS = {
    "model_name",
    "dataset_name",
    "emb_type",
    "save_dir",
    "fold",
}
QUESTION_LEVEL_MODELS = {"denoisekt", "qdkt", "qikt", "rkt"}
SEQUENCE_LENGTH_MODELS = {
    "saint",
    "sakt",
    "atdkt",
    "simplekt",
    "stablekt",
    "datakt",
    "folibikt",
    "cskt",
    "fluckt",
    "fa_kt",
    "denoisekt",
    "ukt",
    "mtkt",
}


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def slug(value):
    return (
        value.lower()
        .replace("+", "_plus")
        .replace("-", "_")
        .replace(" ", "_")
    )


def read_jobs(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def filter_jobs(jobs, models=None, datasets=None, folds=None):
    selected_folds = set(folds) if folds is not None else None
    return [
        job
        for job in jobs
        if (not models or job["paper_model"] in models)
        and (not datasets or job["dataset"] in datasets)
        and (selected_folds is None or int(job["fold"]) in selected_folds)
    ]


def result_path(job):
    return Path(job["output_dir"]) / "result.json"


def failed_path(job):
    return Path(job["output_dir"]) / "FAILED"


def process_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_matches_baseline_job(pid, job):
    """Return whether a live PID is the worker recorded for this exact fold."""
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    expected_job_file = str(Path(job["output_dir"]) / "job.json").encode()
    return b"worker" in arguments and expected_job_file in arguments


def running_marker_pid(lock_path):
    try:
        for line in lock_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("pid="):
                return int(line.partition("=")[2])
    except (OSError, ValueError):
        return None
    return None


def recover_stale_running_lock(job, stale_after_seconds):
    """Preserve a dead worker marker so its fold can be scheduled again."""
    if stale_after_seconds <= 0:
        return False
    lock_path = Path(job["output_dir"]) / "RUNNING"
    if not lock_path.is_file():
        return False
    pid = running_marker_pid(lock_path)
    if (
        pid is not None
        and process_is_alive(pid)
        and process_matches_baseline_job(pid, job)
    ):
        print(
            f"KEEP live marker pid={pid} job={job['job_id']}",
            flush=True,
        )
        return False
    age_seconds = time.time() - lock_path.stat().st_mtime
    # A marker with a recorded PID becomes retryable as soon as that process
    # exits.  The age threshold remains a conservative fallback for legacy
    # markers that did not record a PID.
    if pid is None and age_seconds < stale_after_seconds:
        return False
    recovered_path = lock_path.with_name(
        f"INTERRUPTED_{int(lock_path.stat().st_mtime)}"
    )
    if recovered_path.exists():
        recovered_path = lock_path.with_name(
            f"INTERRUPTED_{int(lock_path.stat().st_mtime)}_{time.time_ns()}"
        )
    lock_path.replace(recovered_path)
    print(
        f"RECOVER stale marker age={age_seconds:.0f}s "
        f"job={job['job_id']} saved={recovered_path}",
        flush=True,
    )
    return True


def parse_integral_sequence_value(value):
    """Accept integer-valued decimals emitted by some official pyKT splitters."""
    try:
        return int(value)
    except (TypeError, ValueError):
        parsed = float(value)
        if not parsed.is_integer():
            raise ValueError(f"Expected an integer-valued sequence item, got {value!r}")
        return int(parsed)


def install_dkt_forget_input_compatibility(pykt_root):
    """Patch pyKT's process-local parser without rewriting large input CSV files."""
    sys.path.insert(0, str(pykt_root))
    from pykt.datasets import dkt_forget_dataloader

    dkt_forget_dataloader.int = parse_integral_sequence_value


def materialize_skipped_job(job, campaign):
    reason = job.get("skip_reason")
    if not reason:
        return False
    payload = {
        "status": "skipped",
        "job_id": job["job_id"],
        "paper_model": job["paper_model"],
        "model_name": job["params"]["model_name"],
        "dataset_name": job["dataset"],
        "fold": job["fold"],
        "input_family": job["input_family"],
        "protocol": campaign["experiment"]["scope"],
        "params": job["params"],
        "input_manifest": campaign["paths"]["input_manifest"],
        "skip_reason": reason,
        "test_auc": None,
        "test_acc": None,
    }
    atomic_json(result_path(job), payload)
    if failed_path(job).exists():
        failed_path(job).unlink()
    return True


def build_jobs(campaign, models, selected_models=None, selected_datasets=None, folds=None):
    selected_models = selected_models or list(models)
    selected_datasets = selected_datasets or list(campaign["datasets"])
    folds = folds if folds is not None else campaign["experiment"]["folds"]
    jobs = []
    for model_name in selected_models:
        if model_name not in models:
            raise KeyError(f"Unknown model: {model_name}")
        spec = models[model_name]
        for dataset in selected_datasets:
            if dataset not in campaign["datasets"]:
                raise KeyError(f"Unknown dataset: {dataset}")
            for fold in folds:
                identity = f"{campaign['experiment']['name']}|{model_name}|{dataset}|{fold}"
                job_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
                output_dir = (
                    Path(campaign["paths"]["output_root"])
                    / slug(model_name)
                    / dataset
                    / f"fold{fold}"
                )
                job = {
                        "job_id": job_id,
                        "paper_model": model_name,
                        "dataset": dataset,
                        "fold": int(fold),
                        "output_dir": str(output_dir),
                        "input_family": spec["input_family"],
                        "params": {
                            **spec["params"],
                            "model_name": spec["pykt_name"],
                            "dataset_name": dataset,
                            "emb_type": spec["emb_type"],
                            "fold": int(fold),
                            "seed": int(campaign["experiment"]["seed"]),
                        },
                    }
                skip_reason = spec.get("unsupported_datasets", {}).get(dataset)
                if skip_reason:
                    job["skip_reason"] = skip_reason
                jobs.append(job)
    return jobs


def command_jobs(args, campaign, models):
    jobs = build_jobs(campaign, models, args.models, args.datasets, args.folds)
    path = Path(campaign["paths"]["jobs_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs),
        encoding="utf-8",
    )
    skipped = sum(materialize_skipped_job(job, campaign) for job in jobs)
    print(
        json.dumps(
            {"jobs": len(jobs), "skipped": skipped, "jobs_file": str(path)},
            ensure_ascii=False,
        )
    )


def disable_pykt_pickle_cache():
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


def rkt_relation_path(job, data_config_path):
    config = json.loads(Path(data_config_path).read_text(encoding="utf-8"))
    entry = config[job["dataset"]]
    train_folds = sorted(set(entry["folds"]) - {int(job["fold"])})
    suffix = "_" + "_".join(str(value) for value in train_folds)
    prefix = "phi_dict" if int(entry["num_q"]) > 100000 else "phi_array"
    return Path(entry["dpath"]) / f"{prefix}{suffix}.pkl", train_folds, entry


def prepare_rkt_fold(job, data_config_path, campaign):
    relation, folds, entry = rkt_relation_path(job, data_config_path)
    if relation.is_file():
        return relation
    command = [
        str(campaign["runtime"]["python"]),
        str(HERE / "build_rkt_phi.py"),
        "--input",
        str(Path(entry["dpath"]) / entry["train_valid_file_quelevel"]),
        "--num-questions",
        str(entry["num_q"]),
        "--train-folds",
        ",".join(str(value) for value in folds),
        "--output",
        str(relation),
        "--pykt-root",
        str(campaign["runtime"]["pykt_root"]),
    ]
    subprocess.run(command, check=True)
    return relation


def cleanup_training_artifacts(result, campaign):
    run_dir = Path(result["run_dir"])
    if not campaign["training"].get("retain_checkpoints", False):
        for path in run_dir.glob("*_model.ckpt"):
            path.unlink()
    if not campaign["training"].get("retain_predictions", False):
        for path in run_dir.glob("*_predictions.txt"):
            path.unlink()


def evaluate_pykt_report_metric(
    result, job, relation_path, pykt_root, evaluation_batch_size=None
):
    """Use pyKT's leakage-safe question-window reporting route."""
    sys.path.insert(0, str(pykt_root))
    from torch.utils.data import DataLoader
    from pykt.datasets.atdkt_dataloader import ATDKTDataset
    from pykt.datasets.data_loader import KTDataset
    from pykt.datasets.dimkt_dataloader import DIMKTDataset
    from pykt.datasets.dkt_forget_dataloader import DktForgetDataset
    from pykt.datasets.que_data_loader import KTQueDataset
    from pykt.models import evaluate, evaluate_question, init_model

    run_dir = Path(result["run_dir"])
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    params = saved["params"]
    model_name = params["model_name"]
    emb_type = params["emb_type"]
    data_config = copy.deepcopy(saved["data_config"])
    data_config["dataset_name"] = params["dataset_name"]
    model_config = copy.deepcopy(saved["model_config"])
    for key in ("use_wandb", "learning_rate", "add_uuid", "l2"):
        model_config.pop(key, None)
    if model_name in SEQUENCE_LENGTH_MODELS:
        model_config["seq_len"] = int(saved["train_config"]["seq_len"])

    model = init_model(model_name, model_config, data_config, emb_type)
    checkpoint = torch.load(
        run_dir / f"{emb_type}_model.ckpt",
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.load_state_dict(checkpoint)
    batch_size = int(
        evaluation_batch_size or saved["train_config"]["batch_size"]
    )
    diff_level = params.get("difficult_levels")

    if model_name in QUESTION_LEVEL_MODELS:
        test_path = Path(data_config["dpath"]) / data_config[
            "test_window_file_quelevel"
        ]
        dataset = KTQueDataset(
            str(test_path),
            input_type=data_config["input_type"],
            folds={-1},
            concept_num=data_config["num_c"],
            max_concepts=data_config["max_concepts"],
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if model_name == "rkt":
            relation = pd.read_pickle(relation_path)
            auc, acc = evaluate(
                model, loader, model_name, relation
            )
        else:
            # Formal runs retain no prediction artifacts. Avoid accumulating
            # full per-row histories solely for an unused prediction export.
            auc, acc = evaluate(model, loader, model_name)
        return {
            "auc": float(auc),
            "acc": float(acc),
            "input_file": str(test_path),
            "aggregation": "direct_question_window_prediction",
            "history": (
                "question_window_fixed_length_"
                f"{int(saved['train_config']['seq_len'])}"
            ),
        }

    test_path = Path(data_config["dpath"]) / data_config[
        "test_question_window_file"
    ]
    if model_name in {"dkt_forget", "bakt_time", "fa_kt", "mtkt"}:
        dataset = DktForgetDataset(
            str(test_path), data_config["input_type"], {-1}, True
        )
    elif model_name == "atdkt":
        dataset = ATDKTDataset(
            str(test_path), data_config["input_type"], {-1}, True
        )
    elif model_name == "dimkt":
        dataset = DIMKTDataset(
            data_config["dpath"],
            str(test_path),
            data_config["input_type"],
            {-1},
            True,
            diff_level=diff_level,
        )
    else:
        dataset = KTDataset(
            str(test_path), data_config["input_type"], {-1}, True
        )
    question_window_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    prediction_path = run_dir / "question_window_predictions.txt"
    aucs, accs = evaluate_question(
        model,
        question_window_loader,
        model_name,
        fusion_type=["late_fusion"],
        save_path=str(prediction_path),
    )
    if "late_mean" not in aucs or "late_mean" not in accs:
        raise ValueError(
            f"{model_name}/{params['dataset_name']} did not produce late_mean"
        )
    return {
        "auc": float(aucs["late_mean"]),
        "acc": float(accs["late_mean"]),
        "input_file": str(test_path),
        "aggregation": "question_window_late_mean",
        "history": (
            "question_window_fixed_length_"
            f"{int(saved['train_config']['seq_len'])}"
        ),
    }


def evaluate_native_pykt_metric(result, job, data_config_path):
    """Keep the model's own pyKT test result rather than imposing QW fusion."""
    if "test_auc" not in result or "test_acc" not in result:
        raise RuntimeError(
            f"{job['paper_model']} did not return its native pyKT test metrics"
        )
    data_config = json.loads(Path(data_config_path).read_text(encoding="utf-8"))
    entry = data_config[job["dataset"]]
    return {
        "auc": float(result["test_auc"]),
        "acc": float(result["test_acc"]),
        "input_file": str(Path(entry["dpath"]) / entry["test_file"]),
        "aggregation": "native_pykt_evaluate_test",
        "history": "model_native_test_file",
    }


def recover_native_test_from_checkpoint(job_file, campaign_path):
    """Evaluate a checkpoint left behind when pyKT failed after training."""
    campaign = load_yaml(campaign_path)
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    if campaign["experiment"].get("evaluation") != "native_pykt_test":
        raise ValueError("Checkpoint recovery only supports native pyKT testing")
    if result_path(job).is_file():
        raise FileExistsError(f"Refusing to overwrite {result_path(job)}")

    checkpoints = sorted(Path(job["output_dir"]).glob("checkpoints/**/config.json"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one saved checkpoint config for {job['job_id']}, "
            f"found {len(checkpoints)}"
        )
    run_dir = checkpoints[0].parent
    saved = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    params = saved["params"]
    model_name = params["model_name"]
    emb_type = params["emb_type"]
    checkpoint_path = run_dir / f"{emb_type}_model.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    pykt_root = Path(campaign["runtime"]["pykt_root"])
    sys.path.insert(0, str(pykt_root))
    from pykt.datasets import init_test_datasets
    from pykt.models import evaluate, init_model

    data_config = json.loads(
        Path(campaign["paths"]["data_config"]).read_text(encoding="utf-8")
    )[job["dataset"]]
    test_data_config = copy.deepcopy(data_config)
    test_data_config["dataset_name"] = job["dataset"]
    test_data_config["fold"] = int(job["fold"])
    model_config = copy.deepcopy(saved["model_config"])
    for key in ("use_wandb", "learning_rate", "add_uuid", "l2"):
        model_config.pop(key, None)
    if model_name in SEQUENCE_LENGTH_MODELS:
        model_config["seq_len"] = int(saved["train_config"]["seq_len"])

    model = init_model(model_name, model_config, test_data_config, emb_type)
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    test_loader, _, _, _ = init_test_datasets(
        test_data_config,
        model_name,
        int(saved["train_config"]["batch_size"]),
        load_window=False,
    )
    prediction_path = run_dir / f"{emb_type}_test_predictions.txt"
    test_auc, test_acc = evaluate(
        model, test_loader, model_name, save_path=str(prediction_path)
    )
    result = {
        "fold": int(job["fold"]),
        "model_name": model_name,
        "dataset_name": job["dataset"],
        "emb_type": emb_type,
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "window_test_auc": None,
        "window_test_acc": None,
        "valid_auc": None,
        "valid_acc": None,
        "best_epoch": None,
        "model_save_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "recovered_test_only": True,
    }
    result["report_metric"] = evaluate_native_pykt_metric(
        result, job, campaign["paths"]["data_config"]
    )
    result.update(
        {
            "job_id": job["job_id"],
            "paper_model": job["paper_model"],
            "input_family": job["input_family"],
            "protocol": campaign["experiment"]["scope"],
            "params": job["params"],
            "input_manifest": campaign["paths"]["input_manifest"],
            "artifacts_retained": {"checkpoint": True, "predictions": True},
        }
    )
    atomic_json(result_path(job), result)
    if failed_path(job).exists():
        failed_path(job).unlink()
    print(json.dumps(result, ensure_ascii=False))


def recover_question_window_test_from_checkpoint(job_file, campaign_path):
    """Finish question-window evaluation from a completed training checkpoint."""
    disable_pykt_pickle_cache()
    campaign = load_yaml(campaign_path)
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    if campaign["experiment"].get("evaluation") not in {
        "pykt_question_window_late_mean",
        "question_window_late_mean",
    }:
        raise ValueError(
            "Question-window checkpoint recovery requires "
            "pykt_question_window_late_mean evaluation"
        )
    if result_path(job).is_file():
        raise FileExistsError(f"Refusing to overwrite {result_path(job)}")

    checkpoint_configs = sorted(
        Path(job["output_dir"]).glob("checkpoints/**/config.json")
    )
    if len(checkpoint_configs) != 1:
        raise RuntimeError(
            f"Expected exactly one saved checkpoint config for {job['job_id']}, "
            f"found {len(checkpoint_configs)}"
        )
    run_dir = checkpoint_configs[0].parent
    saved = json.loads(checkpoint_configs[0].read_text(encoding="utf-8"))
    params = saved["params"]
    checkpoint_path = run_dir / f"{params['emb_type']}_model.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    result = {
        "fold": int(job["fold"]),
        "model_name": params["model_name"],
        "dataset_name": job["dataset"],
        "emb_type": params["emb_type"],
        "test_auc": None,
        "test_acc": None,
        "window_test_auc": None,
        "window_test_acc": None,
        "valid_auc": None,
        "valid_acc": None,
        "best_epoch": None,
        "model_save_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "recovered_test_only": True,
    }
    report_metric = evaluate_pykt_report_metric(
        result,
        job,
        relation_path=None,
        pykt_root=Path(campaign["runtime"]["pykt_root"]),
        evaluation_batch_size=None,
    )
    result["test_auc"] = report_metric["auc"]
    result["test_acc"] = report_metric["acc"]
    result["report_metric"] = report_metric
    result.update(
        {
            "job_id": job["job_id"],
            "paper_model": job["paper_model"],
            "input_family": job["input_family"],
            "protocol": campaign["experiment"]["scope"],
            "params": job["params"],
            "input_manifest": campaign["paths"]["input_manifest"],
            "artifacts_retained": {
                "checkpoint": bool(
                    campaign["training"].get("retain_checkpoints", False)
                ),
                "predictions": bool(
                    campaign["training"].get("retain_predictions", False)
                ),
            },
        }
    )
    atomic_json(result_path(job), result)
    cleanup_training_artifacts(result, campaign)
    running_path = Path(job["output_dir"]) / "RUNNING"
    if running_path.exists():
        running_path.rename(
            running_path.with_name(f"INTERRUPTED_{int(time.time())}")
        )
    if failed_path(job).exists():
        failed_path(job).unlink()
    print(json.dumps(result, ensure_ascii=False))


def _worker_impl(job_file, campaign_path):
    campaign = load_yaml(campaign_path)
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    # The long-running scheduler may have materialized its job list before a
    # compatibility constraint was recorded. Re-resolve the skip policy here
    # so it cannot later run a scientifically unsupported stale job.
    model_registry = Path(campaign.get("models_config", HERE / "models.yaml"))
    live_models = load_yaml(model_registry)["models"]
    live_models.update(campaign.get("model_extensions", {}))
    live_skip = live_models[job["paper_model"]].get(
        "unsupported_datasets", {}
    ).get(job["dataset"])
    if live_skip:
        job["skip_reason"] = live_skip
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pykt_root = Path(campaign["runtime"]["pykt_root"])
    examples = pykt_root / "examples"
    data_config_path = Path(campaign["paths"]["data_config"])
    auxiliary_relation = None
    model_name = job["params"]["model_name"]
    if materialize_skipped_job(job, campaign):
        print(json.dumps(json.loads(result_path(job).read_text()), ensure_ascii=False))
        return
    if model_name == "rkt":
        auxiliary_relation = prepare_rkt_fold(job, data_config_path, campaign)

    disable_pykt_pickle_cache()
    sys.path.insert(0, str(examples))
    sys.path.insert(0, str(pykt_root))
    if model_name == "dkt_forget":
        install_dkt_forget_input_compatibility(pykt_root)
    os.chdir(examples)
    from wandb_train import main as train_main

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    params = dict(job["params"])
    evaluation = campaign["experiment"].get(
        "evaluation", "pykt_question_window_late_mean"
    )
    native_test = evaluation == "native_pykt_test"
    params.update(
        {
            "save_dir": str(checkpoint_dir),
            "config": str(HERE / "kt_config_v10.json"),
            "data_config": str(data_config_path),
            "batch_size": int(campaign["training"]["batch_size"]),
            "num_epochs": int(campaign["training"]["num_epochs"]),
            "use_wandb": 0,
            "add_uuid": 0,
            "save_model": 1,
            "evaluate_test": int(native_test),
            "evaluate_window_test": 0,
        }
    )
    if "gradient_accumulation_steps" in campaign["training"]:
        params["gradient_accumulation_steps"] = int(
            campaign["training"]["gradient_accumulation_steps"]
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    result = train_main(params)
    atomic_json(
        output_dir / "TRAINING_COMPLETE.json",
        {"job_id": job["job_id"], "completed_at": time.time()},
    )
    report_metric = (
        evaluate_native_pykt_metric(result, job, data_config_path)
        if native_test
        else evaluate_pykt_report_metric(
            result, job, auxiliary_relation, pykt_root
        )
    )
    result["test_auc"] = report_metric["auc"]
    result["test_acc"] = report_metric["acc"]
    result["report_metric"] = report_metric
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result["peak_gpu_memory_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
    result["wall_time_seconds"] = float(time.perf_counter() - started_at)
    result.update(
        {
            "job_id": job["job_id"],
            "paper_model": job["paper_model"],
            "input_family": job["input_family"],
            "protocol": campaign["experiment"]["scope"],
            "params": {
                **job["params"],
                "batch_size": int(campaign["training"]["batch_size"]),
                "gradient_accumulation_steps": int(
                    campaign["training"].get("gradient_accumulation_steps", 1)
                ),
            },
            "input_manifest": campaign["paths"]["input_manifest"],
            "artifacts_retained": {
                "checkpoint": bool(campaign["training"].get("retain_checkpoints", False)),
                "predictions": bool(campaign["training"].get("retain_predictions", False)),
            },
        }
    )
    atomic_json(output_dir / "result.json", result)
    cleanup_training_artifacts(result, campaign)
    if auxiliary_relation and auxiliary_relation.exists():
        auxiliary_relation.unlink()
    print(json.dumps(result, ensure_ascii=False))


def worker(job_file, campaign_path):
    """Run one job once, even when multiple queue launchers share the campaign."""
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if result_path(job).is_file():
        print(f"SKIP completed job={job['job_id']}", flush=True)
        return

    lock_path = output_dir / "RUNNING"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"SKIP active job={job['job_id']}", flush=True)
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    try:
        _worker_impl(job_file, campaign_path)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def select_job(pending, active):
    rkt_running = any(item["job"]["params"]["model_name"] == "rkt" for item in active)
    for index, job in enumerate(pending):
        is_rkt = job["params"]["model_name"] == "rkt"
        if is_rkt and active:
            continue
        if not is_rkt and rkt_running:
            continue
        return index
    return None


def command_run(args, campaign):
    if not Path(campaign["paths"]["data_config"]).is_file():
        raise FileNotFoundError("Run prepare_aligned_data.py before launching jobs")
    if torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is unavailable; no formal baseline jobs were started")
    jobs = filter_jobs(
        read_jobs(campaign["paths"]["jobs_file"]),
        models=args.models,
        datasets=args.datasets,
        folds=args.folds,
    )
    pending = []
    for job in jobs:
        if result_path(job).is_file():
            continue
        recover_stale_running_lock(
            job,
            float(args.recover_stale_running_hours) * 60 * 60,
        )
        if (Path(job["output_dir"]) / "RUNNING").is_file():
            continue
        if failed_path(job).is_file() and not args.retry_failed:
            continue
        pending.append(job)
    gpus = args.gpus or [str(value) for value in campaign["runtime"]["gpus"]]
    if any(int(gpu) >= torch.cuda.device_count() for gpu in gpus):
        raise ValueError(f"Requested GPUs {gpus}; visible count is {torch.cuda.device_count()}")
    max_parallel = min(args.max_parallel or len(gpus), len(gpus))
    active = []
    failures = []
    while pending or active:
        used = {item["gpu"] for item in active}
        free = [gpu for gpu in gpus if gpu not in used]
        while pending and free and len(active) < max_parallel:
            index = select_job(pending, active)
            if index is None:
                break
            job = pending.pop(index)
            gpu = free.pop(0)
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            job_file = output_dir / "job.json"
            atomic_json(job_file, job)
            log = (output_dir / "train.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                [
                    str(campaign["runtime"]["python"]),
                    "-u",
                    str(Path(__file__).resolve()),
                    "--config",
                    str(Path(args.config).resolve()),
                    "worker",
                    "--job-file",
                    str(job_file),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append(
                {"process": process, "job": job, "gpu": gpu, "log": log}
            )
            print(
                f"START gpu={gpu} model={job['paper_model']} dataset={job['dataset']} fold={job['fold']}",
                flush=True,
            )
        if active:
            time.sleep(int(campaign["runtime"]["poll_seconds"]))
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            job = item["job"]
            active.remove(item)
            if code != 0:
                failed_path(job).write_text(f"exit_code={code}\n", encoding="utf-8")
                failures.append(job)
            elif failed_path(job).exists():
                failed_path(job).unlink()
            print(
                f"DONE exit={code} gpu={item['gpu']} model={job['paper_model']} dataset={job['dataset']} fold={job['fold']}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{len(failures)} baseline jobs failed")


def summarize(campaign, jobs):
    paper = {}
    with (HERE / "paper_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            paper[(row["model"], row["dataset"])] = row
    grouped = defaultdict(list)
    skipped = defaultdict(list)
    for job in jobs:
        path = result_path(job)
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            destination = skipped if result.get("status") == "skipped" else grouped
            destination[(job["paper_model"], job["dataset"])].append(result)
    rows = []
    for job in jobs:
        key = (job["paper_model"], job["dataset"])
        if any(row["model"] == key[0] and row["dataset"] == key[1] for row in rows):
            continue
        values = grouped.get(key, [])
        skipped_values = skipped.get(key, [])
        auc = np.asarray([value["test_auc"] for value in values], dtype=float)
        acc = np.asarray([value["test_acc"] for value in values], dtype=float)
        reference = paper.get(key)
        rows.append(
            {
                "model": key[0],
                "dataset": key[1],
                "completed_folds": len(values),
                "skipped_folds": len(skipped_values),
                "complete": len(values) == 5,
                "terminal": len(values) + len(skipped_values) == 5,
                "skip_reason": (
                    skipped_values[0]["skip_reason"] if skipped_values else ""
                ),
                "auc_mean": float(auc.mean()) if len(auc) else "",
                "auc_std": float(auc.std(ddof=0)) if len(auc) else "",
                "acc_mean": float(acc.mean()) if len(acc) else "",
                "acc_std": float(acc.std(ddof=0)) if len(acc) else "",
                "paper_auc": float(reference["auc_mean"]) if reference else "",
                "paper_acc": float(reference["acc_mean"]) if reference else "",
                "auc_delta_vs_paper": (
                    float(auc.mean()) - float(reference["auc_mean"])
                    if len(auc) and reference
                    else ""
                ),
            }
        )
    summary_csv = Path(campaign["paths"]["summary_csv"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successful_jobs = sum(len(values) for values in grouped.values())
    skipped_jobs = sum(len(values) for values in skipped.values())
    payload = {
        "expected_jobs": len(jobs),
        # Keep completed_jobs terminal-based so an already-running legacy launcher
        # can finish after scientifically unsupported cells are recorded as N/A.
        "completed_jobs": successful_jobs + skipped_jobs,
        "successful_jobs": successful_jobs,
        "skipped_jobs": skipped_jobs,
        "terminal_jobs": successful_jobs + skipped_jobs,
        "failed_jobs": sum(
            failed_path(job).is_file() and not result_path(job).is_file()
            for job in jobs
        ),
        "complete_groups": sum(bool(row["complete"]) for row in rows),
        "terminal_groups": sum(bool(row["terminal"]) for row in rows),
        "expected_groups": len(rows),
        "rows": rows,
    }
    atomic_json(campaign["paths"]["summary_json"], payload)
    return payload


def command_status(campaign):
    jobs = read_jobs(campaign["paths"]["jobs_file"])
    payload = summarize(campaign, jobs)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    jobs = subparsers.add_parser("jobs")
    jobs.add_argument("--models", nargs="*")
    jobs.add_argument("--datasets", nargs="*")
    jobs.add_argument("--folds", nargs="*", type=int)
    run = subparsers.add_parser("run")
    run.add_argument("--gpus", nargs="*")
    run.add_argument("--max-parallel", type=int)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--recover-stale-running-hours",
        type=float,
        default=0,
        help="Rename stale RUNNING markers to INTERRUPTED_* before scheduling.",
    )
    run.add_argument("--models", nargs="*")
    run.add_argument("--datasets", nargs="*")
    run.add_argument("--folds", nargs="*", type=int)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--job-file", type=Path, required=True)
    recover = subparsers.add_parser("recover-native-test")
    recover.add_argument("--job-file", type=Path, required=True)
    recover_qw = subparsers.add_parser("recover-question-window-test")
    recover_qw.add_argument("--job-file", type=Path, required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("summarize")
    return parser.parse_args()


def main():
    args = parse_args()
    campaign = load_yaml(args.config)
    model_registry = Path(campaign.get("models_config", HERE / "models.yaml"))
    models = load_yaml(model_registry)["models"]
    models.update(campaign.get("model_extensions", {}))
    if args.command == "jobs":
        command_jobs(args, campaign, models)
    elif args.command == "run":
        command_run(args, campaign)
    elif args.command == "worker":
        worker(args.job_file, args.config)
    elif args.command == "recover-native-test":
        recover_native_test_from_checkpoint(args.job_file, args.config)
    elif args.command == "recover-question-window-test":
        recover_question_window_test_from_checkpoint(args.job_file, args.config)
    elif args.command in {"status", "summarize"}:
        command_status(campaign)


if __name__ == "__main__":
    main()
