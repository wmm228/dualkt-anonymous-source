#!/usr/bin/env python3
"""Prepare the exact model-specific pyKT input routes used by DenoiseKT."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config_baseline_reproduction.yaml"
REQUIRED_FILES = (
    "train_valid_original_file",
    "train_valid_file",
    "test_original_file",
    "test_file",
    "train_valid_original_file_quelevel",
    "train_valid_file_quelevel",
    "test_file_quelevel",
    "test_question_window_file",
)


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


def resolve_dpath(entry, pykt_root):
    path = Path(entry["dpath"])
    if path.is_absolute():
        return path.resolve()
    return (Path(pykt_root) / "examples" / path).resolve()


def missing_files(entry):
    dpath = Path(entry["dpath"])
    return [
        key
        for key in REQUIRED_FILES
        if key not in entry or not (dpath / entry[key]).is_file()
    ]


def reuse_question_graph(source_entry, prepared_entry, pykt_root):
    """Reuse the dataset-level DenoiseKT graph across sequence lengths."""
    source = resolve_dpath(source_entry, pykt_root) / "questions_concepts.pt"
    target = Path(prepared_entry["dpath"]).resolve() / "questions_concepts.pt"
    if not source.is_file() or source == target:
        return None
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            pass
    if not target.is_file():
        raise FileNotFoundError(f"failed to reuse question graph: {target}")
    return str(source)


def official_split(dataset, raw_path, output_dir, config_path, pykt_root, maxlen):
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{dataset} needs a maxlen={maxlen} official re-split, but {raw_path} is missing"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, {})
    sys.path.insert(0, str(pykt_root))
    from pykt.preprocess.split_datasets import main as split_concept
    from pykt.preprocess.split_datasets_que import main as split_question

    split_concept(
        str(output_dir), str(raw_path), dataset, str(config_path), 3, maxlen, 5
    )
    split_question(
        str(output_dir), str(raw_path), dataset, str(config_path), 3, maxlen, 5
    )
    generated = json.loads(config_path.read_text(encoding="utf-8"))[dataset]
    generated["dpath"] = str(output_dir.resolve())
    return generated


def prepare_raw_assist2009(raw_csv, output_dir, config_path, pykt_root, maxlen):
    """Create an isolated pyKT ASSIST2009 split from the user-provided CSV."""
    raw_csv = Path(raw_csv).resolve()
    if not raw_csv.is_file():
        raise FileNotFoundError(f"ASSIST2009 raw CSV is missing: {raw_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = output_dir / "data.txt"
    sys.path.insert(0, str(pykt_root))
    from pykt.preprocess.assist2009_preprocess import read_data_from_csv

    read_data_from_csv(str(raw_csv), str(normalized))
    return official_split(
        "assist2009", normalized, output_dir, config_path, pykt_root, maxlen
    )


GENERATED_SEQUENCE_COLUMNS = {
    "selectmasks",
    "qidxs",
    "rest",
    "orirow",
    "cidxs",
}


def _is_padding(value):
    try:
        return float(value) == -1.0
    except (TypeError, ValueError):
        return False


def restore_full_sequences(path):
    """Undo pyKT's contiguous max-length chunks without changing uid/fold."""
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"fold", "uid", "responses"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path} is missing columns needed to restore sequences: {missing}")
    sequence_columns = [
        key
        for key in frame.columns
        if key not in {"fold", "uid"} | GENERATED_SEQUENCE_COLUMNS
    ]
    restored = OrderedDict()
    for row in frame.to_dict("records"):
        uid = str(row["uid"])
        fold = int(row["fold"])
        response_values = row["responses"].split(",")
        masks = row.get("selectmasks", "").split(",")
        if masks != [""] and len(masks) != len(response_values):
            raise ValueError(f"selectmask length mismatch for uid={uid}: {path}")
        valid = [
            index
            for index, response in enumerate(response_values)
            if not _is_padding(response)
            and (masks == [""] or not _is_padding(masks[index]))
        ]
        if uid not in restored:
            restored[uid] = {
                "fold": fold,
                "uid": uid,
                **{key: [] for key in sequence_columns},
            }
        elif restored[uid]["fold"] != fold:
            raise ValueError(f"uid={uid} occurs in multiple folds: {path}")
        for key in sequence_columns:
            values = row[key].split(",")
            if len(values) != len(response_values):
                raise ValueError(f"{key} length mismatch for uid={uid}: {path}")
            restored[uid][key].extend(values[index] for index in valid)
    rows = []
    for item in restored.values():
        rows.append({
            "fold": item["fold"],
            "uid": item["uid"],
            **{key: ",".join(item[key]) for key in sequence_columns},
        })
    return pd.DataFrame(rows, columns=["fold", "uid", *sequence_columns])


def preserve_split_resplit(
    dataset, source_entry, output_dir, config_path, pykt_root, maxlen
):
    """Regenerate windows while preserving the official students and folds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pykt_root))
    from pykt.preprocess.split_datasets import (
        ALL_KEYS,
        generate_question_sequences,
        generate_sequences,
        generate_window_sequences,
        get_inter_qidx,
    )
    from pykt.preprocess.split_datasets_que import (
        generate_sequences as generate_question_level_sequences,
        generate_window_sequences as generate_question_level_windows,
    )

    source_dir = Path(source_entry["dpath"])
    standard_train = restore_full_sequences(
        source_dir / source_entry["train_valid_file"]
    )
    standard_test = restore_full_sequences(
        source_dir / source_entry["test_original_file"]
    )
    question_train = restore_full_sequences(
        source_dir / source_entry["train_valid_file_quelevel"]
    )
    question_test_key = source_entry.get(
        "test_original_file_quelevel", source_entry["test_file_quelevel"]
    )
    question_test = restore_full_sequences(source_dir / question_test_key)

    standard_keys = set(standard_train.columns)
    standard_columns = [key for key in ALL_KEYS if key in standard_keys]
    standard_train[standard_columns].to_csv(
        output_dir / "train_valid.csv", index=False
    )
    generate_sequences(standard_train, standard_keys, 3, maxlen).to_csv(
        output_dir / "train_valid_sequences.csv", index=False
    )

    standard_test["fold"] = -1
    standard_test["cidxs"] = get_inter_qidx(standard_test)
    test_keys = list(standard_keys) + ["cidxs"]
    generate_sequences(standard_test, test_keys, 3, maxlen).to_csv(
        output_dir / "test_sequences.csv", index=False
    )
    generate_window_sequences(standard_test, test_keys, maxlen).to_csv(
        output_dir / "test_window_sequences.csv", index=False
    )
    _, question_sequences = generate_question_sequences(
        standard_test.copy(), standard_keys, False, 3, maxlen
    )
    _, question_windows = generate_question_sequences(
        standard_test.copy(), standard_keys, True, 3, maxlen
    )
    question_sequences.to_csv(
        output_dir / "test_question_sequences.csv", index=False
    )
    question_windows.to_csv(
        output_dir / "test_question_window_sequences.csv", index=False
    )
    standard_test[[*standard_columns, "cidxs"]].to_csv(
        output_dir / "test.csv", index=False
    )

    question_keys = set(question_train.columns)
    question_columns = [key for key in ALL_KEYS if key in question_keys]
    question_train[question_columns].to_csv(
        output_dir / "train_valid_quelevel.csv", index=False
    )
    generate_question_level_sequences(
        question_train, question_keys, 3, maxlen
    ).to_csv(output_dir / "train_valid_sequences_quelevel.csv", index=False)
    question_test["fold"] = -1
    question_test[question_columns].to_csv(
        output_dir / "test_quelevel.csv", index=False
    )
    generate_question_level_sequences(
        question_test, question_keys, 3, maxlen
    ).to_csv(output_dir / "test_sequences_quelevel.csv", index=False)
    generate_question_level_windows(
        question_test, question_keys, maxlen
    ).to_csv(output_dir / "test_window_sequences_quelevel.csv", index=False)

    generated = dict(source_entry)
    generated.update({
        "dpath": str(output_dir.resolve()),
        "maxlen": int(maxlen),
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv",
        "test_question_file": "test_question_sequences.csv",
        "test_question_window_file": "test_question_window_sequences.csv",
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
    })
    atomic_json(config_path, {dataset: generated})
    return generated


def fold_membership(path):
    csv.field_size_limit(sys.maxsize)
    mapping = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = str(row["uid"])
            fold = int(row["fold"])
            previous = mapping.setdefault(uid, fold)
            if previous != fold:
                raise ValueError(f"uid {uid} occurs in folds {previous} and {fold}: {path}")
    digest = hashlib.sha256()
    for uid, fold in sorted(mapping.items()):
        digest.update(f"{uid}\t{fold}\n".encode("utf-8"))
    return mapping, digest.hexdigest()


def audit_entry(dataset, entry, source_mode):
    missing = missing_files(entry)
    if missing:
        raise FileNotFoundError(f"{dataset} is missing pyKT files: {missing}")
    dpath = Path(entry["dpath"])
    standard_train, standard_digest = fold_membership(
        dpath / entry["train_valid_file"]
    )
    question_train, question_digest = fold_membership(
        dpath / entry["train_valid_file_quelevel"]
    )
    standard_test, standard_test_digest = fold_membership(
        dpath / entry["test_file"]
    )
    question_test, question_test_digest = fold_membership(
        dpath / entry["test_file_quelevel"]
    )
    train_match = standard_train == question_train
    test_match = standard_test == question_test
    if not train_match or not test_match:
        raise ValueError(
            f"{dataset} standard/question pyKT views do not share the same student folds"
        )
    if set(standard_train.values()) != set(range(5)):
        raise ValueError(f"{dataset} train/validation folds are not 0-4")
    if set(standard_test.values()) != {-1}:
        raise ValueError(f"{dataset} test fold is not -1")
    return {
        "dataset": dataset,
        "source_mode": source_mode,
        "dpath": str(dpath),
        "maxlen": int(entry["maxlen"]),
        "train_students": len(standard_train),
        "test_students": len(standard_test),
        "standard_train_fold_sha256": standard_digest,
        "question_train_fold_sha256": question_digest,
        "standard_test_fold_sha256": standard_test_digest,
        "question_test_fold_sha256": question_test_digest,
        "student_fold_alignment": True,
        "files": {
            key: {
                "path": str(dpath / entry[key]),
                "bytes": (dpath / entry[key]).stat().st_size,
            }
            for key in REQUIRED_FILES
        },
    }


def prepare_dataset(dataset, source_entry, campaign):
    pykt_root = Path(campaign["runtime"]["pykt_root"])
    maxlen = int(campaign["experiment"]["sequence_length"])
    entry = dict(source_entry)
    source_dir = resolve_dpath(entry, pykt_root)
    entry["dpath"] = str(source_dir)
    raw_sources = campaign.get("raw_sources", {})
    if dataset in raw_sources:
        raw_csv = raw_sources[dataset].get("raw_csv")
        if dataset != "assist2009" or not raw_csv:
            raise ValueError(
                "Only ASSIST2009 raw_csv overrides are supported by this protocol"
            )
        output_dir = Path(campaign["paths"]["data_root"]) / dataset
        generated_config_path = output_dir / "data_config.json"
        if generated_config_path.is_file():
            generated_config = json.loads(
                generated_config_path.read_text(encoding="utf-8")
            ).get(dataset)
            if generated_config:
                generated_config["dpath"] = str(output_dir.resolve())
                if (
                    int(generated_config.get("maxlen", -1)) == maxlen
                    and not missing_files(generated_config)
                ):
                    return generated_config, "user_supplied_assist2009_raw_csv"
        generated = prepare_raw_assist2009(
            raw_csv, output_dir, generated_config_path, pykt_root, maxlen
        )
        return generated, "user_supplied_assist2009_raw_csv"

    if int(entry.get("maxlen", -1)) == maxlen and not missing_files(entry):
        return entry, "existing_official_pykt"

    reusable_root = campaign["paths"].get("reusable_data_root")
    if reusable_root:
        reusable_config_path = Path(reusable_root) / dataset / "data_config.json"
        if reusable_config_path.is_file():
            reusable = json.loads(
                reusable_config_path.read_text(encoding="utf-8")
            ).get(dataset)
            if reusable:
                reusable["dpath"] = str(
                    (Path(reusable_root) / dataset).resolve()
                )
                if (
                    int(reusable.get("maxlen", -1)) == maxlen
                    and not missing_files(reusable)
                ):
                    return reusable, "local_official_pykt_cache"

    output_dir = Path(campaign["paths"]["data_root"]) / dataset
    generated_config_path = output_dir / "data_config.json"
    if generated_config_path.is_file():
        generated_config = json.loads(
            generated_config_path.read_text(encoding="utf-8")
        ).get(dataset)
        if generated_config:
            generated_config["dpath"] = str(output_dir.resolve())
            if (
                int(generated_config.get("maxlen", -1)) == maxlen
                and not missing_files(generated_config)
            ):
                return generated_config, "official_pykt_splitters"
    dataset_options = campaign["datasets"].get(dataset, {})
    if dataset_options.get("preserve_official_split"):
        entry = preserve_split_resplit(
            dataset,
            entry,
            output_dir,
            generated_config_path,
            pykt_root,
            maxlen,
        )
        return entry, "preserved_official_split_reslice"
    entry = official_split(
        dataset, source_dir / "data.txt", output_dir, generated_config_path,
        pykt_root, maxlen,
    )
    return entry, "official_pykt_splitters"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="only add reusable DenoiseKT question graphs to prepared data",
    )
    args = parser.parse_args()
    campaign = load_yaml(args.config)
    source_config_path = Path(campaign["runtime"]["source_data_config"]).resolve()
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    selected = args.datasets or list(campaign["datasets"])
    unknown = sorted(set(selected) - set(campaign["datasets"]))
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}")

    if args.graphs_only:
        data_config_path = Path(campaign["paths"]["data_config"])
        prepared = json.loads(data_config_path.read_text(encoding="utf-8"))
        for dataset in selected:
            source = reuse_question_graph(
                source_config[dataset], prepared[dataset], campaign["runtime"]["pykt_root"]
            )
            print(
                f"GRAPH dataset={dataset} target={prepared[dataset]['dpath']} "
                f"source={source or 'already-local-or-unavailable'}",
                flush=True,
            )
        return

    prepared = {}
    manifests = {}
    for dataset in selected:
        print(f"PREPARE dataset={dataset}", flush=True)
        prepared[dataset], source_mode = prepare_dataset(
            dataset, source_config[dataset], campaign
        )
        graph_source = reuse_question_graph(
            source_config[dataset], prepared[dataset], campaign["runtime"]["pykt_root"]
        )
        manifests[dataset] = audit_entry(dataset, prepared[dataset], source_mode)
        if graph_source:
            manifests[dataset]["question_graph_source"] = graph_source

    atomic_json(campaign["paths"]["data_config"], prepared)
    atomic_json(
        campaign["paths"]["input_manifest"],
        {
            "protocol": {
                "source": "DenoiseKT/pyKT model-specific preprocessing",
                "max_sequence_length": int(campaign["experiment"]["sequence_length"]),
                "standard_models_train": "train_valid_file",
                "question_models_train": "train_valid_file_quelevel",
                "standard_models_report": "test_question_window_file + late_mean fusion",
                "question_models_report": "test_window_file_quelevel + direct prediction",
                "split_assertion": "standard and question views share uid-to-fold mappings",
            },
            "datasets": manifests,
        },
    )
    print(
        json.dumps(
            {
                "datasets": selected,
                "data_config": campaign["paths"]["data_config"],
                "input_manifest": campaign["paths"]["input_manifest"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
