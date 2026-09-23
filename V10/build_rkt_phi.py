#!/usr/bin/env python3
"""Build the fold-local RKT Phi relation without validation/test leakage."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_values(value):
    return [int(item) for item in value.split(",")]


def build_sparse(input_path, train_folds, output_path):
    counts = defaultdict(lambda: [0, 0, 0, 0])
    current_uid = None
    completed = set()
    last_response = {}
    rows = 0
    interactions = 0
    pairs = 0
    with Path(input_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            uid = row["uid"]
            if uid != current_uid:
                if current_uid is not None:
                    completed.add(current_uid)
                if uid in completed:
                    raise ValueError(f"Rows for uid {uid} are not contiguous")
                current_uid = uid
                last_response = {}
            if int(row["fold"]) not in train_folds:
                continue
            questions = parse_values(row["questions"])
            responses = parse_values(row["responses"])
            for target, response in zip(questions, responses):
                if target < 0 or response < 0:
                    break
                for previous, previous_response in last_response.items():
                    counts[(target, previous)][2 * previous_response + response] += 1
                pairs += len(last_response)
                interactions += 1
                last_response[target] = response
    relation = {}
    for (target, previous), (n00, n01, n10, n11) in counts.items():
        denominator = math.sqrt(
            (n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10)
        )
        if denominator == 0:
            continue
        value = (n11 * n00 - n01 * n10) / denominator
        if value:
            relation.setdefault(target, {})[previous] = float(value)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    pd.to_pickle(relation, temporary)
    temporary.replace(output_path)
    return {
        "format": "sparse_dict",
        "rows": rows,
        "interactions": interactions,
        "pairs": pairs,
        "nonzero": sum(len(values) for values in relation.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, required=True)
    parser.add_argument("--train-folds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pykt-root", type=Path, required=True)
    args = parser.parse_args()
    folds = {int(value) for value in args.train_folds.split(",") if value}
    if not folds:
        raise ValueError("At least one training fold is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.num_questions > 100000:
        result = build_sparse(args.input, folds, args.output)
    else:
        builder = (
            args.pykt_root
            / "experiments"
            / "fakt_reproduction"
            / "generate_rkt_phi.py"
        )
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--input",
                str(args.input),
                "--num-questions",
                str(args.num_questions),
                "--train-folds",
                ",".join(str(value) for value in sorted(folds)),
                "--output",
                str(args.output),
            ],
            check=True,
        )
        matrix = pd.read_pickle(args.output)
        result = {
            "format": "dense_array",
            "shape": list(matrix.shape),
            "nonzero": int((matrix != 0).sum()),
        }
    print(json.dumps({"output": str(args.output), **result}))


if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)
    main()
