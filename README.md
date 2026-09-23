# DualKT Source Code

This repository contains the Python implementation of DualKT and the scripts
used to train, evaluate, and test the model. It intentionally excludes
datasets, experiment configurations, checkpoints, prediction files, logs, and
reported experimental results.

## Code structure

- `V7/` contains the shared sequence backbones and training utilities on which
  the final model is built.
- `V8/` contains the final DualKT architecture, the question-level evaluation
  protocol, training and TPE-search entry points, and focused unit tests.
- `V10/` contains data-alignment, baseline orchestration, and sequence-length
  study utilities.

`V8.DualKT` is the public alias of the final `ModularEvidenceKT` model. Some
internal command-line identifiers retain the historical name `matra4kt` for
compatibility with the training pipeline; they refer to DualKT in this release.

## Script reference

### `V7`: shared model and trainer

- `V7/__init__.py`: marks the shared backbone directory as a Python package.
- `V7/model.py`: implements the reusable Mamba/Transformer sequence encoders,
  attention layers, graph components, and auxiliary model blocks used by
  DualKT.
- `V7/trainer.py`: implements optimization, validation, checkpoint recovery,
  early stopping, and sequence-level evaluation utilities.

### `V8`: DualKT

- `V8/__init__.py`: exports the final model and the `DualKT` alias.
- `V8/model.py`: implements the complete DualKT architecture, including
  dual-timescale state construction, target-conditioned retrieval,
  orthogonal decomposition, target-conditioned fusion, and local transition
  refinement.
- `V8/denoisekt_xes_protocol.py`: implements question-window aggregation and
  question-level AUC/accuracy evaluation.
- `V8/train_denoisekt_xes.py`: trains and evaluates one DualKT fold using the
  pyKT question-level input contract.
- `V8/run_unified_denoisekt_bayes.py`: runs validation-only Optuna TPE search
  and the subsequent fixed-configuration fold evaluation.
- `V8/run_single_model_4dataset_5fold.py`: orchestrates repeated five-fold runs
  for a selected model and dataset collection.
- `V8/run_question_semantic_generalization.py`: loads screening definitions
  and dispatches question-semantic evaluation jobs used by the multi-fold
  runner.
- `V8/tests/test_v8_invariants.py`: checks architectural shapes, causality,
  masks, fusion modes, and module invariants.
- `V8/tests/test_question_window_recovery.py`: checks question-window
  evaluation and restart/recovery behavior.
- `V8/tests/test_unified_bayes_protocol.py`: checks search/final-evaluation
  separation and TPE orchestration behavior.

### `V10`: experiment orchestration

- `V10/__init__.py`: marks the orchestration directory as a Python package.
- `V10/prepare_aligned_data.py`: converts source datasets into aligned
  question-level inputs expected by the training pipeline.
- `V10/build_rkt_phi.py`: constructs the relation matrix required by RKT.
- `V10/run_baseline_suite.py`: prepares, launches, resumes, and summarizes
  fixed-configuration baseline jobs.
- `V10/run_sequence_length_study.py`: prepares and executes one
  maximum-sequence-length study.
- `V10/run_sequence_length_matrix.py`: coordinates sequence-length studies
  across models and datasets.

## Environment

The frozen environment used Python 3.10, PyTorch 2.5.1, CUDA 12.4, and
Mamba-SSM 2.3.2. Install a PyTorch build compatible with the local CUDA
toolchain, then install the remaining packages from `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The TPE runner uses `fcntl`, so the released orchestration scripts are intended
for Linux. Training scripts accept an external YAML file through `--config`;
configuration files are deliberately not included in this source-only release.

## Tests

From the repository root, run:

```bash
pytest V8/tests -q
```

