# Contributing

## Current scope

This repository currently maintains only the WSL2/CUDA temporal sliding-window fall benchmark. Before adding a task, open an Issue describing its purpose, data scope, contract changes, expected deliverables, and acceptance method. Wear-location classification, Android development, the annotation UI, cloud services, and product alert policies belong in their respective repositories or later workstreams.

## Checks before committing

Run inside WSL2:

```bash
imu-bench test
imu-bench validate-data
imu-bench plan configs/experiments/temporal_smoke_v1.yaml
imu-bench run configs/experiments/onnx_preflight_v1.yaml --resume
imu-bench run configs/experiments/formal_pipeline_smoke_v1.yaml --resume
```

For pure Python or contract unit-test changes, another Linux development host may run:

```bash
uv sync --dev
uv run ruff check src tests
uv run pytest -q
```

Formal training, `doctor`, and `smoke` still require WSL2 with NVIDIA CUDA.

GitHub Actions runs only the shell syntax check, Ruff, and data-independent pytest suite on a CPU runner. A green CI check does not replace WSL2/CUDA, GCS, or real-data acceptance.

## Data and contract rules

- Do not commit `*.h5`, caches, runs, model checkpoints, cloud credentials, or the local `TODO.md`.
- Manage data through immutable GCS snapshots and explicit `current.json` pointers.
- Never replace an already published snapshot object in place.
- Base data uses folds `0..4`; team data must use the training-only fold `-1`.
- Changes to sampling rate, windows, stride, labels, folds, or metrics must increment the relevant schema, contract, or configuration version and add regression tests.
- Never change a seed, split, threshold-selection rule, or data filter silently to improve results.
- Preserve the resolved configuration, source provenance, environment, logs, and machine-readable metrics for every result.
- Stage immutable snapshot files first and switch `current.json` only as a separate reviewed operation.

## Code rules

- Keep the public CLI small. Prefer extending the existing `setup / data / doctor / test / smoke / plan / run / report` workflow.
- Store configurations under `configs/`; do not scatter protocol constants across command-line flags.
- Use English for code identifiers, comments, and tracked documentation.
- Write generated content to `IMU_BENCH_WORK_ROOT`, not into the repository.
- Keep commits single-purpose. Avoid mixing data-contract, model-logic, and documentation rewrites into one unreviewable commit.

## Reporting results

A smoke run proves only that the workflow executes. ONNX preflight proves conversion and Python Runtime parity, not Android or model quality. Formal comparison requires a clean source, exact snapshot, frozen folds, seeds, recipes, alarm policies, retained OOF scores, and participant-level uncertainty analysis. Do not call `fall_score` a probability before completing a separate calibration process.
