# IMU Fall Benchmark

[![CI](https://github.com/Kscii/imu-fall-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Kscii/imu-fall-benchmark/actions/workflows/ci.yml)

This repository provides a six-axis IMU fall-detection benchmark for internal team use. It versions the training code, data contract, folds, experiment configurations, tests, and result formats. HDF5 datasets are not stored in Git; they are downloaded from the project's Google Cloud Storage (GCS) bucket.

The current release covers one task only: training and evaluating causal sliding-window fall classifiers using temporal interval labels. It also evaluates a fixed catalog of causal N-of-M and cooldown alarm policies. Wear-location classification, recording-level MIL, Android implementation, and automated deployment are outside the current scope.

The model output is named `fall_score`. It must not be described as a fall probability unless a separate probability-calibration process is completed later.

## Runtime environment

Training and evaluation officially support only:

- Windows 11 with WSL2;
- an NVIDIA CUDA GPU, with an RTX 4070 SUPER 12 GB used as the reference machine;
- a repository checkout in the WSL Linux filesystem, such as `~/projects/imu-fall-benchmark`, not under `/mnt/c`;
- Python 3.12, CUDA 12.9, RAPIDS 26.08, XGBoost 3.4.0, and PyTorch 2.8.0.

Install only the WSL2-compatible NVIDIA driver on Windows. Do not install a second Linux display driver inside WSL.

## First-time setup

Install the minimal system dependencies inside WSL2:

```bash
sudo apt update
sudo apt install --yes git curl ca-certificates tar
```

Clone the public repository into the WSL Linux filesystem, then run:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Kscii/imu-fall-benchmark.git
cd ~/projects/imu-fall-benchmark
./setup
imu-bench data pull
imu-bench doctor
imu-bench smoke
```

`./setup` is the only command launched through Bash. It installs or reuses pinned Miniforge, Google Cloud CLI, and CUDA Python environments under `~/imu-fall-work`, then registers `imu-bench` at `~/.local/bin/imu-bench`. It does not modify the system Python installation or shell profile. If `~/.local/bin` is not already on `PATH`, the script prints one `export` command for the current terminal. A first installation normally takes 20–60 minutes and should have at least 25 GiB of free disk space.

Running `./setup` from another checkout intentionally makes that checkout the active `imu-bench` source. Dependency environments are keyed by file contents, so compatible checkouts share the large CUDA environment even when their paths differ.

The first `data pull` asks the user to sign in to Google. It downloads the immutable snapshot referenced by `current.json`, then checks each file's SHA-256, HDF5 v3.1 structure, logical fingerprint, statistics, and bundled participant split. An experiment can pin an exact snapshot without following `current.json`:

```bash
imu-bench data pull --base-snapshot imu_25hz_snapshot_v2
```

Git LFS is not required. Each WSL user signs in only once: the first command must be run directly in an interactive WSL terminal. If WSL cannot open a browser and reports a `gio` error, open the displayed URL in a Windows browser and complete the login there. The URL printed after `gio:` is the same OAuth request repeated by the failed Linux browser opener, not a second login link. Non-interactive SSH or Codex automation neither copies host credentials nor continues when authentication requires input. Later pulls can run automatically after sign-in.

## Routine commands

```bash
imu-bench data status
imu-bench data pull
imu-bench validate-data
imu-bench test
imu-bench doctor
imu-bench smoke
```

- `data status`: compare the active local manifest with the remote current pointer;
- `data pull`: atomically install current or explicitly named base/team snapshots;
- `validate-data`: verify all HDF5 files, hashes, statistics, and participant folds;
- `test`: run Ruff and pytest;
- `doctor`: verify WSL2, CUDA, the GPU, and all seven model backends;
- `smoke`: run a bounded fold-0, seven-model end-to-end check with compatible checkpoint reuse.

Interactive terminals use Rich progress bars. Redirected output and CI use stable line-oriented progress instead. Override this with `--progress auto|plain|off`; the flag may appear before or after the command. Progress and diagnostics always use stderr. Normal stdout is a compact human-readable summary. Use `--json` for the complete machine-readable payload, for example:

```bash
imu-bench data pull --progress plain
imu-bench smoke --json --progress off > smoke-result.json
imu-bench --version
```

## Data contract

The machine-readable contract is [`configs/contracts/imu_benchmark_contract_v2.json`](configs/contracts/imu_benchmark_contract_v2.json). Its key rules are:

- HDF5 schema: `3.1.0`;
- input: 25 Hz, six-axis IMU, `float32`, sensor-local coordinates, gravity retained;
- window: 50 frames, or 2 seconds;
- stride: 0.5 seconds in physical time, using half-up grid starts `0, 13, 25, 38, ...` at 25 Hz;
- decision time: the final frame of a window;
- positive sample: the decision time is inside the half-open interval `[fall_start, fall_stop)`;
- a window intersecting an explicit `exclude` interval is removed;
- a window whose decision time is after a fall interval but that still contains its tail is removed, avoiding negative samples that use post-fall information;
- reference alarm rule: one positive window triggers an alarm episode, followed by a 10-second cooldown;
- evaluated alternatives: 1-of-1, consecutive 2, 2-of-3, and 3-of-5, each with 10- or 30-second cooldowns. Pareto candidates are selected from validation data only.

The public base snapshot contains nine HDF5 shards. KFall and UNIVRFall provide usable temporal fall intervals. In datasets with recording-level labels only, ADL recordings can provide negative windows, while fall recordings without temporal intervals are not used as temporal positives. Future CW12EU snapshots from the annotation platform must contain temporal intervals.

## Folds and team data

Public data uses fixed, participant-disjoint five-fold evaluation. For test fold `k`, validation uses `(k + 1) mod 5`, and the other three folds are used for training. Thresholds are selected only from validation-fold Balanced Accuracy. The split CSV is part of the immutable base snapshot. A sticky split update keeps every existing participant in the same fold and assigns only new participants:

```bash
imu-bench split propose --version participant_5fold_v5
```

Team snapshots produced by the annotation platform use `fold_id = -1`:

- they may be added to the training set of every fold;
- they must never enter validation or test sets;
- this allows limited device-specific data to assist training without inflating evaluation by testing on participants or recordings seen during training.

Team recordings must be assigned manually to a development or sealed-holdout role before they are used as product evidence. The current small team snapshot remains training-only; a sealed holdout is not considered ready until it contains at least three participants. This readiness rule is non-blocking for the public-data baseline.

Any change to a data snapshot, fold, contract, or experiment configuration changes the cache and run fingerprints. Old runs are never reused silently.

## Experiment configurations

The retained engineering and formal entry points are:

| Configuration | Purpose |
|---|---|
| `temporal_smoke_v1.yaml` | Bounded end-to-end check; not model evidence |
| `onnx_preflight_v1.yaml` | Seven-model ONNX opset-18 conversion and Python Runtime parity check |
| `onnx_full_parity_preflight_v1.yaml` | One-fold timing gate over complete validation/test splits at batch 256 |
| `formal_pipeline_smoke_v1.yaml` | Bounded check of both recipes, alarms, OOF shards, and statistics |
| `formal_baseline_main_v1.yaml` | Main five-fold, five-seed public-data baseline; 265 jobs |
| `formal_baseline_temporal_core_v1.yaml` | KFall + UNIVRFall single-seed ablation; 65 jobs |
| `formal_baseline_temporal_core_onnx_v1.yaml` | Same 65 jobs with complete validation/test ONNX parity |

Inspect a plan before running it:

```bash
imu-bench run configs/experiments/onnx_preflight_v1.yaml --resume
imu-bench run configs/experiments/onnx_full_parity_preflight_v1.yaml --resume
imu-bench run configs/experiments/formal_pipeline_smoke_v1.yaml --resume
imu-bench plan configs/experiments/formal_baseline_main_v1.yaml
imu-bench plan configs/experiments/formal_baseline_temporal_core_v1.yaml
imu-bench plan configs/experiments/formal_baseline_temporal_core_onnx_v1.yaml
imu-bench run configs/experiments/formal_baseline_temporal_core_onnx_v1.yaml --resume
```

The seven models are Threshold Impact, cuML Logistic Regression, cuML Random Forest, CUDA XGBoost, PyTorch 1D CNN, PyTorch LSTM, and PyTorch CNN-LSTM. The three tabular models consume 158 engineered features; the three deep sequence models consume raw `50 x 6` windows; Threshold Impact reads the raw window directly.

The current execution scope is the 65-job temporal-core experiment. The 265-job main configuration remains available but is not part of this run. Both compare natural sampling with participant/class-balanced sampling. Temporal-core uses seed `3888`, FP32, at most 100 epochs, patience 12, and 5,000 hierarchical bootstrap replicates. Deterministic Threshold jobs are not repeated across recipes.

Formal configurations require a clean Git commit or immutable source snapshot. The source fingerprint, engine schema, resolved configuration, data snapshot, and split fingerprint all participate in run/checkpoint identity. Engineering smoke runs may use a local source copy but do not become formal evidence.

ONNX export is deliberately split into two stages. The bounded preflight proves that every current model type can be represented at opset 18 and supports dynamic batches 1, 7, and 256. The full-parity preflight measures the cost of streaming complete validation and test splits through Python ONNX Runtime CPU at batch 256. If its projected 65-job overhead is at most 15 minutes, the ONNX temporal-core configuration exports every fold/recipe model and compares every score with an explicit `rtol=1e-4` and `atol=2e-3`. The absolute tolerance covers the measured FP32 recurrent-model conversion tail while `onnx_parity.csv` retains max, mean, and p99 error for every split. These cross-validation files are audit artefacts, not final Android packages. After later model selection, shortlisted combinations still require a separate fixed refit and Android Runtime validation.

## Outputs and recovery

Generated content is stored outside the repository by default:

```text
~/imu-fall-work/
├── data/                 # Verified immutable snapshots and active.json
├── cache/                # Content-addressed 50x6 windows and 158 features
├── envs/                 # Pinned dependency environments
├── runs/<run-id>/        # Configs, checkpoints, metrics, logs, and reports
└── toolchains/           # Miniforge and Google Cloud CLI
```

Set `IMU_BENCH_WORK_ROOT` to another absolute path if needed. Each completed job writes its checkpoint atomically. After an interruption, `--resume` reuses only results whose source, engine, configuration, data, split, and job fingerprints match exactly.

Primary outputs include `resolved_config.yaml`, `environment.json`, `provenance.json`, `events.jsonl`, `jobs/*.npz`, native models under `models/`, `metrics.csv`, `event_metrics.csv`, `alarm_metrics.csv`, `subgroup_metrics.csv`, `performance.json`, and `report.md`. Formal runs also write `oof_manifest.json`, `participant_metrics.csv`, `aggregate_metrics.csv`, `paired_comparisons.csv`, and `statistical_manifest.json`. ONNX-enabled runs add `models/*/model.onnx` and `onnx_parity.csv`.

The formal confidence intervals use participant-cluster and seed hierarchical bootstrap. Paired comparisons are limited to each candidate against Threshold Impact and balanced-versus-natural sampling within the same model. Window false positives and product alarm episodes are reported separately.

## Data publication layout

The default bucket is `gs://soft3888-label`. Override it with `IMU_BENCH_DATA_BUCKET=gs://bucket-name`. The benchmark uses only this prefix:

```text
benchmark-datasets/
├── base/<snapshot-id>/datasets/*.h5
├── base/<snapshot-id>/splits/*.csv
├── base/<snapshot-id>/manifest.json
├── base/current.json
├── team/cw12eu/<snapshot-id>/datasets/*.h5
├── team/cw12eu/<snapshot-id>/manifest.json
└── team/cw12eu/current.json
```

`benchmark-datasets/` is a GCS managed folder. Team read-only IAM should be bound only to this resource, not to the entire bucket. Snapshot objects are immutable; `current.json` is a small, explicit pointer. Publishing the public base is a two-stage maintainer operation. The first command stages immutable files without changing `current.json`; the second performs an explicit compare-and-switch after review:

```bash
imu-bench data publish-base \
  --source-dir /path/to/reviewed/imu_25hz \
  --split-dir data/splits \
  --manifest configs/data/base_imu25_v2.json

imu-bench data activate-base \
  --manifest configs/data/base_imu25_v2.json \
  --expected-current imu_25hz_snapshot_v1
```

Routine users only run `data pull` and do not need bucket write access. Credentials, login caches, HDF5 files, run outputs, and the local `TODO.md` must never be committed to Git.

## Immutable result publication

Maintainers may publish only a clean, complete 65-job temporal-core run that uses the exact `imu_25hz_snapshot_v2` snapshot and has PASS statistical outputs:

```bash
imu-bench results publish <run-id>
imu-bench results verify <run-id>
imu-bench results pull <run-id>
```

Each run is stored under `gs://soft3888-label/benchmark-results/temporal-core/<run-id>/` as a no-clobber `run.tar.gz`, a SHA-256 manifest, and directly readable report/summary files. The bundle contains native models, optional ONNX models, checkpoints, OOF scores, metrics, configuration, environment, and provenance. There is no mutable results `current` pointer.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for collaboration rules.
