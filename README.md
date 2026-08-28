# IMU Fall Benchmark

This repository is a self-contained WSL2/CUDA benchmark for six-axis IMU fall detection. It
versions the six HDF5 v3 datasets, participant folds, experiment definitions, training code, and
tests needed by another team member to clone the repository and reproduce a run.

The current primary task is temporally supervised fall-window detection on KFall. Five public
recording-labelled datasets remain available for explicitly marked research views: they may add
ADL negatives to KFall training, or support a recording-level Multiple Instance Learning (MIL)
experiment. Position classification and the old position-model matrix are outside the current
public scope.

This is a research benchmark, not the Android alert product. It outputs a `fall_score`; the score
must not be described as a calibrated probability unless a later calibration contract proves it.

## Reference environment

The verified reference host is Windows 11 with WSL2 Ubuntu 22.04.5, an NVIDIA RTX 4070 SUPER
(12 GB), Python 3.12, CUDA 12.9, RAPIDS 26.08, XGBoost 3.4.0, and PyTorch 2.8.0 CUDA 12.9.
Compute commands intentionally reject native Linux, macOS, Windows Python, WSL1, and repositories
stored under `/mnt/`.

Keep the clone in the WSL Linux filesystem, for example `~/projects/imu-fall-benchmark`. Install
the NVIDIA display driver on Windows; do not install a second Linux display driver inside WSL.

## First-time setup

```bash
sudo apt update
sudo apt install --yes git git-lfs curl ca-certificates

mkdir -p ~/projects
cd ~/projects
git clone <private-repository-url> imu-fall-benchmark
cd imu-fall-benchmark
./benchmark setup
```

`setup` pulls and verifies Git LFS objects, installs or reuses a pinned Miniforge toolchain, and
creates an immutable environment keyed by the dependency manifests. A fresh installation commonly
takes 20–60 minutes depending on network speed and requires at least 25 GiB free space.

## Normal workflow

Run these commands from the repository root:

```bash
./benchmark doctor
./benchmark test
./benchmark validate-data
./benchmark smoke
```

- `doctor` checks WSL2, paths, disk space, CUDA, BF16 support, and all seven public model paths.
- `test` runs Ruff and pytest inside the same pinned environment.
- `validate-data` checks Git LFS bytes, HDF5 v3 structure, snapshot hashes, and participant folds.
- `smoke` runs the default seven-model, fold-0, FP32 KFall experiment and resumes compatible jobs.

The first smoke builds a unified derived cache. It reads every source HDF5 once and stores raw
60×6 windows, 158 engineered features, recording labels, temporal labels, and fold metadata. Cache
writes are flushed in batches of 16,384 windows. Later runs reuse the content-addressed cache and
materialise each required input array once per invocation.

On the reference RTX 4070 SUPER machine, the fresh unified cache took 97.2 seconds to build for
468,728 windows. The first seven-model smoke invocation, including that build, took 115.7 seconds;
a fully resumed engine invocation took 2.34 seconds. These timings are engineering acceptance
evidence, not model-quality evidence. The previous B3 implementation spent about 144 seconds on
its public-data cache and 41 seconds on its separate KFall cache.

## Versioned experiments

Experiments are YAML files under `configs/experiments/`. Inspect a plan without starting CUDA work:

```bash
./benchmark plan configs/experiments/kfall_reproduce_v1.yaml
```

Run and resume the complete five-fold KFall experiment:

```bash
./benchmark run configs/experiments/kfall_reproduce_v1.yaml --resume
```

Regenerate a report without retraining:

```bash
./benchmark report <run-id>
```

The configuration is split deliberately into three small layers:

- `configs/experiments/`: folds, seeds, precision, GPU mode, runtime limits, and selected models;
- `configs/data_views/`: which datasets may train, supplement, and evaluate a model;
- `configs/models/`: the seven public model definitions and fixed hyperparameters.

The main data views are:

| Data view | Training | Validation/test | Status |
|---|---|---|---|
| `kfall_temporal_v1` | temporally labelled KFall windows | participant-disjoint KFall | primary |
| `kfall_public_adl_v1` | KFall plus public ADL negatives | KFall only | research only |
| `public_recording_mil_v1` | five recording-labelled datasets | public folds plus frozen KFall transfer | research only |

For fold `k`, fold `k` is test, fold `(k + 1) mod 5` is validation, and the remaining three folds
train the model. Threshold selection uses validation Balanced Accuracy only. Updating the dataset
snapshot therefore requires rerunning every affected experiment; old run directories remain tied
to their original config and cache fingerprints.

## Models and precision

The public model set is:

1. Threshold Impact baseline;
2. cuML Logistic Regression;
3. cuML Random Forest;
4. CUDA XGBoost;
5. PyTorch 1D CNN;
6. PyTorch LSTM;
7. PyTorch CNN-LSTM.

Tabular models use the 158 engineered features. Sequence models read the 60×6 raw window. FP32 is
the default. BF16 is available only for PyTorch forward and loss computation; source arrays,
normalisation statistics, thresholds, and reported scores remain FP32 or float64 as appropriate.
Compare BF16 only against a run with the same data view, fold, seed, sampling, and model settings.

`gpu_mode` can be `auto`, `resident`, or `streaming`. Auto mode reserves the larger of 4 GiB or
40% of total VRAM, then keeps prepared tensors resident only if the estimate fits. Streaming uses
pinned host memory and asynchronous batch transfers. A resident allocation OOM in auto mode is
retried once in streaming mode and recorded in job metadata; forced resident mode fails rather
than silently changing policy. The two sequence-only acceptance configs make both paths directly
testable.

## Fixed data and window contract

- Input: 30 Hz, six axes, gravity retained, sensor-local axes.
- Window: 60 frames (2 seconds), stride 15 frames (0.5 seconds), causal decision at the last frame.
- Temporal positive: the decision frame is inside the fall activity half-open interval.
- Temporal exclusion: explicit exclude overlap and post-segment overlap are removed.
- Recording MIL: the mean of the highest-scoring 10% of windows is the recording score.
- Alarm policy: one above-threshold window detects an event; alert merging and N-of-M are future
  product-policy experiments.
- Seed: `3888` by default; PyTorch deterministic algorithms are enabled.

The machine-readable protocol is
[`configs/contracts/imu_benchmark_contract_v1.json`](configs/contracts/imu_benchmark_contract_v1.json).
The immutable six-file snapshot is [`data/snapshot_v1.json`](data/snapshot_v1.json). Experiment
YAML cannot override contract-owned sampling, window, label, or metric rules.

KFall currently remains `provisional_kfall_adapter_v1`: onset and impact may be shifted by one
30 Hz sample, ADL codes are placeholders, and the fall activity tail ends one second after impact.
The unified cache enforces the current regression of 53,365 KFall windows, including 8,027 positive
and 45,338 negative windows. Until the adapter is corrected and re-versioned, runs are implementation
and exploration evidence rather than formal model-validation evidence.

## Metrics and artefacts

Each run records Balanced Accuracy, sensitivity, specificity, precision, F1, MCC, AUROC, AUPRC,
and confusion counts. Temporal runs also record event sensitivity, ADL false-positive decision
windows per hour, and onset/impact-relative latency. Dataset and body-location subgroup CSVs are
generated when those groups exist in the selected test view.

Stable run IDs are derived from the resolved configuration and data-cache fingerprint. Outputs are
written below `~/imu-fall-work/runs/<run-id>/` by default:

```text
resolved_config.yaml
environment.json
provenance.json
plan.json
events.jsonl
jobs/*.npz
metrics.csv
event_metrics.csv
subgroup_metrics.csv
external_metrics.csv
performance.json
run_manifest.json
report.md
```

`subgroup_metrics.csv` and `external_metrics.csv` are emitted only when the selected data view
produces those result scopes.

Set `IMU_BENCH_WORK_ROOT` to an absolute path to move both cache and run output. Keeping generated
state outside the clone allows source snapshots to be replaced without losing resumable jobs.

See [`data/README.md`](data/README.md) for dataset sources, fields, supervision boundaries, and how
to add a future HDF5 v3 dataset.
