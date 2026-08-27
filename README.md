# IMU Fall Benchmark

This repository reproduces the current chest/waist IMU fall-detection benchmark on an
NVIDIA GPU. It is self-contained: five recording-labelled training datasets, the temporally
labelled KFall external dataset, fixed participant folds, experiment configurations, and training
code are versioned together.

The repository is a research benchmark, not an online fall-alert product. The five training
datasets do not share reliable onset/impact intervals, so model fitting uses recording-level
Multiple Instance Learning (MIL). A separate provisional KFall workflow evaluates the frozen
models against onset/impact labels. It still does not estimate false alarms per hour or test an
N-of-M alert policy.

## Reference environment

The verified reference host is Windows 11 with WSL2 Ubuntu 22.04.5, an NVIDIA RTX 4070 SUPER
(12 GB), Python 3.12, CUDA 12.9, RAPIDS 26.08, XGBoost 3.4.0, and PyTorch 2.8.0 CUDA 12.9.
Other NVIDIA GPU and WSL combinations are not yet claimed as verified. Compute commands reject
native Linux, macOS, Windows Python, WSL1, and repositories stored under `/mnt/`.

Keep the repository in the WSL Linux filesystem, such as `~/projects`. Do not run it from
`/mnt/c`, where filesystem I/O can be substantially slower. Install the NVIDIA display driver on
Windows; do not install a second Linux display driver inside WSL.

## First-time setup in WSL2

Install the small system prerequisites inside WSL. A Git LFS installation in Git for Windows does
not replace the WSL installation.

```bash
sudo apt update
sudo apt install --yes git git-lfs curl ca-certificates

mkdir -p ~/projects
cd ~/projects
git clone <private-repository-url> imu-fall-benchmark
cd imu-fall-benchmark
./benchmark setup
```

`setup` installs/pulls/verifies Git LFS data, reuses `~/miniforge3` when available, or installs the
pinned Miniforge distribution without `sudo`. It creates an immutable Conda environment keyed by
the dependency manifests. Repeating `setup` with unchanged dependencies reuses that environment.
Do not replace CUDA, RAPIDS, XGBoost, or PyTorch versions without changing the dependency
manifests and recording a new experiment version.

## Reproduce the benchmark

Run the three main commands from the repository root:

```bash
./benchmark doctor
./benchmark test
./benchmark smoke
./benchmark reproduce --resume
```

- `doctor` verifies WSL2, paths, disk space, CUDA access, and all seven tabular/MLP GPU backends.
- `test` runs the tracked Ruff and pytest checks in the same WSL environment.
- `smoke` validates data, builds or reuses the window cache, and runs 22 jobs on fold 0 with
  controlled sampling and two training epochs.
- `reproduce` runs the fixed five-fold, 110-job conclusion recheck. `--resume` reuses compatible
  checkpoints after an interruption.

On the reference RTX 4070 SUPER, the previous data snapshot required about three minutes to build
the window cache, less than one minute for smoke, and about 70 minutes for the complete run.
For the current corrected data, allow approximately 75–100 minutes after installation. A fresh
Conda environment commonly adds another 20–60 minutes depending on network speed. At least
25 GiB free space is required before a fresh toolchain installation.

The recheck intentionally uses the current corrected HDF5 files. It evaluates whether the model
ranking and the universal-versus-dedicated conclusions remain supported; it does not promise
byte-for-byte reproduction of the earlier report's numbers.

## Provisional KFall external evaluation

KFall is never included in model fitting, normalisation, or early stopping. The workflow trains
Threshold Impact, 1D CNN, LSTM, and CNN-LSTM variants on the five recording-labelled datasets,
then performs frozen-weight inference on lower-back KFall recordings.

```bash
./benchmark kfall-plan --profile smoke
./benchmark kfall-prepare
./benchmark kfall-smoke
./benchmark kfall-evaluate --resume
./benchmark kfall-report --profile evaluate
```

- Universal and waist-only training are evaluated separately.
- Each model uses four old-data participant folds for training and one for validation.
- The full run contains 40 jobs: 2 suites × 4 models × 5 validation folds.
- A 2-second window is positive only when its final, causal decision sample is between fall onset
  and impact, inclusive. A window that reaches the fall but decides after impact is excluded.
- One above-threshold positive window counts as an event detection. The 33 short fall events with
  no valid decision window remain in the denominator and therefore count as misses.
- Zero-shot recording metrics transfer the threshold selected on the old validation fold. The
  temporal comparison separately calibrates only the threshold with five participant-grouped
  KFall folds; KFall never updates model weights.

The current KFall HDF5 is deliberately marked `provisional_kfall_adapter_v1`: onset/impact points
may be shifted by one 30 Hz sample, the derived post-impact activity tail is ignored, and ADL task
codes are placeholders. Results from this workflow are exploratory rather than formal project
validation evidence.

## Experiment contract

- Input: 30 Hz, six-axis, sensor-local IMU windows with gravity retained.
- Window: 60 frames (2 seconds), stride 15 frames (0.5 seconds), 75% overlap.
- Split: fixed five-fold participant split; a participant never crosses train, validation, and test.
- Tasks: paired chest/waist position, universal fall, chest-only fall, and waist-only fall.
- Position models: seven engineered-feature models and three raw-window temporal networks.
- Fall models: Threshold Impact, 1D CNN, LSTM, and CNN-LSTM using recording-level MIL.
- MIL pooling: mean of the highest-scoring 10% of windows in each location sequence.
- Primary metric: Balanced Accuracy, with confusion counts, sensitivity, specificity, precision,
  F1, MCC, ROC AUC, Average Precision, macro-dataset BAcc, and matched participant bootstrap
  comparisons retained.
- Seed: `3888`; PyTorch deterministic algorithms are enabled and the execution environment is
  recorded in each run manifest.
- Source provenance: each run records a Git commit or snapshot digest and dirty/unknown warnings.
  Warnings do not block exploratory or full runs, but they must be considered during review.
- Data schema: HDF5 v3 (`3.0.0`); derived window caches are fingerprints of source HDF5 content,
  split files, and the applicable temporal policy.

## Advanced commands

These commands are useful for inspection and targeted experiments but are not the main
reproduction path:

```bash
./benchmark validate-data
./benchmark prepare
./benchmark plan --profile reproduce
./benchmark run --profile smoke --models torch_cnn_lstm --suites fall_universal --folds 0
./benchmark report --profile reproduce
```

`run` accepts comma-separated `--models`, `--suites`, and `--folds`. Such a run is a custom
experiment and must not be described as the fixed 110-job reproduction.

## Persistent work directory

Derived windows are written to `~/imu-fall-work/cache`. Checkpoints, status files, CSV metrics,
comparisons, subgroup results, manifests, and Markdown summaries are written below
`~/imu-fall-work/runs`. Set `IMU_BENCH_WORK_ROOT` to an absolute path to override both locations.
Keeping these files outside the clone allows source snapshots to be replaced without losing cache
or resume state. The test suite is versioned with the source; generated test caches are not.

See [`data/README.md`](data/README.md) for dataset sources, counts, field definitions, and the HDF5
layout.
