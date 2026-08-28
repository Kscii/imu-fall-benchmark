# Distributed IMU Data

This repository distributes six pipeline-generated HDF5 v3 files. KFall under `data/external/`
is the primary temporally supervised dataset. Five recording-labelled datasets under
`data/imu_30hz/` are retained for research-only ADL supplementation and recording-level MIL.
Raw archives and ingestion adapters are intentionally outside this benchmark repository.

## Sources

| Dataset ID | Role | Source | Participants | Locations used | Source frequency |
|---|---|---|---:|---|---:|
| `cgu_bes` | research: recording/ADL | [CGU-BES on Figshare](https://figshare.com/articles/dataset/CGU-BES_Dataset_for_Fall_and_Activity_of_Daily_Life/7016306) | 15 | chest | 200 Hz |
| `uci_455` | research: recording/ADL | [UCI Simulated Falls and ADLs](https://archive.ics.uci.edu/dataset/455/simulated+falls+and+daily+living+activities+data+set) | 17 | chest, waist | 25 Hz |
| `umafall` | research: recording/ADL | [UMAFall on Figshare](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | 19 IDs | chest, waist | 20 Hz |
| `sisfall` | research: recording/ADL | [SisFall paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/) | 38 | waist | 200 Hz |
| `upfall` | research: recording/ADL | [UP-Fall official site](https://sites.google.com/up.edu.mx/har-up/) | 17 | belt mapped to waist | irregular, about 15–21 Hz |
| `kfall` | primary temporal | [KFall official site](https://sites.google.com/view/kfalldataset) | 32 used | lower back | 100 Hz |

Licensing, attribution, source hashes, exclusion counts, resampling methods, and processing notes
are stored as root HDF5 attributes. `data/checksums.sha256` protects the distributed bytes, while
`logical_content_sha256` identifies the logical content of each HDF5 file independently of its
physical encoding.

## Current snapshot

| Dataset ID | Sequences | 30 Hz frames | Annotations | Fall events | Approximate size |
|---|---:|---:|---:|---:|---:|
| `cgu_bes` | 195 | 96,488 | 0 | 0 | 2.0 MiB |
| `uci_455` | 5,705 | 3,270,374 | 0 | 0 | 65 MiB |
| `umafall` | 1,362 | 608,590 | 0 | 0 | 13 MiB |
| `sisfall` | 4,505 | 2,378,936 | 0 | 0 | 48 MiB |
| `upfall` | 551 | 485,830 | 1,165 | 0 | 9.2 MiB |
| **Training total** | **12,318** | **6,840,218** | **1,165** | **0** | **about 137 MiB** |
| `kfall` external | 5,075 | 1,200,811 | 14,459 | 2,346 | 27 MiB |

The training split contains 106 source-namespaced participants. The independent KFall split
contains 32 participants. A physical recording may contain more than one body-location sequence,
so the sequence count can exceed the recording count.

## HDF5 v3 contract

Each file has exactly three root datasets:

```text
/<dataset_id>.h5
├── samples       float32 [N, 6]
├── sequences     compound [S]
└── annotations   compound [A]
```

`samples` columns and units are:

1. `acceleration_x_mps2` (`m/s²`)
2. `acceleration_y_mps2` (`m/s²`)
3. `acceleration_z_mps2` (`m/s²`)
4. `angular_velocity_x_rad_s` (`rad/s`)
5. `angular_velocity_y_rad_s` (`rad/s`)
6. `angular_velocity_z_rad_s` (`rad/s`)

`sequences` fields are:

| Field | Meaning |
|---|---|
| `sample_start`, `sample_stop` | Half-open sample range `[start, stop)` in `samples` |
| `source_file` | Source archive member used to build the sequence |
| `participant_id` | Dataset-namespaced participant ID |
| `recording_id` | Dataset-namespaced physical recording ID |
| `body_location` | `chest`, `waist`, or external `lower_back` |
| `activity_code` | Source activity code or documented placeholder |
| `is_fall` | Recording-level fall label |
| `supervision_kind` | `recording` or `temporal` |
| `source_sampling_rate_hz` | Sampling rate before canonical resampling |

`annotations` fields are `sequence_index`, `kind`, `start_sample`, `stop_sample`, and `code`.
`activity` and `exclude` are half-open intervals. `onset` and `impact` are points represented by
`start_sample == stop_sample`. Annotation sample numbers are relative to their sequence, not the
global `samples` dataset.

Acceleration retains gravity, and axes remain in each device's sensor-local coordinate frame.
The benchmark does not remove gravity or rotate data into a common body coordinate frame.

## Supervision and KFall boundary

KFall has an independent participant split in `kfall_external_folds_v1.csv`. The current adapter output is
provisional because onset/impact may be shifted by one 30 Hz sample, ADL codes are placeholders,
and the activity end is derived as impact plus one second.

The benchmark derives 2-second windows at a 0.5-second stride. A fall window is positive only if
its final decision sample lies within the fall activity segment, from onset through the
impact-plus-one-second tail. A window that overlaps the segment but decides after its end is
excluded. The resulting cache contains 53,365 retained windows: 8,027 positive and 45,338
negative. It excludes 9,120 post-segment overlap windows; 4 of 2,346 events have no retained
positive decision window and count as misses in event-level sensitivity.

The five other datasets do not contain reliable fall intervals. Their fall recordings must not be
silently relabelled as positive temporal windows. The primary view excludes those recordings;
the mixed view may use only their recording-labelled ADL windows as training negatives, while the
separate MIL view keeps recording labels and top-10% pooling. Both alternatives are research-only.

The exact distributed snapshot is declared in `data/snapshot_v1.json`. The machine-readable
window, temporal-supervision, resampling, MIL, fold-evolution, and metric rules are declared in
`configs/contracts/imu_benchmark_contract_v1.json`. Changing source bytes, logical HDF5 content,
split files, or the contract intentionally creates a different derived-cache fingerprint.

## Validation and adding data

After `git lfs pull`, run:

```bash
./benchmark validate-data
git lfs fsck
```

The validator checks exact file membership, SHA-256 values, v3 schema/dtypes, contiguous sequence
ranges, annotation bounds and ordering, declared counts, and participant-fold coverage.

Do not copy an arbitrary HDF5 file into these directories. A new distributed dataset requires:

1. a v3-compatible, immutable HDF5 output and documented licence/attribution;
2. dataset-namespaced participant and recording IDs;
3. a participant fold CSV with a new explicit split version;
4. a snapshot entry and matching `data/checksums.sha256` row;
5. a data-view YAML that states whether it may train, supplement, or evaluate;
6. `./benchmark validate-data`, cache regression, and smoke acceptance before use.

The generic cache discovers datasets from the snapshot and split manifests. A new temporal HDF5
therefore does not require a dataset-specific training runner, but ingestion and HDF5 validation
remain separate responsibilities. Preserve all existing fold assignments when adding participants;
review and version the new split rather than overwriting the old one in place. Any source, split,
contract, or snapshot change intentionally creates a new cache and run fingerprint.
