# IMU HDF5 3.2.0 Contract

Status: frozen with the first public 3.2 dataset release

Canonical repository: `Kscii/imu-fall-benchmark`

Profiles: `training_dataset`, `client_delivery`

This document defines the physical HDF5 contract shared by the benchmark,
collector/annotation service, client viewer, and SOFT3888 data pipeline. Published
artifacts are immutable. A breaking change requires a new schema major/minor and new
object identity; readers must not silently upgrade an older file in place.

## 1. Common root attributes

Both profiles require:

- `imu_schema_version = "3.2.0"`
- `artifact_profile = "training_dataset" | "client_delivery"`
- `dataset_id`, `sampling_rate_hz = 25.0`, `axis_frame = "sensor_local"`
- `hdf5_compatibility = "1.14"`, `evaluation_role`
- `feature_columns`, `sequence_count`, `sample_count`, `annotation_count`
- `logical_content_sha256`, which fingerprints the logical core content only

`logical_content_sha256` is the SHA-256 of the ordered sequence-local stream used
by the benchmark writer. For each sequence, hash the length-prefixed canonical JSON
metadata (including annotations without global `sequence_index`), then the little-
endian `float32` sequence shape and sample bytes. Global `sample_start`,
`sample_stop`, and `sequence_index` values are container offsets and are not part of
the logical identity. Producers and consumers must use the shared conformance
fixture; an aggregate hash of the three flat tables is not compatible.

The whole-file SHA-256 is external: it is stored in the immutable manifest, GCS
object metadata, and download response headers. A file never contains its own
whole-file digest.

The exact root layouts are:

```text
training_dataset: /samples /sequences /annotations
client_delivery:  /samples /sequences /annotations /media /labels
```

Empty media or label placeholders are forbidden in `training_dataset`.

## 2. Shared core

`/samples` is `float32 [N, 6]` at 25 Hz. Columns are, in order:

```text
acceleration_x_mps2
acceleration_y_mps2
acceleration_z_mps2
angular_velocity_x_rad_s
angular_velocity_y_rad_s
angular_velocity_z_rad_s
```

Units are `m/s^2` for acceleration and `rad/s` for angular velocity. Gravity is
retained and axes remain in the verified sensor-local frame.

`/sequences` is a one-dimensional compound dataset with fields:

```text
sample_start int64
sample_stop int64
source_file UTF-8
participant_id UTF-8
recording_id UTF-8
body_location UTF-8
activity_code UTF-8
is_fall bool
supervision_kind UTF-8
source_sampling_rate_hz float64
```

Sequence sample ranges are non-empty, contiguous half-open ranges that cover all
rows in `/samples` exactly once.

`/annotations` is a one-dimensional compound dataset with fields:

```text
sequence_index int32
kind UTF-8                 # activity | onset | impact | exclude
start_sample int64         # sequence-relative
stop_sample int64          # sequence-relative
code UTF-8
```

Activity and exclude rows use `[start_sample, stop_sample)`. Onset and impact are
points with equal start and stop. Temporal activity/exclude intervals cover their
sequence without gaps or overlap. Each fall activity has one onset at its start and
one impact strictly inside it. Reserved exclude codes are `sync_tap` and `other`;
they are contract codes, not taxonomy activities.

## 3. `training_dataset`

This is the only profile eligible for benchmark catalogs and training snapshots.
Its root contains exactly the shared core. Its filename is `<dataset_id>.h5` and its
manifest descriptor includes:

```json
{
  "hdf5_schema_version": "3.2.0",
  "artifact_profile": "training_dataset"
}
```

The benchmark may expose a low-level opt-in reader for the common core of a client
delivery, but official data discovery and validation must reject any profile other
than `training_dataset`.

## 4. `client_delivery`

Each core sequence maps one-to-one to an original MP4, a timing table, and one
frozen taxonomy version.

### 4.1 Media index and video bytes

`/media/index` is a one-dimensional compound dataset ordered by `sequence_index`:

```text
sequence_index int32
byte_length int64
file_offset int64
sha256 UTF-8
content_type UTF-8
container UTF-8
media_duration_ns int64
sample_zero_recording_time_ns int64
sample_zero_media_time_ns int64
```

There is exactly one row per sequence and `sequence_index` covers `0..S-1` once.
Recording identity is derived from `/sequences`; the index does not duplicate
`recording_id` or a dataset path.

`/media/videos/<sequence_index>` is a contiguous, uncompressed, one-dimensional
`uint8` dataset containing the original MP4 bytes. Chunking, filters, external
storage, virtual datasets, and variable-length frame arrays are forbidden. Dataset
physical byte ranges must not overlap and must remain inside the HDF5 file. The
index `file_offset`, `byte_length`, and `sha256` must match the reopened file and
the MP4 payload.

### 4.2 Timing

`/media/timing/<sequence_index>` is `int64 [F, 2]`:

```text
recording_time_ns, media_time_ns
```

Both columns are non-empty, equal length by construction, strictly increasing, and
inside the sequence recording and media ranges. Viewer synchronization uses
piecewise-linear interpolation of this mapping, not a constant-offset assumption.

### 4.3 Frozen labels

`/labels/catalog` is a one-dimensional compound dataset with fields:

```text
taxonomy_id UTF-8
taxonomy_version UTF-8
code UTF-8
name UTF-8
is_fall bool
active bool
```

Rows are unique by `(taxonomy_id, taxonomy_version, code)`. Names and codes are
non-empty. The catalog includes the immutable code/name/status rows required to
interpret every activity code used by the delivery.

`/labels/sequence_versions` is a one-dimensional compound dataset ordered by
`sequence_index`:

```text
sequence_index int32
taxonomy_id UTF-8
taxonomy_version UTF-8
```

It contains exactly one taxonomy version assignment for each sequence and covers
`0..S-1` once. Every non-exclude annotation code must resolve in the assigned frozen
version. Reserved exclude codes are resolved by this contract and are not required
in the activity catalog.

## 5. Construction and validation

- Generate into a unique temporary file on the same filesystem, close it, reopen it,
  and run the complete logical and physical validator before publication.
- Do not use `h5repack`, in-place mutation, or any operation that changes physical
  offsets after validation.
- For client delivery, re-read every MP4 by physical range after closing HDF5 and
  verify its SHA-256.
- Upload the final HDF5 first; verify remote size and SHA-256; write the immutable
  delivery manifest last.
- HTTP downloads use `application/x-hdf5`, support byte ranges, and expose the same
  whole-file SHA-256 as the manifest and GCS metadata.
