from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .contract import DEFAULT_CONTRACT_PATH, default_active_snapshot_path, load_contract_snapshot

EXPECTED_SCHEMA_VERSION = "3.1.0"
FEATURE_COLUMNS = (
    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "acceleration_z_mps2",
    "angular_velocity_x_rad_s",
    "angular_velocity_y_rad_s",
    "angular_velocity_z_rad_s",
)
FEATURE_UNITS = ("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")
SEQUENCE_FIELDS = (
    "sample_start",
    "sample_stop",
    "source_file",
    "participant_id",
    "recording_id",
    "body_location",
    "activity_code",
    "is_fall",
    "supervision_kind",
    "source_sampling_rate_hz",
)
ANNOTATION_FIELDS = ("sequence_index", "kind", "start_sample", "stop_sample", "code")


@dataclass(frozen=True, slots=True)
class Annotation:
    kind: str
    start_sample: int
    stop_sample: int
    code: str


@dataclass(frozen=True, slots=True)
class FallEvent:
    onset_sample: int
    impact_sample: int

    @property
    def onset_time_s(self) -> float:
        return self.onset_sample / 25.0

    @property
    def impact_time_s(self) -> float:
        return self.impact_sample / 25.0


@dataclass(frozen=True, slots=True)
class IMURecording:
    dataset_id: str
    participant_id: str
    recording_id: str
    body_location: str
    activity: str
    is_fall: bool
    supervision_kind: str
    values: np.ndarray
    annotations: tuple[Annotation, ...]
    fall_event: FallEvent | None


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_utf8(dtype: np.dtype) -> bool:
    info = h5py.check_string_dtype(dtype)
    return info is not None and info.encoding == "utf-8"


def _check_compound_dtype(
    dataset: h5py.Dataset,
    expected: tuple[tuple[str, np.dtype | None], ...],
    path: Path,
) -> None:
    if dataset.dtype.names != tuple(name for name, _dtype in expected):
        raise ValueError(f"{path.name}: invalid {dataset.name} field order")
    for name, expected_dtype in expected:
        actual = dataset.dtype.fields[name][0]
        valid = _is_utf8(actual) if expected_dtype is None else actual == expected_dtype
        if not valid:
            raise ValueError(f"{path.name}: invalid {dataset.name}.{name} dtype")


def _data_files(data_root: Path, expected_ids: Collection[str]) -> tuple[Path, ...]:
    files = tuple(sorted(data_root.glob("*.h5")))
    expected = {f"{dataset_id}.h5" for dataset_id in expected_ids}
    actual = {path.name for path in files}
    if actual != expected:
        raise ValueError(
            f"Expected exactly {sorted(expected)} in {data_root}, found {sorted(actual)}"
        )
    for path in files:
        with path.open("rb") as source:
            prefix = source.read(40)
        if prefix.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"{path} is not an HDF5 file (a Git LFS pointer was found)")
    return files


def _sequence_annotations(rows: np.ndarray, sequence_index: int) -> tuple[Annotation, ...]:
    return tuple(
        Annotation(
            kind=_text(row["kind"]),
            start_sample=int(row["start_sample"]),
            stop_sample=int(row["stop_sample"]),
            code=_text(row["code"]),
        )
        for row in rows
        if int(row["sequence_index"]) == sequence_index
    )


def _check_annotation_rows(
    path: Path, sequence_rows: np.ndarray, annotation_rows: np.ndarray
) -> None:
    previous: tuple[int, int, int, int, str] | None = None
    kind_order = {"activity": 0, "onset": 1, "impact": 2, "exclude": 3}
    for row in annotation_rows:
        sequence_index = int(row["sequence_index"])
        kind = _text(row["kind"])
        start = int(row["start_sample"])
        stop = int(row["stop_sample"])
        code = _text(row["code"])
        if sequence_index < 0 or sequence_index >= len(sequence_rows):
            raise ValueError(f"{path.name}: annotation sequence index is out of bounds")
        if kind not in kind_order or not code:
            raise ValueError(f"{path.name}: invalid annotation kind or code")
        length = int(sequence_rows[sequence_index]["sample_stop"]) - int(
            sequence_rows[sequence_index]["sample_start"]
        )
        if start < 0 or start >= length:
            raise ValueError(f"{path.name}: annotation start is out of bounds")
        if kind in {"onset", "impact"}:
            if stop != start:
                raise ValueError(f"{path.name}: point annotation must have start == stop")
        elif not start < stop <= length:
            raise ValueError(f"{path.name}: annotation interval is invalid")
        key = (sequence_index, start, kind_order[kind], stop, code)
        if previous is not None and key < previous:
            raise ValueError(f"{path.name}: annotations are not deterministically sorted")
        previous = key

    for sequence_index, sequence in enumerate(sequence_rows):
        supervision = _text(sequence["supervision_kind"])
        if supervision != "temporal":
            continue
        annotations = _sequence_annotations(annotation_rows, sequence_index)
        length = int(sequence["sample_stop"]) - int(sequence["sample_start"])
        intervals = sorted(
            (item.start_sample, item.stop_sample)
            for item in annotations
            if item.kind in {"activity", "exclude"}
        )
        cursor = 0
        for start, stop in intervals:
            if start != cursor:
                raise ValueError(
                    f"{path.name}: temporal activity/exclude intervals must cover the sequence"
                )
            cursor = stop
        if cursor != length:
            raise ValueError(
                f"{path.name}: temporal activity/exclude intervals must cover the sequence"
            )
        onsets = [item for item in annotations if item.kind == "onset"]
        impacts = [item for item in annotations if item.kind == "impact"]
        if bool(sequence["is_fall"]) != bool(onsets) or len(onsets) != len(impacts):
            raise ValueError(f"{path.name}: temporal event labels are inconsistent")
        for onset, impact in zip(onsets, impacts, strict=True):
            if onset.code != impact.code or onset.start_sample >= impact.start_sample:
                raise ValueError(f"{path.name}: temporal onset/impact order is invalid")


def _check_file(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"samples", "sequences", "annotations"}:
            raise ValueError(f"{path.name}: v3 requires samples, sequences, and annotations")
        dataset_id = _text(handle.attrs.get("dataset_id", ""))
        if dataset_id != path.stem:
            raise ValueError(f"{path.name}: dataset_id does not match the filename")
        for name, expected in (
            ("imu_schema_version", EXPECTED_SCHEMA_VERSION),
            ("axis_frame", "sensor_local"),
            ("hdf5_compatibility", "1.14"),
        ):
            if _text(handle.attrs.get(name, "")) != expected:
                raise ValueError(f"{path.name}: invalid {name}")
        if float(handle.attrs.get("sampling_rate_hz", 0.0)) != 25.0:
            raise ValueError(f"{path.name}: expected 25 Hz data")
        evaluation_role = _text(handle.attrs.get("evaluation_role", ""))
        if evaluation_role not in {"cross_validation", "training_only"}:
            raise ValueError(f"{path.name}: invalid evaluation_role")
        if tuple(json.loads(_text(handle.attrs.get("feature_columns", "[]")))) != FEATURE_COLUMNS:
            raise ValueError(f"{path.name}: unexpected feature columns")

        samples = handle["samples"]
        sequences = handle["sequences"]
        annotations = handle["annotations"]
        if samples.dtype != np.dtype("float32") or samples.ndim != 2 or samples.shape[1] != 6:
            raise ValueError(f"{path.name}: samples must be float32 [N, 6]")
        if tuple(json.loads(_text(samples.attrs.get("columns", "[]")))) != FEATURE_COLUMNS:
            raise ValueError(f"{path.name}: sample columns do not match")
        if tuple(json.loads(_text(samples.attrs.get("units", "[]")))) != FEATURE_UNITS:
            raise ValueError(f"{path.name}: sample units do not match")
        _check_compound_dtype(
            sequences,
            (
                ("sample_start", np.dtype("int64")),
                ("sample_stop", np.dtype("int64")),
                ("source_file", None),
                ("participant_id", None),
                ("recording_id", None),
                ("body_location", None),
                ("activity_code", None),
                ("is_fall", np.dtype("bool")),
                ("supervision_kind", None),
                ("source_sampling_rate_hz", np.dtype("float64")),
            ),
            path,
        )
        _check_compound_dtype(
            annotations,
            (
                ("sequence_index", np.dtype("int32")),
                ("kind", None),
                ("start_sample", np.dtype("int64")),
                ("stop_sample", np.dtype("int64")),
                ("code", None),
            ),
            path,
        )
        sequence_rows = np.asarray(sequences)
        annotation_rows = np.asarray(annotations)
        if not len(sequence_rows):
            raise ValueError(f"{path.name}: at least one sequence is required")
        starts = np.asarray(sequence_rows["sample_start"], dtype=np.int64)
        stops = np.asarray(sequence_rows["sample_stop"], dtype=np.int64)
        if (
            starts[0] != 0
            or not np.array_equal(starts[1:], stops[:-1])
            or stops[-1] != len(samples)
            or np.any(stops - starts < 2)
        ):
            raise ValueError(f"{path.name}: invalid contiguous sequence ranges")
        if not np.isfinite(sequence_rows["source_sampling_rate_hz"]).all() or np.any(
            sequence_rows["source_sampling_rate_hz"] <= 0
        ):
            raise ValueError(f"{path.name}: invalid source sampling rates")
        for row in sequence_rows:
            if _text(row["supervision_kind"]) not in {"recording", "temporal"}:
                raise ValueError(f"{path.name}: invalid supervision_kind")
            for field in SEQUENCE_FIELDS[2:7]:
                if not _text(row[field]):
                    raise ValueError(f"{path.name}: empty sequences.{field}")
        _check_annotation_rows(path, sequence_rows, annotation_rows)
        declared = {
            "sequence_count": len(sequence_rows),
            "sample_count": len(samples),
            "annotation_count": len(annotation_rows),
        }
        for name, value in declared.items():
            if int(handle.attrs.get(name, -1)) != value:
                raise ValueError(f"{path.name}: {name} attribute mismatch")
        logical_hash = _text(handle.attrs.get("logical_content_sha256", ""))
        if len(logical_hash) != 64 or any(c not in "0123456789abcdef" for c in logical_hash):
            raise ValueError(f"{path.name}: invalid logical content fingerprint")
        kinds = Counter(_text(row["kind"]) for row in annotation_rows)
        participants = {_text(value) for value in sequence_rows["participant_id"]}
        return {
            "dataset_id": dataset_id,
            "sequences": len(sequence_rows),
            "rows": len(samples),
            "annotations": len(annotation_rows),
            "events": kinds["onset"],
            "segments": kinds["activity"] + kinds["exclude"],
            "participants": participants,
            "supervision": dict(Counter(_text(x) for x in sequence_rows["supervision_kind"])),
            "body_locations": dict(Counter(_text(x) for x in sequence_rows["body_location"])),
            "fall_sequences": int(np.count_nonzero(sequence_rows["is_fall"])),
            "logical_content_sha256": logical_hash,
            "evaluation_role": evaluation_role,
        }


def validate_hdf5_file(path: Path) -> dict[str, object]:
    result = _check_file(path)
    participants = result.pop("participants")
    if not isinstance(participants, set):
        raise ValueError(f"Invalid participant summary: {path}")
    return {**result, "participants": len(participants)}


def _check_splits(
    project_root: Path,
    manifests: object,
    participants: set[tuple[str, str]],
) -> dict[str, object]:
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("Base data requires participant split manifests")
    assignments: dict[tuple[str, str], int] = {}
    versions: list[str] = []
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise ValueError("Invalid participant split manifest")
        split_path = project_root / str(manifest["path"])
        if not split_path.is_file() or _sha256(split_path) != manifest["sha256"]:
            raise ValueError(f"Participant split checksum mismatch: {split_path}")
        version = str(manifest["version"])
        versions.append(version)
        with split_path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["split_version"] != version:
                    raise ValueError(f"Unexpected split version: {row['split_version']}")
                key = (row["dataset_id"], row["participant_id"])
                if key in assignments:
                    raise ValueError(f"Duplicate split assignment: {key}")
                assignments[key] = int(row["fold_id"])
    if set(assignments) != participants:
        missing = sorted(participants - set(assignments))
        extra = sorted(set(assignments) - participants)
        raise ValueError(f"Split participant mismatch; missing={missing}, extra={extra}")
    if set(assignments.values()) != set(range(5)):
        raise ValueError("Participant folds must cover 0..4")
    return {
        "versions": versions,
        "participants": len(assignments),
        "fold_counts": {
            str(fold): sum(value == fold for value in assignments.values()) for fold in range(5)
        },
    }


def _validate_collection(
    active_root: Path,
    collection: dict[str, object],
    expected_role: str,
) -> tuple[dict[str, int], set[tuple[str, str]], list[dict[str, object]]]:
    data_root = active_root / str(collection["data_path"])
    entries = collection["datasets"]
    if not isinstance(entries, list):
        raise ValueError("Invalid snapshot dataset entries")
    entry_by_id = {
        str(item["dataset_id"]): item for item in entries if isinstance(item, dict)
    }
    expected_ids = tuple(entry_by_id)
    totals = {"sequences": 0, "rows": 0, "annotations": 0, "events": 0, "segments": 0}
    participants: set[tuple[str, str]] = set()
    datasets: list[dict[str, object]] = []
    for path in _data_files(data_root, expected_ids):
        result = _check_file(path)
        dataset_id = str(result["dataset_id"])
        entry = entry_by_id[dataset_id]
        expected_path = (data_root / str(entry["path"])).resolve()
        if path.resolve() != expected_path:
            raise ValueError(f"Snapshot path mismatch for {dataset_id}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Snapshot size mismatch for {dataset_id}")
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"Snapshot SHA-256 mismatch for {dataset_id}")
        dataset_participants = result.pop("participants")
        if not isinstance(dataset_participants, set):
            raise ValueError(f"Invalid participant summary for {dataset_id}")
        if "participants" in entry and len(dataset_participants) != int(entry["participants"]):
            raise ValueError(f"Participant count mismatch for {dataset_id}")
        for field in (
            "sequences",
            "rows",
            "annotations",
            "logical_content_sha256",
            "evaluation_role",
        ):
            if result[field] != entry[field]:
                raise ValueError(f"Snapshot {field} mismatch for {dataset_id}")
        for field in ("events", "segments", "fall_sequences", "supervision", "body_locations"):
            if field in entry and result[field] != entry[field]:
                raise ValueError(f"Snapshot {field} mismatch for {dataset_id}")
        if result["evaluation_role"] != expected_role:
            raise ValueError(f"Unexpected evaluation role for {dataset_id}")
        participants.update((dataset_id, value) for value in dataset_participants)
        for name in totals:
            totals[name] += int(result[name])
        datasets.append(result)
    return totals, participants, datasets


def validate_data(
    project_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    snapshot_path: Path | None = None,
) -> dict[str, object]:
    active_path = default_active_snapshot_path() if snapshot_path is None else snapshot_path
    contract, snapshot, contract_sha256, snapshot_sha256 = load_contract_snapshot(
        project_root, contract_path=contract_path, snapshot_path=active_path
    )
    collections = snapshot["collections"]
    if not isinstance(collections, dict):
        raise ValueError("Invalid snapshot collections")
    base_collection = collections["base"]
    if not isinstance(base_collection, dict):
        raise ValueError("Invalid base collection")
    base_totals, base_participants, base_datasets = _validate_collection(
        active_path.parent,
        base_collection,
        "cross_validation",
    )
    base_split = _check_splits(
        project_root,
        base_collection["splits"],
        base_participants,
    )
    team_summary = None
    if "team" in collections:
        team_collection = collections["team"]
        if not isinstance(team_collection, dict):
            raise ValueError("Invalid team collection")
        team_totals, team_participants, team_datasets = _validate_collection(
            active_path.parent,
            team_collection,
            "training_only",
        )
        team_summary = {
            "files": len(team_datasets),
            **team_totals,
            "participants": len(team_participants),
            "fold_id": -1,
            "datasets": team_datasets,
        }
    return {
        "status": "PASS",
        "schema_version": contract["data_schema_version"],
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256,
        "snapshot_version": snapshot["snapshot_version"],
        "snapshot_sha256": snapshot_sha256,
        "base_snapshot_id": snapshot["base_snapshot_id"],
        "team_snapshot_id": snapshot.get("team_snapshot_id"),
        "base": {
            "files": len(base_datasets),
            **base_totals,
            "split": base_split,
            "datasets": base_datasets,
        },
        "team": team_summary,
    }


def iter_recordings(
    data_root: Path,
    *,
    expected_dataset_ids: Collection[str],
) -> Iterator[IMURecording]:
    seen: set[tuple[str, str]] = set()
    for path in _data_files(data_root, expected_dataset_ids):
        with h5py.File(path, "r") as handle:
            dataset_id = _text(handle.attrs["dataset_id"])
            sequences = np.asarray(handle["sequences"])
            annotations = np.asarray(handle["annotations"])
            for index, row in enumerate(sequences):
                recording_id = _text(row["recording_id"])
                body_location = _text(row["body_location"])
                key = (recording_id, body_location)
                if key in seen:
                    raise ValueError(f"Duplicate location sequence: {key}")
                seen.add(key)
                rows = _sequence_annotations(annotations, index)
                onsets = [item for item in rows if item.kind == "onset"]
                impacts = [item for item in rows if item.kind == "impact"]
                event = None
                if len(onsets) == len(impacts) == 1:
                    event = FallEvent(onsets[0].start_sample, impacts[0].start_sample)
                start = int(row["sample_start"])
                stop = int(row["sample_stop"])
                yield IMURecording(
                    dataset_id=dataset_id,
                    participant_id=_text(row["participant_id"]),
                    recording_id=recording_id,
                    body_location=body_location,
                    activity=_text(row["activity_code"]),
                    is_fall=bool(row["is_fall"]),
                    supervision_kind=_text(row["supervision_kind"]),
                    values=np.asarray(handle["samples"][start:stop], dtype=np.float32),
                    annotations=rows,
                    fall_event=event,
                )
