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

from .contract import DEFAULT_CONTRACT_PATH, DEFAULT_SNAPSHOT_PATH, load_contract_snapshot

TRAINING_DATASET_IDS = (
    "cgu_bes",
    "ipqm_fall",
    "sfu_ipml",
    "sisfall",
    "uci_455",
    "umafall",
    "univrfall",
    "upfall",
)
EXTERNAL_DATASET_IDS = ("kfall",)
EXPECTED_SCHEMA_VERSION = "3.0.0"
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
        return self.onset_sample / 30.0

    @property
    def impact_time_s(self) -> float:
        return self.impact_sample / 30.0


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
            raise ValueError(f"{path} is a Git LFS pointer; run git lfs pull")
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
        if float(handle.attrs.get("sampling_rate_hz", 0.0)) != 30.0:
            raise ValueError(f"{path.name}: expected 30 Hz data")
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
        }


def validate_hdf5_file(path: Path) -> dict[str, object]:
    result = _check_file(path)
    participants = result.pop("participants")
    if not isinstance(participants, set):
        raise ValueError(f"Invalid participant summary: {path}")
    return {**result, "participants": len(participants)}


def _snapshot_files(snapshot: dict[str, object]) -> dict[str, tuple[str, int | None]]:
    expected: dict[str, tuple[str, int | None]] = {}
    collections = snapshot["collections"]
    if not isinstance(collections, dict):
        raise ValueError("Invalid snapshot collections")
    for collection in collections.values():
        if not isinstance(collection, dict):
            raise ValueError("Invalid snapshot collection")
        split = collection["split"]
        if not isinstance(split, dict):
            raise ValueError("Invalid snapshot split")
        expected[str(split["path"])] = (str(split["sha256"]), None)
        datasets = collection["datasets"]
        if not isinstance(datasets, list):
            raise ValueError("Invalid snapshot datasets")
        for item in datasets:
            if not isinstance(item, dict):
                raise ValueError("Invalid snapshot dataset")
            expected[str(item["path"])] = (str(item["sha256"]), int(item["size_bytes"]))
    return expected


def _check_checksums(project_root: Path, snapshot: dict[str, object]) -> None:
    manifest = project_root / "data/checksums.sha256"
    if not manifest.is_file():
        raise ValueError(f"Missing checksum manifest: {manifest}")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest
    snapshot_files = _snapshot_files(snapshot)
    if set(expected) != set(snapshot_files):
        raise ValueError("Checksum manifest does not exactly cover the distributed data")
    for relative, (snapshot_digest, expected_size) in sorted(snapshot_files.items()):
        digest = expected[relative]
        if digest != snapshot_digest:
            raise ValueError(f"Checksum manifest disagrees with snapshot: {relative}")
        path = project_root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"File size mismatch: {relative}")


def _check_split(
    split_path: Path,
    participants: set[tuple[str, str]],
    *,
    split_version: str,
) -> dict[str, object]:
    assignments: dict[tuple[str, str], int] = {}
    with split_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["split_version"] != split_version:
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
        "version": split_version,
        "participants": len(assignments),
        "fold_counts": {
            str(fold): sum(value == fold for value in assignments.values()) for fold in range(5)
        },
    }


def _validate_collection(
    project_root: Path, collection: dict[str, object], expected_ids: Collection[str]
) -> tuple[dict[str, int], set[tuple[str, str]], list[dict[str, object]]]:
    data_root = project_root / str(collection["data_path"])
    entries = collection["datasets"]
    if not isinstance(entries, list):
        raise ValueError("Invalid snapshot dataset entries")
    entry_by_id = {
        str(item["dataset_id"]): item for item in entries if isinstance(item, dict)
    }
    if set(entry_by_id) != set(expected_ids):
        raise ValueError("Snapshot dataset membership does not match the collection")
    totals = {"sequences": 0, "rows": 0, "annotations": 0, "events": 0, "segments": 0}
    participants: set[tuple[str, str]] = set()
    datasets: list[dict[str, object]] = []
    for path in _data_files(data_root, expected_ids):
        result = _check_file(path)
        dataset_id = str(result["dataset_id"])
        entry = entry_by_id[dataset_id]
        expected_path = (project_root / str(entry["path"])).resolve()
        if path.resolve() != expected_path:
            raise ValueError(f"Snapshot path mismatch for {dataset_id}")
        dataset_participants = result.pop("participants")
        if not isinstance(dataset_participants, set):
            raise ValueError(f"Invalid participant summary for {dataset_id}")
        if len(dataset_participants) != int(entry["participants"]):
            raise ValueError(f"Participant count mismatch for {dataset_id}")
        for field in (
            "sequences",
            "rows",
            "annotations",
            "events",
            "segments",
            "fall_sequences",
            "logical_content_sha256",
            "supervision",
            "body_locations",
        ):
            if result[field] != entry[field]:
                raise ValueError(f"Snapshot {field} mismatch for {dataset_id}")
        participants.update((dataset_id, value) for value in dataset_participants)
        for name in totals:
            totals[name] += int(result[name])
        datasets.append(result)
    return totals, participants, datasets


def validate_data(
    project_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
) -> dict[str, object]:
    contract, snapshot, contract_sha256, snapshot_sha256 = load_contract_snapshot(
        project_root, contract_path=contract_path, snapshot_path=snapshot_path
    )
    _check_checksums(project_root, snapshot)
    collections = snapshot["collections"]
    if not isinstance(collections, dict):
        raise ValueError("Invalid snapshot collections")
    training_collection = collections["training"]
    external_collection = collections["external"]
    if not isinstance(training_collection, dict) or not isinstance(external_collection, dict):
        raise ValueError("Invalid snapshot collection shape")
    training_ids = tuple(str(item["dataset_id"]) for item in training_collection["datasets"])
    training_totals, training_participants, training_datasets = _validate_collection(
        project_root, training_collection, training_ids
    )
    training_split = training_collection["split"]
    if not isinstance(training_split, dict):
        raise ValueError("Invalid training split manifest")
    training_split = _check_split(
        project_root / str(training_split["path"]),
        training_participants,
        split_version=str(training_split["version"]),
    )
    external_ids = tuple(str(item["dataset_id"]) for item in external_collection["datasets"])
    external_totals, external_participants, external_datasets = _validate_collection(
        project_root, external_collection, external_ids
    )
    kfall = next((item for item in external_datasets if item["dataset_id"] == "kfall"), None)
    if kfall is not None:
        if kfall["supervision"] != {"temporal": 5075}:
            raise ValueError("KFall must contain 5,075 temporal sequences")
        if kfall["body_locations"] != {"lower_back": 5075}:
            raise ValueError("KFall must contain lower_back sequences only")
    external_split_manifest = external_collection["split"]
    if not isinstance(external_split_manifest, dict):
        raise ValueError("Invalid external split manifest")
    external_split = _check_split(
        project_root / str(external_split_manifest["path"]),
        external_participants,
        split_version=str(external_split_manifest["version"]),
    )
    return {
        "status": "PASS",
        "schema_version": contract["data_schema_version"],
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256,
        "snapshot_version": snapshot["snapshot_version"],
        "snapshot_sha256": snapshot_sha256,
        "training": {
            "files": len(training_datasets),
            **training_totals,
            "split": training_split,
            "datasets": training_datasets,
        },
        "external": {
            "files": len(external_datasets),
            **external_totals,
            "split": external_split,
            "datasets": external_datasets,
        },
    }


def iter_recordings(
    data_root: Path,
    *,
    expected_dataset_ids: Collection[str] = TRAINING_DATASET_IDS,
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
