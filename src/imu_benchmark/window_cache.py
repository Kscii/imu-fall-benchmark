from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .configuration import canonical_sha256
from .data import FEATURE_NAMES, extract_window_features
from .dataset import IMURecording, iter_recordings
from .performance import PhaseTimer
from .progress import NullProgressReporter, ProgressReporter
from .protocol import segment_decision_time_labels

WINDOW_CACHE_SCHEMA = "unified_fall_windows_v5_25hz"


def _text_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in dataset]
    )


def _split_assignments(
    project_root: Path,
    active_root: Path,
    snapshot: dict[str, Any],
) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    split_root = (
        project_root if snapshot["schema_version"] == "imu_benchmark_active_v1" else active_root
    )
    for split in snapshot["collections"]["base"]["splits"]:
        path = split_root / split["path"]
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["split_version"] != split["version"]:
                    raise ValueError(f"Unexpected split version in {path}")
                key = (row["dataset_id"], row["participant_id"])
                if key in result:
                    raise ValueError(f"Duplicate participant split: {key}")
                result[key] = int(row["fold_id"])
    return result


def _recording_collections(
    project_root: Path,
    active_root: Path,
    snapshot: dict[str, Any],
) -> Iterator[tuple[IMURecording, int]]:
    assignments = _split_assignments(project_root, active_root, snapshot)
    for collection_name, collection in snapshot["collections"].items():
        dataset_ids = tuple(item["dataset_id"] for item in collection["datasets"])
        for recording in iter_recordings(
            active_root / collection["data_path"], expected_dataset_ids=dataset_ids
        ):
            if collection_name == "team":
                yield recording, -1
            else:
                key = (recording.dataset_id, recording.participant_id)
                if key not in assignments:
                    raise ValueError(f"Missing participant fold for {key}")
                yield recording, assignments[key]


def _recordings_with_progress(
    recordings: Iterator[tuple[IMURecording, int]],
    progress: ProgressReporter,
    total: int,
) -> Iterator[tuple[IMURecording, int]]:
    with progress.task(
        "Building the unified sliding-window cache",
        total=total,
        unit="sequences",
    ) as task:
        for recording, fold in recordings:
            task.update(detail=f"{recording.dataset_id}/{recording.recording_id}")
            yield recording, fold
            task.update(advance=1)


def _snapshot_sequence_count(snapshot: dict[str, Any]) -> int:
    return sum(
        int(dataset["sequences"])
        for collection in snapshot["collections"].values()
        for dataset in collection["datasets"]
    )


def _window_starts(
    sample_count: int,
    length: int,
    sampling_rate_hz: float,
    stride_seconds: float,
) -> np.ndarray:
    last_start = sample_count - length
    if last_start < 0:
        return np.empty(0, dtype=np.int32)
    count = int(np.floor((last_start / sampling_rate_hz) / stride_seconds + 1e-12)) + 1
    starts = np.floor(
        np.arange(count, dtype=np.float64) * stride_seconds * sampling_rate_hz + 0.5
    ).astype(np.int32)
    starts = starts[starts <= last_start]
    if len(starts) and (starts[0] != 0 or np.any(np.diff(starts) <= 0)):
        raise ValueError("Window start grid is not strictly increasing from zero")
    return starts


def _windows(
    values: np.ndarray,
    length: int,
    sampling_rate_hz: float,
    stride_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    starts = _window_starts(len(values), length, sampling_rate_hz, stride_seconds)
    if not len(starts):
        return np.empty((0, length, 6), dtype=np.float32), starts
    windows = np.stack([values[start : start + length] for start in starts])
    return windows.astype(np.float32, copy=False), starts


def _resize_write(dataset: h5py.Dataset, values: np.ndarray) -> None:
    start = len(dataset)
    stop = start + len(values)
    dataset.resize((stop, *dataset.shape[1:]))
    dataset[start:stop] = values


class _WindowBuffer:
    def __init__(self, datasets: dict[str, h5py.Dataset], flush_windows: int) -> None:
        self.datasets = datasets
        self.flush_windows = flush_windows
        self.parts: dict[str, list[np.ndarray]] = {name: [] for name in datasets}
        self.size = 0

    def append(self, values: dict[str, np.ndarray]) -> None:
        if set(values) != set(self.datasets):
            raise ValueError("Window-cache buffer fields do not match the schema")
        count = len(next(iter(values.values())))
        if any(len(value) != count for value in values.values()):
            raise ValueError("Window-cache buffer columns have different lengths")
        for name, value in values.items():
            self.parts[name].append(value)
        self.size += count
        if self.size >= self.flush_windows:
            self.flush()

    def flush(self) -> None:
        if not self.size:
            return
        for name, dataset in self.datasets.items():
            _resize_write(dataset, np.concatenate(self.parts[name], axis=0))
            self.parts[name].clear()
        self.size = 0


def _cache_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "schema": WINDOW_CACHE_SCHEMA,
        "contract_sha256": config["contract_sha256"],
        "snapshot_sha256": config["snapshot_sha256"],
        "window_samples": config["contract"]["window"]["samples"],
        "stride_seconds": config["contract"]["window"]["stride_seconds"],
        "feature_names": FEATURE_NAMES,
        "flush_windows": config["cache_flush_windows"],
    }
    return canonical_sha256(payload)


def prepare_unified_window_store(
    *,
    project_root: Path,
    cache_root: Path,
    config: dict[str, Any],
    progress: ProgressReporter | None = None,
) -> tuple[Path, dict[str, Any]]:
    reporter = progress or NullProgressReporter()
    fingerprint = _cache_fingerprint(config)
    destination = cache_root / "windows" / WINDOW_CACHE_SCHEMA / f"{fingerprint[:16]}.h5"
    if destination.exists():
        with reporter.task("Reusing the unified sliding-window cache", total=1) as task:
            with h5py.File(destination, "r") as handle:
                manifest = json.loads(str(handle.attrs["manifest_json"]))
            task.update(advance=1)
        if manifest.get("data_split_fingerprint") != fingerprint:
            raise ValueError(f"Conflicting unified cache: {destination}")
        return destination, {**manifest, "cache_reused_this_invocation": True}

    window_samples = int(config["contract"]["window"]["samples"])
    stride_seconds = float(config["contract"]["window"]["stride_seconds"])
    sampling_rate_hz = float(config["contract"]["canonical_signal"]["sampling_rate_hz"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".h5.tmp-{os.getpid()}")
    started = time.perf_counter()
    phases = PhaseTimer()
    skipped_short = 0
    skipped_post_segment_overlap = 0
    skipped_exclusion = 0
    event_count = 0
    events_without_positive_window = 0
    sequence_count = 0
    window_count = 0
    positive_temporal = 0
    negative_temporal = 0
    dataset_windows: dict[str, int] = {}
    dataset_sequences: dict[str, int] = {}
    dataset_positive_temporal: dict[str, int] = {}
    dataset_negative_temporal: dict[str, int] = {}
    dataset_events: dict[str, int] = {}
    dataset_events_without_positive: dict[str, int] = {}
    dataset_skipped_post_segment: dict[str, int] = {}
    text_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w", libver=("earliest", "v114")) as handle:
            handle.attrs.update(
                {
                    "window_schema_version": WINDOW_CACHE_SCHEMA,
                    "feature_schema_version": config["contract"]["window"][
                        "feature_schema_version"
                    ],
                    "data_split_fingerprint": fingerprint,
                    "contract_sha256": config["contract_sha256"],
                    "snapshot_sha256": config["snapshot_sha256"],
                    "sampling_rate_hz": sampling_rate_hz,
                    "window_samples": window_samples,
                    "stride_seconds": stride_seconds,
                    "feature_names": json.dumps(FEATURE_NAMES),
                    "hdf5_compatibility": "1.14",
                }
            )
            sequences = handle.create_group("sequences")
            text_fields = (
                "dataset_id",
                "participant_id",
                "recording_id",
                "body_location",
                "activity",
                "supervision_kind",
            )
            for name in text_fields:
                sequences.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
                )
            for name, dtype in (("is_fall", "?"), ("fold_id", "i1")):
                sequences.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
                )

            events_group = handle.create_group("events")
            for name, dtype in (
                ("sequence_index", "i4"),
                ("onset_sample", "i8"),
                ("impact_sample", "i8"),
                ("stop_sample", "i8"),
            ):
                events_group.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
                )
            events_group.create_dataset(
                "code", shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
            )

            windows_group = handle.create_group("windows")
            window_datasets = {
                "raw": windows_group.create_dataset(
                    "raw",
                    shape=(0, window_samples, 6),
                    maxshape=(None, window_samples, 6),
                    dtype="f4",
                    chunks=(1024, window_samples, 6),
                    compression="lzf",
                    shuffle=True,
                ),
                "features": windows_group.create_dataset(
                    "features",
                    shape=(0, len(FEATURE_NAMES)),
                    maxshape=(None, len(FEATURE_NAMES)),
                    dtype="f4",
                    chunks=(2048, len(FEATURE_NAMES)),
                    compression="lzf",
                    shuffle=True,
                ),
            }
            for name, dtype in (
                ("sequence_index", "i4"),
                ("start_sample", "i4"),
                ("end_sample", "i4"),
                ("fold_id", "i1"),
                ("bag_label", "i1"),
                ("temporal_label", "i1"),
            ):
                window_datasets[name] = windows_group.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
                )
            buffer = _WindowBuffer(window_datasets, int(config["cache_flush_windows"]))

            recordings = phases.iterate(
                "source_hdf5_read_seconds",
                _recording_collections(
                    project_root,
                    Path(config["active_data_root"]),
                    config["snapshot"],
                ),
            )
            recordings = _recordings_with_progress(
                recordings,
                reporter,
                _snapshot_sequence_count(config["snapshot"]),
            )
            for recording, fold in recordings:
                with phases.track("window_generation_seconds"):
                    raw, starts = _windows(
                        recording.values,
                        window_samples,
                        sampling_rate_hz,
                        stride_seconds,
                    )
                if not len(raw):
                    skipped_short += 1
                    continue
                ends = starts + window_samples
                temporal = np.full(len(starts), -1, dtype=np.int8)
                keep = np.ones(len(starts), dtype=np.bool_)
                with phases.track("window_label_seconds"):
                    if recording.supervision_kind == "temporal":
                        temporal, keep, intervals = segment_decision_time_labels(
                            starts, ends, recording.annotations
                        )
                        event_count += len(intervals)
                        dataset_events[recording.dataset_id] = dataset_events.get(
                            recording.dataset_id, 0
                        ) + len(intervals)
                        exclusions = [
                            (item.start_sample, item.stop_sample)
                            for item in recording.annotations
                            if item.kind == "exclude"
                        ]
                        excluded = np.asarray(
                            [
                                any(
                                    start < excluded_stop and end > excluded_start
                                    for excluded_start, excluded_stop in exclusions
                                )
                                for start, end in zip(starts, ends, strict=True)
                            ],
                            dtype=np.bool_,
                        )
                        skipped_exclusion += int(np.count_nonzero(excluded))
                        skipped_post = int(np.count_nonzero((~keep) & (~excluded)))
                        skipped_post_segment_overlap += skipped_post
                        dataset_skipped_post_segment[recording.dataset_id] = (
                            dataset_skipped_post_segment.get(recording.dataset_id, 0) + skipped_post
                        )
                    elif not recording.is_fall:
                        temporal[:] = 0
                    raw = raw[keep]
                    starts = starts[keep]
                    ends = ends[keep]
                    temporal = temporal[keep]
                if recording.supervision_kind == "temporal" and recording.is_fall:
                    decisions = ends - 1
                    missing_positive = sum(
                        int(
                            not np.any(
                                (decisions >= event.onset_sample) & (decisions < event.stop_sample)
                            )
                        )
                        for event in recording.fall_events
                    )
                    events_without_positive_window += missing_positive
                    dataset_events_without_positive[recording.dataset_id] = (
                        dataset_events_without_positive.get(recording.dataset_id, 0)
                        + missing_positive
                    )
                if len(raw):
                    with phases.track("feature_extraction_seconds"):
                        features = extract_window_features(raw, sampling_rate_hz)
                else:
                    features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
                with phases.track("sequence_metadata_write_seconds"):
                    for name, value in (
                        ("dataset_id", recording.dataset_id),
                        ("participant_id", recording.participant_id),
                        ("recording_id", recording.recording_id),
                        ("body_location", recording.body_location),
                        ("activity", recording.activity),
                        ("supervision_kind", recording.supervision_kind),
                        ("is_fall", recording.is_fall),
                        ("fold_id", fold),
                    ):
                        _resize_write(sequences[name], np.asarray([value]))
                    if recording.fall_events:
                        count = len(recording.fall_events)
                        _resize_write(
                            events_group["sequence_index"],
                            np.full(count, sequence_count, dtype=np.int32),
                        )
                        for name, values in (
                            ("onset_sample", [item.onset_sample for item in recording.fall_events]),
                            (
                                "impact_sample",
                                [item.impact_sample for item in recording.fall_events],
                            ),
                            ("stop_sample", [item.stop_sample for item in recording.fall_events]),
                            ("code", [item.code for item in recording.fall_events]),
                        ):
                            _resize_write(events_group[name], np.asarray(values))
                with phases.track("cache_hdf5_write_seconds"):
                    buffer.append(
                        {
                            "raw": raw,
                            "features": features,
                            "sequence_index": np.full(len(raw), sequence_count, dtype=np.int32),
                            "start_sample": starts,
                            "end_sample": ends,
                            "fold_id": np.full(len(raw), fold, dtype=np.int8),
                            "bag_label": np.full(len(raw), int(recording.is_fall), dtype=np.int8),
                            "temporal_label": temporal,
                        }
                    )
                sequence_count += 1
                window_count += len(raw)
                positive_temporal += int(np.count_nonzero(temporal == 1))
                negative_temporal += int(np.count_nonzero(temporal == 0))
                dataset_positive_temporal[recording.dataset_id] = dataset_positive_temporal.get(
                    recording.dataset_id, 0
                ) + int(np.count_nonzero(temporal == 1))
                dataset_negative_temporal[recording.dataset_id] = dataset_negative_temporal.get(
                    recording.dataset_id, 0
                ) + int(np.count_nonzero(temporal == 0))
                dataset_windows[recording.dataset_id] = dataset_windows.get(
                    recording.dataset_id, 0
                ) + len(raw)
                dataset_sequences[recording.dataset_id] = (
                    dataset_sequences.get(recording.dataset_id, 0) + 1
                )
            with phases.track("cache_hdf5_write_seconds"):
                buffer.flush()
                handle.flush()
            build_seconds = time.perf_counter() - started
            phase_seconds = phases.to_dict()
            phase_seconds["container_overhead_seconds"] = max(
                0.0, build_seconds - sum(phase_seconds.values())
            )
            manifest = {
                "window_schema_version": WINDOW_CACHE_SCHEMA,
                "feature_schema_version": config["contract"]["window"]["feature_schema_version"],
                "data_split_fingerprint": fingerprint,
                "contract_sha256": config["contract_sha256"],
                "snapshot_sha256": config["snapshot_sha256"],
                "sampling_rate_hz": sampling_rate_hz,
                "window_samples": window_samples,
                "stride_seconds": stride_seconds,
                "cache_flush_windows": int(config["cache_flush_windows"]),
                "sequences": sequence_count,
                "windows": window_count,
                "features": len(FEATURE_NAMES),
                "fall_events": event_count,
                "positive_temporal_windows": positive_temporal,
                "negative_temporal_windows": negative_temporal,
                "events_without_positive_window": events_without_positive_window,
                "skipped_short_sequences": skipped_short,
                "skipped_post_segment_overlap_windows": skipped_post_segment_overlap,
                "skipped_exclusion_windows": skipped_exclusion,
                "dataset_sequences": dict(sorted(dataset_sequences.items())),
                "dataset_windows": dict(sorted(dataset_windows.items())),
                "dataset_positive_temporal_windows": dict(
                    sorted(dataset_positive_temporal.items())
                ),
                "dataset_negative_temporal_windows": dict(
                    sorted(dataset_negative_temporal.items())
                ),
                "dataset_events": dict(sorted(dataset_events.items())),
                "dataset_events_without_positive_window": dict(
                    sorted(dataset_events_without_positive.items())
                ),
                "build_seconds": build_seconds,
                "build_phase_seconds": phase_seconds,
            }
            if "kfall" in dataset_windows:
                manifest["kfall_regression_candidate"] = {
                    "windows": dataset_windows["kfall"],
                    "positive": dataset_positive_temporal["kfall"],
                    "negative": dataset_negative_temporal["kfall"],
                    "events": dataset_events["kfall"],
                    "events_without_positive_window": dataset_events_without_positive.get(
                        "kfall", 0
                    ),
                    "skipped_post_segment_overlap_windows": dataset_skipped_post_segment.get(
                        "kfall", 0
                    ),
                }
            handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
            handle.flush()
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination, {**manifest, "cache_reused_this_invocation": False}


@dataclass(frozen=True, slots=True)
class UnifiedWindowStore:
    path: Path
    sequence_index: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray
    fold_id: np.ndarray
    bag_label: np.ndarray
    temporal_label: np.ndarray
    dataset_id: np.ndarray
    participant_id: np.ndarray
    recording_id: np.ndarray
    body_location: np.ndarray
    supervision_kind: np.ndarray
    sequence_is_fall: np.ndarray
    sequence_fold_id: np.ndarray
    event_sequence_index: np.ndarray
    event_onset_sample: np.ndarray
    event_impact_sample: np.ndarray
    event_stop_sample: np.ndarray
    event_code: np.ndarray
    manifest: dict[str, Any]

    @property
    def size(self) -> int:
        return len(self.sequence_index)

    def materialize(self, kind: str) -> np.ndarray:
        if kind not in {"raw", "features"}:
            raise ValueError(f"Unknown window input kind: {kind}")
        with h5py.File(self.path, "r") as handle:
            return np.asarray(handle[f"windows/{kind}"], dtype=np.float32)

    def window_datasets(self) -> np.ndarray:
        return self.dataset_id[self.sequence_index]

    def window_participants(self) -> np.ndarray:
        return self.participant_id[self.sequence_index]


def load_unified_window_store(path: Path) -> UnifiedWindowStore:
    with h5py.File(path, "r") as handle:
        manifest = json.loads(str(handle.attrs["manifest_json"]))
        store = UnifiedWindowStore(
            path=path,
            sequence_index=np.asarray(handle["windows/sequence_index"], dtype=np.int32),
            start_sample=np.asarray(handle["windows/start_sample"], dtype=np.int32),
            end_sample=np.asarray(handle["windows/end_sample"], dtype=np.int32),
            fold_id=np.asarray(handle["windows/fold_id"], dtype=np.int8),
            bag_label=np.asarray(handle["windows/bag_label"], dtype=np.int8),
            temporal_label=np.asarray(handle["windows/temporal_label"], dtype=np.int8),
            dataset_id=_text_array(handle["sequences/dataset_id"]),
            participant_id=_text_array(handle["sequences/participant_id"]),
            recording_id=_text_array(handle["sequences/recording_id"]),
            body_location=_text_array(handle["sequences/body_location"]),
            supervision_kind=_text_array(handle["sequences/supervision_kind"]),
            sequence_is_fall=np.asarray(handle["sequences/is_fall"], dtype=np.bool_),
            sequence_fold_id=np.asarray(handle["sequences/fold_id"], dtype=np.int8),
            event_sequence_index=np.asarray(handle["events/sequence_index"], dtype=np.int32),
            event_onset_sample=np.asarray(handle["events/onset_sample"], dtype=np.int64),
            event_impact_sample=np.asarray(handle["events/impact_sample"], dtype=np.int64),
            event_stop_sample=np.asarray(handle["events/stop_sample"], dtype=np.int64),
            event_code=_text_array(handle["events/code"]),
            manifest=manifest,
        )
    if store.size != manifest["windows"]:
        raise ValueError("Unified cache size does not match its manifest")
    if np.any(store.end_sample - store.start_sample != manifest["window_samples"]):
        raise ValueError("Unified cache contains invalid window lengths")
    event_lengths = {
        len(store.event_sequence_index),
        len(store.event_onset_sample),
        len(store.event_impact_sample),
        len(store.event_stop_sample),
        len(store.event_code),
    }
    if event_lengths != {int(manifest["fall_events"])}:
        raise ValueError("Unified cache event table does not match its manifest")
    if np.any(
        (store.event_sequence_index < 0)
        | (store.event_sequence_index >= len(store.dataset_id))
        | (store.event_onset_sample >= store.event_impact_sample)
        | (store.event_impact_sample >= store.event_stop_sample)
    ):
        raise ValueError("Unified cache contains invalid fall events")
    return store
