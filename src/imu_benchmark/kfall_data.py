from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .contract import load_contract_bundle
from .dataset import EXTERNAL_DATASET_IDS, iter_recordings
from .performance import PhaseTimer
from .protocol import segment_decision_time_labels


def load_kfall_config(path: Path) -> dict[str, Any]:
    bundle = load_contract_bundle(path)
    config = {**bundle.effective_values(), **bundle.experiment}
    if config.get("data_schema_version") != "3.0.0":
        raise ValueError("KFall evaluation requires HDF5 schema 3.0.0")
    if config.get("sampling_rate_hz") != 30:
        raise ValueError("KFall evaluation requires canonical 30 Hz input")
    if config.get("window_samples") != 60 or config.get("stride_samples") != 15:
        raise ValueError("KFall evaluation requires 60-sample windows with stride 15")
    if config.get("temporal_policy") != "decision_time_within_fall_activity_segment":
        raise ValueError("Unexpected KFall temporal policy")
    if config.get("post_segment_overlap_policy") != "exclude":
        raise ValueError("Post-segment overlap windows must be excluded")
    if config.get("alarm_policy") != "single_window":
        raise ValueError("The provisional evaluation supports single-window alerts only")
    if set(config.get("profiles", {})) != {"smoke", "evaluate"}:
        raise ValueError("KFall profiles must be exactly smoke and evaluate")
    if config.get("external_dataset_id") != "kfall":
        raise ValueError("The external dataset must be KFall")
    if tuple(config.get("training_dataset_ids", ())) != (
        "cgu_bes",
        "sisfall",
        "uci_455",
        "umafall",
        "upfall",
    ):
        raise ValueError("Unexpected KFall training dataset scope")
    if tuple(config.get("suites", ())) != ("fall_universal", "fall_waist_only"):
        raise ValueError("KFall suites must be universal and waist-only")
    if tuple(config.get("models", ())) != (
        "threshold_impact",
        "torch_1d_cnn",
        "torch_lstm",
        "torch_cnn_lstm",
    ):
        raise ValueError("Unexpected KFall model scope")
    for profile_name, profile in config["profiles"].items():
        folds = tuple(int(value) for value in profile.get("validation_folds", ()))
        if not folds or len(set(folds)) != len(folds) or set(folds) - set(range(5)):
            raise ValueError(f"Invalid KFall folds in profile {profile_name}")
        for field in ("runtime_budget_seconds", "max_epochs", "patience"):
            if int(profile.get(field, 0)) <= 0:
                raise ValueError(f"Invalid KFall {field} in profile {profile_name}")
    return config


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _external_folds(path: Path, split_version: str) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["split_version"] != split_version or row["dataset_id"] != "kfall":
                raise ValueError("Unexpected KFall split row")
            participant = row["participant_id"]
            if participant in result:
                raise ValueError(f"Duplicate KFall participant: {participant}")
            result[participant] = int(row["fold_id"])
    if set(result.values()) != set(range(5)):
        raise ValueError("KFall participant folds must cover 0..4")
    return result


def _source_fingerprint(data_root: Path) -> str:
    path = data_root / "kfall.h5"
    with h5py.File(path, "r") as handle:
        logical = str(handle.attrs["logical_content_sha256"])
        schema = str(handle.attrs["imu_schema_version"])
    return hashlib.sha256(f"kfall:{schema}:{logical}".encode()).hexdigest()


def _windows(values: np.ndarray, length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(values) - length + 1, stride, dtype=np.int32)
    if not len(starts):
        return np.empty((0, length, 6), dtype=np.float32), starts
    return (
        np.stack([values[start : start + length] for start in starts]).astype(np.float32),
        starts,
    )


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    start = len(dataset)
    stop = start + len(values)
    dataset.resize((stop, *dataset.shape[1:]))
    dataset[start:stop] = values


def prepare_kfall_window_store(
    *, project_root: Path, cache_root: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    data_root = project_root / config["external_data_path"]
    split_path = project_root / config["external_split_path"]
    source_fingerprint = _source_fingerprint(data_root)
    split_fingerprint = _file_sha256(split_path)
    policy_payload = {
        "source": source_fingerprint,
        "split": split_fingerprint,
        "window_samples": config["window_samples"],
        "stride_samples": config["stride_samples"],
        "temporal_policy": config["temporal_policy"],
        "post_segment_overlap_policy": config["post_segment_overlap_policy"],
        "contract": config["contract_sha256"],
        "snapshot": config["snapshot_sha256"],
    }
    data_split_fingerprint = hashlib.sha256(
        json.dumps(policy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    schema = str(config["kfall_window_schema_version"])
    destination = cache_root / "windows" / schema / f"{data_split_fingerprint[:16]}.h5"
    if destination.exists():
        with h5py.File(destination, "r") as handle:
            manifest = json.loads(str(handle.attrs["manifest_json"]))
        if manifest.get("data_split_fingerprint") != data_split_fingerprint:
            raise ValueError(f"Conflicting KFall cache: {destination}")
        return destination, {**manifest, "cache_reused_this_invocation": True}

    folds = _external_folds(split_path, str(config["external_split_version"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".h5.tmp-{os.getpid()}")
    started = time.perf_counter()
    counters = {
        "sequences": 0,
        "windows": 0,
        "positive_windows": 0,
        "negative_windows": 0,
        "fall_events": 0,
        "events_without_positive_window": 0,
        "skipped_short_sequences": 0,
        "skipped_post_segment_overlap_windows": 0,
        "skipped_exclusion_windows": 0,
    }
    phases = PhaseTimer()
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w", libver=("earliest", "v114")) as handle:
        handle.attrs.update(
            {
                "window_schema_version": schema,
                "source_fingerprint": source_fingerprint,
                "split_fingerprint": split_fingerprint,
                "data_split_fingerprint": data_split_fingerprint,
                "sampling_rate_hz": config["sampling_rate_hz"],
                "window_samples": config["window_samples"],
                "stride_samples": config["stride_samples"],
                "temporal_policy": config["temporal_policy"],
                "post_segment_overlap_policy": config["post_segment_overlap_policy"],
                "contract_version": config["contract_version"],
                "contract_sha256": config["contract_sha256"],
                "snapshot_version": config["snapshot_version"],
                "snapshot_sha256": config["snapshot_sha256"],
                "data_quality_status": config["data_quality_status"],
                "hdf5_compatibility": "1.14",
            }
        )
        sequences = handle.create_group("sequences")
        for name in ("participant_id", "recording_id"):
            sequences.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
            )
        for name, dtype in (
            ("is_fall", "?"),
            ("fold_id", "i1"),
            ("event_onset_sample", "i4"),
            ("event_impact_sample", "i4"),
            ("event_fall_stop_sample", "i4"),
            ("event_has_decision_window", "?"),
        ):
            sequences.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
        windows_group = handle.create_group("windows")
        raw_dataset = windows_group.create_dataset(
            "raw",
            shape=(0, config["window_samples"], 6),
            maxshape=(None, config["window_samples"], 6),
            dtype="f4",
            chunks=(1024, config["window_samples"], 6),
            compression="lzf",
            shuffle=True,
        )
        scalar_datasets = {
            name: windows_group.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
            )
            for name, dtype in (
                ("sequence_index", "i4"),
                ("start_sample", "i4"),
                ("end_sample", "i4"),
                ("fold_id", "i1"),
                ("temporal_label", "i1"),
            )
        }

        recordings = iter_recordings(data_root, expected_dataset_ids=EXTERNAL_DATASET_IDS)
        for recording in phases.iterate("source_hdf5_read_seconds", recordings):
            if recording.supervision_kind != "temporal" or recording.body_location != "lower_back":
                raise ValueError(f"Unexpected KFall sequence contract: {recording.recording_id}")
            fold = folds.get(recording.participant_id)
            if fold is None:
                raise ValueError(f"Missing KFall participant fold: {recording.participant_id}")
            with phases.track("window_generation_seconds"):
                raw_windows, starts = _windows(
                    recording.values, config["window_samples"], config["stride_samples"]
                )
            if not len(raw_windows):
                counters["skipped_short_sequences"] += 1
                continue
            ends = starts + int(config["window_samples"])
            labels = np.zeros(len(starts), dtype=np.int8)
            keep = np.ones(len(starts), dtype=np.bool_)
            onset = impact = fall_stop = -1
            has_positive = False
            label_started = time.perf_counter()
            if recording.is_fall:
                counters["fall_events"] += 1
                if recording.fall_event is None:
                    raise ValueError(f"KFall fall lacks onset/impact: {recording.recording_id}")
                onset = recording.fall_event.onset_sample
                impact = recording.fall_event.impact_sample
                labels, temporal_keep, intervals = segment_decision_time_labels(
                    starts, ends, recording.annotations
                )
                if len(intervals) != 1 or intervals[0][0] != onset:
                    raise ValueError(
                        f"KFall fall must have one onset-aligned segment: {recording.recording_id}"
                    )
                fall_stop = intervals[0][1]
                exclusions = [
                    (item.start_sample, item.stop_sample)
                    for item in recording.annotations
                    if item.kind == "exclude"
                ]
                excluded = np.asarray(
                    [
                        any(
                            start < stop_excluded and end > start_excluded
                            for start_excluded, stop_excluded in exclusions
                        )
                        for start, end in zip(starts, ends, strict=True)
                    ],
                    dtype=np.bool_,
                )
                counters["skipped_exclusion_windows"] += int(np.count_nonzero(excluded))
                counters["skipped_post_segment_overlap_windows"] += int(
                    np.count_nonzero((~temporal_keep) & (~excluded))
                )
                keep &= temporal_keep
                has_positive = bool(np.any(labels[keep] == 1))
                if not has_positive:
                    counters["events_without_positive_window"] += 1
            else:
                labels, keep, intervals = segment_decision_time_labels(
                    starts, ends, recording.annotations
                )
                if intervals:
                    raise ValueError(f"KFall ADL contains fall segments: {recording.recording_id}")
            phases.seconds["window_label_seconds"] = phases.seconds.get(
                "window_label_seconds", 0.0
            ) + (time.perf_counter() - label_started)

            selected_raw = raw_windows[keep]
            selected_starts = starts[keep]
            selected_ends = ends[keep]
            selected_labels = labels[keep]
            sequence_index = counters["sequences"]
            values = {
                "sequence_index": np.full(len(selected_raw), sequence_index, dtype=np.int32),
                "start_sample": selected_starts,
                "end_sample": selected_ends,
                "fold_id": np.full(len(selected_raw), fold, dtype=np.int8),
                "temporal_label": selected_labels,
            }
            with phases.track("cache_hdf5_write_seconds"):
                for name, value in (
                    ("participant_id", recording.participant_id),
                    ("recording_id", recording.recording_id),
                    ("is_fall", recording.is_fall),
                    ("fold_id", fold),
                    ("event_onset_sample", onset),
                    ("event_impact_sample", impact),
                    ("event_fall_stop_sample", fall_stop),
                    ("event_has_decision_window", has_positive),
                ):
                    _append(sequences[name], np.asarray([value]))
                _append(raw_dataset, selected_raw)
                for name, array in values.items():
                    _append(scalar_datasets[name], array)
            counters["sequences"] += 1
            counters["windows"] += len(selected_raw)
            counters["positive_windows"] += int(np.count_nonzero(selected_labels == 1))
            counters["negative_windows"] += int(np.count_nonzero(selected_labels == 0))

        build_seconds = time.perf_counter() - started
        phase_seconds = phases.to_dict()
        phase_seconds["container_overhead_seconds"] = max(
            0.0, build_seconds - sum(phase_seconds.values())
        )
        manifest = {
            "window_schema_version": schema,
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "data_split_fingerprint": data_split_fingerprint,
            "temporal_policy": config["temporal_policy"],
            "post_segment_overlap_policy": config["post_segment_overlap_policy"],
            "contract_version": config["contract_version"],
            "contract_sha256": config["contract_sha256"],
            "snapshot_version": config["snapshot_version"],
            "snapshot_sha256": config["snapshot_sha256"],
            "sampling_rate_hz": config["sampling_rate_hz"],
            "window_samples": config["window_samples"],
            "stride_samples": config["stride_samples"],
            "data_quality_status": config["data_quality_status"],
            **counters,
            "build_seconds": build_seconds,
            "build_phase_seconds": phase_seconds,
        }
        handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
        handle.flush()
    os.replace(temporary, destination)
    return destination, {**manifest, "cache_reused_this_invocation": False}


@dataclass(frozen=True, slots=True)
class KFallWindowStore:
    path: Path
    sequence_index: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray
    fold_id: np.ndarray
    temporal_label: np.ndarray
    participant_id: np.ndarray
    recording_id: np.ndarray
    sequence_is_fall: np.ndarray
    sequence_fold_id: np.ndarray
    event_onset_sample: np.ndarray
    event_impact_sample: np.ndarray
    event_fall_stop_sample: np.ndarray
    event_has_decision_window: np.ndarray
    manifest: dict[str, Any]

    @property
    def size(self) -> int:
        return len(self.sequence_index)

    def load_raw(self, indices: np.ndarray | None = None) -> np.ndarray:
        selected = (
            np.arange(self.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(selected) and np.any(np.diff(selected) <= 0):
            raise ValueError("KFall window indices must be strictly increasing")
        with h5py.File(self.path, "r") as handle:
            dataset = handle["windows/raw"]
            if not len(selected):
                return np.empty((0, *dataset.shape[1:]), dtype=np.float32)
            result = np.empty((len(selected), *dataset.shape[1:]), dtype=np.float32)
            blocks: list[tuple[int, int]] = []
            block_start = 0
            for position in range(1, len(selected)):
                if (
                    selected[position] - selected[position - 1] > 64
                    or selected[position] - selected[block_start] >= 65_536
                ):
                    blocks.append((block_start, position))
                    block_start = position
            blocks.append((block_start, len(selected)))
            for output_start, output_stop in blocks:
                first = int(selected[output_start])
                last = int(selected[output_stop - 1]) + 1
                span = np.asarray(dataset[first:last], dtype=np.float32)
                result[output_start:output_stop] = span[selected[output_start:output_stop] - first]
            return result

    def window_participants(self) -> np.ndarray:
        return self.participant_id[self.sequence_index]


def load_kfall_window_store(path: Path) -> KFallWindowStore:
    with h5py.File(path, "r") as handle:
        manifest = json.loads(str(handle.attrs["manifest_json"]))

        def text_values(name: str) -> np.ndarray:
            return np.asarray(
                [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in handle[f"sequences/{name}"]
                ]
            )

        store = KFallWindowStore(
            path=path,
            sequence_index=np.asarray(handle["windows/sequence_index"], dtype=np.int32),
            start_sample=np.asarray(handle["windows/start_sample"], dtype=np.int32),
            end_sample=np.asarray(handle["windows/end_sample"], dtype=np.int32),
            fold_id=np.asarray(handle["windows/fold_id"], dtype=np.int8),
            temporal_label=np.asarray(handle["windows/temporal_label"], dtype=np.int8),
            participant_id=text_values("participant_id"),
            recording_id=text_values("recording_id"),
            sequence_is_fall=np.asarray(handle["sequences/is_fall"], dtype=np.bool_),
            sequence_fold_id=np.asarray(handle["sequences/fold_id"], dtype=np.int8),
            event_onset_sample=np.asarray(
                handle["sequences/event_onset_sample"], dtype=np.int32
            ),
            event_impact_sample=np.asarray(
                handle["sequences/event_impact_sample"], dtype=np.int32
            ),
            event_fall_stop_sample=np.asarray(
                handle["sequences/event_fall_stop_sample"], dtype=np.int32
            ),
            event_has_decision_window=np.asarray(
                handle["sequences/event_has_decision_window"], dtype=np.bool_
            ),
            manifest=manifest,
        )
    if store.size != manifest["windows"] or np.any(store.end_sample - store.start_sample != 60):
        raise ValueError("Invalid KFall window store")
    if int(np.count_nonzero(store.temporal_label == 1)) != manifest["positive_windows"]:
        raise ValueError("KFall positive window count does not match its manifest")
    return store
