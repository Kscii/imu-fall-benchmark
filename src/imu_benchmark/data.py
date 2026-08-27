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
from .dataset import iter_recordings
from .performance import PhaseTimer
from .protocol import segment_decision_time_labels

SIGNAL_NAMES = (
    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "acceleration_z_mps2",
    "angular_velocity_x_rad_s",
    "angular_velocity_y_rad_s",
    "angular_velocity_z_rad_s",
    "acceleration_norm_mps2",
    "angular_velocity_norm_rad_s",
)
TIME_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "peak_to_peak",
    "mean_abs_first_difference",
    "skewness",
    "excess_kurtosis",
    "mean_square_energy",
)
FREQUENCY_STATISTICS = (
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
    "relative_power_0_0p5_hz",
    "relative_power_0p5_3_hz",
    "relative_power_3_8_hz",
    "relative_power_8_15_hz",
)

FEATURE_NAMES = (
    *(f"{signal}__{stat}" for signal in SIGNAL_NAMES for stat in TIME_STATISTICS),
    "acceleration_sma",
    "angular_velocity_sma",
    "acceleration_peak_relative_time",
    "angular_velocity_peak_relative_time",
    "acceleration_jerk_norm__mean",
    "acceleration_jerk_norm__std",
    "acceleration_jerk_norm__max",
    "acceleration_jerk_norm__rms",
    "acceleration_xy_correlation",
    "acceleration_xz_correlation",
    "acceleration_yz_correlation",
    "angular_velocity_xy_correlation",
    "angular_velocity_xz_correlation",
    "angular_velocity_yz_correlation",
    *(f"{signal}__{stat}" for signal in SIGNAL_NAMES for stat in FREQUENCY_STATISTICS),
)


def load_config(path: Path) -> dict[str, Any]:
    bundle = load_contract_bundle(path)
    config = {**bundle.effective_values(), **bundle.experiment}
    if config["sampling_rate_hz"] != 30:
        raise ValueError("The benchmark requires canonical 30 Hz input")
    if config["window_samples"] != 60 or config["stride_samples"] != 15:
        raise ValueError("The reproduction protocol requires 60-sample windows and stride 15")
    if set(config["profiles"]) != {"smoke", "reproduce"}:
        raise ValueError("Profiles must be exactly smoke and reproduce")
    for profile_name, profile in config["profiles"].items():
        folds = tuple(int(value) for value in profile.get("outer_folds", ()))
        if not folds or len(set(folds)) != len(folds) or set(folds) - set(range(5)):
            raise ValueError(f"Invalid outer folds in profile {profile_name}")
    return config


def _folds(path: Path, split_version: str) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["split_version"] != split_version:
                raise ValueError(f"Unexpected split version: {row['split_version']}")
            result[(row["dataset_id"], row["participant_id"])] = int(row["fold_id"])
    if set(result.values()) != set(range(5)):
        raise ValueError("Participant folds must cover 0..4")
    return result


def _source_fingerprint(processed_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(processed_root.glob("*.h5"))
    if not files:
        raise ValueError(f"No canonical HDF5 shards found at {processed_root}")
    for path in files:
        with h5py.File(path, "r") as handle:
            dataset_id = str(handle.attrs["dataset_id"])
            logical_hash = str(handle.attrs["logical_content_sha256"])
        digest.update(dataset_id.encode())
        digest.update(logical_hash.encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_standardized_moments(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)
    standardized = np.divide(
        signals - mean,
        std,
        out=np.zeros_like(signals, dtype=np.float64),
        where=std > 1e-12,
    )
    return np.mean(standardized**3, axis=1), np.mean(standardized**4, axis=1) - 3.0


def _correlation(windows: np.ndarray, left: int, right: int) -> np.ndarray:
    x = windows[:, :, left].astype(np.float64)
    y = windows[:, :, right].astype(np.float64)
    x -= np.mean(x, axis=1, keepdims=True)
    y -= np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def extract_window_features(windows: np.ndarray, sampling_rate_hz: float = 30.0) -> np.ndarray:
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (60, 6):
        raise ValueError(f"Expected windows with shape (n, 60, 6), got {raw.shape}")
    acceleration_norm = np.linalg.norm(raw[:, :, :3], axis=2, keepdims=True)
    angular_velocity_norm = np.linalg.norm(raw[:, :, 3:], axis=2, keepdims=True)
    signals = np.concatenate((raw, acceleration_norm, angular_velocity_norm), axis=2)
    q25, median, q75 = np.percentile(signals, (25, 50, 75), axis=1)
    skewness, kurtosis = _safe_standardized_moments(signals)
    time_values = np.stack(
        (
            np.mean(signals, axis=1),
            np.std(signals, axis=1),
            np.min(signals, axis=1),
            np.max(signals, axis=1),
            median,
            q75 - q25,
            np.ptp(signals, axis=1),
            np.mean(np.abs(np.diff(signals, axis=1)), axis=1),
            skewness,
            kurtosis,
            np.mean(signals**2, axis=1),
        ),
        axis=2,
    ).reshape(len(raw), -1)

    acceleration_sma = np.mean(np.sum(np.abs(raw[:, :, :3]), axis=2), axis=1)
    angular_velocity_sma = np.mean(np.sum(np.abs(raw[:, :, 3:]), axis=2), axis=1)
    denominator = raw.shape[1] - 1
    acceleration_peak_time = np.argmax(acceleration_norm[:, :, 0], axis=1) / denominator
    angular_velocity_peak_time = np.argmax(angular_velocity_norm[:, :, 0], axis=1) / denominator
    jerk = np.diff(raw[:, :, :3], axis=1) * sampling_rate_hz
    jerk_norm = np.linalg.norm(jerk, axis=2)
    jerk_values = np.column_stack(
        (
            np.mean(jerk_norm, axis=1),
            np.std(jerk_norm, axis=1),
            np.max(jerk_norm, axis=1),
            np.sqrt(np.mean(jerk_norm**2, axis=1)),
        )
    )
    pairs = ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5))
    correlations = np.column_stack(tuple(_correlation(raw, left, right) for left, right in pairs))

    taper = np.hanning(raw.shape[1])[None, :, None]
    tapered = (signals - np.mean(signals, axis=1, keepdims=True)) * taper
    spectrum = np.fft.rfft(tapered, axis=1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(raw.shape[1], d=1.0 / sampling_rate_hz)
    total_power = np.sum(power, axis=1)
    safe_total = np.where(total_power > 1e-18, total_power, 1.0)
    non_dc = power[:, 1:, :]
    dominant = frequencies[1:][np.argmax(non_dc, axis=1)]
    centroid = np.sum(power * frequencies[None, :, None], axis=1) / safe_total
    normalized_power = power / safe_total[:, None, :]
    log_power = np.zeros_like(normalized_power)
    np.log2(normalized_power, out=log_power, where=normalized_power > 0)
    entropy = -np.sum(normalized_power * log_power, axis=1) / np.log2(power.shape[1])
    bands = ((0.0, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, 15.000001))
    relative_bands = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        relative_bands.append(np.sum(power[:, mask, :], axis=1) / safe_total)
    frequency_values = np.stack((dominant, centroid, entropy, *relative_bands), axis=2).reshape(
        len(raw), -1
    )

    result = np.column_stack(
        (
            time_values,
            acceleration_sma,
            angular_velocity_sma,
            acceleration_peak_time,
            angular_velocity_peak_time,
            jerk_values,
            correlations,
            frequency_values,
        )
    ).astype(np.float32)
    if result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError(f"Invalid engineered feature matrix: {result.shape}")
    return result


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    start = len(dataset)
    end = start + len(values)
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values


def _windows(values: np.ndarray, length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(values) - length + 1, stride, dtype=np.int32)
    if not len(starts):
        return np.empty((0, length, 6), dtype=np.float32), starts
    windows = np.stack([values[start : start + length] for start in starts]).astype(np.float32)
    return windows, starts


def prepare_window_store(
    *, project_root: Path, cache_root: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    processed_root = project_root / config["data_path"]
    source_fingerprint = _source_fingerprint(processed_root)
    split_path = project_root / config["split_path"]
    split_fingerprint = _file_sha256(split_path)
    fingerprint_payload = {
        "source": source_fingerprint,
        "split": split_fingerprint,
        "contract": config["contract_sha256"],
        "snapshot": config["snapshot_sha256"],
    }
    data_split_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    schema = str(config["window_schema_version"])
    destination = cache_root / "windows" / schema / f"{data_split_fingerprint[:16]}.h5"
    if destination.exists():
        with h5py.File(destination, "r") as handle:
            manifest = json.loads(str(handle.attrs["manifest_json"]))
        if manifest["source_fingerprint"] != source_fingerprint:
            raise ValueError(f"Conflicting window cache: {destination}")
        if manifest.get("split_fingerprint") != split_fingerprint:
            raise ValueError(f"Conflicting split for window cache: {destination}")
        return destination, {**manifest, "cache_reused_this_invocation": True}

    folds = _folds(split_path, config["split_version"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".h5.tmp-{os.getpid()}")
    started = time.perf_counter()
    sequence_count = 0
    window_count = 0
    skipped_short = 0
    event_count = 0
    phases = PhaseTimer()
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w", libver=("earliest", "v114")) as handle:
        handle.attrs.update(
            {
                "window_schema_version": schema,
                "feature_schema_version": config["feature_schema_version"],
                "source_fingerprint": source_fingerprint,
                "split_fingerprint": split_fingerprint,
                "data_split_fingerprint": data_split_fingerprint,
                "contract_version": config["contract_version"],
                "contract_sha256": config["contract_sha256"],
                "snapshot_version": config["snapshot_version"],
                "snapshot_sha256": config["snapshot_sha256"],
                "sampling_rate_hz": config["sampling_rate_hz"],
                "window_samples": config["window_samples"],
                "stride_samples": config["stride_samples"],
                "feature_names": json.dumps(FEATURE_NAMES),
                "hdf5_compatibility": "1.14",
            }
        )
        sequences = handle.create_group("sequences")
        for name in ("dataset_id", "participant_id", "recording_id", "body_location", "activity"):
            sequences.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=text_dtype, chunks=True
            )
        sequences.create_dataset("is_fall", shape=(0,), maxshape=(None,), dtype="?", chunks=True)
        sequences.create_dataset("fold_id", shape=(0,), maxshape=(None,), dtype="i1", chunks=True)
        sequences.create_dataset(
            "event_onset_sample", shape=(0,), maxshape=(None,), dtype="f8", chunks=True
        )
        sequences.create_dataset(
            "event_impact_sample", shape=(0,), maxshape=(None,), dtype="f8", chunks=True
        )
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
        feature_dataset = windows_group.create_dataset(
            "features",
            shape=(0, len(FEATURE_NAMES)),
            maxshape=(None, len(FEATURE_NAMES)),
            dtype="f4",
            chunks=(2048, len(FEATURE_NAMES)),
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
                ("bag_label", "i1"),
                ("temporal_label", "i1"),
                ("position_label", "i1"),
            )
        }
        recordings = phases.iterate(
            "source_hdf5_read_seconds", iter_recordings(processed_root)
        )
        for recording in recordings:
            fold_key = (recording.dataset_id, recording.participant_id)
            if fold_key not in folds:
                raise ValueError(f"Missing participant fold for {fold_key}")
            with phases.track("window_generation_seconds"):
                raw_windows, starts = _windows(
                    recording.values, config["window_samples"], config["stride_samples"]
                )
            if not len(raw_windows):
                skipped_short += 1
                continue
            with phases.track("window_label_seconds"):
                ends = starts + config["window_samples"]
                temporal = np.full(len(starts), -1, dtype=np.int8)
                keep = np.ones(len(starts), dtype=np.bool_)
                if recording.supervision_kind == "temporal":
                    temporal, keep, fall_intervals = segment_decision_time_labels(
                        starts, ends, recording.annotations
                    )
                    event_count += len(fall_intervals)
                elif not recording.is_fall:
                    temporal[:] = 0
                raw_windows = raw_windows[keep]
                starts = starts[keep]
                ends = ends[keep]
                temporal = temporal[keep]
            if not len(raw_windows):
                skipped_short += 1
                continue
            with phases.track("feature_extraction_seconds"):
                features = extract_window_features(raw_windows, config["sampling_rate_hz"])
            position = np.full(len(starts), -1, dtype=np.int8)
            if recording.body_location == "chest":
                position[:] = 0
            elif recording.body_location == "waist":
                position[:] = 1
            values = {
                "sequence_index": np.full(len(starts), sequence_count, dtype=np.int32),
                "start_sample": starts,
                "end_sample": ends,
                "fold_id": np.full(len(starts), folds[fold_key], dtype=np.int8),
                "bag_label": np.full(len(starts), int(recording.is_fall), dtype=np.int8),
                "temporal_label": temporal,
                "position_label": position,
            }
            with phases.track("cache_hdf5_write_seconds"):
                for name, value in (
                    ("dataset_id", recording.dataset_id),
                    ("participant_id", recording.participant_id),
                    ("recording_id", recording.recording_id),
                    ("body_location", recording.body_location),
                    ("activity", recording.activity),
                    ("is_fall", recording.is_fall),
                    ("fold_id", folds[fold_key]),
                    (
                        "event_onset_sample",
                        -1.0
                        if recording.fall_event is None
                        else recording.fall_event.onset_time_s * config["sampling_rate_hz"],
                    ),
                    (
                        "event_impact_sample",
                        -1.0
                        if recording.fall_event is None
                        else recording.fall_event.impact_time_s * config["sampling_rate_hz"],
                    ),
                ):
                    _append(sequences[name], np.asarray([value]))
                _append(raw_dataset, raw_windows)
                _append(feature_dataset, features)
                for name, array in values.items():
                    _append(scalar_datasets[name], array)
            sequence_count += 1
            window_count += len(starts)

        build_seconds = time.perf_counter() - started
        phase_seconds = phases.to_dict()
        phase_seconds["container_overhead_seconds"] = max(
            0.0, build_seconds - sum(phase_seconds.values())
        )
        manifest = {
            "window_schema_version": schema,
            "feature_schema_version": config["feature_schema_version"],
            "source_fingerprint": source_fingerprint,
            "split_fingerprint": split_fingerprint,
            "data_split_fingerprint": data_split_fingerprint,
            "contract_version": config["contract_version"],
            "contract_sha256": config["contract_sha256"],
            "snapshot_version": config["snapshot_version"],
            "snapshot_sha256": config["snapshot_sha256"],
            "sampling_rate_hz": config["sampling_rate_hz"],
            "window_samples": config["window_samples"],
            "stride_samples": config["stride_samples"],
            "sequences": sequence_count,
            "windows": window_count,
            "features": len(FEATURE_NAMES),
            "events": event_count,
            "skipped_short_sequences": skipped_short,
            "build_seconds": build_seconds,
            "build_phase_seconds": phase_seconds,
        }
        handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
        handle.flush()
    os.replace(temporary, destination)
    return destination, {**manifest, "cache_reused_this_invocation": False}


@dataclass(frozen=True, slots=True)
class WindowStore:
    path: Path
    sequence_index: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray
    fold_id: np.ndarray
    bag_label: np.ndarray
    temporal_label: np.ndarray
    position_label: np.ndarray
    dataset_id: np.ndarray
    participant_id: np.ndarray
    recording_id: np.ndarray
    body_location: np.ndarray
    activity: np.ndarray
    sequence_is_fall: np.ndarray
    event_onset_sample: np.ndarray
    event_impact_sample: np.ndarray
    manifest: dict[str, Any]

    @property
    def size(self) -> int:
        return len(self.sequence_index)

    def load(self, kind: str, indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64)
        if len(selected) and np.any(np.diff(selected) <= 0):
            raise ValueError("Window indices must be strictly increasing for HDF5 reads")
        if kind not in {"raw", "features"}:
            raise ValueError(f"Unknown window array kind: {kind}")
        with h5py.File(self.path, "r") as handle:
            dataset = handle[f"windows/{kind}"]
            if not len(selected):
                return np.empty((0, *dataset.shape[1:]), dtype=np.float32)
            # Read bounded spans. Nearby selected rows are cheaper as one HDF5 read,
            # while large gaps and 65k-row boundaries prevent whole-cache materialization.
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
            result = np.empty((len(selected), *dataset.shape[1:]), dtype=np.float32)
            for output_start, output_stop in blocks:
                first = int(selected[output_start])
                last = int(selected[output_stop - 1]) + 1
                span = np.asarray(dataset[first:last], dtype=np.float32)
                result[output_start:output_stop] = span[selected[output_start:output_stop] - first]
            return result

    def window_participants(self) -> np.ndarray:
        return self.participant_id[self.sequence_index]

    def window_recordings(self) -> np.ndarray:
        return self.recording_id[self.sequence_index]

    def window_datasets(self) -> np.ndarray:
        return self.dataset_id[self.sequence_index]


def load_window_store(path: Path) -> WindowStore:
    with h5py.File(path, "r") as handle:
        manifest = json.loads(str(handle.attrs["manifest_json"]))
        sequence_index = np.asarray(handle["windows/sequence_index"], dtype=np.int32)

        def sequence_text(name: str) -> np.ndarray:
            values = handle[f"sequences/{name}"]
            return np.asarray(
                [value.decode() if isinstance(value, bytes) else str(value) for value in values]
            )

        store = WindowStore(
            path=path,
            sequence_index=sequence_index,
            start_sample=np.asarray(handle["windows/start_sample"], dtype=np.int32),
            end_sample=np.asarray(handle["windows/end_sample"], dtype=np.int32),
            fold_id=np.asarray(handle["windows/fold_id"], dtype=np.int8),
            bag_label=np.asarray(handle["windows/bag_label"], dtype=np.int8),
            temporal_label=np.asarray(handle["windows/temporal_label"], dtype=np.int8),
            position_label=np.asarray(handle["windows/position_label"], dtype=np.int8),
            dataset_id=sequence_text("dataset_id"),
            participant_id=sequence_text("participant_id"),
            recording_id=sequence_text("recording_id"),
            body_location=sequence_text("body_location"),
            activity=sequence_text("activity"),
            sequence_is_fall=np.asarray(handle["sequences/is_fall"], dtype=np.bool_),
            event_onset_sample=np.asarray(handle["sequences/event_onset_sample"], dtype=np.float64),
            event_impact_sample=np.asarray(
                handle["sequences/event_impact_sample"], dtype=np.float64
            ),
            manifest=manifest,
        )
    if store.size != manifest["windows"] or np.any(store.end_sample - store.start_sample != 60):
        raise ValueError("Invalid HDF5 window store")
    return store
