from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from imu_benchmark.configuration import load_experiment
from imu_benchmark.contract import load_contract_snapshot
from imu_benchmark.data import FEATURE_NAMES, extract_window_features
from imu_benchmark.dataset import FEATURE_COLUMNS, FEATURE_UNITS, Annotation, validate_hdf5_file
from imu_benchmark.evaluation import false_positive_windows_per_hour
from imu_benchmark.protocol import linear_resample_to_grid, segment_decision_time_labels
from imu_benchmark.window_cache import _window_starts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_contract_and_active_manifest_are_single_protocol_source(
    active_manifest_path: Path,
) -> None:
    contract, active, contract_hash, active_hash = load_contract_snapshot(
        PROJECT_ROOT, snapshot_path=active_manifest_path
    )
    assert contract["contract_version"] == "imu_benchmark_contract_v2"
    assert active["snapshot_version"] == "imu_25hz_snapshot_v2"
    assert contract["canonical_signal"]["sampling_rate_hz"] == 25
    assert contract["window"]["samples"] == 50
    assert contract["window"]["stride_seconds"] == 0.5
    assert len(contract_hash) == len(active_hash) == 64


def test_all_experiment_configs_resolve_against_active_data(active_manifest_path: Path) -> None:
    for path in sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml")):
        config = load_experiment(PROJECT_ROOT, path, snapshot_path=active_manifest_path)
        assert config["contract"]["canonical_signal"]["sampling_rate_hz"] == 25
        assert config["contract"]["window"]["samples"] == 50
        assert config["data_view"]["objective"] == "temporal_supervised"


def test_half_up_stride_grid_has_alternating_12_and_13_sample_steps() -> None:
    starts = _window_starts(150, 50, 25.0, 0.5)
    assert starts.tolist() == [0, 13, 25, 38, 50, 63, 75, 88, 100]
    assert set(np.diff(starts).tolist()) == {12, 13}


def test_reference_resampling_is_identity_on_a_25_hz_grid() -> None:
    timestamps = np.arange(6, dtype=np.float64) / 25.0
    values = np.arange(36, dtype=np.float64).reshape(6, 6)
    output_timestamps, output = linear_resample_to_grid(timestamps, values)
    np.testing.assert_allclose(output_timestamps, timestamps, atol=1e-12)
    np.testing.assert_allclose(output, values, atol=1e-7)
    with pytest.raises(ValueError, match="strictly increasing"):
        linear_resample_to_grid(np.asarray([0.0, 0.04, 0.04]), np.zeros((3, 6)))


def test_engineered_feature_schema_is_deterministic() -> None:
    window = (np.arange(300, dtype=np.float32).reshape(1, 50, 6) - 150.0) / 37.0
    first = extract_window_features(window)
    second = extract_window_features(window)
    assert first.shape == (1, 158)
    assert len(FEATURE_NAMES) == 158
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(
        second.tobytes()
    ).hexdigest()
    assert np.isfinite(first).all()


def test_decision_time_labels_are_causal_and_exclude_post_segment_overlap() -> None:
    starts = np.asarray([0, 13, 25, 38, 50, 63], dtype=np.int32)
    rows = (
        Annotation("activity", 0, 55, "ADL"),
        Annotation("activity", 55, 91, "F01"),
        Annotation("onset", 55, 55, "F01"),
        Annotation("impact", 80, 80, "F01"),
        Annotation("activity", 91, 140, "ADL"),
    )
    labels, keep, intervals = segment_decision_time_labels(starts, starts + 50, rows)
    assert labels.tolist() == [0, 1, 1, 1, 0, 0]
    assert keep.tolist() == [True, True, True, True, False, False]
    assert intervals == ((55, 91),)


def test_false_positive_window_hour_metric_uses_physical_stride() -> None:
    count, hours, rate = false_positive_windows_per_hour(
        np.asarray([0.1, 0.8, 0.7, 0.2]), threshold=0.5, stride_seconds=0.5
    )
    assert count == 2
    assert hours == pytest.approx(2.0 / 3600.0)
    assert rate == pytest.approx(3600.0)


def test_synthetic_hdf5_v31_fixture(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.h5"
    text = h5py.string_dtype(encoding="utf-8")
    sequence_dtype = np.dtype(
        [
            ("sample_start", "<i8"),
            ("sample_stop", "<i8"),
            ("source_file", text),
            ("participant_id", text),
            ("recording_id", text),
            ("body_location", text),
            ("activity_code", text),
            ("is_fall", "?"),
            ("supervision_kind", text),
            ("source_sampling_rate_hz", "<f8"),
        ]
    )
    annotation_dtype = np.dtype(
        [
            ("sequence_index", "<i4"),
            ("kind", text),
            ("start_sample", "<i8"),
            ("stop_sample", "<i8"),
            ("code", text),
        ]
    )
    sequences = np.asarray(
        [(0, 100, "source.csv", "p01", "r01", "waist", "F01", True, "temporal", 25.0)],
        dtype=sequence_dtype,
    )
    annotations = np.asarray(
        [
            (0, "activity", 0, 40, "ADL"),
            (0, "activity", 40, 81, "F01"),
            (0, "onset", 40, 40, "F01"),
            (0, "impact", 70, 70, "F01"),
            (0, "activity", 81, 100, "ADL"),
        ],
        dtype=annotation_dtype,
    )
    samples = np.zeros((100, 6), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "dataset_id": "synthetic",
                "imu_schema_version": "3.1.0",
                "sampling_rate_hz": 25.0,
                "axis_frame": "sensor_local",
                "hdf5_compatibility": "1.14",
                "feature_columns": json.dumps(FEATURE_COLUMNS),
                "evaluation_role": "cross_validation",
                "sequence_count": 1,
                "sample_count": 100,
                "annotation_count": len(annotations),
                "logical_content_sha256": "a" * 64,
            }
        )
        sample_data = handle.create_dataset("samples", data=samples)
        sample_data.attrs["columns"] = json.dumps(FEATURE_COLUMNS)
        sample_data.attrs["units"] = json.dumps(FEATURE_UNITS)
        handle.create_dataset("sequences", data=sequences)
        handle.create_dataset("annotations", data=annotations)
    result = validate_hdf5_file(path)
    assert result["sequences"] == 1
    assert result["events"] == 1
    assert result["segments"] == 3
    assert result["evaluation_role"] == "cross_validation"
