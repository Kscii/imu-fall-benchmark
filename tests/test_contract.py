from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from imu_benchmark.contract import load_contract_bundle, load_contract_snapshot
from imu_benchmark.data import FEATURE_NAMES, extract_window_features
from imu_benchmark.dataset import FEATURE_COLUMNS, FEATURE_UNITS, Annotation, validate_hdf5_file
from imu_benchmark.evaluation import false_positive_windows_per_hour
from imu_benchmark.folds import extend_fold_assignments, plan_fold_assignments
from imu_benchmark.protocol import linear_resample_to_grid, segment_decision_time_labels
from imu_benchmark.sequence_models import aggregate_bag_scores

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (PROJECT_ROOT / "tests/fixtures/temporal_contract_v1.json").read_text(encoding="utf-8")
)


def _annotations() -> tuple[Annotation, ...]:
    return tuple(Annotation(**item) for item in FIXTURE["temporal"]["annotations"])


def test_contract_and_snapshot_are_single_protocol_source() -> None:
    contract, snapshot, contract_hash, snapshot_hash = load_contract_snapshot(PROJECT_ROOT)
    assert contract["contract_version"] == "imu_benchmark_contract_v1"
    assert snapshot["snapshot_version"] == "imu_30hz_snapshot_v1"
    assert contract["canonical_signal"]["sampling_rate_hz"] == 30
    assert contract["window"]["samples"] == 60
    assert contract["window"]["stride_samples"] == 15
    assert len(contract_hash) == 64
    assert len(snapshot_hash) == 64


def test_experiment_cannot_override_protocol_fields(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bad = configs / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "contract_path": "configs/contracts/imu_benchmark_contract_v1.json",
                "snapshot_path": "data/snapshot_v1.json",
                "sampling_rate_hz": 25,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must come from the contract"):
        load_contract_bundle(bad)


def test_experiment_cannot_override_contract_digest(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bad = configs / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "contract_path": "configs/contracts/imu_benchmark_contract_v1.json",
                "snapshot_path": "data/snapshot_v1.json",
                "contract_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must come from the contract"):
        load_contract_bundle(bad)


def test_reference_25_to_30_hz_resampling() -> None:
    case = FIXTURE["resampling"]
    timestamps, values = linear_resample_to_grid(
        np.asarray(case["timestamps_s"]), np.asarray(case["values"]), target_rate_hz=30.0
    )
    np.testing.assert_allclose(timestamps, case["expected_timestamps_s"], atol=1e-12)
    np.testing.assert_allclose(values[:, 0], case["expected_first_channel"], atol=1e-7)
    with pytest.raises(ValueError, match="strictly increasing"):
        linear_resample_to_grid(
            np.asarray([0.0, 0.1, 0.1]), np.zeros((3, 6)), target_rate_hz=30.0
        )


def test_engineered_feature_schema_matches_golden_digest() -> None:
    case = FIXTURE["engineered_features"]
    window = (np.arange(360, dtype=np.float32).reshape(1, 60, 6) - 180.0) / 37.0
    features = extract_window_features(window)
    assert list(features.shape) == case["shape"]
    values_digest = hashlib.sha256(
        np.asarray(features, dtype="<f4").tobytes()
    ).hexdigest()
    names_digest = hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest()
    assert values_digest == case["values_sha256"]
    assert names_digest == case["names_sha256"]


def test_top_ten_percent_mil_pooling_matches_golden_boundary() -> None:
    scores = np.concatenate((np.arange(10), np.arange(20))).astype(np.float64)
    sequence_index = np.concatenate(
        (np.full(10, 3, dtype=np.int32), np.full(20, 7, dtype=np.int32))
    )
    sequence_ids, pooled = aggregate_bag_scores(scores, sequence_index, 0.1)
    assert sequence_ids.tolist() == [3, 7]
    np.testing.assert_allclose(pooled, [9.0, 18.5])


def test_segment_decision_labels_match_golden_boundaries() -> None:
    case = FIXTURE["temporal"]
    starts = np.asarray(case["starts"], dtype=np.int32)
    labels, keep, intervals = segment_decision_time_labels(
        starts, starts + int(case["window_samples"]), _annotations()
    )
    assert labels.tolist() == case["expected_labels"]
    assert keep.tolist() == case["expected_keep"]
    assert intervals == (tuple(case["expected_fall_interval"]),)


def test_exclusion_overlap_and_invalid_segment_are_rejected() -> None:
    starts = np.asarray([0, 15, 30], dtype=np.int32)
    rows = (*_annotations(), Annotation("exclude", 10, 20, "bad_signal"))
    _labels, keep, _intervals = segment_decision_time_labels(starts, starts + 60, rows)
    assert keep.tolist() == [False, False, True]
    mismatched = tuple(
        Annotation(item.kind, item.start_sample, item.stop_sample, "wrong")
        if item.kind == "onset"
        else item
        for item in _annotations()
    )
    with pytest.raises(ValueError, match="exactly one"):
        segment_decision_time_labels(starts, starts + 60, mismatched)


def test_false_positive_window_hour_metric_is_explicit() -> None:
    count, hours, rate = false_positive_windows_per_hour(
        np.asarray([0.1, 0.8, 0.7, 0.2]),
        threshold=0.5,
        stride_samples=15,
        sampling_rate_hz=30.0,
    )
    assert count == 2
    assert hours == pytest.approx(2.0 / 3600.0)
    assert rate == pytest.approx(3600.0)


def test_sticky_fold_extension_is_deterministic_and_balanced() -> None:
    existing = {("dataset", "p0"): 0, ("dataset", "p1"): 1}
    participants = set(existing) | {("dataset", f"p{index}") for index in range(2, 12)}
    first, added_first = extend_fold_assignments(participants, existing)
    second, added_second = extend_fold_assignments(participants, existing)
    assert first == second
    assert added_first == added_second
    assert first[("dataset", "p0")] == 0
    assert first[("dataset", "p1")] == 1
    counts = [sum(value == fold for value in first.values()) for fold in range(5)]
    assert max(counts) - min(counts) <= 1


def test_current_fold_plan_is_a_read_only_noop() -> None:
    training = plan_fold_assignments(PROJECT_ROOT, "training")
    external = plan_fold_assignments(PROJECT_ROOT, "external")
    assert training["new_participants"] == 0
    assert training["candidate_version"] == "participant_5fold_v2"
    assert external["new_participants"] == 0
    assert external["candidate_version"] == "kfall_external_5fold_v1"


def test_synthetic_hdf5_v3_fixture(tmp_path: Path) -> None:
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
    sequence = np.asarray(
        [
            (
                0,
                180,
                "synthetic.csv",
                "synthetic:p01",
                "synthetic:r01",
                "waist",
                "F01",
                True,
                "temporal",
                25.0,
            )
        ],
        dtype=sequence_dtype,
    )
    annotations = np.asarray(
        [(0, item.kind, item.start_sample, item.stop_sample, item.code) for item in _annotations()],
        dtype=annotation_dtype,
    )
    samples = np.arange(180 * 6, dtype=np.float32).reshape(180, 6) / 100.0
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "dataset_id": "synthetic",
                "imu_schema_version": "3.0.0",
                "sampling_rate_hz": 30.0,
                "axis_frame": "sensor_local",
                "hdf5_compatibility": "1.14",
                "feature_columns": json.dumps(FEATURE_COLUMNS),
                "sequence_count": 1,
                "sample_count": 180,
                "annotation_count": len(annotations),
                "logical_content_sha256": "a" * 64,
            }
        )
        sample_data = handle.create_dataset("samples", data=samples)
        sample_data.attrs["columns"] = json.dumps(FEATURE_COLUMNS)
        sample_data.attrs["units"] = json.dumps(FEATURE_UNITS)
        handle.create_dataset("sequences", data=sequence)
        handle.create_dataset("annotations", data=annotations)
    result = validate_hdf5_file(path)
    assert result["sequences"] == 1
    assert result["events"] == 1
    assert result["segments"] == 3
    assert result["participants"] == 1
