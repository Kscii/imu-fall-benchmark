from pathlib import Path

import numpy as np

from imu_benchmark.metrics import binary_classification_metrics, temporal_event_metrics
from imu_benchmark.window_cache import UnifiedWindowStore


def test_zero_window_fall_sequence_remains_an_event_miss() -> None:
    store = UnifiedWindowStore(
        path=Path("unused.h5"),
        sequence_index=np.asarray([0], dtype=np.int32),
        start_sample=np.asarray([0], dtype=np.int32),
        end_sample=np.asarray([50], dtype=np.int32),
        fold_id=np.asarray([0], dtype=np.int8),
        bag_label=np.asarray([1], dtype=np.int8),
        temporal_label=np.asarray([1], dtype=np.int8),
        dataset_id=np.asarray(["kfall", "kfall"]),
        participant_id=np.asarray(["p1", "p2"]),
        recording_id=np.asarray(["fall_with_window", "fall_without_window"]),
        body_location=np.asarray(["waist", "waist"]),
        supervision_kind=np.asarray(["temporal", "temporal"]),
        sequence_is_fall=np.asarray([True, True]),
        sequence_fold_id=np.asarray([0, 0], dtype=np.int8),
        event_onset_sample=np.asarray([0, 0], dtype=np.int64),
        event_impact_sample=np.asarray([25, 25], dtype=np.int64),
        event_stop_sample=np.asarray([50, 50], dtype=np.int64),
        manifest={"sampling_rate_hz": 25, "stride_seconds": 0.5, "windows": 1},
    )
    metrics = temporal_event_metrics(
        store,
        np.asarray([0], dtype=np.int64),
        np.asarray([0.9], dtype=np.float64),
        0.5,
        sequence_scope=np.asarray([0, 1], dtype=np.int64),
    )
    assert metrics["fall_events"] == 2
    assert metrics["detected_events"] == 1
    assert metrics["events_without_positive_decision_window"] == 1
    assert metrics["event_sensitivity"] == 0.5


def test_single_class_subgroup_does_not_claim_balanced_metrics() -> None:
    metrics = binary_classification_metrics(
        np.zeros(4, dtype=np.int8),
        np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        0.5,
    )
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["mcc"])
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["auprc"])
    assert metrics["tn"] == 2
    assert metrics["fp"] == 2
