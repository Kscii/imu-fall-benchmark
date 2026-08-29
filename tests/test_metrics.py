from pathlib import Path

import numpy as np

from imu_benchmark.metrics import (
    _alarm_positions,
    alarm_policy_metrics,
    binary_classification_metrics,
    pareto_alarm_policy_ids,
    temporal_event_metrics,
)
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


def test_alarm_k_of_n_and_cooldown_are_causal() -> None:
    positive = np.asarray([False, True, True, False, True, True, True])
    decisions = np.arange(len(positive), dtype=np.int64) * 13
    policy = {
        "required_positive_windows": 2,
        "lookback_windows": 3,
        "consecutive": False,
        "cooldown_seconds": 1.5,
    }
    assert _alarm_positions(positive, decisions, policy, 25.0).tolist() == [2, 5]


def test_alarm_pareto_selection_does_not_use_test_metrics() -> None:
    rows = [
        {
            "alarm_policy_id": "sensitive",
            "event_sensitivity": 1.0,
            "adl_alarm_episodes_per_hour": 3.0,
            "onset_latency_p95_s": 1.0,
        },
        {
            "alarm_policy_id": "balanced",
            "event_sensitivity": 0.9,
            "adl_alarm_episodes_per_hour": 1.0,
            "onset_latency_p95_s": 1.5,
        },
        {
            "alarm_policy_id": "dominated",
            "event_sensitivity": 0.8,
            "adl_alarm_episodes_per_hour": 2.0,
            "onset_latency_p95_s": 2.0,
        },
    ]
    assert pareto_alarm_policy_ids(rows) == ["balanced", "sensitive"]


def test_alarm_metrics_ignore_recording_only_fall_as_temporal_event() -> None:
    store = UnifiedWindowStore(
        path=Path("unused.h5"),
        sequence_index=np.asarray([0], dtype=np.int32),
        start_sample=np.asarray([0], dtype=np.int32),
        end_sample=np.asarray([50], dtype=np.int32),
        fold_id=np.asarray([0], dtype=np.int8),
        bag_label=np.asarray([0], dtype=np.int8),
        temporal_label=np.asarray([0], dtype=np.int8),
        dataset_id=np.asarray(["kfall", "recording_only"]),
        participant_id=np.asarray(["p1", "p2"]),
        recording_id=np.asarray(["adl", "fall_without_interval"]),
        body_location=np.asarray(["waist", "chest"]),
        supervision_kind=np.asarray(["temporal", "recording"]),
        sequence_is_fall=np.asarray([False, True]),
        sequence_fold_id=np.asarray([0, 0], dtype=np.int8),
        event_onset_sample=np.asarray([-1, -1], dtype=np.int64),
        event_impact_sample=np.asarray([-1, -1], dtype=np.int64),
        event_stop_sample=np.asarray([-1, -1], dtype=np.int64),
        manifest={"sampling_rate_hz": 25, "stride_seconds": 0.5, "windows": 1},
    )
    result = alarm_policy_metrics(
        store,
        np.asarray([0], dtype=np.int64),
        np.asarray([0.1], dtype=np.float64),
        0.5,
        {
            "id": "reference",
            "required_positive_windows": 1,
            "lookback_windows": 1,
            "consecutive": True,
            "cooldown_seconds": 10.0,
        },
        sequence_scope=np.asarray([0, 1], dtype=np.int64),
    )
    assert result["fall_events"] == 0
    assert result["adl_recordings"] == 1
