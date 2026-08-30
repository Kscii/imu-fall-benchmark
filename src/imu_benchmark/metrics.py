from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .evaluation import false_positive_windows_per_hour
from .window_cache import UnifiedWindowStore


def binary_classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int | float]:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or values.shape != truth.shape or not np.isfinite(values).all():
        raise ValueError("Binary labels and scores must be aligned finite vectors")
    predictions = (values >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=(0, 1)).ravel()
    has_both = len(np.unique(truth)) == 2
    return {
        "n": len(truth),
        "positive": int(np.count_nonzero(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(truth, predictions)) if has_both else float("nan")
        ),
        "sensitivity": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predictions)) if has_both else float("nan"),
        "auroc": float(roc_auc_score(truth, values)) if has_both else float("nan"),
        "auprc": float(average_precision_score(truth, values)) if has_both else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def temporal_event_metrics(
    store: UnifiedWindowStore,
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    sequence_scope: np.ndarray | None = None,
    decision_interval_seconds: float | None = None,
) -> dict[str, int | float]:
    selected = np.asarray(indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(selected) != len(values):
        raise ValueError("Temporal event scores do not match selected windows")
    by_global = dict(zip(selected.tolist(), values.tolist(), strict=True))
    if sequence_scope is None:
        sequence_ids = np.unique(store.sequence_index[selected])
    else:
        sequence_ids = np.unique(np.asarray(sequence_scope, dtype=np.int64))
        if np.any((sequence_ids < 0) | (sequence_ids >= len(store.dataset_id))):
            raise ValueError("Temporal event sequence scope contains an invalid sequence ID")
    fall_events = detected_events = no_positive_window = 0
    adl_recordings = adl_false_recordings = 0
    adl_scores: list[np.ndarray] = []
    onset_latency: list[float] = []
    impact_offset: list[float] = []
    for sequence_id in sequence_ids:
        local_indices = selected[store.sequence_index[selected] == sequence_id]
        local_scores = np.asarray([by_global[int(index)] for index in local_indices])
        decisions = store.end_sample[local_indices].astype(np.int64) - 1
        event_indices = np.flatnonzero(store.event_sequence_index == sequence_id)
        for event_index in event_indices:
            fall_events += 1
            onset = int(store.event_onset_sample[event_index])
            stop = int(store.event_stop_sample[event_index])
            positive = (decisions >= onset) & (decisions < stop)
            if not np.any(positive):
                no_positive_window += 1
                continue
            alerted = np.flatnonzero(positive & (local_scores >= threshold))
            if len(alerted):
                detected_events += 1
                first_global = int(local_indices[alerted[0]])
                decision_sample = int(store.end_sample[first_global] - 1)
                onset_latency.append(
                    (decision_sample - onset) / float(store.manifest["sampling_rate_hz"])
                )
                impact_offset.append(
                    (decision_sample - int(store.event_impact_sample[event_index]))
                    / float(store.manifest["sampling_rate_hz"])
                )
        if not store.sequence_is_fall[sequence_id]:
            adl_recordings += 1
            valid = store.temporal_label[local_indices] == 0
            adl_false_recordings += int(np.any(local_scores[valid] >= threshold))
        negative = store.temporal_label[local_indices] == 0
        adl_scores.append(local_scores[negative])
    joined = np.concatenate(adl_scores) if adl_scores else np.empty(0, dtype=np.float64)
    false_windows, negative_hours, false_per_hour = false_positive_windows_per_hour(
        joined,
        threshold=threshold,
        stride_seconds=(
            float(store.manifest["stride_seconds"])
            if decision_interval_seconds is None
            else float(decision_interval_seconds)
        ),
    )
    latency = np.asarray(onset_latency, dtype=np.float64)
    impact = np.asarray(impact_offset, dtype=np.float64)
    return {
        "fall_events": fall_events,
        "detected_events": detected_events,
        "events_without_positive_decision_window": no_positive_window,
        "event_sensitivity": detected_events / fall_events if fall_events else 0.0,
        "adl_recordings": adl_recordings,
        "adl_false_positive_recordings": adl_false_recordings,
        "adl_recording_false_positive_rate": (
            adl_false_recordings / adl_recordings if adl_recordings else 0.0
        ),
        "adl_negative_window_hours": negative_hours,
        "adl_false_positive_windows": false_windows,
        "adl_false_positive_windows_per_hour": false_per_hour,
        "onset_latency_median_s": float(np.median(latency)) if len(latency) else float("nan"),
        "onset_latency_p95_s": (
            float(np.percentile(latency, 95)) if len(latency) else float("nan")
        ),
        "impact_offset_median_s": float(np.median(impact)) if len(impact) else float("nan"),
    }


def _alarm_positions(
    positive: np.ndarray,
    decision_samples: np.ndarray,
    policy: dict[str, Any],
    sampling_rate_hz: float,
) -> np.ndarray:
    required = int(policy["required_positive_windows"])
    lookback = int(policy["lookback_windows"])
    cooldown_samples = int(round(float(policy["cooldown_seconds"]) * sampling_rate_hz))
    result: list[int] = []
    last_alarm_sample: int | None = None
    for index in range(len(positive)):
        start = max(0, index - lookback + 1)
        recent = positive[start : index + 1]
        triggered = len(recent) == lookback and int(np.count_nonzero(recent)) >= required
        if policy["consecutive"]:
            triggered = triggered and bool(np.all(recent))
        if not triggered:
            continue
        decision = int(decision_samples[index])
        if last_alarm_sample is None or decision - last_alarm_sample >= cooldown_samples:
            result.append(index)
            last_alarm_sample = decision
    return np.asarray(result, dtype=np.int64)


def alarm_policy_metrics(
    store: UnifiedWindowStore,
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    policy: dict[str, Any],
    *,
    sequence_scope: np.ndarray | None = None,
    decision_interval_seconds: float | None = None,
) -> dict[str, int | float | str]:
    selected = np.asarray(indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(selected) != len(values):
        raise ValueError("Alarm policy scores do not match selected windows")
    by_global = dict(zip(selected.tolist(), values.tolist(), strict=True))
    sequence_ids = (
        np.unique(store.sequence_index[selected])
        if sequence_scope is None
        else np.unique(np.asarray(sequence_scope, dtype=np.int64))
    )
    rate = float(store.manifest["sampling_rate_hz"])
    stride = (
        float(store.manifest["stride_seconds"])
        if decision_interval_seconds is None
        else float(decision_interval_seconds)
    )
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError("Alarm decision interval must be finite and positive")
    fall_events = detected_events = no_positive_window = 0
    adl_recordings = adl_false_recordings = adl_alarm_episodes = 0
    adl_windows = 0
    onset_latency: list[float] = []
    impact_offset: list[float] = []
    for sequence_id in sequence_ids:
        event_indices = np.flatnonzero(store.event_sequence_index == sequence_id)
        local_indices = selected[store.sequence_index[selected] == sequence_id]
        if not len(local_indices):
            fall_events += len(event_indices)
            no_positive_window += len(event_indices)
            continue
        order = np.argsort(store.end_sample[local_indices], kind="stable")
        local_indices = local_indices[order]
        local_scores = np.asarray([by_global[int(index)] for index in local_indices])
        decision_samples = store.end_sample[local_indices].astype(np.int64) - 1
        alarms = _alarm_positions(
            local_scores >= threshold,
            decision_samples,
            policy,
            rate,
        )
        for event_index in event_indices:
            fall_events += 1
            onset = int(store.event_onset_sample[event_index])
            stop = int(store.event_stop_sample[event_index])
            positive = (decision_samples >= onset) & (decision_samples < stop)
            if not np.any(positive):
                no_positive_window += 1
            qualifying = alarms[
                (decision_samples[alarms] >= onset) & (decision_samples[alarms] < stop)
            ]
            if len(qualifying):
                detected_events += 1
                decision = int(decision_samples[qualifying[0]])
                onset_latency.append((decision - onset) / rate)
                impact_offset.append(
                    (decision - int(store.event_impact_sample[event_index])) / rate
                )
        event_membership = np.zeros(len(local_indices), dtype=np.bool_)
        for event_index in event_indices:
            event_membership |= (decision_samples >= int(store.event_onset_sample[event_index])) & (
                decision_samples < int(store.event_stop_sample[event_index])
            )
        negative_windows = store.temporal_label[local_indices] == 0
        negative_alarms = alarms[~event_membership[alarms]]
        adl_windows += int(np.count_nonzero(negative_windows))
        adl_alarm_episodes += len(negative_alarms)
        if not store.sequence_is_fall[sequence_id]:
            adl_recordings += 1
            adl_false_recordings += int(bool(len(negative_alarms)))
    negative_hours = adl_windows * stride / 3600.0
    latency = np.asarray(onset_latency, dtype=np.float64)
    impact = np.asarray(impact_offset, dtype=np.float64)
    return {
        "alarm_policy_id": str(policy["id"]),
        "fall_events": fall_events,
        "detected_events": detected_events,
        "events_without_positive_decision_window": no_positive_window,
        "event_sensitivity": detected_events / fall_events if fall_events else 0.0,
        "adl_recordings": adl_recordings,
        "adl_false_positive_recordings": adl_false_recordings,
        "adl_recording_false_positive_rate": (
            adl_false_recordings / adl_recordings if adl_recordings else 0.0
        ),
        "adl_negative_window_hours": negative_hours,
        "adl_alarm_episodes": adl_alarm_episodes,
        "adl_alarm_episodes_per_hour": (
            adl_alarm_episodes / negative_hours if negative_hours else 0.0
        ),
        "onset_latency_median_s": float(np.median(latency)) if len(latency) else float("nan"),
        "onset_latency_p95_s": (
            float(np.percentile(latency, 95)) if len(latency) else float("nan")
        ),
        "impact_offset_median_s": float(np.median(impact)) if len(impact) else float("nan"),
    }


def pareto_alarm_policy_ids(rows: list[dict[str, Any]]) -> list[str]:
    result = []
    for candidate in rows:
        candidate_latency = float(candidate["onset_latency_p95_s"])
        if not np.isfinite(candidate_latency):
            candidate_latency = float("inf")
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_latency = float(other["onset_latency_p95_s"])
            if not np.isfinite(other_latency):
                other_latency = float("inf")
            no_worse = (
                float(other["event_sensitivity"]) >= float(candidate["event_sensitivity"])
                and float(other["adl_alarm_episodes_per_hour"])
                <= float(candidate["adl_alarm_episodes_per_hour"])
                and other_latency <= candidate_latency
            )
            strictly_better = (
                float(other["event_sensitivity"]) > float(candidate["event_sensitivity"])
                or float(other["adl_alarm_episodes_per_hour"])
                < float(candidate["adl_alarm_episodes_per_hour"])
                or other_latency < candidate_latency
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["alarm_policy_id"]))
    return sorted(result)


def subgroup_metrics(
    store: UnifiedWindowStore,
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    selected = np.asarray(indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(selected) != len(values):
        raise ValueError("Subgroup scores do not match selected windows")
    rows: list[dict[str, Any]] = []
    for field, groups in (
        ("dataset_id", store.dataset_id[store.sequence_index[selected]]),
        ("body_location", store.body_location[store.sequence_index[selected]]),
    ):
        for value in sorted(set(groups.tolist())):
            mask = groups == value
            labels = store.temporal_label[selected[mask]]
            valid = labels >= 0
            if np.any(valid):
                rows.append(
                    {
                        "subgroup_field": field,
                        "subgroup_value": value,
                        **binary_classification_metrics(
                            labels[valid], values[mask][valid], threshold
                        ),
                    }
                )
    return rows
