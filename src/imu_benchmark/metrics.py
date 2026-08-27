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
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "sensitivity": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predictions)),
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
) -> dict[str, int | float]:
    selected = np.asarray(indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(selected) != len(values):
        raise ValueError("Temporal event scores do not match selected windows")
    by_global = dict(zip(selected.tolist(), values.tolist(), strict=True))
    sequence_ids = np.unique(store.sequence_index[selected])
    fall_events = detected_events = no_positive_window = 0
    adl_recordings = adl_false_recordings = 0
    adl_scores: list[np.ndarray] = []
    onset_latency: list[float] = []
    impact_offset: list[float] = []
    for sequence_id in sequence_ids:
        local_indices = selected[store.sequence_index[selected] == sequence_id]
        local_scores = np.asarray([by_global[int(index)] for index in local_indices])
        if store.sequence_is_fall[sequence_id]:
            fall_events += 1
            positive = store.temporal_label[local_indices] == 1
            if not np.any(positive):
                no_positive_window += 1
                continue
            alerted = np.flatnonzero(positive & (local_scores >= threshold))
            if len(alerted):
                detected_events += 1
                first_global = int(local_indices[alerted[0]])
                decision_sample = int(store.end_sample[first_global] - 1)
                onset_latency.append(
                    (decision_sample - int(store.event_onset_sample[sequence_id]))
                    / float(store.manifest["sampling_rate_hz"])
                )
                impact_offset.append(
                    (decision_sample - int(store.event_impact_sample[sequence_id]))
                    / float(store.manifest["sampling_rate_hz"])
                )
        else:
            adl_recordings += 1
            adl_scores.append(local_scores)
            adl_false_recordings += int(np.any(local_scores >= threshold))
    joined = np.concatenate(adl_scores) if adl_scores else np.empty(0, dtype=np.float64)
    false_windows, negative_hours, false_per_hour = false_positive_windows_per_hour(
        joined,
        threshold=threshold,
        stride_samples=int(store.manifest["stride_samples"]),
        sampling_rate_hz=float(store.manifest["sampling_rate_hz"]),
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
