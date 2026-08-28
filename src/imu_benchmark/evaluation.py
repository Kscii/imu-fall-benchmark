from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_curve


def false_positive_windows_per_hour(
    scores: np.ndarray,
    *,
    threshold: float,
    stride_seconds: float,
) -> tuple[int, float, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Window scores must be a finite one-dimensional array")
    if stride_seconds <= 0:
        raise ValueError("Stride duration must be positive")
    false_positives = int(np.count_nonzero(values >= threshold))
    negative_hours = len(values) * stride_seconds / 3600.0
    rate = false_positives / negative_hours if negative_hours else 0.0
    return false_positives, negative_hours, rate


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, scores, drop_intermediate=False
    )
    balanced = (true_positive_rate + (1.0 - false_positive_rate)) / 2.0
    best_value = float(np.max(balanced))
    tied = np.flatnonzero(np.isclose(balanced, best_value, rtol=0.0, atol=1e-12))
    maximum = float(np.max(scores))
    candidates = [
        float(thresholds[index])
        if np.isfinite(thresholds[index])
        else float(np.nextafter(maximum, np.inf))
        for index in tied
    ]
    threshold = min(candidates, key=lambda value: (abs(value - 0.5), value))
    predictions = (scores >= threshold).astype(np.int8)
    return threshold, best_value, float(matthews_corrcoef(labels, predictions))
