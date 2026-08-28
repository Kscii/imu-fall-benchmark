from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np


class TemporalAnnotation(Protocol):
    kind: str
    start_sample: int
    stop_sample: int
    code: str


def linear_resample_to_grid(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    target_rate_hz: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate six-axis values onto a regular grid without extrapolation."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    signals = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("At least two one-dimensional timestamps are required")
    if signals.shape != (len(timestamps), 6):
        raise ValueError("Values must have shape (n, 6)")
    if not np.isfinite(timestamps).all() or not np.isfinite(signals).all():
        raise ValueError("Timestamps and values must be finite")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    if not np.isfinite(target_rate_hz) or target_rate_hz <= 0:
        raise ValueError("Target rate must be finite and positive")
    relative = timestamps - timestamps[0]
    count = int(np.floor(relative[-1] * target_rate_hz + 1e-9)) + 1
    grid_relative = np.arange(count, dtype=np.float64) / target_rate_hz
    grid_relative = grid_relative[grid_relative <= relative[-1] + 1e-12]
    output = np.column_stack(
        tuple(np.interp(grid_relative, relative, signals[:, channel]) for channel in range(6))
    ).astype(np.float32)
    return timestamps[0] + grid_relative, output


def segment_decision_time_labels(
    starts: np.ndarray,
    ends: np.ndarray,
    annotations: Iterable[TemporalAnnotation],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Label causal windows from fall activity segments matched to onset points."""
    starts_array = np.asarray(starts, dtype=np.int32)
    ends_array = np.asarray(ends, dtype=np.int32)
    if starts_array.shape != ends_array.shape or np.any(ends_array <= starts_array):
        raise ValueError("Window starts and ends must be aligned non-empty intervals")
    rows = tuple(annotations)
    exclusions = tuple(
        (int(item.start_sample), int(item.stop_sample))
        for item in rows
        if item.kind == "exclude"
    )
    activities = tuple(item for item in rows if item.kind == "activity")
    onsets = tuple(item for item in rows if item.kind == "onset")
    intervals: list[tuple[int, int]] = []
    for onset in onsets:
        matches = tuple(
            activity
            for activity in activities
            if activity.start_sample == onset.start_sample
            and activity.stop_sample > onset.start_sample
            and activity.code == onset.code
        )
        if len(matches) != 1:
            raise ValueError("Each onset must match exactly one fall activity segment")
        intervals.append((int(onset.start_sample), int(matches[0].stop_sample)))
    decision = ends_array - 1
    labels = np.zeros(len(starts_array), dtype=np.int8)
    for start, stop in intervals:
        labels[(decision >= start) & (decision < stop)] = 1
    keep = np.ones(len(starts_array), dtype=np.bool_)
    for excluded_start, excluded_stop in exclusions:
        keep &= ~((starts_array < excluded_stop) & (ends_array > excluded_start))
    post_segment_overlap = np.zeros(len(starts_array), dtype=np.bool_)
    for _start, stop in intervals:
        post_segment_overlap |= (starts_array < stop) & (decision >= stop)
    keep &= ~(post_segment_overlap & (labels == 0))
    return labels, keep, tuple(intervals)
