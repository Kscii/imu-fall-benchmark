from __future__ import annotations

import numpy as np

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
    "relative_power_8_12p5_hz",
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


def extract_window_features(windows: np.ndarray, sampling_rate_hz: float = 25.0) -> np.ndarray:
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (50, 6):
        raise ValueError(f"Expected windows with shape (n, 50, 6), got {raw.shape}")
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
    bands = (
        (0.0, 0.5),
        (0.5, 3.0),
        (3.0, 8.0),
        (8.0, sampling_rate_hz / 2.0 + 1e-9),
    )
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
