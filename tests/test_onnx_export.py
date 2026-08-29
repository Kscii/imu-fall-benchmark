from __future__ import annotations

from pathlib import Path

import numpy as np

from imu_benchmark.onnx_export import export_and_validate_onnx
from imu_benchmark.sequence_models import threshold_impact_scores


def test_threshold_onnx_matches_native_scores(tmp_path: Path) -> None:
    generator = np.random.default_rng(3888)
    windows = generator.normal(size=(8, 50, 6)).astype(np.float32)
    result = export_and_validate_onnx(
        None,
        "threshold_impact",
        windows,
        threshold_impact_scores(windows),
        tmp_path / "threshold.onnx",
    )
    assert result["status"] == "PASS"
    assert result["opset"] == 18
    assert result["maximum_absolute_error"] < 1e-5


def test_threshold_onnx_streams_complete_validation_and_test_splits(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(5171)
    validation = generator.normal(size=(513, 50, 6)).astype(np.float32)
    test = generator.normal(size=(259, 50, 6)).astype(np.float32)
    result = export_and_validate_onnx(
        None,
        "threshold_impact",
        validation,
        threshold_impact_scores(validation),
        tmp_path / "threshold-full.onnx",
        additional_splits={"test": (test, threshold_impact_scores(test))},
        batch_size=256,
        max_samples=None,
    )
    assert result["status"] == "PASS"
    assert result["samples"] == 772
    assert result["batches"] == 5
    assert result["parity_splits"]["validation"]["samples"] == 513
    assert result["parity_splits"]["validation"]["batches"] == 3
    assert result["parity_splits"]["test"]["samples"] == 259
    assert result["parity_splits"]["test"]["batches"] == 2
    assert result["relative_tolerance"] == 1e-4
    assert result["absolute_tolerance"] == 1e-4
    assert {1, 3, 7, 256}.issubset(result["validated_batch_sizes"])
    assert result["maximum_absolute_error"] < 1e-5
