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
