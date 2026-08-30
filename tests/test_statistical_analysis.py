from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from imu_benchmark.statistical_analysis import write_statistical_outputs
from imu_benchmark.window_cache import UnifiedWindowStore


def _store() -> UnifiedWindowStore:
    return UnifiedWindowStore(
        path=Path("unused.h5"),
        sequence_index=np.repeat(np.arange(4, dtype=np.int32), 2),
        start_sample=np.tile(np.asarray([0, 25], dtype=np.int32), 4),
        end_sample=np.tile(np.asarray([50, 75], dtype=np.int32), 4),
        fold_id=np.zeros(8, dtype=np.int8),
        bag_label=np.repeat(np.asarray([1, 0, 1, 0], dtype=np.int8), 2),
        temporal_label=np.repeat(np.asarray([1, 0, 1, 0], dtype=np.int8), 2),
        dataset_id=np.asarray(["temporal"] * 4),
        participant_id=np.asarray(["p1", "p1", "p2", "p2"]),
        recording_id=np.asarray(["p1_fall", "p1_adl", "p2_fall", "p2_adl"]),
        body_location=np.asarray(["waist"] * 4),
        supervision_kind=np.asarray(["temporal"] * 4),
        sequence_is_fall=np.asarray([True, False, True, False]),
        sequence_fold_id=np.zeros(4, dtype=np.int8),
        event_sequence_index=np.asarray([0, 2], dtype=np.int32),
        event_onset_sample=np.asarray([0, 0], dtype=np.int64),
        event_impact_sample=np.asarray([25, 25], dtype=np.int64),
        event_stop_sample=np.asarray([100, 100], dtype=np.int64),
        event_code=np.asarray(["fall", "fall"]),
        manifest={
            "sampling_rate_hz": 25,
            "stride_seconds": 0.5,
            "windows": 8,
            "fall_events": 2,
        },
    )


def _result(
    run_dir: Path,
    model_id: str,
    recipe: str,
    scores: np.ndarray,
) -> dict[str, object]:
    key = f"{model_id}-{recipe}"
    path = run_dir / f"{key}.npz"
    np.savez_compressed(
        path,
        window_index=np.arange(8, dtype=np.int64),
        label=np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8),
        fall_score=np.asarray(scores, dtype=np.float32),
    )
    return {
        "path": str(path),
        "metadata": {
            "job_key": key,
            "job": {
                "model_id": model_id,
                "training_recipe": recipe,
                "fold": 0,
                "seed": 3888,
            },
            "selected_threshold": 0.5,
        },
    }


def test_hierarchical_outputs_include_oof_aggregates_and_paired_rows(
    tmp_path: Path,
) -> None:
    scores = np.asarray([0.9, 0.8, 0.1, 0.2, 0.9, 0.8, 0.1, 0.2])
    results = [
        _result(tmp_path, "threshold_impact", "not_applicable", scores),
        _result(tmp_path, "cuml_random_forest", "natural", scores),
        _result(
            tmp_path,
            "cuml_random_forest",
            "participant_class_balanced",
            scores,
        ),
    ]
    config = {
        "bootstrap_replicates": 20,
        "alarm_policy": {
            "reference_policy": "one_of_one_cooldown_10s",
            "policies": [
                {
                    "id": "one_of_one_cooldown_10s",
                    "required_positive_windows": 1,
                    "lookback_windows": 1,
                    "consecutive": True,
                    "cooldown_seconds": 10.0,
                }
            ],
        },
    }
    manifest = write_statistical_outputs(tmp_path, _store(), config, results)
    assert manifest is not None
    assert manifest["status"] == "PASS"
    assert manifest["complete_groups"] == 3
    assert (tmp_path / "oof_manifest.json").is_file()
    with (tmp_path / "aggregate_metrics.csv").open(newline="") as source:
        assert len(list(csv.DictReader(source))) == 15
    with (tmp_path / "paired_comparisons.csv").open(newline="") as source:
        comparisons = {row["comparison"] for row in csv.DictReader(source)}
    assert comparisons == {"model_vs_threshold", "balanced_minus_natural"}
