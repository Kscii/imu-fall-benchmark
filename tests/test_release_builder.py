import json
from pathlib import Path

import numpy as np

from imu_benchmark.artifact_contract import validate_model_marker_v1
from imu_benchmark.release_builder import (
    SEQUENCE_MODELS,
    _choose_policy,
    _release_metadata,
    _source_artifacts,
    load_release_build_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_build_config_freezes_three_natural_sequence_candidates() -> None:
    config = load_release_build_config(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/releases/public_temporal_android_rc1.yaml",
    )

    assert tuple(item["model_id"] for item in config["models"]) == SEQUENCE_MODELS
    assert config["training_recipe"] == "natural"
    assert config["deployment_inference_interval_seconds"] == 1.0


def test_policy_selection_applies_sensitivity_guard_then_alarm_rate_latency_id() -> None:
    rows = [
        {
            "alarm_policy_id": "sensitive",
            "event_sensitivity": 0.95,
            "adl_alarm_episodes_per_hour": 4.0,
            "onset_latency_p95_s": 1.0,
        },
        {
            "alarm_policy_id": "b-policy",
            "event_sensitivity": 0.94,
            "adl_alarm_episodes_per_hour": 1.0,
            "onset_latency_p95_s": 2.0,
        },
        {
            "alarm_policy_id": "a-policy",
            "event_sensitivity": 0.94,
            "adl_alarm_episodes_per_hour": 1.0,
            "onset_latency_p95_s": 2.0,
        },
        {
            "alarm_policy_id": "too-insensitive",
            "event_sensitivity": 0.939,
            "adl_alarm_episodes_per_hour": 0.0,
            "onset_latency_p95_s": 0.0,
        },
    ]

    result = _choose_policy(rows, tolerance_percentage_points=1.0)

    assert result["eligible_policy_ids"] == ["a-policy", "b-policy", "sensitive"]
    assert result["selected"]["alarm_policy_id"] == "a-policy"


def test_validation_selection_source_does_not_read_job_test_score_shards(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "source-run"
    for model_id in SEQUENCE_MODELS:
        for fold in range(5):
            directory = run_dir / "models" / f"{model_id}-{fold}"
            directory.mkdir(parents=True)
            (directory / "model_spec.json").write_text(
                json.dumps(
                    {
                        "job": {
                            "model_id": model_id,
                            "training_recipe": "natural",
                            "fold": fold,
                            "seed": 3888,
                        }
                    }
                ),
                encoding="utf-8",
            )
            for name in ("model.onnx", "normalization.npz", "training_history.json"):
                (directory / name).write_bytes(b"selection-source")
    jobs = run_dir / "jobs"
    jobs.mkdir()
    test_scores = jobs / "test-score-shard.npz"
    test_scores.write_bytes(b"before")
    before = _source_artifacts(run_dir)

    test_scores.write_bytes(b"mutated-test-scores")
    after = _source_artifacts(run_dir)

    assert before == after


def test_release_metadata_embeds_selection_proof_and_final_refit_scope(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture-model")
    snapshot = "b" * 64
    split = "c" * 64
    source_commit = "a" * 40
    values = np.zeros((50, 6), dtype=np.float32).tolist()
    metadata = _release_metadata(
        release={
            "release_id": "public-temporal-1d-cnn-natural-rc1",
            "model_id": "torch_1d_cnn",
            "name": "Public Temporal 1D CNN Natural RC1",
        },
        build_config={"seed": 3888, "known_limitations": ["research only"]},
        source_config={
            "snapshot": {"base_snapshot_id": "imu_25hz_snapshot_v2"},
            "snapshot_sha256": snapshot,
            "data_view": {"id": "all_temporal_v1"},
            "alarm_policy": {
                "policies": [
                    {
                        "id": "reference",
                        "required_positive_windows": 1,
                        "lookback_windows": 1,
                        "consecutive": True,
                        "cooldown_seconds": 10.0,
                    }
                ]
            },
        },
        source_run_manifest={
            "run_id": "formal-source-run",
            "source": {"commit": source_commit},
        },
        source={"kind": "git", "commit": source_commit, "dirty": False},
        run_id="final-refit-run",
        selection={
            "participant_proof": {
                "status": "PASS",
                "participant_count": 5,
                "appearances_per_participant": 1,
                "validation_fold_participant_counts": [1, 1, 1, 1, 1],
                "assignment_sha256": "d" * 64,
            },
            "threshold_selection_method": "maximum_validation_balanced_accuracy",
            "score_threshold": 0.5,
            "alarm_policy_selection": {
                "selected_policy_id": "reference",
                "selected": {"event_sensitivity": 0.9},
            },
            "window_metrics": {"balanced_accuracy": 0.8},
        },
        data_split_fingerprint=split,
        epochs_by_fold=[2, 3, 4, 3, 2],
        final_epochs=3,
        training_indices=np.arange(4),
        training_labels=np.asarray([0, 1, 0, 1], dtype=np.int8),
        mean=np.zeros(6, dtype=np.float32),
        scale=np.ones(6, dtype=np.float32),
        parity={
            "status": "PASS",
            "scope": "all_final_training_windows",
            "samples": 4,
            "golden_fixtures": [
                {
                    "fixture_id": fixture_id,
                    "input_values": values,
                    "expected_fall_score": 0.5,
                    "rtol": 1e-5,
                    "atol": 1e-5,
                }
                for fixture_id in ("stationary", "adl-like", "impact-like")
            ],
        },
        model_path=model_path,
    )

    validate_model_marker_v1(metadata)
    assert metadata["source"]["selection_evidence"]["selection_eligible"] is True
    assert metadata["metrics"]["final_model_independently_evaluated"] is False
    assert metadata["source"]["final_training"]["actual_epochs"] == 3
