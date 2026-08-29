from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from imu_benchmark import engine
from imu_benchmark.configuration import load_experiment
from imu_benchmark.engine import (
    _job_hash,
    _result_rows,
    build_jobs,
    plan_experiment,
    training_recipe_indices,
)
from imu_benchmark.models import CuMLAdapter, ModelConvergenceError
from imu_benchmark.specs import MODEL_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(active_manifest_path: Path) -> dict:
    return load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/temporal_smoke_v1.yaml",
        snapshot_path=active_manifest_path,
    )


def test_versioned_experiments_have_no_total_runtime_budget(active_manifest_path: Path) -> None:
    for path in sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "runtime_budget_seconds" not in payload
        load_experiment(PROJECT_ROOT, path, snapshot_path=active_manifest_path)


def test_logistic_regression_convergence_state_is_machine_readable(
    active_manifest_path: Path,
) -> None:
    params = dict(MODEL_SPECS["cuml_logistic_regression"].fixed_params)
    config = _config(active_manifest_path)
    assert config["model_catalog"]["models"]["cuml_logistic_regression"]["params"] == params
    adapter = CuMLAdapter("cuml_logistic_regression", params, random_seed=3888)
    adapter.estimator = SimpleNamespace(n_iter_=np.asarray([267]))
    assert adapter.optimization_metadata()["converged"]
    adapter.estimator = SimpleNamespace(n_iter_=np.asarray([500]))
    with pytest.raises(ModelConvergenceError, match="500/500"):
        adapter.assert_optimization_converged()


def test_checkpoint_schema_version_changes_job_hash(
    monkeypatch: pytest.MonkeyPatch, active_manifest_path: Path
) -> None:
    config = _config(active_manifest_path)
    assert plan_experiment(config)["job_checkpoint_schema_version"] == (
        engine.JOB_CHECKPOINT_SCHEMA_VERSION
    )
    job = build_jobs(config)[0]
    original = _job_hash(config, "cache-fingerprint", "source-a", job)
    monkeypatch.setattr(
        engine,
        "JOB_CHECKPOINT_SCHEMA_VERSION",
        engine.JOB_CHECKPOINT_SCHEMA_VERSION + 1,
    )
    assert _job_hash(config, "cache-fingerprint", "source-a", job) != original


def test_source_fingerprint_changes_job_hash(active_manifest_path: Path) -> None:
    config = _config(active_manifest_path)
    job = build_jobs(config)[0]
    first = _job_hash(config, "cache-fingerprint", "source-a", job)
    second = _job_hash(config, "cache-fingerprint", "source-b", job)
    assert first != second


def test_formal_run_rejects_unknown_source_before_data_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clean Git commit or immutable source snapshot"):
        engine.run_experiment(
            project_root=tmp_path,
            cache_root=tmp_path / "cache",
            runs_root=tmp_path / "runs",
            config={"data_quality_status": "internal_formal_baseline"},
            resume=True,
            environment={},
            source={"kind": "unknown", "dirty": None},
            warnings=["source_unknown"],
        )


def test_result_rows_distinguish_compute_and_classification_precision() -> None:
    results = [
        {
            "metadata": {
                "job": {
                    "data_view_id": "all_temporal_v1",
                    "objective": "temporal_supervised",
                    "model_id": "torch_1d_cnn",
                    "fold": 0,
                    "seed": 3888,
                    "precision": "fp32",
                },
                "selected_threshold": 0.4,
                "test_metrics": {"precision": 0.75, "balanced_accuracy": 0.8},
                "event_metrics": {"event_sensitivity": 0.9},
                "subgroup_metrics": [],
                "external_metrics": None,
            }
        }
    ]
    metric_rows, event_rows, subgroup_rows, external_rows = _result_rows(results)
    assert metric_rows[0]["compute_precision"] == "fp32"
    assert metric_rows[0]["precision"] == 0.75
    assert event_rows[0]["compute_precision"] == "fp32"
    assert subgroup_rows == []
    assert external_rows == []


def test_participant_class_balanced_recipe_is_deterministic() -> None:
    store = SimpleNamespace(
        temporal_label=np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int8),
        sequence_index=np.arange(6, dtype=np.int32),
        dataset_id=np.asarray(["d"] * 6),
        participant_id=np.asarray(["a", "a", "b", "b", "c", "d"]),
    )
    indices = np.arange(6, dtype=np.int64)
    first, metadata = training_recipe_indices(
        store, indices, "participant_class_balanced", 3888
    )
    second, repeated_metadata = training_recipe_indices(
        store, indices, "participant_class_balanced", 3888
    )
    np.testing.assert_array_equal(first, second)
    assert metadata == repeated_metadata
    labels = store.temporal_label[first]
    assert np.count_nonzero(labels == 0) == np.count_nonzero(labels == 1) == 3


def test_formal_job_matrices_match_the_reviewed_scope(active_manifest_path: Path) -> None:
    main = load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/formal_baseline_main_v1.yaml",
        snapshot_path=active_manifest_path,
    )
    temporal_core = load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/formal_baseline_temporal_core_v1.yaml",
        snapshot_path=active_manifest_path,
    )
    assert len(build_jobs(main)) == 265
    assert len(build_jobs(temporal_core)) == 65
    assert len([job for job in build_jobs(main) if job.model_id == "threshold_impact"]) == 5
