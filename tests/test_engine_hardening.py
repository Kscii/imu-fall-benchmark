from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from imu_benchmark import engine
from imu_benchmark.configuration import load_experiment
from imu_benchmark.engine import _job_hash, _result_rows, build_jobs, plan_experiment
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
    original = _job_hash(config, "cache-fingerprint", job)
    monkeypatch.setattr(
        engine,
        "JOB_CHECKPOINT_SCHEMA_VERSION",
        engine.JOB_CHECKPOINT_SCHEMA_VERSION + 1,
    )
    assert _job_hash(config, "cache-fingerprint", job) != original


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
