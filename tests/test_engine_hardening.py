from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from imu_benchmark import engine
from imu_benchmark.configuration import load_experiment
from imu_benchmark.engine import _job_hash, build_jobs, plan_experiment
from imu_benchmark.models import CuMLAdapter, ModelConvergenceError
from imu_benchmark.specs import MODEL_SPECS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_versioned_experiments_have_no_total_runtime_budget() -> None:
    for path in sorted((PROJECT_ROOT / "configs/experiments").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "runtime_budget_seconds" not in payload
        load_experiment(PROJECT_ROOT, path)


def test_logistic_regression_convergence_state_is_machine_readable() -> None:
    params = dict(MODEL_SPECS["cuml_logistic_regression"].fixed_params)
    config = load_experiment(
        PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/kfall_smoke_v1.yaml"
    )
    assert config["model_catalog"]["models"]["cuml_logistic_regression"]["params"] == params
    adapter = CuMLAdapter("cuml_logistic_regression", params, random_seed=3888)
    adapter.estimator = SimpleNamespace(n_iter_=np.asarray([267]))
    assert adapter.optimization_metadata() == {
        "converged": True,
        "iterations": 267,
        "iteration_limit": 500,
    }
    adapter.assert_optimization_converged()
    adapter.estimator = SimpleNamespace(n_iter_=np.asarray([500]))
    with pytest.raises(ModelConvergenceError, match="500/500"):
        adapter.assert_optimization_converged()


def test_checkpoint_schema_version_changes_job_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment(
        PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/kfall_smoke_v1.yaml"
    )
    plan = plan_experiment(config)
    assert plan["job_checkpoint_schema_version"] == engine.JOB_CHECKPOINT_SCHEMA_VERSION
    job = build_jobs(config)[0]
    original = _job_hash(config, "cache-fingerprint", job)
    monkeypatch.setattr(
        engine,
        "JOB_CHECKPOINT_SCHEMA_VERSION",
        engine.JOB_CHECKPOINT_SCHEMA_VERSION + 1,
    )
    assert _job_hash(config, "cache-fingerprint", job) != original
