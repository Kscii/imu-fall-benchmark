from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imu_benchmark import runtime
from imu_benchmark.data import (
    extract_window_features,
    load_config,
    load_window_store,
    prepare_window_store,
)
from imu_benchmark.dataset import Annotation, validate_data
from imu_benchmark.device import CudaUnavailable, _parse_nvidia_smi_line
from imu_benchmark.evaluation import best_threshold
from imu_benchmark.kfall_data import (
    load_kfall_config,
    load_kfall_window_store,
    prepare_kfall_window_store,
)
from imu_benchmark.kfall_runner import _training_splits, plan_kfall_experiment
from imu_benchmark.performance import (
    PERFORMANCE_SCHEMA_VERSION,
    PhaseTimer,
    aggregate_job_performance,
    process_delta,
    process_snapshot,
)
from imu_benchmark.protocol import segment_decision_time_labels
from imu_benchmark.runner import plan_experiment
from imu_benchmark.runtime import (
    require_compute_runtime,
    resolve_work_paths,
    source_provenance,
)
from imu_benchmark.specs import MODEL_IDS, SUITES, build_jobs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = resolve_work_paths().cache


def test_distributed_data_and_split_validate() -> None:
    result = validate_data(PROJECT_ROOT)
    assert result["status"] == "PASS"
    assert result["training"]["files"] == 5
    assert result["training"]["sequences"] == 12_318
    assert result["training"]["rows"] == 6_840_218
    assert result["training"]["split"]["participants"] == 106
    assert result["external"]["files"] == 1
    assert result["external"]["events"] == 2_346
    assert result["external"]["split"]["participants"] == 32


def test_reproduction_plan_contains_110_jobs() -> None:
    config = load_config(PROJECT_ROOT / "configs/data_contract_v1_validation.json")
    result = plan_experiment(
        config=config,
        profile_name="reproduce",
        suites=SUITES,
        models=MODEL_IDS,
    )
    assert result["scheduled_jobs"] == 110
    assert result["jobs_by_suite"] == {
        "position_paired": 50,
        "fall_universal": 20,
        "fall_chest_only": 20,
        "fall_waist_only": 20,
    }


def test_engineered_window_features_are_finite() -> None:
    values = np.random.default_rng(3888).normal(size=(4, 60, 6)).astype(np.float32)
    features = extract_window_features(values)
    assert features.shape == (4, 158)
    assert np.isfinite(features).all()


def test_threshold_tie_break_is_deterministic() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    first = best_threshold(labels, scores)
    second = best_threshold(labels, scores)
    assert first == second
    assert first[0] == 0.6
    assert first[1] == 1.0


def test_segment_decision_time_boundaries() -> None:
    starts = np.asarray([0, 15, 30, 45, 60], dtype=np.int32)
    ends = starts + 60
    labels, keep, intervals = segment_decision_time_labels(
        starts, ends, (
            Annotation("activity", 70, 101, "F01"),
            Annotation("onset", 70, 70, "F01"),
            Annotation("impact", 100, 100, "F01"),
        )
    )
    assert labels.tolist() == [0, 1, 1, 0, 0]
    assert keep.tolist() == [True, True, True, False, False]
    assert intervals == ((70, 101),)


def test_kfall_workload_and_cache_contract() -> None:
    config = load_kfall_config(PROJECT_ROOT / "configs/kfall_segment_v1_validation.json")
    smoke = plan_kfall_experiment(config, "smoke")
    evaluate = plan_kfall_experiment(config, "evaluate")
    assert smoke["scheduled_jobs"] == 8
    assert evaluate["scheduled_jobs"] == 40
    path, manifest = prepare_kfall_window_store(
        project_root=PROJECT_ROOT,
        cache_root=CACHE_ROOT,
        config=config,
    )
    store = load_kfall_window_store(path)
    assert manifest["windows"] == store.size
    assert manifest["windows"] == 53_365
    assert manifest["positive_windows"] == 8_027
    assert manifest["negative_windows"] == 45_338
    assert manifest["fall_events"] == 2_346
    assert manifest["events_without_positive_window"] == 4
    assert manifest["skipped_post_segment_overlap_windows"] == 9_120


def test_kfall_training_fold_is_participant_disjoint() -> None:
    config = load_kfall_config(PROJECT_ROOT / "configs/kfall_segment_v1_validation.json")
    path, manifest = prepare_window_store(
        project_root=PROJECT_ROOT,
        cache_root=CACHE_ROOT,
        config=config,
    )
    assert manifest["sequences"] == 12_317
    assert manifest["windows"] == 415_363
    assert manifest["features"] == 158
    store = load_window_store(path)
    job = build_jobs(
        suites=("fall_universal",), models=("threshold_impact",), folds=(0,)
    )[0]
    splits = _training_splits(
        store,
        job,
        profile=config["profiles"]["smoke"],
        seed=int(config["random_seed"]),
    )
    train = set(store.window_participants()[splits["train"]])
    validation = set(store.window_participants()[splits["validation"]])
    assert train.isdisjoint(validation)


def test_work_paths_use_visible_default_and_absolute_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("IMU_BENCH_WORK_ROOT", raising=False)
    default = resolve_work_paths()
    assert default.root == Path.home() / "imu-fall-work"
    monkeypatch.setenv("IMU_BENCH_WORK_ROOT", str(tmp_path))
    overridden = resolve_work_paths()
    assert overridden.root == tmp_path
    assert overridden.cache == tmp_path / "cache"
    assert overridden.runs == tmp_path / "runs"
    monkeypatch.setenv("IMU_BENCH_WORK_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        resolve_work_paths()


def test_only_compute_commands_require_wsl2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "is_wsl2", lambda: False)
    require_compute_runtime("validate-data", PROJECT_ROOT)
    require_compute_runtime("plan", PROJECT_ROOT)
    require_compute_runtime("report", PROJECT_ROOT)
    with pytest.raises(CudaUnavailable, match="requires WSL2"):
        require_compute_runtime("doctor", PROJECT_ROOT)
    with pytest.raises(CudaUnavailable, match="requires WSL2"):
        require_compute_runtime("smoke", PROJECT_ROOT)


def test_snapshot_source_provenance_and_dirty_warning(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "snapshot",
        "commit": "abc123",
        "dirty": True,
        "snapshot_sha256": "a" * 64,
    }
    (tmp_path / ".imu-source.json").write_text(json.dumps(payload), encoding="utf-8")
    source, warnings = source_provenance(tmp_path)
    assert source == {
        "kind": "snapshot",
        "commit": "abc123",
        "dirty": True,
        "snapshot_sha256": "a" * 64,
    }
    assert warnings == ["source_tree_dirty"]


def test_invalid_or_missing_source_is_explicit(tmp_path: Path) -> None:
    source, warnings = source_provenance(tmp_path)
    assert source["kind"] == "unknown"
    assert warnings == ["source_unknown"]
    (tmp_path / ".imu-source.json").write_text("{}", encoding="utf-8")
    source, warnings = source_provenance(tmp_path)
    assert source["kind"] == "unknown"
    assert warnings == ["source_manifest_invalid", "source_unknown"]


def test_performance_phases_and_process_usage_are_machine_readable() -> None:
    started = process_snapshot()
    phases = PhaseTimer()
    with phases.track("small_phase_seconds"):
        sum(range(100))
    stopped = process_snapshot()
    assert phases.to_dict()["small_phase_seconds"] >= 0.0
    usage = process_delta(started, stopped)
    assert usage["user_seconds"] >= 0.0
    assert usage["system_seconds"] >= 0.0
    assert usage["max_rss_bytes"] > 0


def test_job_performance_aggregation_preserves_model_and_phase() -> None:
    results = [
        {
            "status": "computed",
            "metadata": {
                "job": {"model_id": "torch_1d_cnn"},
                "performance": {
                    "schema_version": PERFORMANCE_SCHEMA_VERSION,
                    "phase_seconds": {
                        "model_fit_seconds": 2.0,
                        "test_inference_seconds": 0.5,
                    },
                },
            },
            "invocation_performance": {
                "phase_seconds": {"checkpoint_write_seconds": 0.25}
            },
        },
        {
            "status": "cached",
            "metadata": {
                "job": {"model_id": "torch_1d_cnn"},
                "performance": {
                    "schema_version": PERFORMANCE_SCHEMA_VERSION,
                    "phase_seconds": {"model_fit_seconds": 1.0},
                },
            },
            "invocation_performance": {
                "phase_seconds": {"checkpoint_read_seconds": 0.1}
            },
        },
    ]
    result = aggregate_job_performance(results)
    assert result["computed_jobs"] == 1
    assert result["cached_jobs"] == 1
    assert result["bottlenecks"][0] == {
        "phase": "model_fit_seconds",
        "total_seconds": 3.0,
    }
    assert result["by_model"]["torch_1d_cnn"]["checkpoint_write_seconds"][
        "total_seconds"
    ] == pytest.approx(0.25)


def test_nvidia_smi_telemetry_parser_handles_optional_power() -> None:
    parsed = _parse_nvidia_smi_line("87, 42, 3198, 188.5")
    assert parsed == {
        "gpu_utilization_percent": 87.0,
        "memory_utilization_percent": 42.0,
        "memory_used_mib": 3198.0,
        "power_watts": 188.5,
    }
    parsed = _parse_nvidia_smi_line("3, 1, 1024, [N/A]")
    assert parsed is not None
    assert parsed["power_watts"] is None
    assert _parse_nvidia_smi_line("malformed") is None
