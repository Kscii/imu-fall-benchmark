from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imu_benchmark import runtime
from imu_benchmark.configuration import PUBLIC_MODEL_IDS, load_experiment
from imu_benchmark.data import (
    extract_window_features,
)
from imu_benchmark.dataset import Annotation, validate_data
from imu_benchmark.device import CudaUnavailable, _parse_nvidia_smi_line
from imu_benchmark.engine import build_jobs, plan_experiment
from imu_benchmark.evaluation import best_threshold
from imu_benchmark.performance import (
    PERFORMANCE_SCHEMA_VERSION,
    PhaseTimer,
    aggregate_job_performance,
    process_delta,
    process_snapshot,
)
from imu_benchmark.protocol import segment_decision_time_labels
from imu_benchmark.runtime import (
    require_compute_runtime,
    resolve_work_paths,
    source_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_kfall_smoke_plan_contains_seven_public_fall_models() -> None:
    config = load_experiment(PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/kfall_smoke_v1.yaml")
    result = plan_experiment(config)
    assert result["scheduled_jobs"] == 7
    assert tuple(result["models"]) == PUBLIC_MODEL_IDS
    assert result["split_protocol"]["validation_fold_by_test_fold"] == {0: 1}
    assert not result["research_only"]
    assert all(job["objective"] == "temporal_supervised" for job in result["jobs"])


def test_research_data_views_are_explicit() -> None:
    mixed = load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/kfall_public_adl_smoke_v1.yaml",
    )
    mil = load_experiment(
        PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/public_mil_smoke_v1.yaml"
    )
    assert mixed["data_view"]["research_only"]
    assert mixed["data_view"]["negative_supplement_datasets"] == [
        "cgu_bes",
        "sisfall",
        "uci_455",
        "umafall",
        "upfall",
    ]
    assert mil["data_view"]["objective"] == "recording_mil"
    assert set(mil["models"]) == {
        "threshold_impact",
        "torch_1d_cnn",
        "torch_lstm",
        "torch_cnn_lstm",
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
        starts,
        ends,
        (
            Annotation("activity", 70, 101, "F01"),
            Annotation("onset", 70, 70, "F01"),
            Annotation("impact", 100, 100, "F01"),
        ),
    )
    assert labels.tolist() == [0, 1, 1, 0, 0]
    assert keep.tolist() == [True, True, True, False, False]
    assert intervals == ((70, 101),)


def test_bf16_config_is_an_apples_to_apples_sequence_comparison() -> None:
    config = load_experiment(
        PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/kfall_bf16_smoke_v1.yaml"
    )
    assert config["precision"] == "bf16"
    assert config["folds"] == [0]
    assert config["seeds"] == [3888]
    assert config["models"] == ["torch_1d_cnn", "torch_lstm", "torch_cnn_lstm"]


def test_full_fold_pilot_configs_are_apples_to_apples() -> None:
    fp32 = load_experiment(
        PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/kfall_fold0_pilot_fp32_v1.yaml"
    )
    bf16 = load_experiment(
        PROJECT_ROOT, PROJECT_ROOT / "configs/experiments/kfall_fold0_pilot_bf16_v1.yaml"
    )
    assert len(build_jobs(fp32)) == 7
    assert len(build_jobs(bf16)) == 3
    for field in (
        "contract_sha256",
        "snapshot_sha256",
        "data_view_sha256",
        "folds",
        "seeds",
        "gpu_mode",
        "max_epochs",
        "patience",
        "max_sequences_per_split",
        "max_windows_per_sequence",
    ):
        assert fp32[field] == bf16[field]
    assert fp32["precision"] == "fp32"
    assert bf16["precision"] == "bf16"
    assert bf16["models"] == ["torch_1d_cnn", "torch_lstm", "torch_cnn_lstm"]
    assert fp32["max_sequences_per_split"] is None
    assert fp32["max_windows_per_sequence"] is None


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
            "invocation_performance": {"phase_seconds": {"checkpoint_write_seconds": 0.25}},
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
            "invocation_performance": {"phase_seconds": {"checkpoint_read_seconds": 0.1}},
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
