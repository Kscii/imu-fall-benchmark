from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imu_benchmark import cloud_data, runtime
from imu_benchmark.cloud_data import _remote_snapshot, _validate_remote_manifest, data_bucket
from imu_benchmark.configuration import PUBLIC_MODEL_IDS, load_experiment
from imu_benchmark.dataset import validate_data
from imu_benchmark.device import CudaUnavailable, _parse_nvidia_smi_line
from imu_benchmark.engine import plan_experiment, split_indices
from imu_benchmark.evaluation import best_threshold
from imu_benchmark.performance import PhaseTimer, process_delta, process_snapshot
from imu_benchmark.runtime import require_compute_runtime, resolve_work_paths, source_provenance
from imu_benchmark.window_cache import UnifiedWindowStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _store_with_team_training_data() -> UnifiedWindowStore:
    folds = np.repeat(np.arange(5, dtype=np.int8), 2)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 5)
    sequence_folds = np.concatenate((folds, np.asarray([-1, -1], dtype=np.int8)))
    temporal = np.concatenate((labels, np.asarray([0, 1], dtype=np.int8)))
    count = len(sequence_folds)
    sequence = np.arange(count, dtype=np.int32)
    event_sequences = np.flatnonzero(temporal).astype(np.int32)
    return UnifiedWindowStore(
        path=Path("unused.h5"),
        sequence_index=sequence,
        start_sample=np.zeros(count, dtype=np.int32),
        end_sample=np.full(count, 50, dtype=np.int32),
        fold_id=sequence_folds.copy(),
        bag_label=temporal.copy(),
        temporal_label=temporal,
        dataset_id=np.asarray(["kfall"] * 10 + ["team_cw12eu"] * 2),
        participant_id=np.asarray([f"p{index}" for index in range(count)]),
        recording_id=np.asarray([f"r{index}" for index in range(count)]),
        body_location=np.asarray(["waist"] * count),
        supervision_kind=np.asarray(["temporal"] * count),
        sequence_is_fall=temporal.astype(np.bool_),
        sequence_fold_id=sequence_folds,
        event_sequence_index=event_sequences,
        event_onset_sample=np.zeros(len(event_sequences), dtype=np.int64),
        event_impact_sample=np.full(len(event_sequences), 25, dtype=np.int64),
        event_stop_sample=np.full(len(event_sequences), 50, dtype=np.int64),
        event_code=np.asarray(["fall"] * len(event_sequences)),
        manifest={
            "sampling_rate_hz": 25,
            "stride_seconds": 0.5,
            "windows": count,
            "fall_events": len(event_sequences),
        },
    )


def test_smoke_plan_contains_seven_public_fall_models(active_manifest_path: Path) -> None:
    config = load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/temporal_smoke_v1.yaml",
        snapshot_path=active_manifest_path,
    )
    result = plan_experiment(config)
    assert result["scheduled_jobs"] == 7
    assert tuple(result["models"]) == PUBLIC_MODEL_IDS
    assert result["split_protocol"]["validation_fold_by_test_fold"] == {0: 1}
    assert all(job["objective"] == "temporal_supervised" for job in result["jobs"])


def test_temporal_core_onnx_plan_contains_65_full_parity_jobs(
    active_manifest_path: Path,
) -> None:
    config = load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/formal_baseline_temporal_core_onnx_v1.yaml",
        snapshot_path=active_manifest_path,
    )
    result = plan_experiment(config)
    assert result["scheduled_jobs"] == 65
    assert config["export_onnx"] is True
    assert config["onnx_parity_splits"] == ["validation", "test"]
    assert config["onnx_parity_batch_size"] == 256
    assert config["onnx_parity_max_samples"] is None
    assert config["onnx_parity_rtol"] == 1e-4
    assert config["onnx_parity_atol"] == 1e-2


def test_team_fold_is_training_only(active_manifest_path: Path) -> None:
    config = load_experiment(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/experiments/temporal_smoke_v1.yaml",
        snapshot_path=active_manifest_path,
    )
    store = _store_with_team_training_data()
    split = split_indices(store, config, fold=0, seed=3888)
    assert {-1}.issubset(set(store.fold_id[split.train].tolist()))
    assert -1 not in store.fold_id[split.validation]
    assert -1 not in store.fold_id[split.test]
    assert set(store.fold_id[split.validation].tolist()) == {1}
    assert set(store.fold_id[split.test].tolist()) == {0}


def test_threshold_tie_break_is_deterministic() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    assert best_threshold(labels, scores) == best_threshold(labels, scores)
    assert best_threshold(labels, scores)[:2] == (0.6, 1.0)


def test_work_paths_keep_data_cache_and_runs_outside_the_clone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMU_BENCH_WORK_ROOT", str(tmp_path))
    paths = resolve_work_paths()
    assert paths.data == tmp_path / "data"
    assert paths.cache == tmp_path / "cache"
    assert paths.runs == tmp_path / "runs"
    monkeypatch.setenv("IMU_BENCH_WORK_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        resolve_work_paths()


def test_only_compute_commands_require_wsl2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "is_wsl2", lambda: False)
    for command in ("data", "validate-data", "plan", "report"):
        require_compute_runtime(command, PROJECT_ROOT)
    for command in ("doctor", "smoke", "run"):
        with pytest.raises(CudaUnavailable, match="requires WSL2"):
            require_compute_runtime(command, PROJECT_ROOT)


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
    assert source["commit"] == "abc123"
    assert warnings == ["source_tree_dirty"]


def test_performance_and_nvidia_telemetry_helpers() -> None:
    started = process_snapshot()
    phases = PhaseTimer()
    with phases.track("small_phase_seconds"):
        sum(range(100))
    usage = process_delta(started, process_snapshot())
    assert phases.to_dict()["small_phase_seconds"] >= 0.0
    assert usage["max_rss_bytes"] > 0
    assert _parse_nvidia_smi_line("87, 42, 3198, 188.5") == {
        "gpu_utilization_percent": 87.0,
        "memory_utilization_percent": 42.0,
        "memory_used_mib": 3198.0,
        "power_watts": 188.5,
    }


def test_versioned_base_manifest_is_cloud_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "configs/data/base_imu25_v2.json").read_text(encoding="utf-8")
    )
    _validate_remote_manifest(manifest, expected_kind="base")
    monkeypatch.setenv("IMU_BENCH_DATA_BUCKET", "gs://team-bucket")
    assert data_bucket() == "gs://team-bucket"
    monkeypatch.setenv("IMU_BENCH_DATA_BUCKET", "gs://team-bucket/prefix")
    with pytest.raises(ValueError, match="bucket URI"):
        data_bucket()


def test_exact_snapshot_resolution_does_not_read_current_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_bytes = (PROJECT_ROOT / "configs/data/base_imu25_v2.json").read_bytes()
    requested = []

    def fake_cat(uri: str, *, optional: bool = False) -> bytes:
        requested.append((uri, optional))
        return manifest_bytes

    monkeypatch.setattr(cloud_data, "_gcloud_cat", fake_cat)
    resolved = _remote_snapshot(
        "gs://team-bucket",
        kind="base",
        current_object="benchmark-datasets/base/current.json",
        optional=False,
        snapshot_id="imu_25hz_snapshot_v2",
    )
    assert resolved is not None
    current, manifest = resolved
    assert manifest["snapshot_id"] == "imu_25hz_snapshot_v2"
    assert current["manifest_object"].endswith("/base/imu_25hz_snapshot_v2/manifest.json")
    assert all(not uri.endswith("/current.json") for uri, _ in requested)


def test_exact_team_snapshot_preserves_collection_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "configs/data/base_imu25_v2.json").read_text(encoding="utf-8")
    )
    manifest["kind"] = "team"
    manifest["snapshot_id"] = "snapshot-team-v1"
    manifest["handoff_contract_version"] = "0.2.0"
    for entry in manifest["files"]:
        entry["evaluation_role"] = "training_only"
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    requested = []

    def fake_cat(uri: str, *, optional: bool = False) -> bytes:
        requested.append((uri, optional))
        return manifest_bytes

    monkeypatch.setattr(cloud_data, "_gcloud_cat", fake_cat)
    resolved = _remote_snapshot(
        "gs://team-bucket",
        kind="team",
        current_object="benchmark-datasets/team/cw12eu/current.json",
        optional=False,
        snapshot_id="snapshot-team-v1",
    )
    assert resolved is not None
    assert resolved[0]["manifest_object"] == (
        "benchmark-datasets/team/cw12eu/snapshot-team-v1/manifest.json"
    )
    assert all(not uri.endswith("/current.json") for uri, _ in requested)


def test_headless_data_pull_requires_prior_gcloud_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cloud_data,
        "_run_gcloud",
        lambda *arguments, **options: SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(
        cloud_data.sys,
        "stdin",
        SimpleNamespace(isatty=lambda: False),
    )
    with pytest.raises(RuntimeError, match="interactive WSL terminal"):
        cloud_data.ensure_gcloud_login(interactive=True)


def test_reviewed_base_hdf5_integration(active_manifest_path: Path) -> None:
    source_value = os.environ.get("IMU_BENCH_INTEGRATION_DATA_DIR")
    if source_value is None:
        pytest.skip("IMU_BENCH_INTEGRATION_DATA_DIR is not set")
    source = Path(source_value).resolve()
    destination = active_manifest_path.parent / "base/imu_25hz_snapshot_v2/datasets"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)
    result = validate_data(PROJECT_ROOT, snapshot_path=active_manifest_path)
    assert result["status"] == "PASS"
    assert result["base"]["files"] == 9
    assert result["base"]["sequences"] == 21_472
    assert result["base"]["rows"] == 8_475_296
    assert result["base"]["events"] == 2_919
