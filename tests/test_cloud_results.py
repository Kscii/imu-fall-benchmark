from __future__ import annotations

import json
from pathlib import Path

import pytest

from imu_benchmark.cloud_results import (
    _extract_bundle,
    _publication_manifest,
    _validate_publication_manifest,
    _validate_run,
    _write_bundle,
)


def _formal_run(tmp_path: Path) -> Path:
    run_id = "formal_baseline_temporal_core_v1-test"
    run_dir = tmp_path / run_id
    jobs = run_dir / "jobs"
    jobs.mkdir(parents=True)
    for index in range(65):
        (jobs / f"{index:016x}.npz").write_bytes(f"job-{index}".encode())
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "aggregate_metrics.csv").write_text("model,value\n", encoding="utf-8")
    (run_dir / "statistical_manifest.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "experiment_id": "formal_baseline_temporal_core_v1",
        "status": "PASS",
        "scheduled_jobs": 65,
        "completed_jobs": 65,
        "failures": [],
        "source": {
            "kind": "git",
            "commit": "a" * 40,
            "dirty": False,
            "snapshot_sha256": None,
        },
        "base_snapshot_id": "imu_25hz_snapshot_v2",
        "snapshot_sha256": "b" * 64,
        "resolved_config_sha256": "c" * 64,
        "statistical_analysis": {"status": "PASS"},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def test_result_bundle_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    run_manifest, entries = _validate_run(run_dir)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_bundle(run_dir, run_dir.name, entries, first)
    _write_bundle(run_dir, run_dir.name, entries, second)
    assert first.read_bytes() == second.read_bytes()
    publication = _publication_manifest(run_manifest, entries, first)
    _validate_publication_manifest(publication, run_id=run_dir.name)
    extracted = _extract_bundle(
        first, tmp_path / "extracted", run_dir.name, publication["files"]
    )
    assert (extracted / "run_manifest.json").is_file()
    assert len(list((extracted / "jobs").glob("*.npz"))) == 65


def test_result_publication_rejects_dirty_or_incomplete_runs(tmp_path: Path) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["dirty"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="clean, identified"):
        _validate_run(run_dir)

    manifest["source"]["dirty"] = False
    manifest["completed_jobs"] = 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="65/65"):
        _validate_run(run_dir)
