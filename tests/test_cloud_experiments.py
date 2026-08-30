from test_cloud_results import _formal_run

from imu_benchmark.cloud_experiments import _metadata


def test_experiment_catalog_is_independent_and_has_direct_onnx(tmp_path) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    import json

    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    result_manifest = {
        "schema_version": "imu_benchmark_result_manifest_v1",
        "run_id": run_dir.name,
        "bundle": {
            "filename": "run.tar.gz",
            "size_bytes": 123,
            "sha256": "a" * 64,
        },
    }

    metadata, uploads, sources = _metadata(
        run_dir, run_manifest, result_manifest
    )

    assert metadata["schema_version"] == "imu_experiment_catalog_v1"
    assert metadata["contract_version"] == "1.0.0"
    assert metadata["publication_id"] == f"{run_dir.name}-catalog-v1"
    assert metadata["result_evidence"]["schema_version"].endswith("_v1")
    assert len(metadata["artifacts"]) == 7
    assert len(uploads) == 7
    assert set(sources) == {item["file_id"] for item in uploads}
    first = metadata["artifacts"][0]
    assert first["metrics"]["metric_split"] == "test"
    assert first["metrics"]["selection_eligible"] is False
    assert first["decision"]["score_threshold"]["selection_split"] == "validation"
    assert first["decision"]["trigger_policies"][0]["cooldown_seconds"] == 10.0
