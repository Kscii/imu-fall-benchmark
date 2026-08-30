import json

from test_cloud_results import _formal_run

from imu_benchmark import cloud_experiments
from imu_benchmark.cloud_experiments import _metadata, verify_experiment_catalog


def test_experiment_catalog_is_independent_and_has_direct_onnx(tmp_path) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
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
        run_dir,
        run_manifest,
        result_manifest,
        publication_id="formal-catalog-v1",
    )

    assert metadata["schema_version"] == "imu_experiment_catalog_v1"
    assert metadata["contract_version"] == "1.0.0"
    assert metadata["publication_id"] == "formal-catalog-v1"
    assert metadata["result_evidence"]["schema_version"].endswith("_v1")
    assert len(metadata["artifacts"]) == 7
    assert len(uploads) == 7
    assert set(sources) == {item["file_id"] for item in uploads}
    first = metadata["artifacts"][0]
    assert first["metrics"]["metric_split"] == "test"
    assert first["metrics"]["selection_eligible"] is False
    assert first["decision"]["score_threshold"]["selection_split"] == "validation"
    assert first["decision"]["trigger_policies"][0]["cooldown_seconds"] == 10.0


def test_remote_verify_hashes_every_published_onnx(tmp_path, monkeypatch) -> None:
    run_dir = _formal_run(tmp_path / "runs")
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
    metadata, _uploads, sources = _metadata(
        run_dir,
        run_manifest,
        result_manifest,
        publication_id="formal-catalog-v1",
    )
    by_sha = {
        artifact["onnx"]["sha256"]: sources[f"onnx-{artifact['artifact_id']}"]
        for artifact in metadata["artifacts"]
    }
    verified = []

    def download(_bucket, descriptor, destination) -> None:
        destination.write_bytes(by_sha[descriptor["sha256"]].read_bytes())
        verified.append(descriptor["filename"])

    monkeypatch.setattr(
        cloud_experiments,
        "ensure_gcloud_login",
        lambda **_kwargs: "user@example.com",
    )
    monkeypatch.setattr(cloud_experiments, "data_bucket", lambda: "gs://fixture")
    monkeypatch.setattr(
        cloud_experiments,
        "_gcloud_cat",
        lambda *_args, **_kwargs: (json.dumps(metadata) + "\n").encode(),
    )
    monkeypatch.setattr(cloud_experiments, "_download_descriptor", download)

    result = verify_experiment_catalog("formal-catalog-v1")

    assert result["verified_onnx_artifacts"] == len(metadata["artifacts"])
    assert len(verified) == len(metadata["artifacts"])
