from pathlib import Path
from types import SimpleNamespace

from imu_benchmark import model_broker


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self) -> dict:
        return self._payload


def test_model_broker_uses_gcloud_identity_and_resumable_session(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    posted: list[tuple[str, dict, dict]] = []
    uploaded: list[tuple[str, bytes, dict]] = []
    monkeypatch.setattr(
        model_broker,
        "_run_gcloud",
        lambda *args: SimpleNamespace(stdout="signed-google-id-token\n"),
    )
    monkeypatch.setenv("IMU_BENCH_UPLOAD_BROKER_URL", "https://upload.example.test")

    def post(url: str, *, json: dict, headers: dict, timeout: int):
        posted.append((url, json, headers))
        if url.endswith("/v1/model-uploads"):
            return _Response(
                {
                    "upload_id": "a" * 32,
                    "sessions": [
                        {
                            "file_id": "model",
                            "object_key": "benchmark-models/packages/example/files/model.onnx",
                            "session_url": "https://storage.example/session",
                            "already_present": False,
                        }
                    ],
                }
            )
        assert timeout == 900
        return _Response(
            {
                "publication_kind": "package",
                "publication_id": "example",
                "marker_object": "benchmark-models/packages/example/publication.json",
                "marker_generation": 1,
                "verified_sha256": True,
            }
        )

    def put(url: str, *, data, headers: dict, timeout: tuple[int, int]):
        uploaded.append((url, data.read(), headers))
        assert timeout == (30, 900)
        return _Response({}, 201)

    monkeypatch.setattr(model_broker.requests, "post", post)
    monkeypatch.setattr(model_broker.requests, "put", put)
    artifact = {
        "file_id": "model",
        "object_key": "benchmark-models/packages/example/files/model.onnx",
        "size_bytes": 4,
        "sha256": "a" * 64,
        "content_type": "application/octet-stream",
    }

    result = model_broker.publish_model_artifacts(
        publication_kind="package",
        publication_id="example",
        marker={"schema_version": "fixture"},
        artifacts=[artifact],
        sources={"model": source},
    )

    assert result["verified_sha256"] is True
    assert posted[0][2] == {"Authorization": "Bearer signed-google-id-token"}
    assert posted[1][1] == {"upload_id": "a" * 32}
    assert uploaded == [
        (
            "https://storage.example/session",
            b"onnx",
            {"Content-Length": "4", "Content-Type": "application/octet-stream"},
        )
    ]


def test_model_broker_rejects_source_descriptor_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    try:
        model_broker.publish_model_artifacts(
            publication_kind="package",
            publication_id="example",
            marker={},
            artifacts=[
                {
                    "file_id": "different",
                    "object_key": "benchmark-models/packages/example/files/model.onnx",
                    "size_bytes": 4,
                    "sha256": "a" * 64,
                    "content_type": "application/octet-stream",
                }
            ],
            sources={"model": source},
        )
    except ValueError as error:
        assert "source files differ" in str(error)
    else:
        raise AssertionError("descriptor/source mismatch should fail before authentication")
