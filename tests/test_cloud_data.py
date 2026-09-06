import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from imu_benchmark import cloud_data


def _description(path: Path, *, content_type: str, metadata: dict[str, str]) -> dict:
    return {
        "name": "benchmark-datasets/base/snapshot/datasets/example.h5",
        "size": path.stat().st_size,
        "content_type": content_type,
        "custom_fields": metadata,
        "md5_hash": base64.b64encode(
            hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()
        ).decode("ascii"),
        "metageneration": 7,
    }


def _completed(arguments: tuple[str, ...], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


def test_upload_sets_and_verifies_explicit_object_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "example.h5"
    source.write_bytes(b"hdf5-fixture")
    metadata = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    calls: list[tuple[str, ...]] = []

    def fake_run(*arguments: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[1:3] == ("objects", "describe"):
            payload = _description(
                source, content_type="application/x-hdf5", metadata=metadata
            )
            return _completed(arguments, json.dumps(payload))
        return _completed(arguments)

    monkeypatch.setattr(cloud_data, "_run_gcloud", fake_run)
    cloud_data._upload_file(
        source,
        "gs://bucket/example.h5",
        immutable=True,
        content_type="application/x-hdf5",
        metadata=metadata,
    )

    upload = calls[0]
    assert upload[:3] == ("storage", "cp", "--no-clobber")
    assert "--content-type=application/x-hdf5" in upload
    assert f"--custom-metadata=sha256={metadata['sha256']}" in upload
    assert not any(call[1:3] == ("objects", "update") for call in calls)


def test_upload_repairs_metadata_only_when_existing_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "example.h5"
    source.write_bytes(b"same-immutable-bytes")
    metadata = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    calls: list[tuple[str, ...]] = []
    descriptions = iter(
        [
            _description(source, content_type="application/mipc", metadata={}),
            _description(
                source, content_type="application/x-hdf5", metadata=metadata
            ),
        ]
    )

    def fake_run(*arguments: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[1:3] == ("objects", "describe"):
            return _completed(arguments, json.dumps(next(descriptions)))
        return _completed(arguments)

    monkeypatch.setattr(cloud_data, "_run_gcloud", fake_run)
    cloud_data._upload_file(
        source,
        "gs://bucket/example.h5",
        immutable=True,
        content_type="application/x-hdf5",
        metadata=metadata,
    )

    update = next(call for call in calls if call[1:3] == ("objects", "update"))
    assert "--if-metageneration-match=7" in update
    assert "--content-type=application/x-hdf5" in update


def test_upload_refuses_to_repair_metadata_for_different_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "example.h5"
    source.write_bytes(b"expected-bytes")
    metadata = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    description = _description(source, content_type="application/mipc", metadata={})
    description["md5_hash"] = base64.b64encode(b"not-the-same-md5").decode("ascii")
    calls: list[tuple[str, ...]] = []

    def fake_run(*arguments: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[1:3] == ("objects", "describe"):
            return _completed(arguments, json.dumps(description))
        return _completed(arguments)

    monkeypatch.setattr(cloud_data, "_run_gcloud", fake_run)
    with pytest.raises(ValueError, match="content type mismatch"):
        cloud_data._upload_file(
            source,
            "gs://bucket/example.h5",
            immutable=True,
            content_type="application/x-hdf5",
            metadata=metadata,
        )

    assert not any(call[1:3] == ("objects", "update") for call in calls)
