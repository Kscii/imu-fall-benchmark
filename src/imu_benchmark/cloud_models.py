"""Validate and publish immutable final ONNX model packages."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cloud_data import (
    _gcloud_cat,
    _object_uri,
    _read_json_bytes,
    _sha256_file,
    data_bucket,
    ensure_gcloud_login,
)
from .model_broker import publish_model_artifacts, restore_model_publication
from .progress import NullProgressReporter, ProgressReporter

PACKAGE_MANIFEST_SCHEMA = "imu_model_package_manifest_v1"
PACKAGE_PUBLICATION_SCHEMA = "imu_model_package_publication_v1"
PACKAGE_STATE_SCHEMA = "imu_model_publication_state_v1"
PACKAGE_PREFIX = "benchmark-models/packages"
REQUIRED_FILES = (
    "model.onnx",
    "manifest.json",
    "runtime_config.json",
    "metrics.json",
    "golden_input.npz",
    "golden_output.npz",
    "checksums.sha256",
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _package_id(model_code: str, logical_digest: str) -> str:
    return f"{model_code}-{logical_digest[:12]}"


def _validate_manifest(package_dir: Path) -> dict[str, Any]:
    manifest = _json(package_dir / "manifest.json")
    required = {
        "schema_version",
        "model_code",
        "display_name",
        "source",
        "input",
        "output",
        "preprocessing",
        "validation",
        "known_limitations",
    }
    if set(manifest) != required or manifest.get("schema_version") != PACKAGE_MANIFEST_SCHEMA:
        raise ValueError("Model package manifest schema is invalid")
    model_code = manifest.get("model_code")
    if not isinstance(model_code, str) or not _IDENTIFIER.fullmatch(model_code):
        raise ValueError("Model package model_code is invalid")
    if not isinstance(manifest.get("display_name"), str) or not manifest["display_name"].strip():
        raise ValueError("Model package display_name is missing")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("dirty") is not False
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
    ):
        raise ValueError("Model package requires a clean source commit")
    input_contract = manifest.get("input")
    if (
        not isinstance(input_contract, dict)
        or input_contract.get("semantic")
        not in {"si_window", "normalized_window", "engineered_features"}
        or input_contract.get("dtype") != "float32"
        or not isinstance(input_contract.get("shape"), list)
    ):
        raise ValueError("Model package input contract is invalid")
    output_contract = manifest.get("output")
    if (
        not isinstance(output_contract, dict)
        or output_contract.get("semantic") != "fall_score"
        or output_contract.get("name") != "fall_score"
        or output_contract.get("dtype") != "float32"
    ):
        raise ValueError("Model package output contract is invalid")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("onnx_checker") != "PASS"
        or validation.get("python_onnxruntime_parity") != "PASS"
        or validation.get("external_runtime") not in {"PASS", "not_tested"}
        or validation.get("device_replay") not in {"PASS", "not_tested"}
    ):
        raise ValueError("Model package validation status is invalid")
    if not isinstance(manifest.get("known_limitations"), list):
        raise ValueError("Model package known_limitations is invalid")
    return manifest


def _parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not _SHA256.fullmatch(parts[0]):
            raise ValueError("checksums.sha256 has an invalid line")
        filename = parts[1]
        if Path(filename).name != filename or filename in result:
            raise ValueError("checksums.sha256 has an unsafe or duplicate filename")
        result[filename] = parts[0]
    return result


def _validate_onnx_and_golden(package_dir: Path) -> None:
    import onnx
    import onnxruntime as ort

    model_path = package_dir / "model.onnx"
    onnx.checker.check_model(onnx.load(model_path))
    with np.load(package_dir / "golden_input.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"input"}:
            raise ValueError("golden_input.npz must contain only input")
        values = np.ascontiguousarray(archive["input"], dtype=np.float32)
    with np.load(package_dir / "golden_output.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"output"}:
            raise ValueError("golden_output.npz must contain only output")
        expected = np.asarray(archive["output"], dtype=np.float32)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("Model package ONNX must have exactly one input")
    observed = np.asarray(
        session.run(["fall_score"], {session.get_inputs()[0].name: values})[0],
        dtype=np.float32,
    )
    if observed.shape != expected.shape or not np.allclose(
        observed, expected, rtol=1e-4, atol=1e-2
    ):
        raise ValueError("Model package golden parity failed")


def validate_model_package(package_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = package_dir.resolve()
    actual = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual != sorted(REQUIRED_FILES):
        raise ValueError("Model package files differ from the required contract")
    manifest = _validate_manifest(root)
    _json(root / "runtime_config.json")
    _json(root / "metrics.json")
    checksums = _parse_checksums(root / "checksums.sha256")
    expected_checked = set(REQUIRED_FILES) - {"checksums.sha256"}
    if set(checksums) != expected_checked:
        raise ValueError("checksums.sha256 does not cover every package payload")
    for filename, digest in checksums.items():
        if _sha256_file(root / filename) != digest:
            raise ValueError(f"Model package SHA-256 differs: {filename}")
    _validate_onnx_and_golden(root)
    files = [
        {
            "file_id": path.stem.replace("_", "-"),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "content_type": (
                "application/json"
                if path.suffix == ".json"
                else "application/octet-stream"
            ),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
    ]
    logical = hashlib.sha256(
        _json_bytes(
            {
                "manifest": manifest,
                "files": [
                    {key: entry[key] for key in ("filename", "size_bytes", "sha256")}
                    for entry in files
                ],
            }
        )
    ).hexdigest()
    package_manifest = {
        **manifest,
        "package_id": _package_id(manifest["model_code"], logical),
        "logical_digest": logical,
    }
    return package_manifest, files


def _write_bundle(package_dir: Path, package_id: str, destination: Path) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for filename in REQUIRED_FILES:
                    path = package_dir / filename
                    info = tarfile.TarInfo(f"{package_id}/{filename}")
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    info.mtime = 0
                    with path.open("rb") as source:
                        archive.addfile(info, source)


def publish_model_package(
    package_dir: Path,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    with reporter.task("Validating the final ONNX model package"):
        manifest, files = validate_model_package(package_dir)
    with reporter.task("Checking Google Cloud sign-in"):
        account = ensure_gcloud_login(interactive=True)
    bucket = data_bucket()
    package_id = manifest["package_id"]
    prefix = f"{PACKAGE_PREFIX}/{package_id}"
    with tempfile.TemporaryDirectory(prefix="imu-model-package-") as temporary:
        root = Path(temporary)
        bundle = root / "package.tar.gz"
        _write_bundle(package_dir, package_id, bundle)
        published_at = datetime.now(UTC).isoformat()
        publication = {
            "schema_version": PACKAGE_PUBLICATION_SCHEMA,
            "package_id": package_id,
            "created_at_utc": published_at,
            "logical_digest": manifest["logical_digest"],
            "manifest": manifest,
            "bundle": {
                "filename": bundle.name,
                "size_bytes": bundle.stat().st_size,
                "sha256": _sha256_file(bundle),
            },
            "files": [
                {
                    **entry,
                    "object_key": f"{prefix}/files/{entry['filename']}",
                }
                for entry in files
            ],
        }
        artifacts = [
            {
                "file_id": "bundle",
                "object_key": f"{prefix}/package.tar.gz",
                "size_bytes": publication["bundle"]["size_bytes"],
                "sha256": publication["bundle"]["sha256"],
                "content_type": "application/gzip",
            },
            *[
                {
                    key: entry[key]
                    for key in (
                        "file_id",
                        "object_key",
                        "size_bytes",
                        "sha256",
                        "content_type",
                    )
                }
                for entry in publication["files"]
            ],
        ]
        sources = {
            "bundle": bundle,
            **{
                entry["file_id"]: package_dir / entry["filename"]
                for entry in publication["files"]
            },
        }
        with reporter.task("Uploading through the constrained team broker"):
            completed = publish_model_artifacts(
                publication_kind="package",
                publication_id=package_id,
                marker=publication,
                artifacts=artifacts,
                sources=sources,
            )
        if completed.get("marker_object") != f"{prefix}/publication.json":
            raise ValueError("Upload broker completed an unexpected model package")
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "package_id": package_id,
        "publication_object": f"{prefix}/publication.json",
        "bundle_sha256": publication["bundle"]["sha256"],
    }


def verify_model_package(package_id: str) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(package_id):
        raise ValueError("Invalid model package ID")
    account = ensure_gcloud_login(interactive=False)
    bucket = data_bucket()
    key = f"{PACKAGE_PREFIX}/{package_id}/publication.json"
    payload = _gcloud_cat(_object_uri(bucket, key), optional=False)
    assert payload is not None
    publication = _read_json_bytes(payload, source=key)
    if (
        publication.get("schema_version") != PACKAGE_PUBLICATION_SCHEMA
        or publication.get("package_id") != package_id
    ):
        raise ValueError("Remote model package publication is invalid")
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "package_id": package_id,
        "logical_digest": publication.get("logical_digest"),
    }


def restore_published_model_package(
    package_id: str, *, expected_generation: int
) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(package_id):
        raise ValueError("Invalid model package ID")
    ensure_gcloud_login(interactive=False)
    return restore_model_publication(
        "package", package_id, expected_generation=expected_generation
    )
