"""Validate and publish immutable two-file ONNX model releases."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import yaml

from .artifact_contract import (
    MODEL_CONTRACT_VERSION,
    MODEL_SCHEMA_V1,
    require_compatible_version,
    validate_model_marker_v1,
)
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

MODEL_RELEASE_SCHEMA = MODEL_SCHEMA_V1
MODEL_RELEASE_CONTRACT_VERSION = MODEL_CONTRACT_VERSION
MODEL_RELEASE_PREFIX = "benchmark-model-catalog/models"
REQUIRED_FILES = ("metadata.json", "model.onnx")
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


def _nonempty_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Model release {name} is invalid")
    return value


def _validate_metadata(root: Path) -> dict[str, Any]:
    metadata = _json(root / "metadata.json")
    required = {
        "schema_version",
        "contract_version",
        "release_id",
        "model_code",
        "name",
        "created_at_utc",
        "release_stage",
        "source",
        "data",
        "input",
        "output",
        "preprocessing",
        "windowing",
        "decision",
        "metrics",
        "verification",
        "validation",
        "known_limitations",
        "model",
    }
    if set(metadata) != required:
        raise ValueError("Model release metadata fields differ from the contract")
    if metadata.get("schema_version") != MODEL_RELEASE_SCHEMA:
        raise ValueError("Model release metadata schema is invalid")
    require_compatible_version(
        metadata.get("contract_version"),
        MODEL_RELEASE_CONTRACT_VERSION,
        name="model release contract version",
    )
    for name in ("release_id", "model_code"):
        value = metadata.get(name)
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"Model release {name} is invalid")
    if not isinstance(metadata.get("name"), str) or not metadata["name"].strip():
        raise ValueError("Model release name is missing")
    validate_model_marker_v1(metadata)
    descriptor = _nonempty_object(metadata.get("model"), "model descriptor")
    expected_key = f"{MODEL_RELEASE_PREFIX}/{metadata['release_id']}/model.onnx"
    if (
        descriptor.get("filename") != "model.onnx"
        or descriptor.get("object_key") != expected_key
        or descriptor.get("content_type") != "application/octet-stream"
        or not isinstance(descriptor.get("size_bytes"), int)
        or descriptor["size_bytes"] <= 0
        or not isinstance(descriptor.get("sha256"), str)
        or not _SHA256.fullmatch(descriptor["sha256"])
    ):
        raise ValueError("Model release model descriptor is invalid")
    return metadata


def _validate_onnx(root: Path, metadata: dict[str, Any]) -> None:
    path = root / "model.onnx"
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Model release ONNX must have exactly one input and one output")
    expected_input = metadata["input"].get("name")
    expected_output = metadata["output"].get("name")
    if (
        session.get_inputs()[0].name != expected_input
        or session.get_outputs()[0].name != expected_output
    ):
        raise ValueError("Model release ONNX names differ from metadata")
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    if (
        input_info.type != "tensor(float)"
        or list(input_info.shape[1:]) != [50, 6]
        or output_info.type != "tensor(float)"
        or len(output_info.shape) != 1
    ):
        raise ValueError("Model release ONNX shape or dtype differs from metadata")
    for fixture in metadata["verification"]["golden_fixtures"]:
        values = np.asarray(fixture["input_values"], dtype=np.float32)[None, :, :]
        output = np.asarray(session.run([expected_output], {expected_input: values})[0])
        if output.shape != (1,):
            raise ValueError("Golden fixture produced an invalid fall_score shape")
        if not np.isclose(
            float(output[0]),
            float(fixture["expected_fall_score"]),
            rtol=float(fixture["rtol"]),
            atol=float(fixture["atol"]),
        ):
            raise ValueError(f"Golden fixture failed: {fixture['fixture_id']}")


def validate_model_release(release_dir: Path) -> dict[str, Any]:
    root = release_dir.resolve()
    if not root.is_dir():
        raise ValueError("Model release directory does not exist")
    actual = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual != list(REQUIRED_FILES):
        raise ValueError("Model release must contain exactly metadata.json and model.onnx")
    metadata = _validate_metadata(root)
    path = root / "model.onnx"
    descriptor = metadata["model"]
    if (
        path.stat().st_size != descriptor["size_bytes"]
        or _sha256_file(path) != descriptor["sha256"]
    ):
        raise ValueError("Model release ONNX differs from metadata")
    _validate_onnx(root, metadata)
    return metadata


def _package_fixtures(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
) -> list[dict[str, Any]]:
    time = np.arange(50, dtype=np.float32) / 25.0
    stationary = np.zeros((50, 6), dtype=np.float32)
    stationary[:, 2] = np.float32(9.80665)
    adl = stationary.copy()
    adl[:, 0] = np.sin(2.0 * np.pi * time).astype(np.float32) * 1.2
    adl[:, 1] = np.cos(2.0 * np.pi * time).astype(np.float32) * 0.8
    adl[:, 5] = np.sin(np.pi * time).astype(np.float32) * 0.4
    impact = stationary.copy()
    impact[24:27, 0] = np.asarray([6.0, 18.0, 5.0], dtype=np.float32)
    impact[24:27, 2] = np.asarray([2.0, 28.0, 7.0], dtype=np.float32)
    impact[24:27, 4] = np.asarray([1.5, 4.0, 1.0], dtype=np.float32)
    fixtures = []
    for fixture_id, values in (
        ("stationary", stationary),
        ("adl_like", adl),
        ("impact_like", impact),
    ):
        score = np.asarray(session.run([output_name], {input_name: values[None, :, :]})[0])
        if score.shape != (1,) or not np.isfinite(score[0]):
            raise ValueError(f"Cannot produce golden fixture: {fixture_id}")
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "input_values": values.tolist(),
                "expected_fall_score": float(score[0]),
                "rtol": 1e-5,
                "atol": 1e-5,
            }
        )
    return fixtures


def package_model_release(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build the immutable two-file payload from one reviewable package spec."""

    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("Invalid model package spec") from error
    if not isinstance(spec, dict) or set(spec) != {
        "schema_version",
        "model_path",
        "metadata",
    }:
        raise ValueError("Model package spec fields differ from the contract")
    if spec.get("schema_version") != "imu_model_package_spec_v1":
        raise ValueError("Model package spec schema is invalid")
    metadata = spec.get("metadata")
    if not isinstance(metadata, dict) or "model" in metadata or "verification" in metadata:
        raise ValueError("Package metadata must omit generated model and verification fields")
    source_path = (spec_path.parent / str(spec["model_path"])).resolve()
    if not source_path.is_file():
        raise ValueError("Package model_path does not exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Model package output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"
    shutil.copyfile(source_path, model_path)
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Model package ONNX must have exactly one input and output")
    input_name = str(metadata.get("input", {}).get("name") or "")
    output_name = str(metadata.get("output", {}).get("name") or "")
    if session.get_inputs()[0].name != input_name or session.get_outputs()[0].name != output_name:
        raise ValueError("Model package ONNX names differ from metadata")
    release_id = metadata.get("release_id")
    if not isinstance(release_id, str) or not _IDENTIFIER.fullmatch(release_id):
        raise ValueError("Model package release_id is invalid")
    completed = {
        **metadata,
        "verification": {"golden_fixtures": _package_fixtures(session, input_name, output_name)},
        "model": {
            "filename": "model.onnx",
            "object_key": f"{MODEL_RELEASE_PREFIX}/{release_id}/model.onnx",
            "size_bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
            "content_type": "application/octet-stream",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_model_release(output_dir)
    return {
        "status": "PASS",
        "release_id": release_id,
        "output_dir": str(output_dir.resolve()),
        "model_sha256": completed["model"]["sha256"],
    }


def publish_model_release(
    release_dir: Path,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    with reporter.task("Validating the two-file ONNX model release"):
        metadata = validate_model_release(release_dir)
    with reporter.task("Checking Google Cloud sign-in"):
        account = ensure_gcloud_login(interactive=True)
    release_id = metadata["release_id"]
    path = release_dir.resolve() / "model.onnx"
    artifact = {
        "file_id": "model",
        **{
            key: metadata["model"][key]
            for key in ("object_key", "size_bytes", "sha256", "content_type")
        },
    }
    with reporter.task("Uploading model.onnx and writing metadata.json last"):
        completed = publish_model_artifacts(
            publication_kind="model",
            publication_id=release_id,
            marker=metadata,
            artifacts=[artifact],
            sources={"model": path},
        )
    expected = f"{MODEL_RELEASE_PREFIX}/{release_id}/metadata.json"
    if completed.get("marker_object") != expected:
        raise ValueError("Upload broker completed an unexpected model release")
    return {
        "status": "PASS",
        "account": account,
        "bucket": data_bucket(),
        "release_id": release_id,
        "metadata_object": expected,
        "model_sha256": metadata["model"]["sha256"],
    }


def verify_model_release(release_id: str) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(release_id):
        raise ValueError("Invalid model release ID")
    account = ensure_gcloud_login(interactive=False)
    bucket = data_bucket()
    key = f"{MODEL_RELEASE_PREFIX}/{release_id}/metadata.json"
    payload = _gcloud_cat(_object_uri(bucket, key), optional=False)
    assert payload is not None
    metadata = _read_json_bytes(payload, source=key)
    if (
        metadata.get("schema_version") != MODEL_RELEASE_SCHEMA
        or metadata.get("contract_version") != MODEL_RELEASE_CONTRACT_VERSION
        or metadata.get("release_id") != release_id
    ):
        raise ValueError("Remote model release metadata is invalid")
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "release_id": release_id,
        "model_sha256": metadata.get("model", {}).get("sha256"),
    }


def restore_model_release(release_id: str, *, expected_generation: int) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(release_id):
        raise ValueError("Invalid model release ID")
    ensure_gcloud_login(interactive=False)
    return restore_model_publication("model", release_id, expected_generation=expected_generation)
