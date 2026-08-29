"""Validate and publish immutable two-file ONNX model releases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

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

MODEL_RELEASE_SCHEMA = "imu_model_release_v0"
MODEL_RELEASE_CONTRACT_VERSION = "0.1.0"
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
        "source",
        "data",
        "input",
        "output",
        "preprocessing",
        "decision",
        "metrics",
        "validation",
        "known_limitations",
        "model",
    }
    if set(metadata) != required:
        raise ValueError("Model release metadata fields differ from the contract")
    if (
        metadata.get("schema_version") != MODEL_RELEASE_SCHEMA
        or metadata.get("contract_version") != MODEL_RELEASE_CONTRACT_VERSION
    ):
        raise ValueError("Model release metadata schema is invalid")
    for name in ("release_id", "model_code"):
        value = metadata.get(name)
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"Model release {name} is invalid")
    if not isinstance(metadata.get("name"), str) or not metadata["name"].strip():
        raise ValueError("Model release name is missing")
    for name in ("source", "data", "input", "output", "preprocessing", "metrics"):
        _nonempty_object(metadata.get(name), name)
    source = metadata["source"]
    if source.get("dirty") is not False or not source.get("commit"):
        raise ValueError("Model release requires a clean source commit")
    output = metadata["output"]
    if output.get("semantic") != "fall_score" or output.get("dtype") != "float32":
        raise ValueError("Model release output contract is invalid")
    decision = _nonempty_object(metadata.get("decision"), "decision")
    threshold = _nonempty_object(decision.get("score_threshold"), "score threshold")
    if (
        not isinstance(threshold.get("value"), (int, float))
        or threshold.get("comparison") != ">="
        or decision.get("anchor") != "window_end"
    ):
        raise ValueError("Model release score threshold is invalid")
    trigger = _nonempty_object(decision.get("trigger_policy"), "trigger policy")
    trigger_required = {
        "policy_id",
        "required_positive_windows",
        "lookback_windows",
        "consecutive",
        "cooldown_seconds",
    }
    if not trigger_required.issubset(trigger):
        raise ValueError("Model release trigger policy is incomplete")
    validation = _nonempty_object(metadata.get("validation"), "validation")
    if (
        validation.get("onnx_checker") != "PASS"
        or validation.get("python_onnxruntime_parity") != "PASS"
        or validation.get("external_runtime") not in {"PASS", "not_tested"}
        or validation.get("device_replay") not in {"PASS", "not_tested"}
    ):
        raise ValueError("Model release validation status is invalid")
    if not isinstance(metadata.get("known_limitations"), list):
        raise ValueError("Model release known_limitations is invalid")
    descriptor = _nonempty_object(metadata.get("model"), "model descriptor")
    expected_key = (
        f"{MODEL_RELEASE_PREFIX}/{metadata['release_id']}/model.onnx"
    )
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
    artifact = {"file_id": "model", **metadata["model"]}
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


def restore_model_release(
    release_id: str, *, expected_generation: int
) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(release_id):
        raise ValueError("Invalid model release ID")
    ensure_gcloud_login(interactive=False)
    return restore_model_publication(
        "model", release_id, expected_generation=expected_generation
    )
