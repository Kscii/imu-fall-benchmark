"""Publish a read-only ONNX experiment catalog without mutating result evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cloud_data import (
    _gcloud_cat,
    _object_uri,
    _read_json_bytes,
    _sha256_file,
    data_bucket,
    ensure_gcloud_login,
)
from .cloud_results import RESULT_PREFIXES, _remote_manifest, _run_dir, _validate_run
from .model_broker import publish_model_artifacts, restore_model_publication
from .model_catalog import (
    EXPERIMENT_CATALOG_CONTRACT_VERSION,
    EXPERIMENT_CATALOG_SCHEMA,
    build_experiment_catalog,
)
from .progress import NullProgressReporter, ProgressReporter

EXPERIMENT_CATALOG_PREFIX = "benchmark-model-catalog/experiments"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _result_prefix(manifest: dict[str, Any]) -> str:
    evidence_level = manifest.get("evidence_level", "formal_cv")
    try:
        return RESULT_PREFIXES[str(evidence_level)]
    except KeyError as error:
        raise ValueError("Remote result has an unsupported evidence level") from error


def _metadata(
    run_dir: Path,
    run_manifest: dict[str, Any],
    result_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    publication_id = str(run_manifest["run_id"])
    prefix = f"{EXPERIMENT_CATALOG_PREFIX}/{publication_id}"
    catalog = build_experiment_catalog(run_dir, run_manifest)
    uploads: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    artifacts: list[dict[str, Any]] = []
    for artifact in catalog["artifacts"]:
        artifact_id = artifact["artifact_id"]
        path = run_dir / artifact["source"]["onnx_run_path"]
        descriptor = {
            "filename": f"{artifact_id}.onnx",
            "object_key": f"{prefix}/onnx/{artifact_id}.onnx",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "content_type": "application/octet-stream",
        }
        if descriptor["sha256"] != artifact["source"]["onnx_sha256"]:
            raise ValueError(f"ONNX catalog descriptor differs: {artifact_id}")
        file_id = f"onnx-{artifact_id}"
        uploads.append(
            {
                "file_id": file_id,
                **{
                    key: descriptor[key]
                    for key in ("object_key", "size_bytes", "sha256", "content_type")
                },
            }
        )
        sources[file_id] = path
        artifacts.append({**artifact, "onnx": descriptor})

    result_prefix = _result_prefix(result_manifest)
    result_id = str(result_manifest["run_id"])
    result_manifest_key = f"{result_prefix}/{result_id}/manifest.json"
    result_evidence = {
        "schema_version": result_manifest["schema_version"],
        "manifest": {
            "filename": "manifest.json",
            "object_key": result_manifest_key,
            "size_bytes": len(_json_bytes(result_manifest)),
            "sha256": hashlib.sha256(_json_bytes(result_manifest)).hexdigest(),
            "content_type": "application/json",
        },
        "bundle": {
            **result_manifest["bundle"],
            "object_key": f"{result_prefix}/{result_id}/run.tar.gz",
            "content_type": "application/gzip",
        },
    }
    metadata = {
        "schema_version": EXPERIMENT_CATALOG_SCHEMA,
        "contract_version": EXPERIMENT_CATALOG_CONTRACT_VERSION,
        "publication_id": publication_id,
        "run_id": result_id,
        "experiment_id": run_manifest["experiment_id"],
        "evidence_level": catalog["evidence_level"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": run_manifest["source"],
        "data": {
            "base_snapshot_id": run_manifest["base_snapshot_id"],
            "snapshot_sha256": run_manifest["snapshot_sha256"],
            "data_view_id": run_manifest.get("data_view_id"),
        },
        "evaluation_fingerprint": catalog["evaluation_fingerprint"],
        "scheduled_jobs": run_manifest["scheduled_jobs"],
        "methods": catalog["methods"],
        "artifacts": artifacts,
        "result_evidence": result_evidence,
        "known_limitations": list(run_manifest.get("known_limitations") or []),
    }
    return metadata, uploads, sources


def publish_experiment_catalog(
    runs_root: Path,
    run_id: str,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    run_dir = _run_dir(runs_root, run_id)
    with reporter.task("Validating the completed ONNX experiment"):
        run_manifest, _entries = _validate_run(run_dir)
    with reporter.task("Checking Google Cloud sign-in"):
        account = ensure_gcloud_login(interactive=True)
    bucket = data_bucket()
    with reporter.task("Verifying the immutable benchmark result evidence"):
        result_manifest = _remote_manifest(bucket, run_id)
    with reporter.task("Building the independent experiment catalog"):
        metadata, uploads, sources = _metadata(
            run_dir, run_manifest, result_manifest
        )
    with reporter.task("Uploading ONNX files and writing metadata last"):
        completed = publish_model_artifacts(
            publication_kind="experiment",
            publication_id=run_id,
            marker=metadata,
            artifacts=uploads,
            sources=sources,
        )
    expected = f"{EXPERIMENT_CATALOG_PREFIX}/{run_id}/metadata.json"
    if completed.get("marker_object") != expected:
        raise ValueError("Upload broker completed an unexpected experiment catalog")
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "publication_id": run_id,
        "metadata_object": expected,
        "onnx_artifacts": len(uploads),
    }


def verify_experiment_catalog(publication_id: str) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(publication_id):
        raise ValueError("Invalid experiment publication ID")
    account = ensure_gcloud_login(interactive=False)
    bucket = data_bucket()
    key = f"{EXPERIMENT_CATALOG_PREFIX}/{publication_id}/metadata.json"
    payload = _gcloud_cat(_object_uri(bucket, key), optional=False)
    assert payload is not None
    metadata = _read_json_bytes(payload, source=key)
    if (
        metadata.get("schema_version") != EXPERIMENT_CATALOG_SCHEMA
        or metadata.get("contract_version") != EXPERIMENT_CATALOG_CONTRACT_VERSION
        or metadata.get("publication_id") != publication_id
    ):
        raise ValueError("Remote experiment catalog metadata is invalid")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Remote experiment catalog has no ONNX artifacts")
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "publication_id": publication_id,
        "onnx_artifacts": len(artifacts),
    }


def restore_experiment_catalog(
    publication_id: str, *, expected_generation: int
) -> dict[str, Any]:
    if not _IDENTIFIER.fullmatch(publication_id):
        raise ValueError("Invalid experiment publication ID")
    ensure_gcloud_login(interactive=False)
    return restore_model_publication(
        "experiment", publication_id, expected_generation=expected_generation
    )
