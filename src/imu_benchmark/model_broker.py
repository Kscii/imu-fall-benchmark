"""Publish model evidence through the constrained team upload broker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from .cloud_data import _run_gcloud

DEFAULT_BROKER_URL = "https://upload.imu.kscii.tech"


def broker_url() -> str:
    value = os.environ.get("IMU_BENCH_UPLOAD_BROKER_URL", DEFAULT_BROKER_URL).rstrip("/")
    if not value.startswith("https://") and not value.startswith("http://127.0.0.1"):
        raise ValueError("IMU_BENCH_UPLOAD_BROKER_URL must use HTTPS or loopback HTTP")
    return value


def identity_token() -> str:
    result = _run_gcloud("auth", "print-identity-token")
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("gcloud did not return a Google identity token")
    return token


def _json_response(response: requests.Response, *, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not response.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        safe_detail = str(detail or "server rejected the request")[:500]
        raise RuntimeError(f"{operation} failed (HTTP {response.status_code}): {safe_detail}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{operation} returned a non-object response")
    return payload


def _put_resumable(session_url: str, source: Path, content_type: str) -> None:
    size = source.stat().st_size
    with source.open("rb") as stream:
        response = requests.put(
            session_url,
            data=stream,
            headers={
                "Content-Length": str(size),
                "Content-Type": content_type,
            },
            timeout=(30, 900),
        )
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"resumable upload failed (HTTP {response.status_code}); retry publication"
        )


def publish_model_artifacts(
    *,
    publication_kind: str,
    publication_id: str,
    marker: dict[str, Any],
    artifacts: list[dict[str, Any]],
    sources: dict[str, Path],
) -> dict[str, Any]:
    if {item["file_id"] for item in artifacts} != set(sources):
        raise ValueError("Model publication source files differ from artifact descriptors")
    token = identity_token()
    headers = {"Authorization": f"Bearer {token}"}
    start = requests.post(
        f"{broker_url()}/v1/model-uploads",
        json={
            "publication_kind": publication_kind,
            "publication_id": publication_id,
            "marker": marker,
            "artifacts": artifacts,
        },
        headers=headers,
        timeout=60,
    )
    plan = _json_response(start, operation="start model publication")
    sessions = plan.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("Upload broker returned an invalid session list")
    descriptors = {item["file_id"]: item for item in artifacts}
    for session in sessions:
        if not isinstance(session, dict) or session.get("file_id") not in sources:
            raise RuntimeError("Upload broker returned an unexpected artifact session")
        if session.get("already_present"):
            continue
        session_url = session.get("session_url")
        if not isinstance(session_url, str) or not session_url.startswith("https://"):
            raise RuntimeError("Upload broker did not return a secure resumable session")
        file_id = session["file_id"]
        _put_resumable(
            session_url,
            sources[file_id],
            descriptors[file_id]["content_type"],
        )
    complete = requests.post(
        f"{broker_url()}/v1/model-uploads/complete",
        json={"upload_id": plan.get("upload_id")},
        headers=headers,
        timeout=900,
    )
    return _json_response(complete, operation="complete model publication")


def restore_model_publication(
    publication_kind: str,
    publication_id: str,
    *,
    expected_generation: int,
) -> dict[str, Any]:
    response = requests.post(
        f"{broker_url()}/v1/model-publications/{publication_kind}/{publication_id}/restore",
        json={"expected_generation": expected_generation},
        headers={"Authorization": f"Bearer {identity_token()}"},
        timeout=60,
    )
    return _json_response(response, operation="restore model publication")
