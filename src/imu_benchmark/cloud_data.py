from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .contract import (
    ACTIVE_SCHEMA_VERSION,
    CONTRACT_VERSION,
    SNAPSHOT_VERSION,
    canonical_json_sha256,
    validate_snapshot_shape,
)
from .dataset import validate_hdf5_file

DEFAULT_BUCKET = "gs://soft3888-label"
BENCHMARK_PREFIX = "benchmark-datasets"
REMOTE_MANIFEST_SCHEMA = "imu_benchmark_dataset_manifest_v1"
CURRENT_SCHEMA = "imu_benchmark_current_v1"
BASE_MANIFEST_PATH = Path("configs/data/base_imu25_v1.json")
BASE_SPLITS_PATH = Path("configs/data/base_splits_v1.json")


def data_bucket() -> str:
    value = os.environ.get("IMU_BENCH_DATA_BUCKET", DEFAULT_BUCKET).rstrip("/")
    if not value.startswith("gs://") or value.count("/") != 2:
        raise ValueError("IMU_BENCH_DATA_BUCKET must be a bucket URI such as gs://name")
    return value


def _run_gcloud(
    *arguments: str,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    if shutil.which("gcloud") is None:
        raise RuntimeError("Google Cloud CLI is missing; run ./benchmark setup")
    try:
        return subprocess.run(
            ("gcloud", *arguments),
            check=check,
            capture_output=capture_output,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"gcloud failed: {message}") from error


def ensure_gcloud_login(*, interactive: bool) -> str:
    result = _run_gcloud(
        "auth",
        "list",
        "--filter=status:ACTIVE",
        "--format=value(account)",
    )
    accounts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not accounts and interactive:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "No active gcloud account in WSL2. Open an interactive WSL terminal "
                "and run ./benchmark data pull once to complete Google sign-in."
            )
        _run_gcloud("auth", "login", capture_output=False)
        return ensure_gcloud_login(interactive=False)
    if not accounts:
        raise RuntimeError("No active gcloud account; run gcloud auth login")
    return accounts[0]


def _object_uri(bucket: str, object_key: str) -> str:
    if not object_key or object_key.startswith("/") or ".." in Path(object_key).parts:
        raise ValueError(f"Invalid GCS object key: {object_key!r}")
    return f"{bucket}/{object_key}"


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON from {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object from {source}")
    return value


def _gcloud_cat(uri: str, *, optional: bool = False) -> bytes | None:
    result = _run_gcloud("storage", "cat", uri, check=False)
    if result.returncode != 0:
        if optional:
            return None
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Cannot read {uri}: {message}")
    return result.stdout.encode()


def _validate_current(payload: dict[str, Any], *, kind: str) -> None:
    required = {
        "schema_version",
        "kind",
        "snapshot_id",
        "manifest_object",
        "manifest_sha256",
        "updated_at_utc",
    }
    if set(payload) != required or payload.get("schema_version") != CURRENT_SCHEMA:
        raise ValueError(f"Invalid {kind} current pointer")
    if payload.get("kind") != kind:
        raise ValueError(f"Current pointer kind differs: expected {kind}")
    for name in ("snapshot_id", "manifest_object", "updated_at_utc"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"Current pointer has invalid {name}")
    digest = payload.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Current pointer has invalid manifest_sha256")


def _validate_remote_manifest(payload: dict[str, Any], *, expected_kind: str) -> None:
    if payload.get("schema_version") != REMOTE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported remote dataset manifest")
    if payload.get("kind") != expected_kind:
        raise ValueError("Remote manifest kind differs from current pointer")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Remote manifest uses a different benchmark contract")
    if not isinstance(payload.get("snapshot_id"), str) or not payload["snapshot_id"]:
        raise ValueError("Remote manifest is missing snapshot_id")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Remote manifest contains no files")
    expected_role = "cross_validation" if expected_kind == "base" else "training_only"
    ids: set[str] = set()
    names: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Remote manifest contains a non-object file entry")
        required = {
            "dataset_id",
            "object_key",
            "filename",
            "size_bytes",
            "sha256",
            "logical_content_sha256",
            "hdf5_schema_version",
            "sampling_rate_hz",
            "evaluation_role",
            "sequences",
            "rows",
            "annotations",
        }
        if not required.issubset(item):
            raise ValueError("Remote manifest contains an incomplete file entry")
        dataset_id = item["dataset_id"]
        filename = item["filename"]
        if (
            not isinstance(dataset_id, str)
            or dataset_id in ids
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".h5")
            or filename in names
        ):
            raise ValueError("Remote manifest contains a duplicate or unsafe identity")
        ids.add(dataset_id)
        names.add(filename)
        if item["hdf5_schema_version"] != "3.1.0":
            raise ValueError("Remote manifest does not contain HDF5 v3.1")
        if float(item["sampling_rate_hz"]) != 25.0:
            raise ValueError("Remote manifest does not contain 25 Hz data")
        if item["evaluation_role"] != expected_role:
            raise ValueError("Remote manifest contains an invalid evaluation role")


def _remote_snapshot(
    bucket: str,
    *,
    kind: str,
    current_object: str,
    optional: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    current_bytes = _gcloud_cat(_object_uri(bucket, current_object), optional=optional)
    if current_bytes is None:
        return None
    current = _read_json_bytes(current_bytes, source=current_object)
    _validate_current(current, kind=kind)
    manifest_bytes = _gcloud_cat(
        _object_uri(bucket, str(current["manifest_object"])),
        optional=False,
    )
    assert manifest_bytes is not None
    if _sha256_bytes(manifest_bytes) != current["manifest_sha256"]:
        raise ValueError(f"{kind} manifest SHA-256 differs from current pointer")
    manifest = _read_json_bytes(manifest_bytes, source=str(current["manifest_object"]))
    _validate_remote_manifest(manifest, expected_kind=kind)
    if manifest["snapshot_id"] != current["snapshot_id"]:
        raise ValueError(f"{kind} snapshot IDs differ between pointer and manifest")
    return current, manifest


def _validate_local_file(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"Local dataset size mismatch: {path}")
    if _sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Local dataset SHA-256 mismatch: {path}")
    observed = validate_hdf5_file(path)
    for name in (
        "dataset_id",
        "sequences",
        "rows",
        "annotations",
        "logical_content_sha256",
        "evaluation_role",
    ):
        if observed[name] != entry[name]:
            raise ValueError(f"Local HDF5 {name} mismatch: {path}")


def _install_snapshot(
    data_root: Path,
    bucket: str,
    *,
    kind: str,
    manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    snapshot_id = str(manifest["snapshot_id"])
    parent = data_root / kind
    final = parent / snapshot_id
    datasets = final / "datasets"
    if final.exists():
        for entry in manifest["files"]:
            _validate_local_file(datasets / str(entry["filename"]), entry)
    else:
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".{snapshot_id}.partial-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging_datasets = staging / "datasets"
        staging_datasets.mkdir(parents=True)
        try:
            for entry in manifest["files"]:
                destination = staging_datasets / str(entry["filename"])
                _run_gcloud(
                    "storage",
                    "cp",
                    _object_uri(bucket, str(entry["object_key"])),
                    str(destination),
                )
                _validate_local_file(destination, entry)
            (staging / "manifest.json").write_bytes(_json_bytes(manifest))
            os.replace(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    active_entries = []
    for entry in manifest["files"]:
        active_entries.append(
            {
                key: value
                for key, value in entry.items()
                if key not in {"object_key", "filename"}
            }
            | {"path": str(entry["filename"])}
        )
    return f"{kind}/{snapshot_id}/datasets", active_entries


def _load_splits(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / BASE_SPLITS_PATH
    payload = _read_json_bytes(path.read_bytes(), source=str(path))
    if payload.get("schema_version") != "imu_benchmark_split_set_v1":
        raise ValueError("Unsupported base split set")
    splits = payload.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("Base split set is empty")
    return splits


def pull_data(project_root: Path, data_root: Path) -> dict[str, Any]:
    account = ensure_gcloud_login(interactive=True)
    bucket = data_bucket()
    base_remote = _remote_snapshot(
        bucket,
        kind="base",
        current_object=f"{BENCHMARK_PREFIX}/base/current.json",
        optional=False,
    )
    assert base_remote is not None
    base_current, base_manifest = base_remote
    base_path, base_entries = _install_snapshot(
        data_root,
        bucket,
        kind="base",
        manifest=base_manifest,
    )
    team_remote = _remote_snapshot(
        bucket,
        kind="team",
        current_object=f"{BENCHMARK_PREFIX}/team/cw12eu/current.json",
        optional=True,
    )
    collections: dict[str, Any] = {
        "base": {
            "data_path": base_path,
            "splits": _load_splits(project_root),
            "datasets": base_entries,
        }
    }
    team_snapshot_id = None
    team_current = None
    if team_remote is not None:
        team_current, team_manifest = team_remote
        team_path, team_entries = _install_snapshot(
            data_root,
            bucket,
            kind="team",
            manifest=team_manifest,
        )
        collections["team"] = {
            "data_path": team_path,
            "fold_id": -1,
            "datasets": team_entries,
        }
        team_snapshot_id = team_manifest["snapshot_id"]
    active = {
        "schema_version": ACTIVE_SCHEMA_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "bucket": bucket,
        "base_snapshot_id": base_manifest["snapshot_id"],
        "team_snapshot_id": team_snapshot_id,
        "resolved": {
            "base_manifest_object": base_current["manifest_object"],
            "base_manifest_sha256": base_current["manifest_sha256"],
            "team_manifest_object": (
                None if team_current is None else team_current["manifest_object"]
            ),
            "team_manifest_sha256": (
                None if team_current is None else team_current["manifest_sha256"]
            ),
        },
        "collections": collections,
    }
    contract = json.loads(
        (project_root / "configs/contracts/imu_benchmark_contract_v2.json").read_text(
            encoding="utf-8"
        )
    )
    validate_snapshot_shape(active, contract)
    data_root.mkdir(parents=True, exist_ok=True)
    destination = data_root / "active.json"
    temporary = data_root / f".active.json.tmp-{os.getpid()}"
    temporary.write_bytes(_json_bytes(active))
    os.replace(temporary, destination)
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "base_snapshot_id": active["base_snapshot_id"],
        "team_snapshot_id": active["team_snapshot_id"],
        "active_manifest": str(destination),
        "snapshot_sha256": canonical_json_sha256(active),
    }


def data_status(project_root: Path, data_root: Path) -> dict[str, Any]:
    account = ensure_gcloud_login(interactive=False)
    bucket = data_bucket()
    base = _remote_snapshot(
        bucket,
        kind="base",
        current_object=f"{BENCHMARK_PREFIX}/base/current.json",
        optional=False,
    )
    assert base is not None
    team = _remote_snapshot(
        bucket,
        kind="team",
        current_object=f"{BENCHMARK_PREFIX}/team/cw12eu/current.json",
        optional=True,
    )
    active_path = data_root / "active.json"
    active = (
        _read_json_bytes(active_path.read_bytes(), source=str(active_path))
        if active_path.is_file()
        else None
    )
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "local_active": active,
        "remote_base_snapshot_id": base[1]["snapshot_id"],
        "remote_team_snapshot_id": None if team is None else team[1]["snapshot_id"],
        "update_available": active is None
        or active.get("base_snapshot_id") != base[1]["snapshot_id"]
        or active.get("team_snapshot_id")
        != (None if team is None else team[1]["snapshot_id"]),
        "project_root": str(project_root),
    }


def _upload_file(source: Path, uri: str, *, immutable: bool) -> None:
    arguments = ["storage", "cp"]
    if immutable:
        arguments.append("--no-clobber")
    arguments.extend((str(source), uri))
    _run_gcloud(*arguments)


def publish_base(project_root: Path, source_dir: Path) -> dict[str, Any]:
    account = ensure_gcloud_login(interactive=True)
    bucket = data_bucket()
    manifest_path = project_root / BASE_MANIFEST_PATH
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_json_bytes(manifest_bytes, source=str(manifest_path))
    _validate_remote_manifest(manifest, expected_kind="base")
    for entry in manifest["files"]:
        source = source_dir / str(entry["filename"])
        _validate_local_file(source, entry)
        _upload_file(
            source,
            _object_uri(bucket, str(entry["object_key"])),
            immutable=True,
        )
    manifest_object = (
        f"{BENCHMARK_PREFIX}/base/{manifest['snapshot_id']}/manifest.json"
    )
    _upload_file(
        manifest_path,
        _object_uri(bucket, manifest_object),
        immutable=True,
    )
    current = {
        "schema_version": CURRENT_SCHEMA,
        "kind": "base",
        "snapshot_id": manifest["snapshot_id"],
        "manifest_object": manifest_object,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "updated_at_utc": manifest["created_at_utc"],
    }
    current_object = f"{BENCHMARK_PREFIX}/base/current.json"
    existing_bytes = _gcloud_cat(_object_uri(bucket, current_object), optional=True)
    if existing_bytes is not None:
        existing = _read_json_bytes(existing_bytes, source=current_object)
        _validate_current(existing, kind="base")
        if existing != current:
            raise RuntimeError(
                "The base current pointer already refers to a different immutable snapshot"
            )
    else:
        with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as temporary:
            temporary.write(_json_bytes(current))
            current_path = Path(temporary.name)
        try:
            _upload_file(
                current_path,
                _object_uri(bucket, current_object),
                immutable=True,
            )
        finally:
            current_path.unlink(missing_ok=True)
    remote = _remote_snapshot(
        bucket,
        kind="base",
        current_object=current_object,
        optional=False,
    )
    assert remote is not None
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "snapshot_id": manifest["snapshot_id"],
        "files": len(manifest["files"]),
        "manifest_sha256": current["manifest_sha256"],
    }
