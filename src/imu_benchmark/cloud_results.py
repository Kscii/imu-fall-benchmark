from __future__ import annotations

import gzip
import json
import os
import re
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from .cloud_data import (
    _gcloud_cat,
    _object_uri,
    _read_json_bytes,
    _run_gcloud,
    _sha256_file,
    data_bucket,
    ensure_gcloud_login,
)
from .model_broker import publish_model_artifacts, restore_model_publication
from .model_catalog import (
    EXPERIMENT_PUBLICATION_SCHEMA,
    build_experiment_catalog,
    evidence_profile,
)
from .progress import NullProgressReporter, ProgressReporter

RESULT_MANIFEST_SCHEMA = EXPERIMENT_PUBLICATION_SCHEMA
RESULT_PREFIXES = {
    "formal_cv": "benchmark-results/temporal-core",
    "engineering": "benchmark-results/engineering",
}
QUICK_FILES = (
    "report.md",
    "aggregate_metrics.csv",
    "paired_comparisons.csv",
    "statistical_manifest.json",
    "performance.json",
    "onnx_parity.csv",
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _run_dir(runs_root: Path, run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid result run ID")
    root = runs_root.resolve()
    path = (root / run_id).resolve()
    if path.parent != root:
        raise ValueError("Result run ID resolves outside the runs directory")
    return path


def _file_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Result run contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Result run contains a non-regular file: {path}")
        entries.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not entries:
        raise ValueError("Result run contains no files")
    return entries


def _validate_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("Run manifest ID differs from its directory")
    _evidence_level, expected_jobs, _expected_folds = evidence_profile(manifest)
    if manifest.get("status") != "PASS":
        raise ValueError("Only PASS runs may be published")
    if (
        manifest.get("scheduled_jobs") != expected_jobs
        or manifest.get("completed_jobs") != expected_jobs
    ):
        raise ValueError(
            f"A published ONNX run must contain {expected_jobs}/{expected_jobs} jobs"
        )
    if manifest.get("failures"):
        raise ValueError("A run with failures cannot be published")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("kind") not in {"git", "snapshot"}
        or source.get("dirty") is not False
        or not source.get("commit")
    ):
        raise ValueError("Published results require a clean, identified source commit")
    if manifest.get("base_snapshot_id") != "imu_25hz_snapshot_v2":
        raise ValueError("Published results must use imu_25hz_snapshot_v2")
    for key in ("snapshot_sha256", "resolved_config_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Run manifest has an invalid {key}")
    if expected_jobs == 65:
        statistics = manifest.get("statistical_analysis")
        if not isinstance(statistics, dict) or statistics.get("status") != "PASS":
            raise ValueError("Published formal results require PASS statistical outputs")
    job_files = sorted((run_dir / "jobs").glob("*.npz"))
    if len(job_files) != expected_jobs:
        raise ValueError(
            f"Published results require exactly {expected_jobs} job checkpoints"
        )
    onnx_files = sorted((run_dir / "models").glob("*/model.onnx"))
    if len(onnx_files) != expected_jobs:
        raise ValueError(
            f"The published run requires exactly {expected_jobs} ONNX models"
        )
    parity_path = run_dir / "onnx_parity.csv"
    expected_parity_lines = expected_jobs * 2 + 1
    if (
        not parity_path.is_file()
        or len(parity_path.read_text(encoding="utf-8").splitlines())
        != expected_parity_lines
    ):
        raise ValueError(
            "The published run requires complete validation and test ONNX parity"
        )
    # Build once during validation so malformed scientific metadata never reaches upload.
    build_experiment_catalog(run_dir, manifest)
    return manifest, _file_entries(run_dir)


def _tar_info(run_id: str, entry: dict[str, Any]) -> tarfile.TarInfo:
    info = tarfile.TarInfo(f"{run_id}/{entry['path']}")
    info.size = int(entry["size_bytes"])
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_bundle(
    run_dir: Path,
    run_id: str,
    entries: list[dict[str, Any]],
    destination: Path,
) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for entry in entries:
                    source = run_dir / str(entry["path"])
                    with source.open("rb") as stream:
                        archive.addfile(_tar_info(run_id, entry), stream)


def _publication_manifest(
    run_manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    bundle_path: Path,
    catalog: dict[str, Any],
    direct_files: list[dict[str, Any]],
) -> dict[str, Any]:
    quick = [entry for entry in entries if entry["path"] in QUICK_FILES]
    return {
        "schema_version": RESULT_MANIFEST_SCHEMA,
        "run_id": run_manifest["run_id"],
        "experiment_id": run_manifest["experiment_id"],
        "evidence_level": catalog["evidence_level"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scheduled_jobs": run_manifest["scheduled_jobs"],
        "source": run_manifest["source"],
        "base_snapshot_id": run_manifest["base_snapshot_id"],
        "snapshot_sha256": run_manifest["snapshot_sha256"],
        "resolved_config_sha256": run_manifest["resolved_config_sha256"],
        "evaluation_fingerprint": catalog["evaluation_fingerprint"],
        "data_quality_status": run_manifest.get("data_quality_status"),
        "known_limitations": list(run_manifest.get("known_limitations") or []),
        "bundle": {
            "filename": "run.tar.gz",
            "size_bytes": bundle_path.stat().st_size,
            "sha256": _sha256_file(bundle_path),
        },
        "files": entries,
        "quick_files": quick,
        "direct_files": direct_files,
        "methods": catalog["methods"],
        "artifacts": catalog["artifacts"],
    }


def _validate_publication_manifest(payload: dict[str, Any], *, run_id: str) -> None:
    expected = {
        "schema_version",
        "run_id",
        "experiment_id",
        "evidence_level",
        "created_at_utc",
        "scheduled_jobs",
        "source",
        "base_snapshot_id",
        "snapshot_sha256",
        "resolved_config_sha256",
        "evaluation_fingerprint",
        "data_quality_status",
        "known_limitations",
        "bundle",
        "files",
        "quick_files",
        "direct_files",
        "methods",
        "artifacts",
    }
    if set(payload) != expected or payload.get("schema_version") != RESULT_MANIFEST_SCHEMA:
        raise ValueError("Invalid result publication manifest")
    if payload.get("run_id") != run_id:
        raise ValueError("Result publication manifest describes a different run")
    if payload.get("evidence_level") not in RESULT_PREFIXES:
        raise ValueError("Result publication manifest has an invalid evidence level")
    if payload.get("scheduled_jobs") not in {7, 65}:
        raise ValueError("Result publication manifest has an invalid job count")
    bundle = payload.get("bundle")
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"filename", "size_bytes", "sha256"}
        or bundle.get("filename") != "run.tar.gz"
        or not isinstance(bundle.get("size_bytes"), int)
        or bundle["size_bytes"] <= 0
        or not isinstance(bundle.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", bundle["sha256"])
    ):
        raise ValueError("Invalid result bundle metadata")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Result publication manifest contains no files")
    paths = set()
    for entry in files:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "size_bytes", "sha256"}
            or not isinstance(entry["path"], str)
            or not entry["path"]
            or Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or entry["path"] in paths
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("Invalid result publication file entry")
        paths.add(entry["path"])
    direct_files = payload.get("direct_files")
    if not isinstance(direct_files, list) or not direct_files:
        raise ValueError("Result publication manifest contains no direct downloads")
    file_ids: set[str] = set()
    for entry in direct_files:
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "file_id",
                "artifact_id",
                "role",
                "filename",
                "object_key",
                "content_type",
                "size_bytes",
                "sha256",
            }
            or not isinstance(entry.get("file_id"), str)
            or not _RUN_ID.fullmatch(entry["file_id"])
            or entry["file_id"] in file_ids
            or not isinstance(entry.get("object_key"), str)
            or ".." in Path(entry["object_key"]).parts
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("Invalid direct result file entry")
        file_ids.add(entry["file_id"])
    if not isinstance(payload.get("methods"), list) or not payload["methods"]:
        raise ValueError("Result publication manifest contains no method summaries")
    if (
        not isinstance(payload.get("artifacts"), list)
        or len(payload["artifacts"]) != payload["scheduled_jobs"]
    ):
        raise ValueError("Result publication manifest contains incomplete ONNX artifacts")


def _prepare_direct_files(
    run_dir: Path,
    prefix: str,
    catalog: dict[str, Any],
    temporary_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    direct: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for artifact in catalog["artifacts"]:
        artifact_id = artifact["artifact_id"]
        onnx_path = run_dir / artifact["source"]["onnx_run_path"]
        onnx_id = f"onnx-{artifact_id}"
        metadata_id = f"metadata-{artifact_id}"
        artifact["files"] = {
            "onnx_file_id": onnx_id,
            "metadata_file_id": metadata_id,
        }
        metadata_path = temporary_root / "metadata" / f"{artifact_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_bytes(_json_bytes(artifact))
        for file_id, role, filename, path, content_type in (
            (onnx_id, "onnx", "model.onnx", onnx_path, "application/octet-stream"),
            (
                metadata_id,
                "metadata",
                "metadata.json",
                metadata_path,
                "application/json",
            ),
        ):
            object_key = f"{prefix}/models/{artifact_id}/{filename}"
            direct.append(
                {
                    "file_id": file_id,
                    "artifact_id": artifact_id,
                    "role": role,
                    "filename": filename,
                    "object_key": object_key,
                    "content_type": content_type,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
            paths[file_id] = path
    return direct, paths


def _quick_content_type(filename: str) -> str:
    if filename.endswith(".json"):
        return "application/json"
    if filename.endswith(".csv"):
        return "text/csv"
    if filename.endswith(".md"):
        return "text/markdown"
    return "application/octet-stream"


def _broker_artifacts(
    publication: dict[str, Any],
    bundle_path: Path,
    run_dir: Path,
    direct_paths: dict[str, Path],
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    artifacts = [
        {
            "file_id": "bundle",
            "object_key": f"{prefix}/run.tar.gz",
            "size_bytes": publication["bundle"]["size_bytes"],
            "sha256": publication["bundle"]["sha256"],
            "content_type": "application/gzip",
        }
    ]
    sources = {"bundle": bundle_path}
    for entry in publication["quick_files"]:
        filename = entry["path"]
        file_id = f"quick-{filename.replace('.', '-')}"
        artifacts.append(
            {
                "file_id": file_id,
                "object_key": f"{prefix}/files/{filename}",
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
                "content_type": _quick_content_type(filename),
            }
        )
        sources[file_id] = run_dir / filename
    for entry in publication["direct_files"]:
        artifacts.append(
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
        )
        sources[entry["file_id"]] = direct_paths[entry["file_id"]]
    return artifacts, sources


def _remote_manifest(bucket: str, run_id: str) -> dict[str, Any]:
    for prefix in RESULT_PREFIXES.values():
        uri = _object_uri(bucket, f"{prefix}/{run_id}/manifest.json")
        payload = _gcloud_cat(uri, optional=True)
        if payload is None:
            continue
        manifest = _read_json_bytes(payload, source=uri)
        _validate_publication_manifest(manifest, run_id=run_id)
        return manifest
    raise FileNotFoundError(f"Published result not found: {run_id}")


def _download(uri: str, destination: Path) -> None:
    _run_gcloud("storage", "cp", uri, str(destination))


def _validate_files(root: Path, entries: list[dict[str, Any]]) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = {str(entry["path"]) for entry in entries}
    if actual != expected:
        raise ValueError("Result directory files differ from the publication manifest")
    for entry in entries:
        path = root / str(entry["path"])
        if path.is_symlink() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Result file metadata differs: {entry['path']}")
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Result file SHA-256 differs: {entry['path']}")


def _extract_bundle(
    bundle_path: Path,
    destination_root: Path,
    run_id: str,
    entries: list[dict[str, Any]],
) -> Path:
    expected = {f"{run_id}/{entry['path']}" for entry in entries}
    with tarfile.open(bundle_path, "r:gz") as archive:
        members = archive.getmembers()
        if {member.name for member in members} != expected:
            raise ValueError("Result bundle members differ from its manifest")
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"Unsafe result bundle member: {member.name}")
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe result bundle path: {member.name}")
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source: BinaryIO | None = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read result bundle member: {member.name}")
            with source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    extracted = destination_root / run_id
    _validate_files(extracted, entries)
    return extracted


def publish_result(
    runs_root: Path,
    run_id: str,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    run_dir = _run_dir(runs_root, run_id)
    with reporter.task("Validating the completed temporal-core run"):
        run_manifest, entries = _validate_run(run_dir)
    with reporter.task("Checking Google Cloud sign-in"):
        account = ensure_gcloud_login(interactive=True)
    bucket = data_bucket()
    catalog = build_experiment_catalog(run_dir, run_manifest)
    prefix = f"{RESULT_PREFIXES[catalog['evidence_level']]}/{run_id}"
    with tempfile.TemporaryDirectory(prefix="imu-result-") as temporary:
        temporary_root = Path(temporary)
        bundle_path = temporary_root / "run.tar.gz"
        with reporter.task("Creating the immutable result bundle"):
            _write_bundle(run_dir, run_id, entries, bundle_path)
            direct_files, direct_paths = _prepare_direct_files(
                run_dir, prefix, catalog, temporary_root
            )
            publication = _publication_manifest(
                run_manifest,
                entries,
                bundle_path,
                catalog,
                direct_files,
            )
            manifest_bytes = _json_bytes(publication)
            artifacts, sources = _broker_artifacts(
                publication,
                bundle_path,
                run_dir,
                direct_paths,
                prefix,
            )
        with reporter.task("Uploading through the constrained team broker"):
            completed = publish_model_artifacts(
                publication_kind="experiment",
                publication_id=run_id,
                marker=publication,
                artifacts=artifacts,
                sources=sources,
            )
        if completed.get("marker_object") != f"{prefix}/manifest.json":
            raise ValueError("Upload broker completed an unexpected result publication")
        if not manifest_bytes:
            raise AssertionError("Result publication marker is empty")
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "run_id": run_id,
        "evidence_level": publication["evidence_level"],
        "manifest_object": f"{prefix}/manifest.json",
        "bundle_sha256": publication["bundle"]["sha256"],
        "files": len(entries),
    }


def verify_result(
    runs_root: Path,
    run_id: str,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    with reporter.task("Checking Google Cloud sign-in"):
        account = ensure_gcloud_login(interactive=False)
    bucket = data_bucket()
    with reporter.task("Reading the immutable result manifest"):
        manifest = _remote_manifest(bucket, run_id)
    prefix = f"{RESULT_PREFIXES[manifest['evidence_level']]}/{run_id}"
    with tempfile.TemporaryDirectory(prefix="imu-result-verify-") as temporary:
        bundle_path = Path(temporary) / "run.tar.gz"
        with reporter.task("Downloading and verifying the result bundle"):
            _download(_object_uri(bucket, f"{prefix}/run.tar.gz"), bundle_path)
            if bundle_path.stat().st_size != manifest["bundle"]["size_bytes"]:
                raise ValueError("Downloaded result bundle size differs")
            if _sha256_file(bundle_path) != manifest["bundle"]["sha256"]:
                raise ValueError("Downloaded result bundle SHA-256 differs")
    local_dir = _run_dir(runs_root, run_id)
    local_validated = False
    if local_dir.is_dir():
        with reporter.task("Verifying the local run against the publication"):
            _validate_files(local_dir, manifest["files"])
        local_validated = True
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "run_id": run_id,
        "bundle_sha256": manifest["bundle"]["sha256"],
        "files": len(manifest["files"]),
        "local_validated": local_validated,
    }


def pull_result(
    runs_root: Path,
    run_id: str,
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    with reporter.task("Checking Google Cloud sign-in"):
        account = ensure_gcloud_login(interactive=True)
    bucket = data_bucket()
    with reporter.task("Reading the immutable result manifest"):
        manifest = _remote_manifest(bucket, run_id)
    runs_root.mkdir(parents=True, exist_ok=True)
    destination = _run_dir(runs_root, run_id)
    if destination.exists():
        with reporter.task("Verifying the existing local result"):
            _validate_files(destination, manifest["files"])
        return {
            "status": "PASS",
            "account": account,
            "bucket": bucket,
            "run_id": run_id,
            "path": str(destination),
            "reused": True,
        }
    prefix = f"{RESULT_PREFIXES[manifest['evidence_level']]}/{run_id}"
    with tempfile.TemporaryDirectory(prefix="imu-result-pull-", dir=runs_root) as temporary:
        temporary_root = Path(temporary)
        bundle_path = temporary_root / "run.tar.gz"
        with reporter.task("Downloading the immutable result bundle"):
            _download(_object_uri(bucket, f"{prefix}/run.tar.gz"), bundle_path)
            if _sha256_file(bundle_path) != manifest["bundle"]["sha256"]:
                raise ValueError("Downloaded result bundle SHA-256 differs")
        with reporter.task("Extracting and validating the result bundle"):
            extracted = _extract_bundle(
                bundle_path,
                temporary_root / "extracted",
                run_id,
                manifest["files"],
            )
            os.replace(extracted, destination)
    return {
        "status": "PASS",
        "account": account,
        "bucket": bucket,
        "run_id": run_id,
        "path": str(destination),
        "reused": False,
    }


def restore_published_result(
    run_id: str, *, expected_generation: int
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid result run ID")
    ensure_gcloud_login(interactive=False)
    return restore_model_publication(
        "experiment", run_id, expected_generation=expected_generation
    )
