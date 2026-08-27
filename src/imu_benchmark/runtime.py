from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .device import CudaUnavailable

COMPUTE_COMMANDS = frozenset(
    {
        "doctor",
        "prepare",
        "smoke",
        "reproduce",
        "run",
        "kfall-prepare",
        "kfall-smoke",
        "kfall-evaluate",
    }
)
FORMAL_COMMANDS = frozenset({"reproduce", "kfall-evaluate"})
SOURCE_MANIFEST = ".imu-source.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class WorkPaths:
    root: Path
    cache: Path
    runs: Path

    def to_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in asdict(self).items()}


def is_wsl2() -> bool:
    return "microsoft-standard-wsl2" in platform.release().lower()


def repository_is_on_linux_filesystem(project_root: Path) -> bool:
    resolved = project_root.resolve()
    return resolved != Path("/mnt") and Path("/mnt") not in resolved.parents


def require_compute_runtime(command: str, project_root: Path) -> None:
    if command not in COMPUTE_COMMANDS:
        return
    if not is_wsl2():
        raise CudaUnavailable(f"{command} requires WSL2 with NVIDIA CUDA")
    if not repository_is_on_linux_filesystem(project_root):
        raise CudaUnavailable(
            "The repository must be stored in the WSL Linux filesystem, not under /mnt/"
        )


def resolve_work_paths() -> WorkPaths:
    raw = os.environ.get("IMU_BENCH_WORK_ROOT", "~/imu-fall-work")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        raise ValueError("IMU_BENCH_WORK_ROOT must be an absolute path")
    root = root.resolve()
    return WorkPaths(root=root, cache=root / "cache", runs=root / "runs")


def _unknown_source(*warnings: str) -> tuple[dict[str, Any], list[str]]:
    values = list(warnings) or ["source_unknown"]
    return (
        {
            "kind": "unknown",
            "commit": None,
            "dirty": None,
            "snapshot_sha256": None,
        },
        values,
    )


def _snapshot_source(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return _unknown_source("source_manifest_invalid", "source_unknown")
    commit = payload.get("commit")
    dirty = payload.get("dirty")
    digest = payload.get("snapshot_sha256")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "snapshot"
        or (commit is not None and not isinstance(commit, str))
        or not isinstance(dirty, bool)
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
    ):
        return _unknown_source("source_manifest_invalid", "source_unknown")
    source = {
        "kind": "snapshot",
        "commit": commit,
        "dirty": dirty,
        "snapshot_sha256": digest,
    }
    return source, ["source_tree_dirty"] if dirty else []


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(project_root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_provenance(project_root: Path) -> tuple[dict[str, Any], list[str]]:
    snapshot_path = project_root / SOURCE_MANIFEST
    if snapshot_path.is_file():
        return _snapshot_source(snapshot_path)
    commit = _git_output(project_root, "rev-parse", "HEAD")
    if not commit:
        return _unknown_source()
    status = _git_output(project_root, "status", "--porcelain", "--untracked-files=normal")
    if status is None:
        return _unknown_source()
    dirty = bool(status)
    source = {
        "kind": "git",
        "commit": commit,
        "dirty": dirty,
        "snapshot_sha256": None,
    }
    return source, ["source_tree_dirty"] if dirty else []
