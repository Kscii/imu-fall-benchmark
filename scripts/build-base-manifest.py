#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from imu_benchmark.dataset import validate_hdf5_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reviewed base snapshot manifest")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument("--split-version", action="append", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if len(args.split) != len(args.split_version):
        raise ValueError("Each --split requires one corresponding --split-version")
    prefix = f"benchmark-datasets/base/{args.snapshot_id}"
    files = []
    for path in sorted(args.source_dir.glob("*.h5")):
        summary = validate_hdf5_file(path)
        files.append(
            {
                "dataset_id": summary["dataset_id"],
                "filename": path.name,
                "object_key": f"{prefix}/datasets/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "hdf5_schema_version": "3.1.0",
                "sampling_rate_hz": 25.0,
                **{
                    key: summary[key]
                    for key in (
                        "logical_content_sha256",
                        "evaluation_role",
                        "participants",
                        "sequences",
                        "rows",
                        "annotations",
                        "events",
                        "segments",
                        "fall_sequences",
                        "supervision",
                        "body_locations",
                    )
                },
            }
        )
    splits = [
        {
            "filename": path.name,
            "object_key": f"{prefix}/splits/{path.name}",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "version": version,
        }
        for path, version in zip(args.split, args.split_version, strict=True)
    ]
    payload = {
        "schema_version": "imu_benchmark_dataset_manifest_v2",
        "kind": "base",
        "contract_version": "imu_benchmark_contract_v2",
        "snapshot_id": args.snapshot_id,
        "created_at_utc": args.created_at_utc,
        "split_set_id": "public_participant_5fold_sticky_v1",
        "files": files,
        "splits": splits,
        "source": {
            "repository": args.source_repository,
            "branch": args.source_branch,
            "data_commit": args.source_commit,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
