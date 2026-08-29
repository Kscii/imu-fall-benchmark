from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def active_manifest_path(tmp_path: Path) -> Path:
    remote = json.loads(
        (PROJECT_ROOT / "configs/data/base_imu25_v2.json").read_text(encoding="utf-8")
    )
    split_root = tmp_path / "base/imu_25hz_snapshot_v2/splits"
    split_root.mkdir(parents=True)
    splits = []
    for item in remote["splits"]:
        source = PROJECT_ROOT / "data/splits" / item["filename"]
        destination = split_root / item["filename"]
        shutil.copy2(source, destination)
        splits.append(
            {
                "path": f"base/imu_25hz_snapshot_v2/splits/{item['filename']}",
                "version": item["version"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )
    entries = [
        {
            key: value
            for key, value in item.items()
            if key not in {"filename", "object_key"}
        }
        | {"path": item["filename"]}
        for item in remote["files"]
    ]
    active = {
        "schema_version": "imu_benchmark_active_v2",
        "snapshot_version": "imu_25hz_snapshot_v2",
        "contract_version": "imu_benchmark_contract_v2",
        "bucket": "gs://unit-test",
        "base_snapshot_id": remote["snapshot_id"],
        "team_snapshot_id": None,
        "collections": {
            "base": {
                "data_path": "base/imu_25hz_snapshot_v2/datasets",
                "splits": splits,
                "datasets": entries,
            }
        },
    }
    destination = tmp_path / "active.json"
    destination.write_text(json.dumps(active), encoding="utf-8")
    return destination
