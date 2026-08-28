from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def active_manifest_path(tmp_path: Path) -> Path:
    remote = json.loads(
        (PROJECT_ROOT / "configs/data/base_imu25_v1.json").read_text(encoding="utf-8")
    )
    splits = json.loads(
        (PROJECT_ROOT / "configs/data/base_splits_v1.json").read_text(encoding="utf-8")
    )["splits"]
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
        "schema_version": "imu_benchmark_active_v1",
        "snapshot_version": "imu_25hz_snapshot_v1",
        "contract_version": "imu_benchmark_contract_v2",
        "bucket": "gs://unit-test",
        "base_snapshot_id": remote["snapshot_id"],
        "team_snapshot_id": None,
        "collections": {
            "base": {
                "data_path": "base/imu_25hz_snapshot_v1/datasets",
                "splits": splits,
                "datasets": entries,
            }
        },
    }
    destination = tmp_path / "active.json"
    destination.write_text(json.dumps(active), encoding="utf-8")
    return destination
