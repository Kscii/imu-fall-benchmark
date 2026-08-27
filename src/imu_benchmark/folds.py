from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import h5py

from .contract import load_contract_snapshot


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def extend_fold_assignments(
    participants: set[tuple[str, str]],
    existing: dict[tuple[str, str], int],
    *,
    seed: int = 3888,
    fold_count: int = 5,
) -> tuple[dict[tuple[str, str], int], tuple[tuple[str, str], ...]]:
    extras = set(existing) - participants
    if extras:
        raise ValueError(f"Split contains participants absent from data: {sorted(extras)}")
    if any(fold not in range(fold_count) for fold in existing.values()):
        raise ValueError("Existing fold IDs are out of range")
    result = dict(existing)
    dataset_counts = Counter(
        (dataset_id, fold) for (dataset_id, _participant), fold in result.items()
    )
    total_counts = Counter(result.values())
    new_participants = participants - set(result)

    def participant_order(key: tuple[str, str]) -> tuple[str, str]:
        dataset_id, participant_id = key
        digest = hashlib.sha256(f"{seed}:{dataset_id}:{participant_id}".encode()).hexdigest()
        return dataset_id, digest

    ordered = tuple(sorted(new_participants, key=participant_order))
    for dataset_id, participant_id in ordered:
        fold = min(
            range(fold_count),
            key=lambda candidate: (
                dataset_counts[(dataset_id, candidate)],
                total_counts[candidate],
                hashlib.sha256(
                    f"{seed}:{dataset_id}:{participant_id}:{candidate}".encode()
                ).hexdigest(),
            ),
        )
        result[(dataset_id, participant_id)] = fold
        dataset_counts[(dataset_id, fold)] += 1
        total_counts[fold] += 1
    return result, ordered


def _next_version(current: str, changed: bool) -> str:
    if not changed:
        return current
    match = re.fullmatch(r"(.+_v)(\d+)", current)
    if match is None:
        raise ValueError("Split version must end in _vN for deterministic evolution")
    return f"{match.group(1)}{int(match.group(2)) + 1}"


def _participants(project_root: Path, datasets: list[dict[str, Any]]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for item in datasets:
        dataset_id = str(item["dataset_id"])
        with h5py.File(project_root / str(item["path"]), "r") as handle:
            values = handle["sequences"]["participant_id"]
            result.update((dataset_id, _text(value)) for value in values)
    return result


def _existing(path: Path, expected_version: str) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["split_version"] != expected_version:
                raise ValueError("Existing split contains an unexpected version")
            key = (row["dataset_id"], row["participant_id"])
            if key in result:
                raise ValueError(f"Duplicate split assignment: {key}")
            result[key] = int(row["fold_id"])
    return result


def plan_fold_assignments(project_root: Path, collection_name: str) -> dict[str, Any]:
    contract, snapshot, contract_sha256, snapshot_sha256 = load_contract_snapshot(project_root)
    if collection_name not in {"training", "external"}:
        raise ValueError("Collection must be training or external")
    collection = snapshot["collections"][collection_name]
    split = collection["split"]
    participants = _participants(project_root, collection["datasets"])
    existing = _existing(project_root / split["path"], str(split["version"]))
    assignments, added = extend_fold_assignments(
        participants,
        existing,
        seed=int(contract["folds"]["seed"]),
        fold_count=int(contract["folds"]["count"]),
    )
    version = _next_version(str(split["version"]), bool(added))
    rows = [
        {
            "split_version": version,
            "dataset_id": dataset_id,
            "participant_id": participant_id,
            "fold_id": assignments[(dataset_id, participant_id)],
        }
        for dataset_id, participant_id in sorted(assignments)
    ]
    return {
        "status": "PASS",
        "collection": collection_name,
        "contract_sha256": contract_sha256,
        "snapshot_sha256": snapshot_sha256,
        "current_version": split["version"],
        "candidate_version": version,
        "existing_participants": len(existing),
        "new_participants": len(added),
        "added": [list(item) for item in added],
        "fold_counts": {
            str(fold): sum(row["fold_id"] == fold for row in rows)
            for fold in range(int(contract["folds"]["count"]))
        },
        "rows": rows,
    }


def fold_plan_csv(plan: dict[str, Any]) -> str:
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(
        destination,
        fieldnames=("split_version", "dataset_id", "participant_id", "fold_id"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(plan["rows"])
    return destination.getvalue()
