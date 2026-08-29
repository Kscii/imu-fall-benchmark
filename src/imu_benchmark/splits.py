from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FOLD_COUNT = 5
SPLIT_SEED = 3888


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True, slots=True)
class ParticipantStats:
    dataset_id: str
    participant_id: str
    sequences: int
    fall_sequences: int
    events: int
    adl_rows: int
    body_locations: tuple[str, ...]


def collect_participant_stats(source_dir: Path) -> dict[tuple[str, str], ParticipantStats]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    files = sorted(source_dir.glob("*.h5"))
    if not files:
        raise ValueError(f"No HDF5 files found in {source_dir}")
    for path in files:
        with h5py.File(path, "r") as handle:
            dataset_id = _text(handle.attrs["dataset_id"])
            sequences = np.asarray(handle["sequences"])
            annotations = np.asarray(handle["annotations"])
            onset_by_sequence = Counter(
                int(row["sequence_index"])
                for row in annotations
                if _text(row["kind"]) == "onset"
            )
            for index, row in enumerate(sequences):
                participant_id = _text(row["participant_id"])
                key = (dataset_id, participant_id)
                item = aggregate.setdefault(
                    key,
                    {
                        "sequences": 0,
                        "fall_sequences": 0,
                        "events": 0,
                        "adl_rows": 0,
                        "body_locations": set(),
                    },
                )
                item["sequences"] += 1
                item["fall_sequences"] += int(bool(row["is_fall"]))
                item["events"] += onset_by_sequence[index]
                if not bool(row["is_fall"]):
                    item["adl_rows"] += int(row["sample_stop"] - row["sample_start"])
                item["body_locations"].add(_text(row["body_location"]))
    return {
        key: ParticipantStats(
            dataset_id=key[0],
            participant_id=key[1],
            sequences=int(value["sequences"]),
            fall_sequences=int(value["fall_sequences"]),
            events=int(value["events"]),
            adl_rows=int(value["adl_rows"]),
            body_locations=tuple(sorted(value["body_locations"])),
        )
        for key, value in aggregate.items()
    }


def read_assignments(paths: list[Path]) -> dict[tuple[str, str], int]:
    assignments: dict[tuple[str, str], int] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"dataset_id", "participant_id", "fold_id"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Invalid participant split CSV: {path}")
            for row in reader:
                key = (row["dataset_id"], row["participant_id"])
                fold = int(row["fold_id"])
                if key in assignments:
                    raise ValueError(f"Duplicate participant split assignment: {key}")
                if fold not in range(FOLD_COUNT):
                    raise ValueError(f"Invalid participant fold: {key} -> {fold}")
                assignments[key] = fold
    return assignments


def _feature_values(stats: ParticipantStats) -> dict[str, float]:
    values = {
        "participants": 1.0,
        "fall_sequences": float(stats.fall_sequences),
        "events": float(stats.events),
        "adl_rows": float(stats.adl_rows),
        f"dataset:{stats.dataset_id}": 1.0,
    }
    for location in stats.body_locations:
        values[f"body_location:{location}"] = 1.0
    return values


def _stable_tie_break(key: tuple[str, str], fold: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key[0]}:{key[1]}:{fold}".encode()).hexdigest()


def propose_assignments(
    stats: dict[tuple[str, str], ParticipantStats],
    existing: dict[tuple[str, str], int],
    *,
    seed: int = SPLIT_SEED,
) -> tuple[dict[tuple[str, str], int], list[tuple[str, str]]]:
    extra = set(existing) - set(stats)
    if extra:
        raise ValueError(f"Existing split contains participants absent from data: {sorted(extra)}")
    assignments = dict(existing)
    loads: dict[int, Counter[str]] = {fold: Counter() for fold in range(FOLD_COUNT)}
    totals: Counter[str] = Counter()
    for key, item in stats.items():
        totals.update(_feature_values(item))
        if key in assignments:
            loads[assignments[key]].update(_feature_values(item))
    new_keys = sorted(
        set(stats) - set(assignments),
        key=lambda key: (
            -stats[key].events,
            -stats[key].fall_sequences,
            -stats[key].adl_rows,
            hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest(),
        ),
    )
    for key in new_keys:
        values = _feature_values(stats[key])
        candidate_scores = []
        for fold in range(FOLD_COUNT):
            score = 0.0
            for feature, total in totals.items():
                target = max(total / FOLD_COUNT, 1.0)
                projected = loads[fold][feature] + values.get(feature, 0.0)
                score += ((projected - target) / target) ** 2
            candidate_scores.append((score, _stable_tie_break(key, fold, seed), fold))
        fold = min(candidate_scores)[2]
        assignments[key] = fold
        loads[fold].update(values)
    return assignments, new_keys


def write_proposal(
    stats: dict[tuple[str, str], ParticipantStats],
    existing: dict[tuple[str, str], int],
    output_dir: Path,
    *,
    version: str,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    assignments, new_keys = propose_assignments(stats, existing, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{version}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("split_version", "dataset_id", "participant_id", "fold_id"))
        for (dataset_id, participant_id), fold in sorted(assignments.items()):
            writer.writerow((version, dataset_id, participant_id, fold))
    fold_summary = {}
    for fold in range(FOLD_COUNT):
        members = [stats[key] for key, value in assignments.items() if value == fold]
        fold_summary[str(fold)] = {
            "participants": len(members),
            "fall_sequences": sum(item.fall_sequences for item in members),
            "events": sum(item.events for item in members),
            "adl_rows": sum(item.adl_rows for item in members),
            "datasets": dict(Counter(item.dataset_id for item in members)),
        }
    report = {
        "schema_version": "imu_benchmark_split_proposal_v1",
        "status": "PASS",
        "version": version,
        "seed": seed,
        "sticky_existing_assignments": len(existing),
        "new_assignments": len(new_keys),
        "changed_existing_assignments": 0,
        "participants": len(assignments),
        "new_participants": [
            {"dataset_id": dataset_id, "participant_id": participant_id}
            for dataset_id, participant_id in new_keys
        ],
        "folds": fold_summary,
        "csv_path": str(csv_path),
    }
    report_path = output_dir / f"{version}.report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def propose_from_active(
    project_root: Path,
    active_path: Path,
    output_dir: Path,
    *,
    version: str,
) -> dict[str, Any]:
    snapshot = json.loads(active_path.read_text(encoding="utf-8"))
    base = snapshot["collections"]["base"]
    source_dir = active_path.parent / base["data_path"]
    split_root = (
        project_root
        if snapshot["schema_version"] == "imu_benchmark_active_v1"
        else active_path.parent
    )
    split_paths = [split_root / item["path"] for item in base["splits"]]
    return write_proposal(
        collect_participant_stats(source_dir),
        read_assignments(split_paths),
        output_dir,
        version=version,
    )
