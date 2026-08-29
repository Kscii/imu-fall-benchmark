from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import alarm_policy_metrics, binary_classification_metrics
from .window_cache import UnifiedWindowStore

COUNT_FIELDS = ("tn", "fp", "fn", "tp", "fall_events", "detected_events")
RATE_METRICS = (
    "balanced_accuracy",
    "window_sensitivity",
    "window_specificity",
    "event_sensitivity",
    "adl_alarm_episodes_per_hour",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rates(counts: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp, fall_events, detected_events, adl_hours, alarms = counts
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "window_sensitivity": sensitivity,
        "window_specificity": specificity,
        "event_sensitivity": (
            detected_events / fall_events if fall_events else 0.0
        ),
        "adl_alarm_episodes_per_hour": alarms / adl_hours if adl_hours else 0.0,
    }


def _participant_rows(
    run_dir: Path,
    store: UnifiedWindowStore,
    config: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    reference = None
    if "alarm_policy" in config:
        reference_id = config["alarm_policy"]["reference_policy"]
        reference = next(
            policy
            for policy in config["alarm_policy"]["policies"]
            if policy["id"] == reference_id
        )
    for result in results:
        checkpoint = Path(result["path"])
        with np.load(checkpoint, allow_pickle=False) as archive:
            window_index = np.asarray(archive["window_index"], dtype=np.int64)
            labels = np.asarray(archive["label"], dtype=np.int8)
            scores = np.asarray(archive["fall_score"], dtype=np.float64)
        metadata = result["metadata"]
        job = metadata["job"]
        threshold = float(metadata["selected_threshold"])
        sequence_ids = store.sequence_index[window_index]
        datasets = store.dataset_id[sequence_ids]
        participants = store.participant_id[sequence_ids]
        participant_keys = np.asarray(
            [
                f"{dataset}\0{participant}"
                for dataset, participant in zip(datasets, participants, strict=True)
            ],
            dtype=object,
        )
        for participant_key in np.unique(participant_keys):
            mask = participant_keys == participant_key
            local_indices = window_index[mask]
            local_labels = labels[mask]
            local_scores = scores[mask]
            dataset_id, participant_id = str(participant_key).split("\0", maxsplit=1)
            classification = binary_classification_metrics(
                local_labels,
                local_scores,
                threshold,
            )
            sequence_scope = np.flatnonzero(
                (store.sequence_fold_id == int(job["fold"]))
                & (store.dataset_id == dataset_id)
                & (store.participant_id == participant_id)
            )
            alarm = (
                None
                if reference is None
                else alarm_policy_metrics(
                    store,
                    local_indices,
                    local_scores,
                    threshold,
                    reference,
                    sequence_scope=sequence_scope,
                )
            )
            rows.append(
                {
                    "model_id": job["model_id"],
                    "training_recipe": job["training_recipe"],
                    "fold": job["fold"],
                    "seed": job["seed"],
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "participant_key": str(participant_key),
                    **{field: classification[field] for field in ("tn", "fp", "fn", "tp")},
                    "fall_events": 0 if alarm is None else alarm["fall_events"],
                    "detected_events": 0 if alarm is None else alarm["detected_events"],
                    "adl_negative_window_hours": (
                        0.0 if alarm is None else alarm["adl_negative_window_hours"]
                    ),
                    "adl_alarm_episodes": (
                        0 if alarm is None else alarm["adl_alarm_episodes"]
                    ),
                }
            )
        oof.append(
            {
                "job_key": metadata["job_key"],
                "model_id": job["model_id"],
                "training_recipe": job["training_recipe"],
                "fold": job["fold"],
                "seed": job["seed"],
                "checkpoint": str(checkpoint.relative_to(run_dir)),
                "checkpoint_sha256": _sha256(checkpoint),
                "windows": len(window_index),
            }
        )
    return rows, oof


def _group_matrices(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["training_recipe"])].append(row)
    output = {}
    for key, group_rows in grouped.items():
        seeds = sorted({int(row["seed"]) for row in group_rows})
        participants = sorted({str(row["participant_key"]) for row in group_rows})
        matrix = np.full((len(seeds), len(participants), 8), np.nan, dtype=np.float64)
        seed_index = {value: index for index, value in enumerate(seeds)}
        participant_index = {value: index for index, value in enumerate(participants)}
        for row in group_rows:
            values = [float(row[field]) for field in COUNT_FIELDS]
            values.extend(
                (
                    float(row["adl_negative_window_hours"]),
                    float(row["adl_alarm_episodes"]),
                )
            )
            matrix[
                seed_index[int(row["seed"])],
                participant_index[str(row["participant_key"])],
            ] = values
        complete = bool(np.isfinite(matrix).all())
        output[key] = {
            "seeds": seeds,
            "participants": participants,
            "matrix": matrix,
            "complete": complete,
        }
    return output


def _bootstrap_group(
    matrix: np.ndarray,
    replicates: int,
    generator: np.random.Generator,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    seed_count, participant_count, _ = matrix.shape
    point = _rates(np.sum(matrix, axis=(0, 1)))
    values = {name: np.empty(replicates, dtype=np.float64) for name in RATE_METRICS}
    for replicate in range(replicates):
        seed_draw = generator.integers(0, seed_count, size=seed_count)
        participant_draw = generator.integers(
            0, participant_count, size=participant_count
        )
        counts = np.sum(matrix[seed_draw][:, participant_draw], axis=(0, 1))
        rates = _rates(counts)
        for name in RATE_METRICS:
            values[name][replicate] = rates[name]
    return point, values


def _paired_bootstrap(
    first: dict[str, Any],
    second: dict[str, Any],
    replicates: int,
    generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    common = sorted(set(first["participants"]) & set(second["participants"]))
    first_positions = [first["participants"].index(value) for value in common]
    second_positions = [second["participants"].index(value) for value in common]
    first_matrix = first["matrix"][:, first_positions]
    second_matrix = second["matrix"][:, second_positions]
    differences = {name: np.empty(replicates, dtype=np.float64) for name in RATE_METRICS}
    for replicate in range(replicates):
        participant_draw = generator.integers(0, len(common), size=len(common))
        first_seed_draw = generator.integers(
            0, len(first["seeds"]), size=len(first["seeds"])
        )
        second_seed_draw = generator.integers(
            0, len(second["seeds"]), size=len(second["seeds"])
        )
        first_counts = np.sum(
            first_matrix[first_seed_draw][:, participant_draw], axis=(0, 1)
        )
        second_counts = np.sum(
            second_matrix[second_seed_draw][:, participant_draw], axis=(0, 1)
        )
        first_rates = _rates(first_counts)
        second_rates = _rates(second_counts)
        for name in RATE_METRICS:
            differences[name][replicate] = second_rates[name] - first_rates[name]
    return differences


def write_statistical_outputs(
    run_dir: Path,
    store: UnifiedWindowStore,
    config: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    replicates = int(config["bootstrap_replicates"])
    if not replicates or not results:
        return None
    participant_rows, oof_rows = _participant_rows(run_dir, store, config, results)
    _write_csv(run_dir / "participant_metrics.csv", participant_rows)
    (run_dir / "oof_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "imu_benchmark_oof_manifest_v1",
                "shards": oof_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    groups = _group_matrices(participant_rows)
    aggregate_rows = []
    for key, group in sorted(groups.items()):
        if not group["complete"]:
            continue
        seed_value = int.from_bytes(
            hashlib.sha256(":".join(key).encode()).digest()[:8],
            byteorder="big",
        )
        point, values = _bootstrap_group(
            group["matrix"],
            replicates,
            np.random.default_rng(3888 + seed_value),
        )
        for metric in RATE_METRICS:
            aggregate_rows.append(
                {
                    "model_id": key[0],
                    "training_recipe": key[1],
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower_95": float(np.percentile(values[metric], 2.5)),
                    "ci_upper_95": float(np.percentile(values[metric], 97.5)),
                    "bootstrap_replicates": replicates,
                    "participants": len(group["participants"]),
                    "seeds": len(group["seeds"]),
                }
            )
    _write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    threshold_key = ("threshold_impact", "not_applicable")
    pairs = []
    if threshold_key in groups and groups[threshold_key]["complete"]:
        pairs.extend(
            (threshold_key, key, "model_vs_threshold")
            for key in groups
            if key != threshold_key and groups[key]["complete"]
        )
    for model_id in sorted({key[0] for key in groups}):
        natural = (model_id, "natural")
        balanced = (model_id, "participant_class_balanced")
        if (
            natural in groups
            and balanced in groups
            and groups[natural]["complete"]
            and groups[balanced]["complete"]
        ):
            pairs.append((natural, balanced, "balanced_minus_natural"))
    paired_rows = []
    for index, (first_key, second_key, comparison) in enumerate(pairs):
        differences = _paired_bootstrap(
            groups[first_key],
            groups[second_key],
            replicates,
            np.random.default_rng(3888 + index),
        )
        for metric in RATE_METRICS:
            values = differences[metric]
            paired_rows.append(
                {
                    "comparison": comparison,
                    "reference_model_id": first_key[0],
                    "reference_recipe": first_key[1],
                    "candidate_model_id": second_key[0],
                    "candidate_recipe": second_key[1],
                    "metric": metric,
                    "difference_median": float(np.median(values)),
                    "ci_lower_95": float(np.percentile(values, 2.5)),
                    "ci_upper_95": float(np.percentile(values, 97.5)),
                    "bootstrap_replicates": replicates,
                }
            )
    _write_csv(run_dir / "paired_comparisons.csv", paired_rows)
    incomplete_groups = [
        {"model_id": key[0], "training_recipe": key[1]}
        for key, group in groups.items()
        if not group["complete"]
    ]
    manifest = {
        "schema_version": "imu_benchmark_statistical_analysis_v1",
        "status": "PASS" if not incomplete_groups else "PARTIAL",
        "method": "participant_cluster_and_seed_hierarchical_bootstrap",
        "bootstrap_replicates": replicates,
        "complete_groups": sum(group["complete"] for group in groups.values()),
        "incomplete_groups": incomplete_groups,
        "oof_manifest_path": str(run_dir / "oof_manifest.json"),
        "participant_metrics_path": str(run_dir / "participant_metrics.csv"),
        "aggregate_metrics_path": str(run_dir / "aggregate_metrics.csv"),
        "paired_comparisons_path": str(run_dir / "paired_comparisons.csv"),
    }
    (run_dir / "statistical_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
