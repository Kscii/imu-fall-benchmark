from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import WindowStore, load_window_store, prepare_window_store
from .device import GpuMemoryMonitor, NvidiaSmiMonitor
from .evaluation import best_threshold
from .models import release_gpu_memory
from .performance import (
    PERFORMANCE_SCHEMA_VERSION,
    PhaseTimer,
    build_performance_report,
    process_delta,
    process_snapshot,
)
from .sequence_models import (
    TorchSequenceAdapter,
    aggregate_bag_scores,
    create_fixed_tabular_adapter,
    threshold_impact_scores,
)
from .specs import (
    MIL_SUITES,
    MODEL_SPECS,
    POSITION_SUITES,
    SEQUENCE_MODEL_IDS,
    TABULAR_MODEL_IDS,
    JobSpec,
    build_jobs,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def registry_hash(config: dict[str, Any]) -> str:
    payload = {
        "config": config,
        "models": {key: value.to_dict() for key, value in MODEL_SPECS.items()},
        "performance_schema_version": PERFORMANCE_SCHEMA_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plan_experiment(
    *, config: dict[str, Any], profile_name: str, suites: tuple[str, ...], models: tuple[str, ...]
) -> dict[str, Any]:
    if profile_name not in config["profiles"]:
        raise ValueError(f"Unknown profile: {profile_name}")
    profile = config["profiles"][profile_name]
    folds = tuple(int(value) for value in profile["outer_folds"])
    jobs = build_jobs(suites=suites, models=models, folds=folds)
    capabilities = {
        model_id: {
            "input_kind": MODEL_SPECS[model_id].input_kind,
            "suites": [
                suite
                for suite in suites
                if (
                    MODEL_SPECS[model_id].position
                    if suite in POSITION_SUITES
                    else MODEL_SPECS[model_id].fall_mil
                )
            ],
        }
        for model_id in models
    }
    return {
        "experiment_version": config["experiment_version"],
        "contract_version": config["contract_version"],
        "contract_sha256": config["contract_sha256"],
        "snapshot_version": config["snapshot_version"],
        "snapshot_sha256": config["snapshot_sha256"],
        "profile": profile_name,
        "runtime_budget_seconds": profile["runtime_budget_seconds"],
        "folds": folds,
        "requested_models": len(models),
        "scheduled_jobs": len(jobs),
        "jobs_by_suite": {suite: sum(job.suite == suite for job in jobs) for suite in suites},
        "capabilities": capabilities,
    }


def _sequence_ids_for_indices(store: WindowStore, indices: np.ndarray) -> np.ndarray:
    return np.unique(store.sequence_index[indices])


def _sequence_labels(store: WindowStore, suite: str, sequence_ids: np.ndarray) -> np.ndarray:
    if suite in MIL_SUITES:
        return store.sequence_is_fall[sequence_ids].astype(np.int8)
    if suite in POSITION_SUITES:
        first_windows = np.searchsorted(store.sequence_index, sequence_ids)
        return store.position_label[first_windows].astype(np.int8)
    raise ValueError(f"Unknown suite: {suite}")


def _limit_sequence_ids(
    store: WindowStore,
    suite: str,
    sequence_ids: np.ndarray,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or len(sequence_ids) <= limit:
        return sequence_ids
    labels = _sequence_labels(store, suite, sequence_ids)
    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    classes = np.unique(labels)
    base = limit // len(classes)
    remainder = limit % len(classes)
    for class_index, label in enumerate(classes):
        candidates = sequence_ids[labels == label]
        count = min(len(candidates), base + int(class_index < remainder))
        selected.append(np.sort(generator.choice(candidates, size=count, replace=False)))
    result = np.sort(np.concatenate(selected))
    if len(result) < limit:
        remaining_ids = np.setdiff1d(sequence_ids, result, assume_unique=True)
        additional = generator.choice(remaining_ids, size=limit - len(result), replace=False)
        result = np.sort(np.concatenate((result, additional)))
    return result


def _limit_windows_per_sequence(
    store: WindowStore, indices: np.ndarray, limit: int | None, suite: str
) -> np.ndarray:
    if limit is None:
        return indices
    selected: list[np.ndarray] = []
    sequence_values = store.sequence_index[indices]
    for sequence_id in np.unique(sequence_values):
        candidates = indices[sequence_values == sequence_id]
        if len(candidates) <= limit:
            selected.append(candidates)
            continue
        positions = np.linspace(0, len(candidates) - 1, limit).astype(int)
        selected.append(candidates[positions])
    return np.sort(np.concatenate(selected)).astype(np.int64)


def select_window_splits(
    store: WindowStore,
    job: JobSpec,
    *,
    config: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, np.ndarray]:
    test_fold = job.outer_fold
    validation_fold = (test_fold + 1) % 5
    window_dataset = store.window_datasets()
    base = np.ones(store.size, dtype=np.bool_)
    if job.suite in POSITION_SUITES:
        base &= store.position_label >= 0
        if job.suite == "position_paired":
            base &= np.isin(window_dataset, config["position_dataset_ids"])
    elif job.suite == "fall_chest_only":
        base &= store.body_location[store.sequence_index] == "chest"
    elif job.suite == "fall_waist_only":
        base &= store.body_location[store.sequence_index] == "waist"
    splits: dict[str, np.ndarray] = {}
    fold_masks = {
        "train": (store.fold_id != test_fold) & (store.fold_id != validation_fold),
        "validation": store.fold_id == validation_fold,
        "test": store.fold_id == test_fold,
    }
    for split_index, (name, fold_mask) in enumerate(fold_masks.items()):
        indices = np.flatnonzero(base & fold_mask)
        if not len(indices):
            splits[name] = indices
            continue
        sequence_ids = _sequence_ids_for_indices(store, indices)
        sequence_ids = _limit_sequence_ids(
            store,
            job.suite,
            sequence_ids,
            profile["max_sequences_per_split"],
            int(config["random_seed"]) + 100 * test_fold + split_index,
        )
        indices = indices[np.isin(store.sequence_index[indices], sequence_ids)]
        splits[name] = _limit_windows_per_sequence(
            store, indices, profile["max_windows_per_sequence"], job.suite
        )
    train_participants = set(store.window_participants()[splits["train"]])
    validation_participants = set(store.window_participants()[splits["validation"]])
    test_participants = set(store.window_participants()[splits["test"]])
    if (
        train_participants & validation_participants
        or train_participants & test_participants
        or validation_participants & test_participants
    ):
        raise ValueError(f"Participant leakage in {job.run_key}")
    return splits


def _window_labels(store: WindowStore, job: JobSpec, indices: np.ndarray) -> np.ndarray:
    if job.suite in POSITION_SUITES:
        return store.position_label[indices]
    if job.suite in MIL_SUITES:
        return store.bag_label[indices]
    raise ValueError(f"Unknown suite: {job.suite}")


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int8)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Binary metrics require both classes")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=(0, 1)).ravel()
    return {
        "n": len(labels),
        "positive": int(np.count_nonzero(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _bag_evaluation(
    store: WindowStore, indices: np.ndarray, window_scores: np.ndarray, top_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence_ids, scores = aggregate_bag_scores(
        window_scores, store.sequence_index[indices], top_fraction
    )
    labels = store.sequence_is_fall[sequence_ids].astype(np.int8)
    return sequence_ids, labels, scores


def _position_evaluation(
    store: WindowStore, indices: np.ndarray, window_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence_ids, scores = aggregate_bag_scores(window_scores, store.sequence_index[indices], 1.0)
    first = np.searchsorted(store.sequence_index, sequence_ids)
    labels = store.position_label[first].astype(np.int8)
    return sequence_ids, labels, scores


def _bag_label_map(store: WindowStore, indices: np.ndarray) -> dict[int, int]:
    sequence_ids = _sequence_ids_for_indices(store, indices)
    return {int(value): int(store.sequence_is_fall[value]) for value in sequence_ids}


def _checkpoint_path(
    root: Path,
    registry_hash: str,
    data_split_fingerprint: str,
    job: JobSpec,
    profile: str,
) -> Path:
    key = f"{profile}:{job.run_key}"
    return (
        root
        / "checkpoints"
        / registry_hash[:16]
        / data_split_fingerprint[:16]
        / f"{hashlib.sha256(key.encode()).hexdigest()}.npz"
    )


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".npz.tmp-{os.getpid()}")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            metadata_json=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
            labels=np.asarray(payload["labels"], dtype=np.int8),
            scores=np.asarray(payload["scores"], dtype=np.float64),
            sequence_ids=np.asarray(payload["sequence_ids"], dtype=np.int32),
        )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "metadata": json.loads(str(archive["metadata_json"])),
            "labels": np.asarray(archive["labels"], dtype=np.int8),
            "scores": np.asarray(archive["scores"], dtype=np.float64),
            "sequence_ids": np.asarray(archive["sequence_ids"], dtype=np.int32),
        }


def _job_hash(
    job: JobSpec,
    profile_name: str,
    profile: dict[str, Any],
    registry_hash: str,
    store: WindowStore,
) -> str:
    payload = {
        "job": job.to_dict(),
        "profile_name": profile_name,
        "profile": profile,
        "registry_hash": registry_hash,
        "source_fingerprint": store.manifest["source_fingerprint"],
        "data_split_fingerprint": store.manifest["data_split_fingerprint"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_job(
    *,
    store: WindowStore,
    job: JobSpec,
    config: dict[str, Any],
    profile_name: str,
    run_root: Path,
    registry_hash: str,
    resume: bool,
    telemetry: NvidiaSmiMonitor | None = None,
) -> dict[str, Any]:
    job_started = time.perf_counter()
    process_started = process_snapshot()
    invocation_phases = PhaseTimer()
    profile = config["profiles"][profile_name]
    expected_hash = _job_hash(job, profile_name, profile, registry_hash, store)
    checkpoint = _checkpoint_path(
        run_root,
        registry_hash,
        store.manifest["data_split_fingerprint"],
        job,
        profile_name,
    )
    if resume and checkpoint.exists():
        with invocation_phases.track("checkpoint_read_seconds"):
            result = _load_checkpoint(checkpoint)
        if result["metadata"].get("job_hash") == expected_hash:
            job_stopped = time.perf_counter()
            return {
                "status": "cached",
                "path": str(checkpoint),
                **result,
                "invocation_performance": {
                    "wall_seconds": job_stopped - job_started,
                    "phase_seconds": invocation_phases.to_dict(),
                    "process_usage": process_delta(process_started, process_snapshot()),
                    "gpu_telemetry": (
                        telemetry.summary(job_started, job_stopped)
                        if telemetry is not None
                        else {"status": "unavailable", "reason": "monitor_not_started"}
                    ),
                },
            }
    job_phases = PhaseTimer()
    with job_phases.track("split_selection_seconds"):
        splits = select_window_splits(store, job, config=config, profile=profile)
    if any(not len(indices) for indices in splits.values()):
        return {
            "status": "unavailable",
            "reason": "selected_split_is_empty",
            "job": job.to_dict(),
        }
    with job_phases.track("split_validation_seconds"):
        for name, indices in splits.items():
            labels = _window_labels(store, job, indices)
            if job.objective == "mil":
                labels = store.sequence_is_fall[
                    _sequence_ids_for_indices(store, indices)
                ].astype(np.int8)
            if set(np.unique(labels).tolist()) != {0, 1}:
                return {
                    "status": "unavailable",
                    "reason": f"{name}_split_lacks_both_classes",
                    "job": job.to_dict(),
                }
    monitor = GpuMemoryMonitor().start() if job.model_id != "threshold_impact" else None
    adapter: Any | None = None
    invocation_performance: dict[str, Any] = {}
    try:
        with job_phases.track("train_hdf5_load_seconds"):
            train_values = store.load(job.input_kind, splits["train"])
        with job_phases.track("validation_hdf5_load_seconds"):
            validation_values = store.load(job.input_kind, splits["validation"])
        with job_phases.track("test_hdf5_load_seconds"):
            test_values = store.load(job.input_kind, splits["test"])
        train_labels = _window_labels(store, job, splits["train"])
        validation_labels = _window_labels(store, job, splits["validation"])
        if job.model_id == "threshold_impact":
            with job_phases.track("validation_inference_seconds"):
                validation_window_scores = threshold_impact_scores(validation_values)
            with job_phases.track("test_inference_seconds"):
                test_window_scores = threshold_impact_scores(test_values)
            model_size = 0
            strict_cuda = False
            best_epoch = None
        elif job.model_id in TABULAR_MODEL_IDS:
            with job_phases.track("model_setup_seconds"):
                adapter = create_fixed_tabular_adapter(
                    job.model_id,
                    max_epochs=int(profile["max_epochs"]),
                    patience=int(profile["patience"]),
                    random_seed=int(config["random_seed"]),
                )
            with job_phases.track("model_fit_seconds"):
                adapter.fit(
                    train_values,
                    train_labels,
                    validation=(validation_values, validation_labels),
                )
            with job_phases.track("validation_inference_seconds"):
                validation_window_scores = adapter.predict_proba(validation_values)
            with job_phases.track("test_inference_seconds"):
                test_window_scores = adapter.predict_proba(test_values)
            adapter.assert_cuda()
            with job_phases.track("model_size_probe_seconds"):
                model_size = adapter.serialized_size()
            strict_cuda = True
            best_epoch = adapter.best_epoch
        elif job.model_id in SEQUENCE_MODEL_IDS:
            with job_phases.track("model_setup_seconds"):
                adapter = TorchSequenceAdapter(
                    job.model_id,
                    max_epochs=int(profile["max_epochs"]),
                    patience=int(profile["patience"]),
                    top_fraction=float(config["mil_top_fraction"]),
                    random_seed=int(config["random_seed"]),
                )
            with job_phases.track("model_fit_seconds"):
                if job.objective == "mil":
                    adapter.fit_mil(
                        train_values,
                        store.sequence_index[splits["train"]],
                        _bag_label_map(store, splits["train"]),
                        validation_values,
                        store.sequence_index[splits["validation"]],
                        _bag_label_map(store, splits["validation"]),
                    )
                else:
                    adapter.fit_supervised(
                        train_values, train_labels, validation_values, validation_labels
                    )
            with job_phases.track("validation_inference_seconds"):
                validation_window_scores = adapter.predict_proba(validation_values)
            with job_phases.track("test_inference_seconds"):
                test_window_scores = adapter.predict_proba(test_values)
            adapter.assert_cuda()
            with job_phases.track("model_size_probe_seconds"):
                model_size = adapter.serialized_size()
            strict_cuda = True
            best_epoch = adapter.best_epoch
        else:
            raise ValueError(f"Unsupported model: {job.model_id}")

        with job_phases.track("evaluation_seconds"):
            if job.objective == "mil":
                _, validation_labels, validation_scores = _bag_evaluation(
                    store,
                    splits["validation"],
                    validation_window_scores,
                    float(config["mil_top_fraction"]),
                )
                sequence_ids, test_labels, test_scores = _bag_evaluation(
                    store,
                    splits["test"],
                    test_window_scores,
                    float(config["mil_top_fraction"]),
                )
            elif job.suite in POSITION_SUITES:
                _, validation_labels, validation_scores = _position_evaluation(
                    store, splits["validation"], validation_window_scores
                )
                sequence_ids, test_labels, test_scores = _position_evaluation(
                    store, splits["test"], test_window_scores
                )
            else:
                validation_scores = validation_window_scores
                test_scores = test_window_scores
                test_labels = _window_labels(store, job, splits["test"])
                sequence_ids = store.sequence_index[splits["test"]]
            threshold, validation_bacc, validation_mcc = best_threshold(
                validation_labels, validation_scores
            )
            metrics = binary_metrics(test_labels, test_scores, threshold)
        peak_bytes = monitor.stop() if monitor is not None else 0
        fit_seconds = job_phases.seconds.get("model_fit_seconds", 0.0)
        core_stopped = time.perf_counter()
        metadata = {
            "job": asdict(job),
            "job_hash": expected_hash,
            "profile": profile_name,
            "selected_threshold": threshold,
            "validation_balanced_accuracy": validation_bacc,
            "validation_mcc": validation_mcc,
            "metrics": metrics,
            "fit_seconds": fit_seconds,
            "total_seconds": core_stopped - job_started,
            "best_epoch": best_epoch,
            "model_size_bytes": model_size,
            "gpu_peak_used_bytes": peak_bytes,
            "gpu_used_bytes_at_start": monitor.start_used_bytes if monitor is not None else 0,
            "gpu_peak_increment_bytes": (
                max(0, peak_bytes - monitor.start_used_bytes) if monitor is not None else 0
            ),
            "strict_cuda_verified": strict_cuda,
            "train_windows": len(splits["train"]),
            "validation_windows": len(splits["validation"]),
            "test_windows": len(splits["test"]),
            "train_sequences": len(_sequence_ids_for_indices(store, splits["train"])),
            "validation_sequences": len(_sequence_ids_for_indices(store, splits["validation"])),
            "test_sequences": len(_sequence_ids_for_indices(store, splits["test"])),
            "source_fingerprint": store.manifest["source_fingerprint"],
            "performance": {
                "schema_version": PERFORMANCE_SCHEMA_VERSION,
                "phase_seconds": job_phases.to_dict(),
            },
        }
        payload = {
            "metadata": metadata,
            "labels": test_labels,
            "scores": test_scores,
            "sequence_ids": sequence_ids,
        }
        with invocation_phases.track("checkpoint_write_seconds"):
            _write_checkpoint(checkpoint, payload)
        result = {
            "status": "computed",
            "path": str(checkpoint),
            **payload,
            "invocation_performance": invocation_performance,
        }
        return result
    finally:
        if monitor is not None:
            monitor.stop()
        adapter = None
        with invocation_phases.track("gpu_cleanup_seconds"):
            if job.model_id != "threshold_impact":
                release_gpu_memory()
        job_stopped = time.perf_counter()
        invocation_performance.update(
            {
                "wall_seconds": job_stopped - job_started,
                "phase_seconds": invocation_phases.to_dict(),
                "process_usage": process_delta(process_started, process_snapshot()),
                "gpu_telemetry": (
                    telemetry.summary(job_started, job_stopped)
                    if telemetry is not None
                    else {"status": "unavailable", "reason": "monitor_not_started"}
                ),
            }
        )


def _subgroup_rows(results: list[dict[str, Any]], store: WindowStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["status"] not in {"computed", "cached"}:
            continue
        metadata = result["metadata"]
        sequence_ids = np.asarray(result["sequence_ids"], dtype=np.int64)
        labels = np.asarray(result["labels"], dtype=np.int8)
        scores = np.asarray(result["scores"], dtype=np.float64)
        threshold = float(metadata["selected_threshold"])
        dimensions = {
            "dataset_id": store.dataset_id[sequence_ids],
            "body_location": store.body_location[sequence_ids],
        }
        for group_type, values in dimensions.items():
            for group_value in sorted(set(values.tolist())):
                selected = values == group_value
                if set(np.unique(labels[selected]).tolist()) != {0, 1}:
                    continue
                rows.append(
                    {
                        "suite": metadata["job"]["suite"],
                        "model_id": metadata["job"]["model_id"],
                        "outer_fold": metadata["job"]["outer_fold"],
                        "profile": metadata["profile"],
                        "group_type": group_type,
                        "group_value": group_value,
                        **binary_metrics(labels[selected], scores[selected], threshold),
                    }
                )
    return rows


def _metric_rows(
    results: list[dict[str, Any]], subgroup_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    macro_dataset = {}
    for subgroup in subgroup_rows:
        if subgroup["group_type"] != "dataset_id":
            continue
        key = (subgroup["suite"], subgroup["model_id"], subgroup["outer_fold"])
        macro_dataset.setdefault(key, []).append(subgroup["balanced_accuracy"])
    rows: list[dict[str, Any]] = []
    for result in results:
        if result["status"] not in {"computed", "cached"}:
            continue
        metadata = result["metadata"]
        row = {
            "suite": metadata["job"]["suite"],
            "model_id": metadata["job"]["model_id"],
            "outer_fold": metadata["job"]["outer_fold"],
            "profile": metadata["profile"],
            "selected_threshold": metadata["selected_threshold"],
            "fit_seconds": metadata["fit_seconds"],
            "total_seconds": metadata["total_seconds"],
            **metadata["metrics"],
        }
        key = (row["suite"], row["model_id"], row["outer_fold"])
        values = macro_dataset.get(key, [])
        row["macro_dataset_balanced_accuracy"] = float(np.mean(values)) if values else float("nan")
        rows.append(row)
    return rows


def _bootstrap_delta(
    labels: np.ndarray,
    universal_predictions: np.ndarray,
    dedicated_predictions: np.ndarray,
    participants: np.ndarray,
    *,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float, float, float]:
    point = float(
        balanced_accuracy_score(labels, dedicated_predictions)
        - balanced_accuracy_score(labels, universal_predictions)
    )
    unique_participants = np.unique(participants)
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        sampled = generator.choice(unique_participants, size=len(unique_participants), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(participants == participant) for participant in sampled]
        )
        sampled_labels = labels[indices]
        if set(np.unique(sampled_labels).tolist()) != {0, 1}:
            continue
        deltas.append(
            balanced_accuracy_score(sampled_labels, dedicated_predictions[indices])
            - balanced_accuracy_score(sampled_labels, universal_predictions[indices])
        )
    if not deltas:
        return point, float("nan"), float("nan")
    low, high = np.percentile(deltas, (2.5, 97.5))
    return point, float(low), float(high)


def _comparison_rows(
    results: list[dict[str, Any]], store: WindowStore, seed: int
) -> list[dict[str, Any]]:
    indexed = {
        (
            result["metadata"]["job"]["suite"],
            result["metadata"]["job"]["model_id"],
            result["metadata"]["job"]["outer_fold"],
        ): result
        for result in results
        if result["status"] in {"computed", "cached"}
    }
    rows: list[dict[str, Any]] = []
    combined: dict[tuple[str, str], list[tuple[np.ndarray, ...]]] = {}
    for dedicated_suite in ("fall_chest_only", "fall_waist_only"):
        for model_id in ("threshold_impact", *SEQUENCE_MODEL_IDS):
            for fold in range(5):
                dedicated = indexed.get((dedicated_suite, model_id, fold))
                universal = indexed.get(("fall_universal", model_id, fold))
                if dedicated is None or universal is None:
                    continue
                universal_lookup = {
                    int(sequence_id): index
                    for index, sequence_id in enumerate(universal["sequence_ids"])
                }
                dedicated_indices = []
                universal_indices = []
                for index, sequence_id in enumerate(dedicated["sequence_ids"]):
                    if int(sequence_id) in universal_lookup:
                        dedicated_indices.append(index)
                        universal_indices.append(universal_lookup[int(sequence_id)])
                if not dedicated_indices:
                    continue
                dedicated_indices_array = np.asarray(dedicated_indices, dtype=np.int64)
                universal_indices_array = np.asarray(universal_indices, dtype=np.int64)
                labels = np.asarray(dedicated["labels"], dtype=np.int8)[dedicated_indices_array]
                universal_labels = np.asarray(universal["labels"], dtype=np.int8)[
                    universal_indices_array
                ]
                if not np.array_equal(labels, universal_labels) or set(
                    np.unique(labels).tolist()
                ) != {0, 1}:
                    continue
                universal_predictions = (
                    np.asarray(universal["scores"])[universal_indices_array]
                    >= universal["metadata"]["selected_threshold"]
                ).astype(np.int8)
                dedicated_predictions = (
                    np.asarray(dedicated["scores"])[dedicated_indices_array]
                    >= dedicated["metadata"]["selected_threshold"]
                ).astype(np.int8)
                universal_bacc = float(balanced_accuracy_score(labels, universal_predictions))
                dedicated_bacc = float(balanced_accuracy_score(labels, dedicated_predictions))
                sequence_ids = np.asarray(dedicated["sequence_ids"], dtype=np.int64)[
                    dedicated_indices_array
                ]
                participants = store.participant_id[sequence_ids]
                rows.append(
                    {
                        "dedicated_suite": dedicated_suite,
                        "model_id": model_id,
                        "scope": f"fold_{fold}",
                        "n": len(labels),
                        "universal_balanced_accuracy": universal_bacc,
                        "dedicated_balanced_accuracy": dedicated_bacc,
                        "delta_dedicated_minus_universal": dedicated_bacc - universal_bacc,
                        "bootstrap_95_ci_low": "",
                        "bootstrap_95_ci_high": "",
                    }
                )
                combined.setdefault((dedicated_suite, model_id), []).append(
                    (labels, universal_predictions, dedicated_predictions, participants)
                )
    for comparison_index, ((dedicated_suite, model_id), parts) in enumerate(
        sorted(combined.items())
    ):
        if len(parts) != 5:
            continue
        labels = np.concatenate([part[0] for part in parts])
        universal_predictions = np.concatenate([part[1] for part in parts])
        dedicated_predictions = np.concatenate([part[2] for part in parts])
        participants = np.concatenate([part[3] for part in parts])
        delta, low, high = _bootstrap_delta(
            labels,
            universal_predictions,
            dedicated_predictions,
            participants,
            seed=seed + comparison_index,
        )
        rows.append(
            {
                "dedicated_suite": dedicated_suite,
                "model_id": model_id,
                "scope": "all_folds_participant_cluster_bootstrap",
                "n": len(labels),
                "universal_balanced_accuracy": float(
                    balanced_accuracy_score(labels, universal_predictions)
                ),
                "dedicated_balanced_accuracy": float(
                    balanced_accuracy_score(labels, dedicated_predictions)
                ),
                "delta_dedicated_minus_universal": delta,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cumulative_elapsed(
    previous_elapsed: float | None, invocation_elapsed: float, computed_jobs: int
) -> float:
    if previous_elapsed is None:
        return invocation_elapsed
    if computed_jobs == 0:
        return previous_elapsed
    return previous_elapsed + invocation_elapsed


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    subgroup_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_path = summary.get("performance_path", str(output_dir / "performance.json"))
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "subgroup_metrics.csv", subgroup_rows)
    _write_csv(output_dir / "comparisons.csv", comparison_rows)
    report_lines = [
        "# Initial validation conclusion recheck",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Completed jobs: {summary['completed_jobs']}/{summary['scheduled_jobs']}",
        f"- Cumulative elapsed: {summary['elapsed_seconds']:.1f} seconds",
        f"- Current invocation: {summary['invocation_elapsed_seconds']:.1f} seconds",
        f"- Checkpoint job seconds: {summary['checkpoint_job_seconds']:.1f} seconds",
        f"- Window cache: `{summary['window_store']}`",
        f"- Performance profile: `{performance_path}`",
        (
            "- The five sources do not share consistent onset/impact labels. Fall suites "
            "therefore use recording-level MIL rather than supervised window labels."
        ),
        "",
        "| Suite | Model | Fold | Balanced accuracy | F1 | MCC | ROC AUC | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {suite} | {model_id} | {outer_fold} | {balanced_accuracy:.4f} | "
            "{f1:.4f} | {mcc:.4f} | {roc_auc:.4f} | {total_seconds:.1f} |".format(**row)
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    aggregate_lines = [
        f"# {summary['experiment_version']} summary",
        "",
        "Results use participant-grouped folds. Fall suites use recording-level MIL and do "
        "not represent online event-level alert evaluation.",
        "",
        "| Suite | Model | Folds | Balanced accuracy | Macro dataset BAcc | F1 | ROC AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["suite"], row["model_id"]), []).append(row)
    for (suite, model_id), values in sorted(grouped.items()):
        formatted = {}
        for name in (
            "balanced_accuracy",
            "macro_dataset_balanced_accuracy",
            "f1",
            "roc_auc",
        ):
            numbers = np.asarray([value[name] for value in values], dtype=np.float64)
            finite = numbers[np.isfinite(numbers)]
            formatted[name] = (
                "N/A" if not len(finite) else f"{np.mean(finite):.4f} ± {np.std(finite):.4f}"
            )

        aggregate_lines.append(
            f"| {suite} | {model_id} | {len(values)} | "
            f"{formatted['balanced_accuracy']} | "
            f"{formatted['macro_dataset_balanced_accuracy']} | "
            f"{formatted['f1']} | {formatted['roc_auc']} |"
        )
    if comparison_rows:
        aggregate_lines.extend(
            [
                "",
                "## Universal vs dedicated on matched test sequences",
                "",
                "| Dedicated suite | Model | N | Universal BAcc | Dedicated BAcc | "
                "Delta | 95% CI |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in comparison_rows:
            if comparison["scope"] != "all_folds_participant_cluster_bootstrap":
                continue
            aggregate_lines.append(
                "| {dedicated_suite} | {model_id} | {n} | "
                "{universal_balanced_accuracy:.4f} | {dedicated_balanced_accuracy:.4f} | "
                "{delta_dedicated_minus_universal:+.4f} | "
                "[{bootstrap_95_ci_low:+.4f}, {bootstrap_95_ci_high:+.4f}] |".format(**comparison)
            )
    (output_dir / "summary.md").write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")
    _atomic_json(output_dir / "run_manifest.json", summary)


def run_experiment(
    *,
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    config: dict[str, Any],
    profile_name: str,
    suites: tuple[str, ...],
    models: tuple[str, ...],
    resume: bool,
    environment: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    invocation_process_started = process_snapshot()
    invocation_phases = PhaseTimer()
    profile = config["profiles"][profile_name]
    with invocation_phases.track("cache_prepare_seconds"):
        window_path, manifest = prepare_window_store(
            project_root=project_root, cache_root=cache_root, config=config
        )
    with invocation_phases.track("store_metadata_load_seconds"):
        store = load_window_store(window_path)
    jobs = build_jobs(
        suites=suites,
        models=models,
        folds=tuple(int(value) for value in profile["outer_folds"]),
    )
    registry_digest = registry_hash(config)
    experiment_version = str(config["experiment_version"])
    fingerprint = store.manifest["data_split_fingerprint"][:16]
    output_dir = run_root / "reports" / fingerprint / profile_name
    status_path = run_root / profile_name / "status.json"
    previous_elapsed = None
    previous_manifest_path = output_dir / "run_manifest.json"
    if resume and previous_manifest_path.is_file():
        try:
            previous_summary = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            previous_summary = {}
        previous_fingerprint = previous_summary.get("window_manifest", {}).get(
            "data_split_fingerprint"
        )
        if (
            previous_summary.get("experiment_version") == experiment_version
            and previous_summary.get("profile") == profile_name
            and previous_summary.get("registry_hash") == registry_digest
            and previous_fingerprint == store.manifest["data_split_fingerprint"]
        ):
            previous_elapsed = float(previous_summary.get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    telemetry = NvidiaSmiMonitor().start()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        with invocation_phases.track("job_execution_seconds"):
            for index, job in enumerate(jobs, start=1):
                elapsed = time.perf_counter() - started
                if elapsed >= float(profile["runtime_budget_seconds"]):
                    failures.append(
                        {
                            "status": "not_started",
                            "reason": "runtime_budget_exhausted",
                            "job": job.to_dict(),
                        }
                    )
                    continue
                _atomic_json(
                    status_path,
                    {
                        "state": "running",
                        "profile": profile_name,
                        "progress": f"{index - 1}/{len(jobs)}",
                        "current_job": job.to_dict(),
                        "elapsed_seconds": elapsed,
                    },
                )
                try:
                    result = run_job(
                        store=store,
                        job=job,
                        config=config,
                        profile_name=profile_name,
                        run_root=run_root,
                        registry_hash=registry_digest,
                        resume=resume,
                        telemetry=telemetry,
                    )
                except Exception as error:
                    result = {
                        "status": "failed",
                        "reason": f"{type(error).__name__}: {error}",
                        "job": job.to_dict(),
                    }
                if result["status"] in {"computed", "cached"}:
                    results.append(result)
                else:
                    failures.append(result)
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(jobs)}",
                            "run_key": job.run_key,
                            "status": result["status"],
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    finally:
        telemetry.stop()
    elapsed = time.perf_counter() - started
    telemetry_summary = telemetry.summary(started, time.perf_counter())
    with invocation_phases.track("result_aggregation_seconds"):
        subgroup_rows = _subgroup_rows(results, store)
        rows = _metric_rows(results, subgroup_rows)
        comparison_rows = _comparison_rows(results, store, int(config["random_seed"]))
    completed_seconds = [row["total_seconds"] for row in rows]
    computed_jobs = sum(result["status"] == "computed" for result in results)
    cached_jobs = sum(result["status"] == "cached" for result in results)
    projected_reproduce_seconds = None
    if profile_name == "smoke" and completed_seconds:
        reproduction_jobs = build_jobs(
            suites=suites,
            models=models,
            folds=tuple(
                int(value) for value in config["profiles"]["reproduce"]["outer_folds"]
            ),
        )
        projected_reproduce_seconds = float(
            np.mean(completed_seconds) * len(reproduction_jobs) * 1.5
        )
    summary = {
        "status": "PASS" if not failures else "PARTIAL",
        "experiment_version": experiment_version,
        "profile": profile_name,
        "scheduled_jobs": len(jobs),
        "completed_jobs": len(results),
        "failed_or_unavailable_jobs": failures,
        "elapsed_seconds": _cumulative_elapsed(previous_elapsed, elapsed, computed_jobs),
        "invocation_elapsed_seconds": elapsed,
        "checkpoint_job_seconds": float(np.sum(completed_seconds)),
        "computed_jobs_this_invocation": computed_jobs,
        "cached_jobs_this_invocation": cached_jobs,
        "runtime_budget_seconds": profile["runtime_budget_seconds"],
        "projected_reproduce_seconds_from_smoke": projected_reproduce_seconds,
        "window_store": str(window_path),
        "window_manifest": manifest,
        "registry_hash": registry_digest,
        "random_seed": int(config["random_seed"]),
        "determinism_policy": "fixed_seed_and_torch_deterministic_algorithms",
        "environment": environment,
        "source": source,
        "warnings": list(warnings or []),
        "source_balance": [
            {
                "dataset_id": dataset_id,
                "body_location": body_location,
                "sequences": int(np.count_nonzero(selected)),
                "fall_sequences": int(np.count_nonzero(store.sequence_is_fall[selected])),
            }
            for dataset_id in sorted(set(store.dataset_id.tolist()))
            for body_location in sorted(
                set(store.body_location[store.dataset_id == dataset_id].tolist())
            )
            for selected in [
                (store.dataset_id == dataset_id) & (store.body_location == body_location)
            ]
        ],
    }
    performance_path = output_dir / "performance.json"
    summary["performance_schema_version"] = PERFORMANCE_SCHEMA_VERSION
    summary["performance_path"] = str(performance_path)
    with invocation_phases.track("report_write_seconds"):
        write_report(output_dir, summary, rows, subgroup_rows, comparison_rows)
    performance = build_performance_report(
        invocation_phases=invocation_phases.to_dict(),
        results=results,
        process_usage=process_delta(invocation_process_started, process_snapshot()),
        gpu_telemetry=telemetry_summary,
        cache_manifests={"training": manifest},
    )
    performance["invocation_wall_seconds"] = time.perf_counter() - invocation_started
    _atomic_json(performance_path, performance)
    _atomic_json(output_dir / "run_manifest.json", summary)
    _atomic_json(
        status_path,
        {
            "state": "finished",
            "profile": profile_name,
            "summary": summary,
            "output_dir": str(output_dir),
        },
    )
    return summary


def regenerate_report(
    *,
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    config: dict[str, Any],
    profile_name: str,
    suites: tuple[str, ...],
    models: tuple[str, ...],
) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    window_path, manifest = prepare_window_store(
        project_root=project_root, cache_root=cache_root, config=config
    )
    store = load_window_store(window_path)
    jobs = build_jobs(
        suites=suites,
        models=models,
        folds=tuple(int(value) for value in profile["outer_folds"]),
    )
    registry_digest = registry_hash(config)
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for job in jobs:
        checkpoint = _checkpoint_path(
            run_root,
            registry_digest,
            store.manifest["data_split_fingerprint"],
            job,
            profile_name,
        )
        if not checkpoint.is_file():
            missing.append(job.run_key)
            continue
        result = _load_checkpoint(checkpoint)
        expected = _job_hash(job, profile_name, profile, registry_digest, store)
        if result["metadata"].get("job_hash") != expected:
            missing.append(job.run_key)
            continue
        results.append({"status": "cached", "path": str(checkpoint), **result})
    if missing:
        raise ValueError(
            f"Cannot regenerate a complete {profile_name} report; missing or stale jobs: "
            f"{len(missing)}"
        )

    subgroup_rows = _subgroup_rows(results, store)
    rows = _metric_rows(results, subgroup_rows)
    comparison_rows = _comparison_rows(results, store, int(config["random_seed"]))
    fingerprint = store.manifest["data_split_fingerprint"][:16]
    output_dir = run_root / "reports" / fingerprint / profile_name
    previous_manifest = output_dir / "run_manifest.json"
    if previous_manifest.is_file():
        summary = json.loads(previous_manifest.read_text(encoding="utf-8"))
    else:
        summary = {
            "status": "PASS",
            "experiment_version": config["experiment_version"],
            "profile": profile_name,
            "scheduled_jobs": len(jobs),
            "completed_jobs": len(results),
            "failed_or_unavailable_jobs": [],
            "elapsed_seconds": float(np.sum([row["total_seconds"] for row in rows])),
            "invocation_elapsed_seconds": 0.0,
            "checkpoint_job_seconds": float(np.sum([row["total_seconds"] for row in rows])),
            "computed_jobs_this_invocation": 0,
            "cached_jobs_this_invocation": len(results),
            "runtime_budget_seconds": profile["runtime_budget_seconds"],
            "projected_reproduce_seconds_from_smoke": None,
            "window_store": str(window_path),
            "window_manifest": manifest,
            "registry_hash": registry_digest,
            "random_seed": int(config["random_seed"]),
            "determinism_policy": "fixed_seed_and_torch_deterministic_algorithms",
            "environment": None,
            "source": None,
            "warnings": ["source_unknown"],
            "source_balance": [],
        }
    summary["report_regenerated"] = True
    write_report(output_dir, summary, rows, subgroup_rows, comparison_rows)
    return {"status": "PASS", "output_dir": str(output_dir), "jobs": len(results)}
