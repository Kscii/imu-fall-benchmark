from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .configuration import PUBLIC_MODEL_IDS, public_config
from .device import CudaUnavailable, GpuMemoryMonitor, NvidiaSmiMonitor
from .evaluation import best_threshold
from .metrics import (
    binary_classification_metrics,
    subgroup_metrics,
    temporal_event_metrics,
)
from .models import release_gpu_memory
from .performance import (
    PERFORMANCE_SCHEMA_VERSION,
    PhaseTimer,
    build_performance_report,
    process_delta,
    process_snapshot,
)
from .sequence_models import aggregate_bag_scores, threshold_impact_scores
from .unified_models import (
    CudaSequenceTrainer,
    feature_normalization,
    fit_tabular_supervised,
    normalize,
    select_execution_mode,
    sequence_normalization,
)
from .window_cache import (
    UnifiedWindowStore,
    load_unified_window_store,
    prepare_unified_window_store,
)

ENGINE_SCHEMA_VERSION = 2
TABULAR_MODELS = (
    "cuml_logistic_regression",
    "cuml_random_forest",
    "xgboost_cuda",
)
SEQUENCE_MODELS = ("torch_1d_cnn", "torch_lstm", "torch_cnn_lstm")


@dataclass(frozen=True, slots=True)
class Job:
    data_view_id: str
    objective: str
    model_id: str
    fold: int
    seed: int
    input_kind: str
    precision: str

    @property
    def key(self) -> str:
        return (
            f"{self.data_view_id}__{self.model_id}__fold_{self.fold}__"
            f"seed_{self.seed}__{self.precision}"
        )


@dataclass(frozen=True, slots=True)
class FoldIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    external: np.ndarray
    validation_fold: int
    test_fold: int


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(path)


def _event(path: Path, name: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time_unix": time.time(), "event": name, **details}
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(config: dict[str, Any]) -> list[Job]:
    catalog = config["model_catalog"]["models"]
    view = config["data_view"]
    jobs = [
        Job(
            data_view_id=view["id"],
            objective=view["objective"],
            model_id=model_id,
            fold=int(fold),
            seed=int(seed),
            input_kind=str(catalog[model_id]["input_kind"]),
            precision=str(config["precision"]),
        )
        for fold in config["folds"]
        for seed in config["seeds"]
        for model_id in config["models"]
    ]
    return sorted(
        jobs,
        key=lambda job: (
            job.data_view_id,
            job.fold,
            job.seed,
            job.input_kind,
            job.precision,
            config["models"].index(job.model_id),
        ),
    )


def plan_experiment(config: dict[str, Any]) -> dict[str, Any]:
    jobs = build_jobs(config)
    groups: dict[str, int] = {}
    for job in jobs:
        key = f"{job.data_view_id}/fold-{job.fold}/seed-{job.seed}/{job.input_kind}/{job.precision}"
        groups[key] = groups.get(key, 0) + 1
    validation = {fold: (fold + 1) % 5 for fold in config["folds"]}
    return {
        "engine_schema_version": ENGINE_SCHEMA_VERSION,
        "experiment_id": config["id"],
        "resolved_config_sha256": config["resolved_config_sha256"],
        "data_view": config["data_view"],
        "models": config["models"],
        "folds": config["folds"],
        "seeds": config["seeds"],
        "precision": config["precision"],
        "gpu_mode": config["gpu_mode"],
        "split_protocol": {
            "test_fold": "k",
            "validation_fold_by_test_fold": validation,
            "training_folds": "all folds except test and validation",
        },
        "scheduled_jobs": len(jobs),
        "execution_groups": groups,
        "research_only": config["data_view"]["research_only"],
        "jobs": [asdict(job) for job in jobs],
    }


def _sequence_ids(store: UnifiedWindowStore, indices: np.ndarray) -> np.ndarray:
    return np.unique(store.sequence_index[indices])


def _limit_sequence_ids(
    store: UnifiedWindowStore,
    sequence_ids: np.ndarray,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or len(sequence_ids) <= limit:
        return np.asarray(sequence_ids, dtype=np.int64)
    labels = store.sequence_is_fall[sequence_ids].astype(np.int8)
    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    classes = np.unique(labels)
    base = limit // len(classes)
    remainder = limit % len(classes)
    for class_index, label in enumerate(classes):
        candidates = sequence_ids[labels == label]
        count = min(len(candidates), base + int(class_index < remainder))
        if count:
            selected.append(generator.choice(candidates, size=count, replace=False))
    result = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    if len(result) < limit:
        remaining = np.setdiff1d(sequence_ids, result, assume_unique=True)
        additional = generator.choice(remaining, size=limit - len(result), replace=False)
        result = np.concatenate((result, additional))
    return np.sort(result).astype(np.int64)


def _limit_windows(store: UnifiedWindowStore, indices: np.ndarray, limit: int | None) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if limit is None:
        return selected
    parts: list[np.ndarray] = []
    sequence_values = store.sequence_index[selected]
    for sequence_id in np.unique(sequence_values):
        candidates = selected[sequence_values == sequence_id]
        if len(candidates) > limit:
            positions = np.linspace(0, len(candidates) - 1, limit).astype(int)
            candidates = candidates[positions]
        parts.append(candidates)
    return np.sort(np.concatenate(parts)).astype(np.int64) if parts else selected[:0]


def _apply_limits(
    store: UnifiedWindowStore,
    indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> np.ndarray:
    sequence_ids = _limit_sequence_ids(
        store,
        _sequence_ids(store, indices),
        config["max_sequences_per_split"],
        seed,
    )
    selected = indices[np.isin(store.sequence_index[indices], sequence_ids)]
    return _limit_windows(store, selected, config["max_windows_per_sequence"])


def _assert_split_integrity(store: UnifiedWindowStore, split: FoldIndices, objective: str) -> None:
    participant_sets = []
    for indices in (split.train, split.validation, split.test):
        values = {
            (store.dataset_id[sequence_id], store.participant_id[sequence_id])
            for sequence_id in _sequence_ids(store, indices)
        }
        participant_sets.append(values)
    if (
        participant_sets[0] & participant_sets[1]
        or participant_sets[0] & participant_sets[2]
        or participant_sets[1] & participant_sets[2]
    ):
        raise ValueError("Participant leakage detected between train, validation, and test")
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        labels = (
            store.temporal_label[indices]
            if objective == "temporal_supervised"
            else store.sequence_is_fall[_sequence_ids(store, indices)].astype(np.int8)
        )
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(f"{name} split does not contain both classes")


def split_indices(
    store: UnifiedWindowStore, config: dict[str, Any], fold: int, seed: int
) -> FoldIndices:
    view = config["data_view"]
    validation_fold = (fold + 1) % 5
    datasets = store.window_datasets()
    primary = np.isin(datasets, view["training_datasets"])
    if view["objective"] == "temporal_supervised":
        primary &= store.temporal_label >= 0
    train_mask = primary & (store.fold_id != fold) & (store.fold_id != validation_fold)
    validation_mask = primary & (store.fold_id == validation_fold)
    test_mask = primary & (store.fold_id == fold)
    train = _apply_limits(store, np.flatnonzero(train_mask), config, seed + 11)
    validation = _apply_limits(store, np.flatnonzero(validation_mask), config, seed + 13)
    test = _apply_limits(store, np.flatnonzero(test_mask), config, seed + 17)
    supplements = view["negative_supplement_datasets"]
    if supplements:
        supplement_mask = (
            np.isin(datasets, supplements) & (store.temporal_label == 0) & (store.bag_label == 0)
        )
        supplement = _apply_limits(store, np.flatnonzero(supplement_mask), config, seed + 19)
        train = np.sort(np.concatenate((train, supplement))).astype(np.int64)
    external = np.flatnonzero(
        (datasets == view["evaluation_dataset"]) & (store.temporal_label >= 0)
    )
    result = FoldIndices(
        train=train,
        validation=validation,
        test=test,
        external=external,
        validation_fold=validation_fold,
        test_fold=fold,
    )
    _assert_split_integrity(store, result, view["objective"])
    return result


def _bag_labels(store: UnifiedWindowStore, indices: np.ndarray) -> dict[int, int]:
    return {
        int(sequence_id): int(store.sequence_is_fall[sequence_id])
        for sequence_id in _sequence_ids(store, indices)
    }


def _job_hash(config: dict[str, Any], cache_fingerprint: str, job: Job) -> str:
    payload = {
        "engine_schema_version": ENGINE_SCHEMA_VERSION,
        "performance_schema_version": PERFORMANCE_SCHEMA_VERSION,
        "resolved_config_sha256": config["resolved_config_sha256"],
        "cache_fingerprint": cache_fingerprint,
        "job": asdict(job),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checkpoint_path(run_dir: Path, job: Job) -> Path:
    digest = hashlib.sha256(job.key.encode()).hexdigest()[:16]
    return run_dir / "jobs" / f"{digest}.npz"


def _write_checkpoint(
    path: Path,
    metadata: dict[str, Any],
    fall_score: np.ndarray,
    external_fall_score: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".npz.tmp-{os.getpid()}")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=True)),
            fall_score=np.asarray(fall_score, dtype=np.float32),
            external_fall_score=np.asarray(external_fall_score, dtype=np.float32),
        )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "metadata": json.loads(str(archive["metadata_json"])),
            "fall_score": np.asarray(archive["fall_score"], dtype=np.float64),
            "external_fall_score": np.asarray(archive["external_fall_score"], dtype=np.float64),
        }


def _is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def _model_scores(
    *,
    store: UnifiedWindowStore,
    arrays: dict[str, np.ndarray],
    split: FoldIndices,
    config: dict[str, Any],
    job: Job,
    phases: PhaseTimer,
    prepared_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    spec = config["model_catalog"]["models"][job.model_id]
    params = dict(spec["params"])
    objective = job.objective
    external_needed = objective == "recording_mil"
    if job.model_id == "threshold_impact":
        raw = arrays["raw"]
        with phases.track("validation_inference_seconds"):
            validation = threshold_impact_scores(raw[split.validation])
        with phases.track("test_inference_seconds"):
            test = threshold_impact_scores(raw[split.test])
        external = (
            threshold_impact_scores(raw[split.external])
            if external_needed
            else np.empty(0, dtype=np.float64)
        )
        return (
            validation,
            test,
            external,
            {
                "strict_cuda_verified": False,
                "best_epoch": None,
                "model_size_bytes": 0,
                "execution": {
                    "requested_mode": config["gpu_mode"],
                    "effective_mode": "cpu_rule",
                    "reason": "threshold_baseline_has_no_trainable_gpu_model",
                },
                "fallback": None,
            },
        )

    already_standardized = bool(spec["standardize"])
    context_key = (
        job.objective,
        job.fold,
        job.seed,
        job.input_kind,
        already_standardized,
    )
    if context_key not in prepared_cache:
        values = arrays[job.input_kind]
        with phases.track("fold_array_selection_seconds"):
            train_values = values[split.train]
            validation_values = values[split.validation]
            test_values = values[split.test]
            external_values = values[split.external] if external_needed else values[:0]
        with phases.track("normalization_seconds"):
            if already_standardized:
                if job.input_kind == "raw":
                    mean, scale = sequence_normalization(train_values)
                else:
                    mean, scale = feature_normalization(train_values)
                train_values = normalize(train_values, mean, scale)
                validation_values = normalize(validation_values, mean, scale)
                test_values = normalize(test_values, mean, scale)
                if external_needed:
                    external_values = normalize(external_values, mean, scale)
        prepared_cache[context_key] = {
            "train": train_values,
            "validation": validation_values,
            "test": test_values,
            "external": external_values,
            "decision": select_execution_mode(
                str(config["gpu_mode"]),
                (train_values, validation_values, test_values, external_values),
            ),
        }
    prepared = prepared_cache[context_key]
    train_values = prepared["train"]
    validation_values = prepared["validation"]
    test_values = prepared["test"]
    external_values = prepared["external"]
    decision = prepared["decision"]
    mode = decision.effective_mode
    fallback: dict[str, Any] | None = None

    def execute(effective_mode: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        if job.model_id in TABULAR_MODELS:
            if effective_mode == "streaming":
                raise CudaUnavailable(
                    f"{job.model_id} requires backend-managed resident feature arrays"
                )
            with phases.track("model_fit_seconds"):
                adapter = fit_tabular_supervised(
                    job.model_id,
                    params,
                    random_seed=job.seed,
                    train=train_values,
                    train_labels=store.temporal_label[split.train],
                    validation=validation_values,
                    validation_labels=store.temporal_label[split.validation],
                    already_standardized=already_standardized,
                )
            with phases.track("validation_inference_seconds"):
                validation_scores = adapter.predict_proba(validation_values)
            with phases.track("test_inference_seconds"):
                test_scores = adapter.predict_proba(test_values)
            external_scores = (
                adapter.predict_proba(external_values)
                if external_needed
                else np.empty(0, dtype=np.float64)
            )
            return adapter, validation_scores, test_scores, external_scores
        if job.model_id not in SEQUENCE_MODELS:
            raise ValueError(f"Unsupported public model: {job.model_id}")
        trainer = CudaSequenceTrainer(
            job.model_id,
            params,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            top_fraction=float(config["contract"]["supervision"]["recording"]["mil_top_fraction"]),
            random_seed=job.seed,
            precision=job.precision,
            execution_mode=effective_mode,
        )
        with phases.track("model_fit_seconds"):
            if objective == "temporal_supervised":
                trainer.fit_supervised(
                    train_values,
                    store.temporal_label[split.train],
                    validation_values,
                    store.temporal_label[split.validation],
                )
            else:
                trainer.fit_mil(
                    train_values,
                    store.sequence_index[split.train],
                    _bag_labels(store, split.train),
                    validation_values,
                    store.sequence_index[split.validation],
                    _bag_labels(store, split.validation),
                )
        with phases.track("validation_inference_seconds"):
            validation_scores = trainer.predict_proba(validation_values)
        with phases.track("test_inference_seconds"):
            test_scores = trainer.predict_proba(test_values)
        with phases.track("external_inference_seconds"):
            external_scores = (
                trainer.predict_proba(external_values)
                if external_needed
                else np.empty(0, dtype=np.float64)
            )
        return trainer, validation_scores, test_scores, external_scores

    try:
        adapter, validation, test, external = execute(mode)
    except Exception as error:
        if config["gpu_mode"] != "auto" or mode != "resident" or not _is_cuda_oom(error):
            raise
        release_gpu_memory()
        fallback = {
            "from": "resident",
            "to": "streaming",
            "reason": f"{type(error).__name__}: {error}",
            "attempts": 1,
        }
        mode = "streaming"
        adapter, validation, test, external = execute(mode)
    metadata = {
        "strict_cuda_verified": True,
        "best_epoch": getattr(adapter, "best_epoch", None),
        "model_size_bytes": adapter.serialized_size(),
        "execution": {**decision.to_dict(), "effective_mode": mode},
        "fallback": fallback,
    }
    return validation, test, external, metadata


def _evaluate(
    *,
    store: UnifiedWindowStore,
    split: FoldIndices,
    config: dict[str, Any],
    job: Job,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    external_scores: np.ndarray,
) -> dict[str, Any]:
    top_fraction = float(config["contract"]["supervision"]["recording"]["mil_top_fraction"])
    if job.objective == "temporal_supervised":
        validation_labels = store.temporal_label[split.validation]
        test_labels = store.temporal_label[split.test]
        if config["max_sequences_per_split"] is None:
            # A full run also evaluates fall events that retained no decision window.
            test_sequence_scope = np.flatnonzero(
                np.isin(store.dataset_id, config["data_view"]["training_datasets"])
                & (store.sequence_fold_id == split.test_fold)
            )
        else:
            # A capped smoke run evaluates only the sampled sequence subset.
            test_sequence_scope = _sequence_ids(store, split.test)
        threshold, validation_bacc, validation_mcc = best_threshold(
            validation_labels, validation_scores
        )
        return {
            "selected_threshold": threshold,
            "threshold_selection": "maximum_validation_balanced_accuracy",
            "validation_balanced_accuracy": validation_bacc,
            "validation_mcc": validation_mcc,
            "test_metrics": binary_classification_metrics(test_labels, test_scores, threshold),
            "event_metrics": temporal_event_metrics(
                store,
                split.test,
                test_scores,
                threshold,
                sequence_scope=test_sequence_scope,
            ),
            "subgroup_metrics": subgroup_metrics(store, split.test, test_scores, threshold),
            "external_metrics": None,
            "external_event_metrics": None,
        }

    validation_ids, validation_bag_scores = aggregate_bag_scores(
        validation_scores, store.sequence_index[split.validation], top_fraction
    )
    test_ids, test_bag_scores = aggregate_bag_scores(
        test_scores, store.sequence_index[split.test], top_fraction
    )
    validation_labels = store.sequence_is_fall[validation_ids].astype(np.int8)
    test_labels = store.sequence_is_fall[test_ids].astype(np.int8)
    threshold, validation_bacc, validation_mcc = best_threshold(
        validation_labels, validation_bag_scores
    )
    external_ids, external_bag_scores = aggregate_bag_scores(
        external_scores, store.sequence_index[split.external], top_fraction
    )
    external_labels = store.sequence_is_fall[external_ids].astype(np.int8)
    external_sequence_scope = np.flatnonzero(
        store.dataset_id == config["data_view"]["evaluation_dataset"]
    )
    return {
        "selected_threshold": threshold,
        "threshold_selection": "maximum_validation_recording_balanced_accuracy",
        "validation_balanced_accuracy": validation_bacc,
        "validation_mcc": validation_mcc,
        "test_metrics": binary_classification_metrics(test_labels, test_bag_scores, threshold),
        "event_metrics": None,
        "subgroup_metrics": [],
        "external_metrics": {
            "recording": binary_classification_metrics(
                external_labels, external_bag_scores, threshold
            ),
            "window": binary_classification_metrics(
                store.temporal_label[split.external], external_scores, threshold
            ),
        },
        "external_event_metrics": temporal_event_metrics(
            store,
            split.external,
            external_scores,
            threshold,
            sequence_scope=external_sequence_scope,
        ),
    }


def _run_job(
    *,
    store: UnifiedWindowStore,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    job: Job,
    run_dir: Path,
    resume: bool,
    telemetry: NvidiaSmiMonitor,
    split_cache: dict[tuple[str, int, int], FoldIndices],
    prepared_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    process_started = process_snapshot()
    invocation = PhaseTimer()
    expected_hash = _job_hash(config, str(store.manifest["data_split_fingerprint"]), job)
    checkpoint = _checkpoint_path(run_dir, job)
    if resume and checkpoint.exists():
        with invocation.track("checkpoint_read_seconds"):
            cached = _load_checkpoint(checkpoint)
        if cached["metadata"].get("job_hash") == expected_hash:
            stopped = time.perf_counter()
            return {
                "status": "cached",
                "path": str(checkpoint),
                "metadata": cached["metadata"],
                "invocation_performance": {
                    "wall_seconds": stopped - started,
                    "phase_seconds": invocation.to_dict(),
                    "process_usage": process_delta(process_started, process_snapshot()),
                    "gpu_telemetry": telemetry.summary(started, stopped),
                },
            }
    phases = PhaseTimer()
    monitor = GpuMemoryMonitor().start() if job.model_id != "threshold_impact" else None
    invocation_result: dict[str, Any] = {}
    try:
        with phases.track("split_selection_seconds"):
            split_key = (job.objective, job.fold, job.seed)
            if split_key not in split_cache:
                split_cache[split_key] = split_indices(store, config, job.fold, job.seed)
            split = split_cache[split_key]
        validation_scores, test_scores, external_scores, model_metadata = _model_scores(
            store=store,
            arrays=arrays,
            split=split,
            config=config,
            job=job,
            phases=phases,
            prepared_cache=prepared_cache,
        )
        with phases.track("evaluation_seconds"):
            evaluation = _evaluate(
                store=store,
                split=split,
                config=config,
                job=job,
                validation_scores=validation_scores,
                test_scores=test_scores,
                external_scores=external_scores,
            )
        peak = monitor.stop() if monitor is not None else 0
        stopped_core = time.perf_counter()
        metadata = {
            "job": asdict(job),
            "job_key": job.key,
            "job_hash": expected_hash,
            "split": {
                "train_windows": len(split.train),
                "validation_windows": len(split.validation),
                "test_windows": len(split.test),
                "external_windows": len(split.external),
                "train_sequences": len(_sequence_ids(store, split.train)),
                "validation_sequences": len(_sequence_ids(store, split.validation)),
                "test_sequences": len(_sequence_ids(store, split.test)),
                "validation_fold": split.validation_fold,
                "test_fold": split.test_fold,
            },
            **model_metadata,
            **evaluation,
            "gpu_peak_used_bytes": peak,
            "gpu_peak_increment_bytes": (
                max(0, peak - monitor.start_used_bytes) if monitor is not None else 0
            ),
            "total_seconds": stopped_core - started,
            "performance": {
                "schema_version": PERFORMANCE_SCHEMA_VERSION,
                "phase_seconds": phases.to_dict(),
            },
        }
        with invocation.track("checkpoint_write_seconds"):
            _write_checkpoint(checkpoint, metadata, test_scores, external_scores)
        return {
            "status": "computed",
            "path": str(checkpoint),
            "metadata": metadata,
            "invocation_performance": invocation_result,
        }
    finally:
        if monitor is not None:
            monitor.stop()
        if job.model_id != "threshold_impact":
            with invocation.track("gpu_cleanup_seconds"):
                release_gpu_memory()
        stopped = time.perf_counter()
        invocation_result.update(
            {
                "wall_seconds": stopped - started,
                "phase_seconds": invocation.to_dict(),
                "process_usage": process_delta(process_started, process_snapshot()),
                "gpu_telemetry": telemetry.summary(started, stopped),
            }
        )


def _result_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ...]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    for result in results:
        metadata = result["metadata"]
        job = metadata["job"]
        base = {
            "data_view": job["data_view_id"],
            "objective": job["objective"],
            "model_id": job["model_id"],
            "fold": job["fold"],
            "seed": job["seed"],
            "precision": job["precision"],
            "threshold": metadata["selected_threshold"],
        }
        metric_rows.append({**base, **metadata["test_metrics"]})
        if metadata["event_metrics"] is not None:
            event_rows.append({**base, **metadata["event_metrics"]})
        for row in metadata["subgroup_metrics"]:
            subgroup_rows.append({**base, **row})
        external = metadata["external_metrics"]
        if external is not None:
            external_rows.append({**base, "scope": "recording", **external["recording"]})
            external_rows.append({**base, "scope": "window", **external["window"]})
            event_rows.append({**base, "scope": "external", **metadata["external_event_metrics"]})
    return metric_rows, event_rows, subgroup_rows, external_rows


def _write_report(run_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    metric_rows, event_rows, subgroup_rows, external_rows = _result_rows(results)
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "event_metrics.csv", event_rows)
    _write_csv(run_dir / "subgroup_metrics.csv", subgroup_rows)
    _write_csv(run_dir / "external_metrics.csv", external_rows)
    lines = [
        f"# Experiment report: {summary['experiment_id']}",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Jobs: {summary['completed_jobs']}/{summary['scheduled_jobs']}",
        f"- Data view: `{summary['data_view_id']}`",
        f"- Research-only view: `{str(summary['research_only']).lower()}`",
        f"- Data quality: `{summary['data_quality_status']}`",
        f"- Precision: `{summary['precision']}`",
        f"- Requested GPU mode: `{summary['gpu_mode']}`",
        "- Thresholds are selected only on the validation fold.",
        "- `fall_score` is a model score; it is not described as a calibrated probability.",
        "",
        "## Test metrics",
        "",
        "| Model | Fold | Seed | BAcc | Sensitivity | Specificity | F1 | MCC | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model_id']} | {row['fold']} | {row['seed']} | "
            f"{row['balanced_accuracy']:.4f} | {row['sensitivity']:.4f} | "
            f"{row['specificity']:.4f} | {row['f1']:.4f} | {row['mcc']:.4f} | "
            f"{row['auroc']:.4f} | {row['auprc']:.4f} |"
        )
    if not metric_rows:
        lines.append("No completed jobs are available.")
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in summary["known_limitations"])
    lines.extend(
        [
            "",
            "## Reproduction artefacts",
            "",
            "The run directory also contains the resolved YAML configuration, source and "
            "environment provenance, per-job score checkpoints, CSV metrics, performance JSON, "
            "and the append-only event log.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_id(config: dict[str, Any], cache_fingerprint: str) -> str:
    digest = hashlib.sha256(
        f"{config['resolved_config_sha256']}:{cache_fingerprint}".encode()
    ).hexdigest()[:12]
    return f"{config['id']}-{digest}"


def run_experiment(
    *,
    project_root: Path,
    cache_root: Path,
    runs_root: Path,
    config: dict[str, Any],
    resume: bool,
    environment: dict[str, Any],
    source: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    process_started = process_snapshot()
    phases = PhaseTimer()
    with phases.track("cache_prepare_seconds"):
        cache_path, cache_manifest = prepare_unified_window_store(
            project_root=project_root, cache_root=cache_root, config=config
        )
    with phases.track("store_metadata_load_seconds"):
        store = load_unified_window_store(cache_path)
    run_id = _run_id(config, str(store.manifest["data_split_fingerprint"]))
    run_dir = runs_root / run_id
    event_path = run_dir / "events.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_yaml(run_dir / "resolved_config.yaml", public_config(config))
    _atomic_json(run_dir / "environment.json", environment)
    _atomic_json(
        run_dir / "provenance.json",
        {
            "source": source,
            "warnings": warnings,
            "cache_path": str(cache_path),
            "cache_manifest": cache_manifest,
        },
    )
    plan = plan_experiment(config)
    _atomic_json(run_dir / "plan.json", plan)
    jobs = build_jobs(config)
    kinds = {job.input_kind for job in jobs}
    if "raw" not in kinds and "threshold_impact" in config["models"]:
        kinds.add("raw")
    arrays: dict[str, np.ndarray] = {}
    with phases.track("run_level_hdf5_materialization_seconds"):
        for kind in sorted(kinds):
            arrays[kind] = store.materialize(kind)
    _event(event_path, "run_started", run_id=run_id, jobs=len(jobs), resume=resume)
    telemetry = NvidiaSmiMonitor().start()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_cache: dict[tuple[str, int, int], FoldIndices] = {}
    prepared_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    started = time.perf_counter()
    try:
        with phases.track("job_execution_seconds"):
            for position, job in enumerate(jobs, start=1):
                elapsed = time.perf_counter() - started
                if elapsed >= float(config["runtime_budget_seconds"]):
                    failure = {
                        "status": "not_started",
                        "reason": "runtime_budget_exhausted",
                        "job": asdict(job),
                    }
                    failures.append(failure)
                    _event(event_path, "job_skipped", **failure)
                    continue
                _event(event_path, "job_started", job=asdict(job), position=position)
                try:
                    result = _run_job(
                        store=store,
                        arrays=arrays,
                        config=config,
                        job=job,
                        run_dir=run_dir,
                        resume=resume,
                        telemetry=telemetry,
                        split_cache=split_cache,
                        prepared_cache=prepared_cache,
                    )
                except Exception as error:
                    result = {
                        "status": "failed",
                        "reason": f"{type(error).__name__}: {error}",
                        "job": asdict(job),
                    }
                if result["status"] in {"computed", "cached"}:
                    results.append(result)
                else:
                    failures.append(result)
                _event(
                    event_path,
                    "job_finished",
                    job=asdict(job),
                    status=result["status"],
                    reason=result.get("reason"),
                )
                print(
                    json.dumps(
                        {
                            "progress": f"{position}/{len(jobs)}",
                            "job": job.key,
                            "status": result["status"],
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    finally:
        telemetry.stop()
    completed_at = time.perf_counter()
    with phases.track("report_generation_seconds"):
        summary = {
            "engine_schema_version": ENGINE_SCHEMA_VERSION,
            "run_id": run_id,
            "experiment_id": config["id"],
            "status": "PASS" if not failures else "PARTIAL",
            "scheduled_jobs": len(jobs),
            "completed_jobs": len(results),
            "computed_jobs_this_invocation": sum(
                result["status"] == "computed" for result in results
            ),
            "cached_jobs_this_invocation": sum(result["status"] == "cached" for result in results),
            "failures": failures,
            "data_view_id": config["data_view"]["id"],
            "research_only": config["data_view"]["research_only"],
            "data_quality_status": config["data_quality_status"],
            "precision": config["precision"],
            "gpu_mode": config["gpu_mode"],
            "known_limitations": config["known_limitations"],
            "source": source,
            "warnings": warnings,
            "cache_manifest": cache_manifest,
            "elapsed_seconds": completed_at - invocation_started,
        }
        _write_report(run_dir, summary, results)
        performance = build_performance_report(
            invocation_phases=phases.to_dict(),
            results=results,
            process_usage=process_delta(process_started, process_snapshot()),
            gpu_telemetry=telemetry.summary(started, completed_at),
            cache_manifests={"unified": cache_manifest},
        )
        _atomic_json(run_dir / "performance.json", performance)
        summary["performance_path"] = str(run_dir / "performance.json")
        summary["report_path"] = str(run_dir / "report.md")
        _atomic_json(run_dir / "run_manifest.json", summary)
    _event(event_path, "run_finished", status=summary["status"])
    return summary


def regenerate_report(runs_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = runs_root / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Unknown run ID: {run_id}")
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for checkpoint in sorted((run_dir / "jobs").glob("*.npz")):
        payload = _load_checkpoint(checkpoint)
        results.append(
            {"status": "cached", "path": str(checkpoint), "metadata": payload["metadata"]}
        )
    _write_report(run_dir, summary, results)
    return {
        "status": "PASS",
        "run_id": run_id,
        "jobs": len(results),
        "report_path": str(run_dir / "report.md"),
    }


def ensure_public_models_only(config: dict[str, Any]) -> None:
    unknown = set(config["models"]) - set(PUBLIC_MODEL_IDS)
    if unknown:
        raise ValueError(f"Unknown public model IDs: {sorted(unknown)}")
