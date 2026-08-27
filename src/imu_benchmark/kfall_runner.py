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
from .device import GpuMemoryMonitor
from .evaluation import best_threshold
from .kfall_data import (
    KFallWindowStore,
    load_kfall_window_store,
    prepare_kfall_window_store,
)
from .models import release_gpu_memory
from .runner import binary_metrics
from .sequence_models import (
    TorchSequenceAdapter,
    aggregate_bag_scores,
    threshold_impact_scores,
)
from .specs import MODEL_SPECS, JobSpec, build_jobs

KFALL_SUITES = ("fall_universal", "fall_waist_only")
KFALL_MODELS = ("threshold_impact", "torch_1d_cnn", "torch_lstm", "torch_cnn_lstm")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _registry_hash(config: dict[str, Any]) -> str:
    payload = {
        "config": config,
        "models": {model: MODEL_SPECS[model].to_dict() for model in KFALL_MODELS},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plan_kfall_experiment(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if profile_name not in config["profiles"]:
        raise ValueError(f"Unknown KFall profile: {profile_name}")
    profile = config["profiles"][profile_name]
    folds = tuple(int(value) for value in profile["validation_folds"])
    jobs = build_jobs(suites=KFALL_SUITES, models=KFALL_MODELS, folds=folds)
    return {
        "experiment_version": config["experiment_version"],
        "profile": profile_name,
        "training_protocol": "four_folds_train_one_fold_validation",
        "external_dataset": "kfall",
        "external_use": "inference_and_participant_grouped_threshold_calibration_only",
        "validation_folds": folds,
        "scheduled_jobs": len(jobs),
        "jobs_by_suite": {
            suite: sum(job.suite == suite for job in jobs) for suite in KFALL_SUITES
        },
        "runtime_budget_seconds": profile["runtime_budget_seconds"],
        "data_quality_status": config["data_quality_status"],
    }


def _sequence_ids(store: WindowStore, indices: np.ndarray) -> np.ndarray:
    return np.unique(store.sequence_index[indices])


def _bag_labels(store: WindowStore, indices: np.ndarray) -> dict[int, int]:
    sequence_ids = _sequence_ids(store, indices)
    return {int(value): int(store.sequence_is_fall[value]) for value in sequence_ids}


def _limit_sequences(
    store: WindowStore,
    sequence_ids: np.ndarray,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or len(sequence_ids) <= limit:
        return sequence_ids
    labels = store.sequence_is_fall[sequence_ids].astype(np.int8)
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
        remaining = np.setdiff1d(sequence_ids, result, assume_unique=True)
        additional = generator.choice(remaining, size=limit - len(result), replace=False)
        result = np.sort(np.concatenate((result, additional)))
    return result


def _limit_windows(
    store: WindowStore, indices: np.ndarray, limit: int | None
) -> np.ndarray:
    if limit is None:
        return indices
    output: list[np.ndarray] = []
    sequence_values = store.sequence_index[indices]
    for sequence_id in np.unique(sequence_values):
        candidates = indices[sequence_values == sequence_id]
        if len(candidates) <= limit:
            output.append(candidates)
        else:
            positions = np.linspace(0, len(candidates) - 1, limit).astype(int)
            output.append(candidates[positions])
    return np.sort(np.concatenate(output)).astype(np.int64)


def _training_splits(
    store: WindowStore,
    job: JobSpec,
    *,
    profile: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    base = np.ones(store.size, dtype=np.bool_)
    if job.suite == "fall_waist_only":
        base &= store.body_location[store.sequence_index] == "waist"
    elif job.suite != "fall_universal":
        raise ValueError(f"Unsupported KFall training suite: {job.suite}")
    masks = {
        "train": store.fold_id != job.outer_fold,
        "validation": store.fold_id == job.outer_fold,
    }
    splits: dict[str, np.ndarray] = {}
    for split_index, (name, mask) in enumerate(masks.items()):
        indices = np.flatnonzero(base & mask)
        sequence_ids = _limit_sequences(
            store,
            _sequence_ids(store, indices),
            profile["max_sequences_per_split"],
            seed + 100 * job.outer_fold + split_index,
        )
        indices = indices[np.isin(store.sequence_index[indices], sequence_ids)]
        splits[name] = _limit_windows(store, indices, profile["max_windows_per_sequence"])
    train_participants = set(store.window_participants()[splits["train"]])
    validation_participants = set(store.window_participants()[splits["validation"]])
    if train_participants & validation_participants:
        raise ValueError(f"Participant leakage in {job.run_key}")
    for name, indices in splits.items():
        labels = store.sequence_is_fall[_sequence_ids(store, indices)].astype(np.int8)
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(f"{job.run_key}: {name} split lacks both classes")
    return splits


def _checkpoint_path(
    run_root: Path,
    registry_hash: str,
    training_fingerprint: str,
    external_fingerprint: str,
    profile_name: str,
    job: JobSpec,
) -> Path:
    key = f"{profile_name}:{job.run_key}"
    return (
        run_root
        / "checkpoints"
        / registry_hash[:16]
        / f"{training_fingerprint[:12]}-{external_fingerprint[:12]}"
        / f"{hashlib.sha256(key.encode()).hexdigest()}.npz"
    )


def _job_hash(
    job: JobSpec,
    profile_name: str,
    profile: dict[str, Any],
    registry_hash: str,
    training_fingerprint: str,
    external_fingerprint: str,
) -> str:
    payload = {
        "job": job.to_dict(),
        "profile": profile_name,
        "profile_config": profile,
        "registry_hash": registry_hash,
        "training_fingerprint": training_fingerprint,
        "external_fingerprint": external_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".npz.tmp-{os.getpid()}")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            metadata_json=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
            external_window_scores=np.asarray(payload["external_window_scores"], dtype=np.float32),
        )
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "metadata": json.loads(str(archive["metadata_json"])),
            "external_window_scores": np.asarray(
                archive["external_window_scores"], dtype=np.float64
            ),
        }


def _run_job(
    *,
    training_store: WindowStore,
    external_store: KFallWindowStore,
    external_values: np.ndarray,
    job: JobSpec,
    config: dict[str, Any],
    profile_name: str,
    run_root: Path,
    registry_hash: str,
    resume: bool,
) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    training_fingerprint = training_store.manifest["data_split_fingerprint"]
    external_fingerprint = external_store.manifest["data_split_fingerprint"]
    expected_hash = _job_hash(
        job,
        profile_name,
        profile,
        registry_hash,
        training_fingerprint,
        external_fingerprint,
    )
    checkpoint = _checkpoint_path(
        run_root,
        registry_hash,
        training_fingerprint,
        external_fingerprint,
        profile_name,
        job,
    )
    if resume and checkpoint.exists():
        result = _load_checkpoint(checkpoint)
        if result["metadata"].get("job_hash") == expected_hash:
            if len(result["external_window_scores"]) != external_store.size:
                raise ValueError(f"Invalid external score count in {checkpoint}")
            return {"status": "cached", "path": str(checkpoint), **result}

    splits = _training_splits(
        training_store,
        job,
        profile=profile,
        seed=int(config["random_seed"]),
    )
    monitor = GpuMemoryMonitor().start() if job.model_id != "threshold_impact" else None
    adapter: TorchSequenceAdapter | None = None
    started = time.perf_counter()
    try:
        validation_values = training_store.load("raw", splits["validation"])
        validation_sequence_index = training_store.sequence_index[splits["validation"]]
        if job.model_id == "threshold_impact":
            validation_window_scores = threshold_impact_scores(validation_values)
            external_window_scores = threshold_impact_scores(external_values)
            fit_seconds = 0.0
            model_size = 0
            strict_cuda = False
            best_epoch = None
        else:
            train_values = training_store.load("raw", splits["train"])
            adapter = TorchSequenceAdapter(
                job.model_id,
                max_epochs=int(profile["max_epochs"]),
                patience=int(profile["patience"]),
                top_fraction=float(config["mil_top_fraction"]),
                random_seed=int(config["random_seed"]),
            )
            fit_started = time.perf_counter()
            adapter.fit_mil(
                train_values,
                training_store.sequence_index[splits["train"]],
                _bag_labels(training_store, splits["train"]),
                validation_values,
                validation_sequence_index,
                _bag_labels(training_store, splits["validation"]),
            )
            fit_seconds = time.perf_counter() - fit_started
            validation_window_scores = adapter.predict_proba(validation_values)
            external_window_scores = adapter.predict_proba(external_values)
            adapter.assert_cuda()
            model_size = adapter.serialized_size()
            strict_cuda = True
            best_epoch = adapter.best_epoch

        validation_sequence_ids, validation_bag_scores = aggregate_bag_scores(
            validation_window_scores,
            validation_sequence_index,
            float(config["mil_top_fraction"]),
        )
        validation_labels = training_store.sequence_is_fall[validation_sequence_ids].astype(np.int8)
        threshold, validation_bacc, validation_mcc = best_threshold(
            validation_labels, validation_bag_scores
        )
        peak_bytes = monitor.stop() if monitor is not None else 0
        metadata = {
            "job": asdict(job),
            "job_hash": expected_hash,
            "profile": profile_name,
            "training_protocol": "four_folds_train_one_fold_validation",
            "selected_recording_threshold": threshold,
            "validation_balanced_accuracy": validation_bacc,
            "validation_mcc": validation_mcc,
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - started,
            "best_epoch": best_epoch,
            "model_size_bytes": model_size,
            "gpu_peak_used_bytes": peak_bytes,
            "strict_cuda_verified": strict_cuda,
            "train_windows": len(splits["train"]),
            "validation_windows": len(splits["validation"]),
            "train_sequences": len(_sequence_ids(training_store, splits["train"])),
            "validation_sequences": len(_sequence_ids(training_store, splits["validation"])),
            "training_fingerprint": training_fingerprint,
            "external_fingerprint": external_fingerprint,
            "external_windows": external_store.size,
            "data_quality_status": config["data_quality_status"],
        }
        payload = {
            "metadata": metadata,
            "external_window_scores": external_window_scores,
        }
        _write_checkpoint(checkpoint, payload)
        return {"status": "computed", "path": str(checkpoint), **payload}
    finally:
        if monitor is not None:
            monitor.stop()
        adapter = None
        release_gpu_memory()


def _classification_metrics_from_predictions(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int8)
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


def _top_window_localization(store: KFallWindowStore, scores: np.ndarray) -> float:
    hits = 0
    fall_count = 0
    for sequence_id in np.flatnonzero(store.sequence_is_fall):
        indices = np.flatnonzero(store.sequence_index == sequence_id)
        fall_count += 1
        if len(indices):
            highest = indices[int(np.argmax(scores[indices]))]
            hits += int(store.temporal_label[highest] == 1)
    return float(hits / fall_count) if fall_count else 0.0


def _zero_shot_rows(
    results: list[dict[str, Any]], store: KFallWindowStore, top_fraction: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_window = store.temporal_label.astype(np.int8)
    for result in results:
        metadata = result["metadata"]
        scores = np.asarray(result["external_window_scores"], dtype=np.float64)
        sequence_ids, recording_scores = aggregate_bag_scores(
            scores, store.sequence_index, top_fraction
        )
        recording_labels = store.sequence_is_fall[sequence_ids].astype(np.int8)
        threshold = float(metadata["selected_recording_threshold"])
        rows.append(
            {
                "suite": metadata["job"]["suite"],
                "model_id": metadata["job"]["model_id"],
                "training_validation_fold": metadata["job"]["outer_fold"],
                "recording_threshold": threshold,
                "window_roc_auc": float(roc_auc_score(labels_window, scores)),
                "window_average_precision": float(
                    average_precision_score(labels_window, scores)
                ),
                "top_window_temporal_hit_rate": _top_window_localization(store, scores),
                **binary_metrics(recording_labels, recording_scores, threshold),
            }
        )
    return rows


def _event_metrics(
    store: KFallWindowStore,
    scores: np.ndarray,
    thresholds_by_fold: dict[int, float],
    selected_folds: set[int],
) -> dict[str, Any]:
    fall_events = detected_events = no_decision_events = 0
    pre_onset_false_recordings = 0
    adl_recordings = adl_false_recordings = 0
    onset_latencies: list[float] = []
    impact_offsets: list[float] = []
    for sequence_id, fold in enumerate(store.sequence_fold_id):
        fold_int = int(fold)
        if fold_int not in selected_folds:
            continue
        threshold = thresholds_by_fold[fold_int]
        indices = np.flatnonzero(store.sequence_index == sequence_id)
        if store.sequence_is_fall[sequence_id]:
            fall_events += 1
            positive = indices[store.temporal_label[indices] == 1]
            if not len(positive):
                no_decision_events += 1
            alerted = positive[scores[positive] >= threshold]
            if len(alerted):
                detected_events += 1
                first_decision = int(np.min(store.end_sample[alerted] - 1))
                onset_latencies.append(
                    (first_decision - int(store.event_onset_sample[sequence_id])) / 30.0
                )
                impact_offsets.append(
                    (first_decision - int(store.event_impact_sample[sequence_id])) / 30.0
                )
            pre_onset = indices[
                store.end_sample[indices] - 1 < store.event_onset_sample[sequence_id]
            ]
            if len(pre_onset) and np.any(scores[pre_onset] >= threshold):
                pre_onset_false_recordings += 1
        else:
            adl_recordings += 1
            if len(indices) and np.any(scores[indices] >= threshold):
                adl_false_recordings += 1
    latency = np.asarray(onset_latencies, dtype=np.float64)
    impact = np.asarray(impact_offsets, dtype=np.float64)
    return {
        "fall_events": fall_events,
        "detected_events": detected_events,
        "events_without_positive_decision_window": no_decision_events,
        "event_sensitivity": detected_events / fall_events if fall_events else 0.0,
        "pre_onset_false_alert_recordings": pre_onset_false_recordings,
        "pre_onset_false_alert_rate": (
            pre_onset_false_recordings / fall_events if fall_events else 0.0
        ),
        "adl_recordings": adl_recordings,
        "adl_false_positive_recordings": adl_false_recordings,
        "adl_recording_false_positive_rate": (
            adl_false_recordings / adl_recordings if adl_recordings else 0.0
        ),
        "onset_latency_median_s": float(np.median(latency)) if len(latency) else float("nan"),
        "onset_latency_p95_s": float(np.percentile(latency, 95)) if len(latency) else float("nan"),
        "impact_offset_median_s": float(np.median(impact)) if len(impact) else float("nan"),
    }


def _calibrated_rows(
    results: list[dict[str, Any]], store: KFallWindowStore
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        metadata = result["metadata"]
        key = (metadata["job"]["suite"], metadata["job"]["model_id"])
        grouped.setdefault(key, []).append(result)
    window_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    ensemble_scores: dict[tuple[str, str], np.ndarray] = {}
    for key, values in sorted(grouped.items()):
        folds = {int(value["metadata"]["job"]["outer_fold"]) for value in values}
        if folds != set(range(5)):
            continue
        scores = np.mean(
            np.stack([np.asarray(value["external_window_scores"]) for value in values]),
            axis=0,
        ).astype(np.float64)
        ensemble_scores[key] = scores
        thresholds: dict[int, float] = {}
        out_of_fold_predictions = np.empty(store.size, dtype=np.int8)
        for fold in range(5):
            calibration = store.fold_id != fold
            test = store.fold_id == fold
            threshold, calibration_bacc, calibration_mcc = best_threshold(
                store.temporal_label[calibration], scores[calibration]
            )
            thresholds[fold] = threshold
            out_of_fold_predictions[test] = (scores[test] >= threshold).astype(np.int8)
            window_rows.append(
                {
                    "suite": key[0],
                    "model_id": key[1],
                    "scope": f"kfall_fold_{fold}",
                    "threshold": threshold,
                    "calibration_balanced_accuracy": calibration_bacc,
                    "calibration_mcc": calibration_mcc,
                    **binary_metrics(store.temporal_label[test], scores[test], threshold),
                }
            )
            event_rows.append(
                {
                    "suite": key[0],
                    "model_id": key[1],
                    "scope": f"kfall_fold_{fold}",
                    "threshold": threshold,
                    **_event_metrics(store, scores, thresholds, {fold}),
                }
            )
        window_rows.append(
            {
                "suite": key[0],
                "model_id": key[1],
                "scope": "pooled_participant_oof",
                "threshold": "fold_specific",
                "calibration_balanced_accuracy": "",
                "calibration_mcc": "",
                **_classification_metrics_from_predictions(
                    store.temporal_label, scores, out_of_fold_predictions
                ),
            }
        )
        event_rows.append(
            {
                "suite": key[0],
                "model_id": key[1],
                "scope": "pooled_participant_oof",
                "threshold": "fold_specific",
                **_event_metrics(store, scores, thresholds, set(range(5))),
            }
        )
    return window_rows, event_rows, ensemble_scores


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(rows: list[dict[str, Any]], field: str) -> str:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def _write_report(
    output_dir: Path,
    summary: dict[str, Any],
    zero_shot_rows: list[dict[str, Any]],
    calibrated_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    ensemble_scores: dict[tuple[str, str], np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics_recording_zero_shot.csv", zero_shot_rows)
    _write_csv(output_dir / "metrics_temporal_calibrated.csv", calibrated_rows)
    _write_csv(output_dir / "event_metrics.csv", event_rows)
    for (suite, model_id), scores in ensemble_scores.items():
        np.savez_compressed(
            output_dir / f"window_scores__{suite}__{model_id}.npz",
            scores=np.asarray(scores, dtype=np.float32),
        )
    lines = [
        "# Provisional KFall external evaluation",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Status: `{summary['status']}`",
        f"- Jobs: {summary['completed_jobs']}/{summary['scheduled_jobs']}",
        f"- Data quality: `{summary['data_quality_status']}`",
        "- KFall was not used for model weights, normalisation, or early stopping.",
        "- Positive window policy: decision time from onset through impact, inclusive.",
        "- One above-threshold positive window is an event detection; N-of-M is not tested.",
        "",
        "## Zero-shot recording transfer",
        "",
        "| Suite | Model | Validation variants | Recording BAcc | Sensitivity | "
        "Specificity | Window ROC AUC | Top-window hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in zero_shot_rows:
        grouped.setdefault((str(row["suite"]), str(row["model_id"])), []).append(row)
    for (suite, model_id), rows in sorted(grouped.items()):
        lines.append(
            f"| {suite} | {model_id} | {len(rows)} | "
            f"{_mean_std(rows, 'balanced_accuracy')} | "
            f"{_mean_std(rows, 'recall_sensitivity')} | "
            f"{_mean_std(rows, 'specificity')} | "
            f"{_mean_std(rows, 'window_roc_auc')} | "
            f"{_mean_std(rows, 'top_window_temporal_hit_rate')} |"
        )
    pooled_windows = [row for row in calibrated_rows if row["scope"] == "pooled_participant_oof"]
    pooled_events = [row for row in event_rows if row["scope"] == "pooled_participant_oof"]
    if pooled_windows:
        lines.extend(
            [
                "",
                "## Frozen-weight participant-grouped threshold calibration",
                "",
                "| Suite | Model | Window BAcc | F1 | MCC | Event sensitivity | "
                "ADL recording FPR | Onset latency median (s) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        event_lookup = {(row["suite"], row["model_id"]): row for row in pooled_events}
        for row in pooled_windows:
            event = event_lookup[(row["suite"], row["model_id"])]
            lines.append(
                f"| {row['suite']} | {row['model_id']} | "
                f"{row['balanced_accuracy']:.4f} | {row['f1']:.4f} | "
                f"{row['mcc']:.4f} | {event['event_sensitivity']:.4f} | "
                f"{event['adl_recording_false_positive_rate']:.4f} | "
                f"{event['onset_latency_median_s']:.3f} |"
            )
    else:
        lines.extend(
            [
                "",
                "## Calibration status",
                "",
                "The smoke profile does not contain all five training variants, so the final "
                "participant-grouped calibrated comparison is intentionally unavailable.",
            ]
        )
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in summary["known_limitations"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(output_dir / "run_manifest.json", summary)


def _report_paths(
    run_root: Path,
    training_store: WindowStore,
    external_store: KFallWindowStore,
    profile_name: str,
) -> Path:
    fingerprint = (
        f"{training_store.manifest['data_split_fingerprint'][:12]}-"
        f"{external_store.manifest['data_split_fingerprint'][:12]}"
    )
    return run_root / "reports" / fingerprint / profile_name


def run_kfall_experiment(
    *,
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    config: dict[str, Any],
    profile_name: str,
    resume: bool,
    environment: dict[str, Any] | None,
) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    training_path, training_manifest = prepare_window_store(
        project_root=project_root, cache_root=cache_root, config=config
    )
    external_path, external_manifest = prepare_kfall_window_store(
        project_root=project_root, cache_root=cache_root, config=config
    )
    training_store = load_window_store(training_path)
    external_store = load_kfall_window_store(external_path)
    external_values = external_store.load_raw()
    jobs = build_jobs(
        suites=KFALL_SUITES,
        models=KFALL_MODELS,
        folds=tuple(int(value) for value in profile["validation_folds"]),
    )
    registry_hash = _registry_hash(config)
    status_path = run_root / profile_name / "status.json"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
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
            result = _run_job(
                training_store=training_store,
                external_store=external_store,
                external_values=external_values,
                job=job,
                config=config,
                profile_name=profile_name,
                run_root=run_root,
                registry_hash=registry_hash,
                resume=resume,
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
    elapsed = time.perf_counter() - started
    zero_shot = _zero_shot_rows(results, external_store, float(config["mil_top_fraction"]))
    calibrated, events, ensemble_scores = _calibrated_rows(results, external_store)
    summary = {
        "status": "PASS" if not failures else "PARTIAL",
        "experiment_version": config["experiment_version"],
        "profile": profile_name,
        "scheduled_jobs": len(jobs),
        "completed_jobs": len(results),
        "computed_jobs_this_invocation": sum(r["status"] == "computed" for r in results),
        "cached_jobs_this_invocation": sum(r["status"] == "cached" for r in results),
        "failed_or_unavailable_jobs": failures,
        "elapsed_seconds": elapsed,
        "checkpoint_job_seconds": float(
            np.sum([r["metadata"]["total_seconds"] for r in results])
        ),
        "runtime_budget_seconds": profile["runtime_budget_seconds"],
        "training_window_store": str(training_path),
        "external_window_store": str(external_path),
        "training_window_manifest": training_manifest,
        "external_window_manifest": external_manifest,
        "registry_hash": registry_hash,
        "random_seed": int(config["random_seed"]),
        "training_protocol": "four_folds_train_one_fold_validation",
        "external_model_weight_use": "none",
        "external_threshold_use": (
            "participant_grouped_calibration_only"
            if len(ensemble_scores)
            else "not_available_in_smoke"
        ),
        "alarm_policy": config["alarm_policy"],
        "data_quality_status": config["data_quality_status"],
        "known_limitations": list(config["known_limitations"]),
        "environment": environment,
    }
    output_dir = _report_paths(run_root, training_store, external_store, profile_name)
    _write_report(output_dir, summary, zero_shot, calibrated, events, ensemble_scores)
    _atomic_json(
        status_path,
        {
            "state": "finished",
            "profile": profile_name,
            "status": summary["status"],
            "completed_jobs": len(results),
            "scheduled_jobs": len(jobs),
            "output_dir": str(output_dir),
            "elapsed_seconds": elapsed,
        },
    )
    return {**summary, "output_dir": str(output_dir)}


def regenerate_kfall_report(
    *,
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    config: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    training_path, training_manifest = prepare_window_store(
        project_root=project_root, cache_root=cache_root, config=config
    )
    external_path, external_manifest = prepare_kfall_window_store(
        project_root=project_root, cache_root=cache_root, config=config
    )
    training_store = load_window_store(training_path)
    external_store = load_kfall_window_store(external_path)
    registry_hash = _registry_hash(config)
    jobs = build_jobs(
        suites=KFALL_SUITES,
        models=KFALL_MODELS,
        folds=tuple(int(value) for value in profile["validation_folds"]),
    )
    results: list[dict[str, Any]] = []
    for job in jobs:
        checkpoint = _checkpoint_path(
            run_root,
            registry_hash,
            training_store.manifest["data_split_fingerprint"],
            external_store.manifest["data_split_fingerprint"],
            profile_name,
            job,
        )
        if not checkpoint.is_file():
            raise ValueError(f"Missing KFall checkpoint: {job.run_key}")
        result = _load_checkpoint(checkpoint)
        results.append({"status": "cached", "path": str(checkpoint), **result})
    zero_shot = _zero_shot_rows(results, external_store, float(config["mil_top_fraction"]))
    calibrated, events, ensemble_scores = _calibrated_rows(results, external_store)
    output_dir = _report_paths(run_root, training_store, external_store, profile_name)
    existing_manifest = output_dir / "run_manifest.json"
    existing_environment = None
    if existing_manifest.is_file():
        existing_environment = json.loads(existing_manifest.read_text(encoding="utf-8")).get(
            "environment"
        )
    summary = {
        "status": "PASS",
        "experiment_version": config["experiment_version"],
        "profile": profile_name,
        "scheduled_jobs": len(jobs),
        "completed_jobs": len(results),
        "computed_jobs_this_invocation": 0,
        "cached_jobs_this_invocation": len(results),
        "failed_or_unavailable_jobs": [],
        "elapsed_seconds": 0.0,
        "checkpoint_job_seconds": float(
            np.sum([r["metadata"]["total_seconds"] for r in results])
        ),
        "runtime_budget_seconds": profile["runtime_budget_seconds"],
        "training_window_store": str(training_path),
        "external_window_store": str(external_path),
        "training_window_manifest": training_manifest,
        "external_window_manifest": external_manifest,
        "registry_hash": registry_hash,
        "random_seed": int(config["random_seed"]),
        "training_protocol": "four_folds_train_one_fold_validation",
        "external_model_weight_use": "none",
        "external_threshold_use": (
            "participant_grouped_calibration_only"
            if len(ensemble_scores)
            else "not_available_in_smoke"
        ),
        "alarm_policy": config["alarm_policy"],
        "data_quality_status": config["data_quality_status"],
        "known_limitations": list(config["known_limitations"]),
        "environment": existing_environment,
    }
    _write_report(output_dir, summary, zero_shot, calibrated, events, ensemble_scores)
    return {**summary, "output_dir": str(output_dir)}
