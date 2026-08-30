"""Build validation-selected, final-refit two-file ONNX research candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml

from .cloud_models import (
    MODEL_RELEASE_CONTRACT_VERSION,
    MODEL_RELEASE_PREFIX,
    MODEL_RELEASE_SCHEMA,
    validate_model_release,
)
from .cloud_results import _run_dir, _validate_run
from .configuration import canonical_sha256, load_experiment
from .engine import split_indices
from .evaluation import best_threshold
from .metrics import alarm_policy_metrics, binary_classification_metrics
from .onnx_export import export_final_sequence_onnx
from .progress import NullProgressReporter, ProgressReporter
from .runtime import WorkPaths
from .unified_models import (
    CudaSequenceTrainer,
    normalize,
    select_execution_mode,
    sequence_normalization,
)
from .window_cache import (
    UnifiedWindowStore,
    load_unified_window_store,
    prepare_unified_window_store,
)

RELEASE_BUILD_SCHEMA = 1
SELECTION_SCHEMA = "imu_validation_oof_selection_v1"
FINAL_TRAINING_SCHEMA = "imu_final_refit_run_v1"
SEQUENCE_MODELS = ("torch_1d_cnn", "torch_lstm", "torch_cnn_lstm")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
UNITS = ("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid YAML file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(project_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must stay inside the project")
    path = (project_root / relative).resolve()
    if project_root.resolve() not in path.parents:
        raise ValueError(f"{name} resolves outside the project")
    return path


def load_release_build_config(project_root: Path, path: Path) -> dict[str, Any]:
    payload = _yaml(path)
    required = {
        "schema_version",
        "id",
        "source_run_id",
        "source_experiment_config",
        "models",
        "seed",
        "training_recipe",
        "precision",
        "deployment_inference_interval_seconds",
        "event_sensitivity_tolerance_percentage_points",
        "onnx_parity_batch_size",
        "onnx_parity_rtol",
        "onnx_parity_atol",
        "known_limitations",
    }
    if set(payload) != required or payload.get("schema_version") != RELEASE_BUILD_SCHEMA:
        raise ValueError("Release build config fields or schema are invalid")
    for name in ("id", "source_run_id"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"Release build {name} is invalid")
    source_experiment_path = _project_path(
        project_root,
        payload["source_experiment_config"],
        name="source_experiment_config",
    )
    models = payload["models"]
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("Release build must define the three sequence candidates")
    model_ids = []
    release_ids = []
    for item in models:
        if not isinstance(item, dict) or set(item) != {"model_id", "release_id", "name"}:
            raise ValueError("Release build model entry is invalid")
        model_ids.append(item["model_id"])
        release_ids.append(item["release_id"])
        if not all(isinstance(item[name], str) and item[name] for name in item):
            raise ValueError("Release build model identity is invalid")
    if tuple(model_ids) != SEQUENCE_MODELS or len(set(release_ids)) != len(release_ids):
        raise ValueError("Release build model order or release IDs are invalid")
    if payload["training_recipe"] != "natural" or payload["precision"] != "fp32":
        raise ValueError("Final release build is fixed to natural FP32 training")
    if int(payload["seed"]) < 0:
        raise ValueError("Release build seed is invalid")
    interval_seconds = float(payload["deployment_inference_interval_seconds"])
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Deployment inference interval must be finite and positive")
    tolerance = float(payload["event_sensitivity_tolerance_percentage_points"])
    if not 0.0 <= tolerance <= 100.0:
        raise ValueError("Event sensitivity tolerance is invalid")
    if int(payload["onnx_parity_batch_size"]) <= 0:
        raise ValueError("ONNX parity batch size is invalid")
    if not isinstance(payload["known_limitations"], list):
        raise ValueError("Release build known_limitations is invalid")
    payload["config_sha256"] = canonical_sha256(payload)
    payload["source_experiment_config_path"] = str(source_experiment_path)
    return payload


def _source_artifacts(run_dir: Path) -> dict[tuple[str, int], dict[str, Path | int]]:
    result: dict[tuple[str, int], dict[str, Path | int]] = {}
    for spec_path in sorted((run_dir / "models").glob("*/model_spec.json")):
        spec = _json(spec_path)
        job = spec.get("job")
        if not isinstance(job, dict):
            continue
        model_id = str(job.get("model_id") or "")
        if model_id not in SEQUENCE_MODELS or job.get("training_recipe") != "natural":
            continue
        try:
            fold = int(job["fold"])
            seed = int(job["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Source model_spec has an invalid job identity") from error
        key = (model_id, fold)
        if key in result:
            raise ValueError(f"Duplicate source natural model artifact: {key}")
        directory = spec_path.parent
        onnx_path = directory / "model.onnx"
        normalization_path = directory / "normalization.npz"
        history_path = directory / "training_history.json"
        if not all(path.is_file() for path in (onnx_path, normalization_path, history_path)):
            raise ValueError(f"Source natural model artifact is incomplete: {key}")
        result[key] = {
            "onnx": onnx_path,
            "normalization": normalization_path,
            "history": history_path,
            "seed": seed,
        }
    expected = {(model_id, fold) for model_id in SEQUENCE_MODELS for fold in range(5)}
    if set(result) != expected:
        raise ValueError("Source run lacks the complete 3-model natural 5-fold artifact set")
    return result


def _scheduled(indices: np.ndarray, store: UnifiedWindowStore, interval_s: float) -> np.ndarray:
    step = int(round(float(store.manifest["sampling_rate_hz"]) * interval_s))
    selected = np.asarray(indices, dtype=np.int64)
    return selected[store.start_sample[selected] % step == 0]


def _onnx_scores(
    onnx_path: Path,
    normalization_path: Path,
    raw: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    with np.load(normalization_path, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float32)
        scale = np.asarray(archive["scale"], dtype=np.float32)
    values = normalize(raw, mean, scale)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "imu" or session.get_outputs()[0].name != "fall_score":
        raise ValueError(f"Source ONNX contract is invalid: {onnx_path}")
    scores = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        scores[start:stop] = np.asarray(
            session.run(["fall_score"], {"imu": values[start:stop]})[0]
        ).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError(f"Source ONNX produced non-finite scores: {onnx_path}")
    return scores


def _participant_proof(
    store: UnifiedWindowStore,
    fold_indices: dict[int, np.ndarray],
    expected_indices: np.ndarray,
) -> dict[str, Any]:
    seen: dict[str, int] = {}
    fold_counts: list[int] = []
    for fold, indices in sorted(fold_indices.items()):
        sequence_ids = np.unique(store.sequence_index[indices])
        participants = {
            f"{store.dataset_id[sequence]}::{store.participant_id[sequence]}"
            for sequence in sequence_ids
        }
        if fold != len(fold_counts):
            raise ValueError("Validation folds must be ordered 0 through 4")
        fold_counts.append(len(participants))
        for participant in participants:
            if participant in seen:
                raise ValueError(
                    f"Validation OOF participant appears more than once: {participant}"
                )
            seen[participant] = fold
    expected_sequences = np.unique(store.sequence_index[expected_indices])
    expected = {
        f"{store.dataset_id[sequence]}::{store.participant_id[sequence]}"
        for sequence in expected_sequences
    }
    if set(seen) != expected:
        raise ValueError("Validation OOF participant coverage is incomplete")
    return {
        "status": "PASS",
        "participant_count": len(seen),
        "appearances_per_participant": 1,
        "validation_fold_participant_counts": fold_counts,
        "assignment_sha256": canonical_sha256(dict(sorted(seen.items()))),
    }


def _choose_policy(
    rows: list[dict[str, Any]], *, tolerance_percentage_points: float
) -> dict[str, Any]:
    maximum = max(float(row["event_sensitivity"]) for row in rows)
    cutoff = maximum - tolerance_percentage_points / 100.0

    def finite_latency(row: dict[str, Any]) -> float:
        value = float(row["onset_latency_p95_s"])
        return value if math.isfinite(value) else float("inf")

    eligible = [row for row in rows if float(row["event_sensitivity"]) >= cutoff - 1e-12]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["adl_alarm_episodes_per_hour"]),
            finite_latency(row),
            str(row["alarm_policy_id"]),
        ),
    )
    return {
        "maximum_validation_event_sensitivity": maximum,
        "eligibility_cutoff": cutoff,
        "eligible_policy_ids": sorted(str(row["alarm_policy_id"]) for row in eligible),
        "selected": selected,
    }


def _build_selection_evidence(
    *,
    run_dir: Path,
    output_dir: Path,
    store: UnifiedWindowStore,
    raw: np.ndarray,
    source_config: dict[str, Any],
    build_config: dict[str, Any],
    artifacts: dict[tuple[str, int], dict[str, Path | int]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = []
    batch_size = int(build_config["onnx_parity_batch_size"])
    interval_s = float(build_config["deployment_inference_interval_seconds"])
    data_filter = source_config["data_view"]["dataset_filter"]
    expected_mask = (store.sequence_fold_id[store.sequence_index] >= 0) & (
        store.temporal_label >= 0
    )
    if data_filter is not None:
        expected_mask &= np.isin(store.window_datasets(), data_filter)
    expected_indices = _scheduled(np.flatnonzero(expected_mask), store, interval_s)
    participant_proof: dict[str, Any] | None = None

    for model_id in SEQUENCE_MODELS:
        pooled_indices = []
        pooled_scores = []
        validation_by_fold: dict[int, np.ndarray] = {}
        source_onnx = []
        for test_fold in range(5):
            artifact = artifacts[(model_id, test_fold)]
            split = split_indices(store, source_config, test_fold, int(artifact["seed"]))
            indices = _scheduled(split.validation, store, interval_s)
            scores = _onnx_scores(
                Path(artifact["onnx"]),
                Path(artifact["normalization"]),
                raw[indices],
                batch_size=batch_size,
            )
            validation_by_fold[split.validation_fold] = indices
            pooled_indices.append(indices)
            pooled_scores.append(scores)
            source_onnx.append(
                {
                    "test_fold": test_fold,
                    "validation_fold": split.validation_fold,
                    "sha256": _sha256(Path(artifact["onnx"])),
                }
            )
        indices = np.concatenate(pooled_indices)
        scores = np.concatenate(pooled_scores)
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        scores = scores[order]
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Validation OOF windows overlap for {model_id}")
        labels = store.temporal_label[indices]
        if participant_proof is None:
            participant_proof = _participant_proof(store, validation_by_fold, expected_indices)
        threshold, balanced_accuracy, mcc = best_threshold(labels, scores)
        policies = source_config["alarm_policy"]["policies"]
        alarm_rows = [
            alarm_policy_metrics(
                store,
                indices,
                scores,
                threshold,
                policy,
                sequence_scope=np.unique(store.sequence_index[expected_indices]),
                decision_interval_seconds=interval_s,
            )
            for policy in policies
        ]
        policy_selection = _choose_policy(
            alarm_rows,
            tolerance_percentage_points=float(
                build_config["event_sensitivity_tolerance_percentage_points"]
            ),
        )
        score_path = output_dir / f"{model_id}.validation_oof.npz"
        temporary = score_path.with_suffix(f".npz.tmp-{os.getpid()}")
        with temporary.open("wb") as destination:
            np.savez_compressed(
                destination,
                window_index=indices,
                label=labels.astype(np.int8),
                fall_score=scores.astype(np.float32),
            )
        temporary.replace(score_path)
        selected_policy = policy_selection["selected"]
        model_rows.append(
            {
                "model_id": model_id,
                "training_recipe": "natural",
                "selection_eligible": True,
                "metric_split": "validation_oof",
                "deployment_inference_interval_seconds": interval_s,
                "windows": len(indices),
                "positive_windows": int(np.count_nonzero(labels == 1)),
                "score_threshold": threshold,
                "threshold_selection_method": "maximum_validation_balanced_accuracy",
                "validation_balanced_accuracy": balanced_accuracy,
                "validation_mcc": mcc,
                "window_metrics": binary_classification_metrics(labels, scores, threshold),
                "alarm_policy_selection": {
                    **policy_selection,
                    "selected_policy_id": selected_policy["alarm_policy_id"],
                },
                "alarm_policy_metrics": alarm_rows,
                "source_onnx": source_onnx,
                "score_artifact": {
                    "filename": score_path.name,
                    "size_bytes": score_path.stat().st_size,
                    "sha256": _sha256(score_path),
                },
            }
        )
    assert participant_proof is not None
    evidence = {
        "schema_version": SELECTION_SCHEMA,
        "selection_scope": "validation_only_oof",
        "selection_eligible": True,
        "source_run_id": run_dir.name,
        "snapshot_sha256": source_config["snapshot_sha256"],
        "data_split_fingerprint": store.manifest["data_split_fingerprint"],
        "window_schema_version": store.manifest["window_schema_version"],
        "source_stride_seconds": interval_s,
        "participant_proof": participant_proof,
        "models": model_rows,
    }
    evidence_path = output_dir / "selection_evidence.json"
    _atomic_json(evidence_path, evidence)
    return {
        **evidence,
        "artifact": {
            "filename": evidence_path.name,
            "size_bytes": evidence_path.stat().st_size,
            "sha256": _sha256(evidence_path),
        },
    }


def _best_epochs(
    artifacts: dict[tuple[str, int], dict[str, Path | int]], model_id: str
) -> tuple[list[int], int]:
    values = []
    for fold in range(5):
        history = _json(Path(artifacts[(model_id, fold)]["history"]))
        epoch = history.get("best_epoch")
        if not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(f"Source CV best epoch is invalid: {model_id}/fold-{fold}")
        values.append(epoch)
    return values, int(statistics.median(values))


def _selection_row(evidence: dict[str, Any], model_id: str) -> dict[str, Any]:
    rows = [item for item in evidence["models"] if item["model_id"] == model_id]
    if len(rows) != 1:
        raise ValueError(f"Selection evidence lacks one row for {model_id}")
    return rows[0]


def _release_metadata(
    *,
    release: dict[str, str],
    build_config: dict[str, Any],
    source_config: dict[str, Any],
    source_run_manifest: dict[str, Any],
    source: dict[str, Any],
    run_id: str,
    selection: dict[str, Any],
    data_split_fingerprint: str,
    epochs_by_fold: list[int],
    final_epochs: int,
    training_indices: np.ndarray,
    training_labels: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    parity: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    selected_alarm = selection["alarm_policy_selection"]["selected"]
    inference_interval_seconds = float(
        build_config["deployment_inference_interval_seconds"]
    )
    source_stride_seconds = float(selection["source_stride_seconds"])
    if inference_interval_seconds != source_stride_seconds:
        raise ValueError("Release inference interval differs from selection source stride")
    training_stride_seconds = float(source_config["contract"]["window"]["stride_seconds"])
    policy_id = selection["alarm_policy_selection"]["selected_policy_id"]
    policy = next(
        item for item in source_config["alarm_policy"]["policies"] if item["id"] == policy_id
    )
    return {
        "schema_version": MODEL_RELEASE_SCHEMA,
        "contract_version": MODEL_RELEASE_CONTRACT_VERSION,
        "release_id": release["release_id"],
        "model_code": release["model_id"],
        "name": release["name"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "release_stage": "research_candidate",
        "source": {
            "selection_evidence": {
                "source_run_id": source_run_manifest["run_id"],
                "source_commit": source_run_manifest["source"].get("commit"),
                "model_id": release["model_id"],
                "training_recipe": "natural",
                "data_snapshot_fingerprint": source_config["snapshot_sha256"],
                "split_fingerprint": data_split_fingerprint,
                "selection_scope": "validation_only_oof",
                "metric_split": "validation_oof",
                "selection_eligible": True,
                "source_stride_seconds": source_stride_seconds,
                "participant_once": selection["participant_proof"],
                "threshold_selection": {
                    "method": selection["threshold_selection_method"],
                    "tie_break": "smallest_threshold_among_equal_balanced_accuracy",
                },
                "trigger_policy_selection": {
                    "method": "event_sensitivity_guard_then_alarm_rate_latency_id",
                    "tie_break": "alarm_rate_then_latency_then_policy_id",
                },
            },
            "final_training": {
                **source,
                "run_id": run_id,
                "model_id": release["model_id"],
                "training_recipe": "natural",
                "seed": int(build_config["seed"]),
                "epochs_by_cv_fold": epochs_by_fold,
                "fixed_epochs": final_epochs,
                "fixed_epoch_source": "median_cv_best_epoch",
                "training_scope": "all_public_temporal_development_participants",
                "actual_epochs": final_epochs,
                "early_stopping": False,
            },
        },
        "data": {
            "base_snapshot_id": source_config["snapshot"]["base_snapshot_id"],
            "snapshot_sha256": source_config["snapshot_sha256"],
            "data_split_fingerprint": data_split_fingerprint,
            "snapshot_fingerprint": source_config["snapshot_sha256"],
            "split_fingerprint": data_split_fingerprint,
            "data_view_id": source_config["data_view"]["id"],
            "training_scope": "all_public_temporal_development_participants",
            "team_data_included": False,
            "training_windows": len(training_indices),
            "positive_training_windows": int(np.count_nonzero(training_labels == 1)),
            "negative_training_windows": int(np.count_nonzero(training_labels == 0)),
        },
        "input": {
            "semantic": "si_window",
            "name": "imu",
            "dtype": "float32",
            "shape": [None, 50, 6],
            "sampling_rate_hz": 25.0,
            "channels": list(CHANNELS),
            "units": list(UNITS),
            "axis_frame": "sensor_local",
            "gravity": "retained",
        },
        "output": {
            "semantic": "fall_score",
            "name": "fall_score",
            "dtype": "float32",
            "shape": [None],
            "probability_calibrated": False,
        },
        "preprocessing": {
            "location": "onnx_graph",
            "normalization": {
                "embedded": True,
                "fit_scope": "all_final_training_windows",
                "mean": mean.tolist(),
                "scale": scale.tolist(),
            },
        },
        "windowing": {
            "window_seconds": 2.0,
            "training_stride_seconds": training_stride_seconds,
            "inference_interval_seconds": inference_interval_seconds,
            "anchor": "window_end",
            "reset_on": ["new_sequence", "stream_gap"],
            "refill_frames_after_reset": 50,
            "cooldown_history_semantics": (
                "trigger history continues during cooldown; a still-triggered policy may emit "
                "again when cooldown expires"
            ),
        },
        "decision": {
            "status": "provisional_validation_derived",
            "score_threshold": {
                "value": selection["score_threshold"],
                "comparison": ">=",
                "selection_method": selection["threshold_selection_method"],
                "selection_split": "validation_oof",
            },
            "trigger_policy": {
                "policy_id": policy["id"],
                "required_positive_windows": policy["required_positive_windows"],
                "lookback_windows": policy["lookback_windows"],
                "consecutive": policy["consecutive"],
                "cooldown_seconds": policy["cooldown_seconds"],
            },
        },
        "metrics": {
            "metric_split": "validation_oof",
            "selection_eligible": True,
            "final_model_independently_evaluated": False,
            "window": selection["window_metrics"],
            "alarm": selected_alarm,
            "interpretation": (
                "Metrics select the threshold and alarm policy from CV validation OOF "
                "scores; they are not an independent accuracy estimate for the final-refit model."
            ),
        },
        "verification": {
            "parity": {key: value for key, value in parity.items() if key != "golden_fixtures"},
            "golden_fixtures": parity["golden_fixtures"],
        },
        "validation": {
            "onnx_checker": {"status": "PASS"},
            "python_onnxruntime_parity": {
                "status": "PASS",
                "scope": parity["scope"],
                "windows": parity["samples"],
            },
            "external_runtime": {"status": "not_tested"},
            "device_replay": {"status": "not_tested"},
        },
        "known_limitations": list(build_config["known_limitations"]),
        "model": {
            "filename": "model.onnx",
            "object_key": (f"{MODEL_RELEASE_PREFIX}/{release['release_id']}/model.onnx"),
            "size_bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
            "content_type": "application/octet-stream",
        },
    }


def plan_model_releases(
    *,
    project_root: Path,
    paths: WorkPaths,
    config_path: Path,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Validate all immutable inputs and describe the release build without training."""

    reporter = progress or NullProgressReporter()
    build_config = load_release_build_config(project_root, config_path)
    source_config = load_experiment(
        project_root,
        Path(build_config["source_experiment_config_path"]),
        snapshot_path=paths.data / "active.json",
    )
    source_run = _run_dir(paths.runs, str(build_config["source_run_id"]))
    with reporter.task("Validating the immutable source benchmark run"):
        source_run_manifest, entries = _validate_run(source_run)
    if source_run_manifest.get("snapshot_sha256") != source_config["snapshot_sha256"]:
        raise ValueError("Source run and active data snapshot differ")
    if source_run_manifest.get("experiment_id") != source_config["id"]:
        raise ValueError("Source run and configured experiment differ")
    artifacts = _source_artifacts(source_run)
    releases = []
    for release in build_config["models"]:
        epochs_by_fold, fixed_epochs = _best_epochs(artifacts, release["model_id"])
        releases.append(
            {
                **release,
                "validation_oof_source_models": 5,
                "epochs_by_cv_fold": epochs_by_fold,
                "fixed_final_refit_epochs": fixed_epochs,
                "payload_files": ["model.onnx", "metadata.json"],
            }
        )
    return {
        "status": "PASS",
        "training_started": False,
        "build_id": build_config["id"],
        "source_run_id": source_run.name,
        "source_run_files": len(entries),
        "source_snapshot_sha256": source_config["snapshot_sha256"],
        "selection_scope": "validation_only_oof",
        "selection_source_models": len(artifacts),
        "final_refit_models": len(releases),
        "input_contract": {
            "semantic": "si_window",
            "dtype": "float32",
            "shape": [None, 50, 6],
            "sampling_rate_hz": 25,
            "normalization": "embedded_in_onnx_graph",
        },
        "releases": releases,
    }


def build_model_releases(
    *,
    project_root: Path,
    paths: WorkPaths,
    config_path: Path,
    source: dict[str, Any],
    environment: dict[str, Any],
    resume: bool,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = progress or NullProgressReporter()
    if source.get("dirty") is not False or not source.get("commit"):
        raise ValueError("Final model releases require a clean committed source tree")
    build_config = load_release_build_config(project_root, config_path)
    source_config = load_experiment(
        project_root,
        Path(build_config["source_experiment_config_path"]),
        snapshot_path=paths.data / "active.json",
    )
    source_run = _run_dir(paths.runs, str(build_config["source_run_id"]))
    with reporter.task("Validating the immutable source benchmark run"):
        source_run_manifest, _entries = _validate_run(source_run)
    if source_run_manifest.get("snapshot_sha256") != source_config["snapshot_sha256"]:
        raise ValueError("Source run and active data snapshot differ")
    if source_run_manifest.get("experiment_id") != source_config["id"]:
        raise ValueError("Source run and configured experiment differ")
    artifacts = _source_artifacts(source_run)
    run_fingerprint = canonical_sha256(
        {
            "config_sha256": build_config["config_sha256"],
            "source_run_id": source_run.name,
            "source_run_manifest_sha256": _sha256(source_run / "run_manifest.json"),
            "snapshot_sha256": source_config["snapshot_sha256"],
            "source_commit": source["commit"],
        }
    )
    run_id = f"{build_config['id']}-{run_fingerprint[:12]}"
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with reporter.task("Building or reusing the multi-event sliding-window cache"):
        cache_path, cache_manifest = prepare_unified_window_store(
            project_root=project_root,
            cache_root=paths.cache,
            config=source_config,
            progress=reporter,
        )
    store = load_unified_window_store(cache_path)
    raw = store.materialize("raw")
    with reporter.task("Reconstructing validation-only OOF selection evidence"):
        selection = _build_selection_evidence(
            run_dir=source_run,
            output_dir=run_dir / "selection",
            store=store,
            raw=raw,
            source_config=source_config,
            build_config=build_config,
            artifacts=artifacts,
        )

    data_filter = source_config["data_view"]["dataset_filter"]
    train_mask = (
        (store.fold_id >= 0)
        & (store.temporal_label >= 0)
        & (store.supervision_kind[store.sequence_index] == "temporal")
    )
    if data_filter is not None:
        train_mask &= np.isin(store.window_datasets(), data_filter)
    training_indices = np.flatnonzero(train_mask)
    training_labels = store.temporal_label[training_indices]
    if set(np.unique(training_labels).tolist()) != {0, 1}:
        raise ValueError("Final-refit training data must contain both classes")
    raw_train = raw[training_indices]
    mean, scale = sequence_normalization(raw_train)
    normalized_train = normalize(raw_train, mean, scale)
    model_catalog = source_config["model_catalog"]["models"]
    releases = []
    computed = 0
    cached = 0
    for release in build_config["models"]:
        release_dir = run_dir / "releases" / release["release_id"]
        if resume and release_dir.is_dir():
            try:
                metadata = validate_model_release(release_dir)
            except (OSError, TypeError, ValueError):
                metadata = None
            if metadata is not None:
                cached += 1
                releases.append(
                    {
                        "release_id": release["release_id"],
                        "status": "cached",
                        "release_dir": str(release_dir),
                        "model_sha256": metadata["model"]["sha256"],
                    }
                )
                continue
        epochs_by_fold, final_epochs = _best_epochs(artifacts, release["model_id"])
        decision = select_execution_mode(str(source_config["gpu_mode"]), (normalized_train,))
        with reporter.task(
            f"Final-refit {release['model_id']}", total=final_epochs, unit="epochs"
        ) as task:

            def on_epoch(epoch: int, maximum: int, loss: float, remaining: int) -> None:
                del maximum, remaining
                task.update(completed=epoch, detail=f"training_loss={loss:.5f}")

            trainer = CudaSequenceTrainer(
                release["model_id"],
                dict(model_catalog[release["model_id"]]["params"]),
                max_epochs=final_epochs,
                patience=final_epochs,
                top_fraction=float(
                    source_config["contract"]["supervision"]["recording"]["mil_top_fraction"]
                ),
                random_seed=int(build_config["seed"]),
                precision="fp32",
                execution_mode=decision.effective_mode,
                use_class_weight=False,
                epoch_callback=on_epoch,
            )
            trainer.fit_supervised_fixed_epochs(
                normalized_train, training_labels, epochs=final_epochs
            )
        native_scores = trainer.predict_proba(normalized_train)
        release_dir.mkdir(parents=True, exist_ok=True)
        model_path = release_dir / "model.onnx"
        with reporter.task(f"Full ONNX parity for {release['model_id']}"):
            parity = export_final_sequence_onnx(
                trainer,
                raw_train,
                mean,
                scale,
                native_scores,
                model_path,
                batch_size=int(build_config["onnx_parity_batch_size"]),
                rtol=float(build_config["onnx_parity_rtol"]),
                atol=float(build_config["onnx_parity_atol"]),
            )
        selected = {
            **_selection_row(selection, release["model_id"]),
            "source_stride_seconds": selection["source_stride_seconds"],
            "participant_proof": selection["participant_proof"],
        }
        metadata = _release_metadata(
            release=release,
            build_config=build_config,
            source_config=source_config,
            source_run_manifest=source_run_manifest,
            source=source,
            run_id=run_id,
            selection=selected,
            data_split_fingerprint=str(store.manifest["data_split_fingerprint"]),
            epochs_by_fold=epochs_by_fold,
            final_epochs=final_epochs,
            training_indices=training_indices,
            training_labels=training_labels,
            mean=mean,
            scale=scale,
            parity=parity,
            model_path=model_path,
        )
        _atomic_json(release_dir / "metadata.json", metadata)
        validated = validate_model_release(release_dir)
        computed += 1
        releases.append(
            {
                "release_id": release["release_id"],
                "status": "computed",
                "release_dir": str(release_dir),
                "model_sha256": validated["model"]["sha256"],
                "fixed_epochs": final_epochs,
            }
        )
    run_manifest = {
        "schema_version": FINAL_TRAINING_SCHEMA,
        "status": "PASS",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "environment": environment,
        "source_run_id": source_run.name,
        "config_path": str(config_path.resolve()),
        "config_sha256": build_config["config_sha256"],
        "snapshot_sha256": source_config["snapshot_sha256"],
        "cache_manifest": cache_manifest,
        "selection_evidence": selection["artifact"],
        "computed_releases_this_invocation": computed,
        "cached_releases_this_invocation": cached,
        "releases": releases,
    }
    _atomic_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest
