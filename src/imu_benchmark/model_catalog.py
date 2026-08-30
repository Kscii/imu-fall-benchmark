"""Build strict, display-ready metadata for published ONNX experiment runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .data import FEATURE_NAMES

EXPERIMENT_PUBLICATION_SCHEMA = "imu_benchmark_result_manifest_v2"
EXPERIMENT_CATALOG_SCHEMA = "imu_experiment_catalog_v0"
EXPERIMENT_CATALOG_CONTRACT_VERSION = "0.1.0"
FORMAL_EXPERIMENT_ID = "formal_baseline_temporal_core_onnx_v1"
ENGINEERING_EXPERIMENT_ID = "onnx_full_parity_preflight_v1"
EVIDENCE_LEVELS = {
    FORMAL_EXPERIMENT_ID: ("formal_cv", 65, 5),
    ENGINEERING_EXPERIMENT_ID: ("engineering", 7, 1),
}
WINDOW_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
WINDOW_UNITS = ("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")

WINDOW_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "mcc",
    "auroc",
    "auprc",
)
EVENT_METRICS = (
    "event_sensitivity",
    "adl_recording_false_positive_rate",
    "adl_false_positive_windows_per_hour",
    "onset_latency_median_s",
    "onset_latency_p95_s",
    "impact_offset_median_s",
)
ALARM_METRICS = (
    "event_sensitivity",
    "adl_recording_false_positive_rate",
    "adl_alarm_episodes_per_hour",
    "onset_latency_median_s",
    "onset_latency_p95_s",
    "impact_offset_median_s",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Required result table is missing: {path.name}")
    with path.open(newline="", encoding="utf-8") as source:
        return [dict(row) for row in csv.DictReader(source)]


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid YAML file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name} in published result table") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name} in published result table")
    return value


def _key(row: dict[str, str]) -> tuple[str, str, int, int]:
    try:
        return (
            row["model_id"],
            row["training_recipe"],
            int(row["fold"]),
            int(row["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Published result row has an invalid job identity") from error


def _truth(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def evidence_profile(run_manifest: dict[str, Any]) -> tuple[str, int, int]:
    experiment_id = str(run_manifest.get("experiment_id") or "")
    try:
        return EVIDENCE_LEVELS[experiment_id]
    except KeyError as error:
        raise ValueError(
            "Only the controlled formal or engineering ONNX run may be published"
        ) from error


def _evaluation_fingerprint(run_dir: Path, run_manifest: dict[str, Any]) -> str:
    provenance = _read_json(run_dir / "provenance.json")
    cache = provenance.get("cache_manifest")
    if not isinstance(cache, dict):
        raise ValueError("Run provenance lacks cache_manifest")
    payload = {
        "base_snapshot_id": run_manifest.get("base_snapshot_id"),
        "snapshot_sha256": run_manifest.get("snapshot_sha256"),
        "data_view_id": run_manifest.get("data_view_id"),
        "contract_sha256": cache.get("contract_sha256"),
        "data_split_fingerprint": cache.get("data_split_fingerprint"),
        "window_schema_version": cache.get("window_schema_version"),
        "feature_schema_version": cache.get("feature_schema_version"),
        "sampling_rate_hz": cache.get("sampling_rate_hz"),
        "window_samples": cache.get("window_samples"),
        "stride_seconds": cache.get("stride_seconds"),
    }
    if any(value is None for value in payload.values()):
        raise ValueError("Run provenance lacks evaluation fingerprint fields")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _input_contract(spec: dict[str, Any], feature_schema_version: str) -> dict[str, Any]:
    input_kind = spec.get("input_kind")
    normalization = spec.get("normalization")
    if not isinstance(normalization, dict) or not isinstance(
        normalization.get("enabled"), bool
    ):
        raise ValueError("model_spec normalization is invalid")
    enabled = bool(normalization["enabled"])
    if input_kind == "raw":
        semantic = "normalized_window" if enabled else "si_window"
        return {
            "semantic": semantic,
            "name": "imu",
            "dtype": "float32",
            "shape": [None, 50, 6],
            "sampling_rate_hz": 25.0,
            "window_seconds": 2.0,
            "channels": list(WINDOW_CHANNELS),
            "units": list(WINDOW_UNITS),
            "preprocessing_location": "runtime" if enabled else "none",
            "normalization": normalization,
        }
    if input_kind != "features":
        raise ValueError(f"Unsupported model input_kind: {input_kind}")
    feature_order_bytes = json.dumps(
        FEATURE_NAMES, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return {
        "semantic": "engineered_features",
        "name": "features",
        "dtype": "float32",
        "shape": [None, len(FEATURE_NAMES)],
        "feature_schema_version": feature_schema_version,
        "feature_order": list(FEATURE_NAMES),
        "feature_order_sha256": hashlib.sha256(feature_order_bytes).hexdigest(),
        "preprocessing_location": "runtime",
        "normalization": normalization,
    }


def _metric_values(row: dict[str, str], names: tuple[str, ...]) -> dict[str, float]:
    return {name: _float(row, name) for name in names}


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def build_experiment_catalog(
    run_dir: Path,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return catalog metadata derived exclusively from auditable run artifacts."""

    evidence_level, expected_jobs, expected_folds = evidence_profile(run_manifest)
    metrics_rows = _read_csv(run_dir / "metrics.csv")
    event_rows = _read_csv(run_dir / "event_metrics.csv")
    all_alarm_rows = _read_csv(run_dir / "alarm_metrics.csv")
    alarm_rows = [
        row
        for row in all_alarm_rows
        if _truth(row.get("reference_policy"))
    ]
    parity_rows = _read_csv(run_dir / "onnx_parity.csv")
    if len(metrics_rows) != expected_jobs or len(event_rows) != expected_jobs:
        raise ValueError("Published run does not have one metric row per job")
    if len(alarm_rows) != expected_jobs:
        raise ValueError("Published run does not have one reference alarm policy per job")
    if len(parity_rows) != expected_jobs * 2:
        raise ValueError("Published run does not have validation and test parity per job")

    metrics = {_key(row): row for row in metrics_rows}
    events = {_key(row): row for row in event_rows}
    alarms = {_key(row): row for row in alarm_rows}
    alarm_policies: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in all_alarm_rows:
        alarm_policies[_key(row)].append(row)
    parity: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in parity_rows:
        parity[_key(row)].append(row)
    if set(metrics) != set(events) or set(metrics) != set(alarms) or set(metrics) != set(parity):
        raise ValueError("Published run tables describe different jobs")

    provenance = _read_json(run_dir / "provenance.json")
    cache = provenance.get("cache_manifest")
    if not isinstance(cache, dict) or not isinstance(cache.get("feature_schema_version"), str):
        raise ValueError("Run provenance lacks feature schema")
    resolved = _read_yaml(run_dir / "resolved_config.yaml")
    alarm_config = resolved.get("alarm_policy")
    if not isinstance(alarm_config, dict):
        raise ValueError("Resolved config lacks alarm_policy")
    policy_rows = alarm_config.get("policies")
    if not isinstance(policy_rows, list) or not policy_rows:
        raise ValueError("Resolved config has no alarm policies")
    policies: dict[str, dict[str, Any]] = {}
    for policy in policy_rows:
        if not isinstance(policy, dict) or not isinstance(policy.get("id"), str):
            raise ValueError("Resolved config has an invalid alarm policy")
        if policy["id"] in policies:
            raise ValueError("Resolved config has duplicate alarm policies")
        policies[policy["id"]] = policy
    if set(policies) != {
        str(row.get("alarm_policy_id") or "") for row in all_alarm_rows
    }:
        raise ValueError("Alarm result policies differ from the resolved config")
    alarm_policy_sha256 = resolved.get("alarm_policy_sha256")
    if not isinstance(alarm_policy_sha256, str) or len(alarm_policy_sha256) != 64:
        raise ValueError("Resolved config lacks alarm_policy_sha256")
    environment = _read_json(run_dir / "environment.json")

    artifacts: list[dict[str, Any]] = []
    artifact_keys: set[tuple[str, str, int, int]] = set()
    for spec_path in sorted((run_dir / "models").glob("*/model_spec.json")):
        artifact_id = spec_path.parent.name
        spec = _read_json(spec_path)
        job = spec.get("job")
        if not isinstance(job, dict):
            raise ValueError(f"model_spec lacks job: {artifact_id}")
        key = _key(
            {
                name: str(job.get(name, ""))
                for name in ("model_id", "training_recipe", "fold", "seed")
            }
        )
        if key not in metrics or key in artifact_keys:
            raise ValueError(
                "model_spec job is missing or duplicated in result tables: "
                f"{artifact_id}"
            )
        artifact_keys.add(key)
        onnx_path = spec_path.parent / "model.onnx"
        if not onnx_path.is_file() or onnx_path.stat().st_size <= 0:
            raise ValueError(f"ONNX artifact is missing: {artifact_id}")
        parity_splits = sorted(parity[key], key=lambda row: row.get("split", ""))
        if [row.get("split") for row in parity_splits] != ["test", "validation"]:
            raise ValueError(f"ONNX parity splits are incomplete: {artifact_id}")
        onnx_sha256 = {row.get("onnx_sha256") for row in parity_splits}
        if len(onnx_sha256) != 1:
            raise ValueError(f"ONNX parity rows disagree on SHA-256: {artifact_id}")
        verified_onnx_sha256 = next(iter(onnx_sha256))
        if verified_onnx_sha256 != _sha256(onnx_path):
            raise ValueError(
                f"ONNX artifact differs from the parity evidence: {artifact_id}"
            )
        threshold = _float(metrics[key], "threshold")
        if any(_float(row, "threshold") != threshold for row in alarm_policies[key]):
            raise ValueError(f"Alarm rows disagree on score threshold: {artifact_id}")
        artifact_policies = []
        for row in sorted(
            alarm_policies[key], key=lambda item: item.get("alarm_policy_id", "")
        ):
            policy_id = str(row.get("alarm_policy_id") or "")
            definition = policies[policy_id]
            artifact_policies.append(
                {
                    "policy_id": policy_id,
                    "required_positive_windows": int(
                        definition["required_positive_windows"]
                    ),
                    "lookback_windows": int(definition["lookback_windows"]),
                    "consecutive": bool(definition["consecutive"]),
                    "cooldown_seconds": float(definition["cooldown_seconds"]),
                    "reference_policy": _truth(row.get("reference_policy")),
                    "validation_pareto": _truth(row.get("validation_pareto")),
                    "metrics": _metric_values(row, ALARM_METRICS),
                }
            )
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "model_id": key[0],
                "training_recipe": key[1],
                "fold": key[2],
                "seed": key[3],
                "backend": spec.get("backend"),
                "input": _input_contract(spec, cache["feature_schema_version"]),
                "output": {
                    "semantic": "fall_score",
                    "name": "fall_score",
                    "dtype": "float32",
                    "shape": [None],
                    "probability_calibrated": False,
                },
                "metrics": {
                    "window": _metric_values(metrics[key], WINDOW_METRICS),
                    "event": _metric_values(events[key], EVENT_METRICS),
                    "alarm": {
                        "policy_id": alarms[key].get("alarm_policy_id"),
                        **_metric_values(alarms[key], ALARM_METRICS),
                    },
                },
                "decision": {
                    "score_threshold": {
                        "value": threshold,
                        "selection_method": "maximum_validation_balanced_accuracy",
                        "selection_split": "validation",
                        "comparison": ">=",
                    },
                    "anchor": "window_end",
                    "alarm_policy_schema_version": alarm_config.get("schema_version"),
                    "alarm_policy_sha256": alarm_policy_sha256,
                    "trigger_policies": artifact_policies,
                },
                "parity": {
                    "status": "PASS",
                    "runtime": {
                        "provider": "CPUExecutionProvider",
                        "onnx": environment.get("onnx"),
                        "onnxruntime": environment.get("onnxruntime"),
                    },
                    "splits": [
                        {
                            "split": row["split"],
                            "samples": int(row["samples"]),
                            "batches": int(row["batches"]),
                            "batch_size": int(row["batch_size"]),
                            "maximum_absolute_error": _float(row, "maximum_absolute_error"),
                            "mean_absolute_error": _float(row, "mean_absolute_error"),
                            "p99_absolute_error": _float(row, "p99_absolute_error"),
                        }
                        for row in parity_splits
                    ],
                },
                "source": {
                    "onnx_run_path": f"models/{artifact_id}/model.onnx",
                    "model_spec_run_path": f"models/{artifact_id}/model_spec.json",
                    "onnx_sha256": verified_onnx_sha256,
                },
            }
        )
    if len(artifacts) != expected_jobs or artifact_keys != set(metrics):
        raise ValueError("Published run does not contain exactly one ONNX artifact per job")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        grouped[(artifact["model_id"], artifact["training_recipe"])].append(artifact)
    methods = []
    for (model_id, recipe), members in sorted(grouped.items()):
        if len(members) != expected_folds:
            raise ValueError(f"Method {model_id}/{recipe} has an incomplete fold set")
        aggregates: dict[str, dict[str, dict[str, float]]] = {}
        for scope, names in (
            ("window", WINDOW_METRICS),
            ("event", EVENT_METRICS),
            ("alarm", ALARM_METRICS),
        ):
            aggregates[scope] = {
                name: _mean_std([member["metrics"][scope][name] for member in members])
                for name in names
            }
        methods.append(
            {
                "method_id": f"{model_id}--{recipe}",
                "model_id": model_id,
                "training_recipe": recipe,
                "fold_count": len(members),
                "input_semantic": members[0]["input"]["semantic"],
                "metrics": aggregates,
                "artifact_ids": [
                    member["artifact_id"]
                    for member in sorted(members, key=lambda item: item["fold"])
                ],
            }
        )

    return {
        "schema_version": EXPERIMENT_CATALOG_SCHEMA,
        "contract_version": EXPERIMENT_CATALOG_CONTRACT_VERSION,
        "evidence_level": evidence_level,
        "evaluation_fingerprint": _evaluation_fingerprint(run_dir, run_manifest),
        "methods": methods,
        "artifacts": artifacts,
    }
