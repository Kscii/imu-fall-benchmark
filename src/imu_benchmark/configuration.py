from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .contract import (
    DEFAULT_CONTRACT_PATH,
    DEFAULT_SNAPSHOT_PATH,
    load_contract_snapshot,
)

CONFIG_SCHEMA_VERSION = 1
PUBLIC_MODEL_IDS = (
    "threshold_impact",
    "cuml_logistic_regression",
    "cuml_random_forest",
    "xgboost_cuda",
    "torch_1d_cnn",
    "torch_lstm",
    "torch_cnn_lstm",
)
OBJECTIVES = ("temporal_supervised", "recording_mil")
GPU_MODES = ("auto", "resident", "streaming")
PRECISIONS = ("fp32", "bf16")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid YAML file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _project_path(project_root: Path, value: object, *, parent: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{parent} must be a non-empty project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{parent} must stay inside the project")
    path = (project_root / relative).resolve()
    if project_root.resolve() not in path.parents:
        raise ValueError(f"{parent} resolves outside the project")
    return path


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_model_catalog(payload: dict[str, Any]) -> None:
    _exact_keys(payload, {"schema_version", "models"}, "model catalog")
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported model catalog schema")
    models = payload["models"]
    if not isinstance(models, dict) or set(models) != set(PUBLIC_MODEL_IDS):
        raise ValueError("Model catalog must define the seven public fall models exactly")
    for model_id, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid model specification: {model_id}")
        _exact_keys(
            spec,
            {"backend", "input_kind", "standardize", "params"},
            f"model {model_id}",
        )
        if spec["backend"] not in {"rule", "cuml", "xgboost", "pytorch"}:
            raise ValueError(f"Unsupported backend for {model_id}")
        if spec["input_kind"] not in {"raw", "features"}:
            raise ValueError(f"Unsupported input kind for {model_id}")
        if not isinstance(spec["standardize"], bool) or not isinstance(spec["params"], dict):
            raise ValueError(f"Invalid model fields for {model_id}")


def _validate_data_view(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "id",
            "objective",
            "research_only",
            "training_datasets",
            "negative_supplement_datasets",
            "evaluation_dataset",
        },
        "data view",
    )
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported data-view schema")
    if payload["objective"] not in OBJECTIVES:
        raise ValueError("Unsupported data-view objective")
    for name in ("training_datasets", "negative_supplement_datasets"):
        values = payload[name]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{name} must be a list of dataset IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate dataset IDs")
    if not payload["training_datasets"]:
        raise ValueError("A data view requires at least one training dataset")
    if set(payload["training_datasets"]) & set(payload["negative_supplement_datasets"]):
        raise ValueError("Primary and supplement datasets must be disjoint")
    if payload["objective"] == "temporal_supervised" and payload["evaluation_dataset"] != "kfall":
        raise ValueError("The current temporal-supervised view must evaluate on KFall")
    if not isinstance(payload["research_only"], bool):
        raise ValueError("research_only must be boolean")


def _validate_experiment(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "id",
            "contract_path",
            "snapshot_path",
            "data_view_path",
            "model_catalog_path",
            "models",
            "folds",
            "seeds",
            "precision",
            "gpu_mode",
            "runtime_budget_seconds",
            "max_epochs",
            "patience",
            "max_sequences_per_split",
            "max_windows_per_sequence",
            "cache_flush_windows",
            "data_quality_status",
            "known_limitations",
        },
        "experiment",
    )
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported experiment schema")
    if payload["precision"] not in PRECISIONS or payload["gpu_mode"] not in GPU_MODES:
        raise ValueError("Unsupported precision or GPU mode")
    models = payload["models"]
    if not isinstance(models, list) or not models or set(models) - set(PUBLIC_MODEL_IDS):
        raise ValueError("Experiment contains an unknown or empty model selection")
    if len(models) != len(set(models)):
        raise ValueError("Experiment model IDs must be unique")
    folds = payload["folds"]
    if not isinstance(folds, list) or not folds or set(folds) - set(range(5)):
        raise ValueError("Experiment folds must be unique values from 0 through 4")
    if len(folds) != len(set(folds)):
        raise ValueError("Experiment folds must be unique")
    seeds = payload["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("Experiment seeds must be a non-empty integer list")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Experiment seeds must be unique")
    for name in ("runtime_budget_seconds", "max_epochs", "patience", "cache_flush_windows"):
        if not isinstance(payload[name], int) or payload[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if payload["cache_flush_windows"] != 16_384:
        raise ValueError("Cache schema v3 fixes cache_flush_windows at 16384")
    for name in ("max_sequences_per_split", "max_windows_per_sequence"):
        if payload[name] is not None and (not isinstance(payload[name], int) or payload[name] <= 0):
            raise ValueError(f"{name} must be null or a positive integer")
    if not isinstance(payload["known_limitations"], list) or any(
        not isinstance(item, str) for item in payload["known_limitations"]
    ):
        raise ValueError("known_limitations must be a list of strings")


def load_experiment(project_root: Path, path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    config_root = (project_root / "configs/experiments").resolve()
    if config_root not in resolved_path.parents:
        raise ValueError("Experiment YAML must be stored under configs/experiments/")
    experiment = _read_yaml(resolved_path)
    _validate_experiment(experiment)
    if Path(str(experiment["contract_path"])) != DEFAULT_CONTRACT_PATH:
        raise ValueError("Experiment must use the canonical contract-v1 path")
    if Path(str(experiment["snapshot_path"])) != DEFAULT_SNAPSHOT_PATH:
        raise ValueError("Experiment must use snapshot-v1")
    contract, snapshot, contract_hash, snapshot_hash = load_contract_snapshot(project_root)
    model_path = _project_path(
        project_root, experiment["model_catalog_path"], parent="model_catalog_path"
    )
    data_view_path = _project_path(
        project_root, experiment["data_view_path"], parent="data_view_path"
    )
    model_catalog = _read_yaml(model_path)
    data_view = _read_yaml(data_view_path)
    _validate_model_catalog(model_catalog)
    _validate_data_view(data_view)
    missing = set(experiment["models"]) - set(model_catalog["models"])
    if missing:
        raise ValueError(f"Selected models are missing from the catalog: {sorted(missing)}")
    selected_specs = [model_catalog["models"][model_id] for model_id in experiment["models"]]
    if experiment["precision"] == "bf16" and any(
        spec["backend"] != "pytorch" for spec in selected_specs
    ):
        raise ValueError("BF16 experiments may select PyTorch models only")
    if data_view["objective"] == "recording_mil" and any(
        model_id not in {"threshold_impact", "torch_1d_cnn", "torch_lstm", "torch_cnn_lstm"}
        for model_id in experiment["models"]
    ):
        raise ValueError("The recording-MIL research view supports threshold and sequence models")
    resolved = deepcopy(experiment)
    resolved.update(
        {
            "config_path": str(resolved_path.relative_to(project_root.resolve())),
            "contract": contract,
            "contract_sha256": contract_hash,
            "snapshot": snapshot,
            "snapshot_sha256": snapshot_hash,
            "data_view": data_view,
            "model_catalog": model_catalog,
            "data_view_sha256": canonical_sha256(data_view),
            "model_catalog_sha256": canonical_sha256(model_catalog),
        }
    )
    resolved["resolved_config_sha256"] = canonical_sha256(resolved)
    return resolved


def public_config(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    result.pop("contract", None)
    result.pop("snapshot", None)
    return result
