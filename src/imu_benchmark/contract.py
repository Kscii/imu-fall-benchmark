from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "imu_benchmark_contract_v1"
SNAPSHOT_VERSION = "imu_30hz_snapshot_v1"
DEFAULT_CONTRACT_PATH = Path("configs/contracts/imu_benchmark_contract_v1.json")
DEFAULT_SNAPSHOT_PATH = Path("data/snapshot_v1.json")


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name != "configs":
        raise ValueError("Experiment configurations must be stored under configs/")
    return resolved.parent.parent


@dataclass(frozen=True, slots=True)
class ContractBundle:
    project_root: Path
    experiment: dict[str, Any]
    contract: dict[str, Any]
    snapshot: dict[str, Any]
    contract_sha256: str
    snapshot_sha256: str

    def effective_values(self) -> dict[str, Any]:
        signal = self.contract["canonical_signal"]
        window = self.contract["window"]
        recording = self.contract["supervision"]["recording"]
        temporal = self.contract["supervision"]["temporal"]
        training = self.snapshot["collections"]["training"]
        external = self.snapshot["collections"]["external"]
        return {
            "contract_version": self.contract["contract_version"],
            "contract_sha256": self.contract_sha256,
            "snapshot_version": self.snapshot["snapshot_version"],
            "snapshot_sha256": self.snapshot_sha256,
            "data_schema_version": self.contract["data_schema_version"],
            "sampling_rate_hz": signal["sampling_rate_hz"],
            "window_schema_version": window["schema_version"],
            "kfall_window_schema_version": window["external_schema_version"],
            "feature_schema_version": window["feature_schema_version"],
            "window_samples": window["samples"],
            "stride_samples": window["stride_samples"],
            "mil_top_fraction": recording["mil_top_fraction"],
            "temporal_policy": temporal["positive_policy"],
            "post_segment_overlap_policy": temporal["post_segment_overlap_policy"],
            "alarm_policy": temporal["event_detection_policy"],
            "fold_seed": self.contract["folds"]["seed"],
            "data_path": training["data_path"],
            "split_path": training["split"]["path"],
            "split_version": training["split"]["version"],
            "external_data_path": external["data_path"],
            "external_split_path": external["split"]["path"],
            "external_split_version": external["split"]["version"],
        }


def load_contract_bundle(config_path: Path) -> ContractBundle:
    project_root = _project_root(config_path)
    experiment = _read_object(config_path.resolve())
    allowed_protocol_keys = {
        "contract_version",
        "contract_sha256",
        "snapshot_version",
        "snapshot_sha256",
        "data_schema_version",
        "sampling_rate_hz",
        "window_schema_version",
        "kfall_window_schema_version",
        "feature_schema_version",
        "window_samples",
        "stride_samples",
        "mil_top_fraction",
        "temporal_policy",
        "post_segment_overlap_policy",
        "alarm_policy",
        "fold_seed",
        "data_path",
        "split_path",
        "split_version",
        "external_data_path",
        "external_split_path",
        "external_split_version",
    }
    overrides = sorted(allowed_protocol_keys & set(experiment))
    if overrides:
        raise ValueError(f"Protocol fields must come from the contract: {overrides}")
    contract_relative = experiment.get("contract_path")
    snapshot_relative = experiment.get("snapshot_path")
    if not isinstance(contract_relative, str) or not isinstance(snapshot_relative, str):
        raise ValueError("Experiment config must reference contract_path and snapshot_path")
    if Path(contract_relative) != DEFAULT_CONTRACT_PATH:
        raise ValueError("Experiment config must reference the contract-v1 canonical path")
    if Path(snapshot_relative) != DEFAULT_SNAPSHOT_PATH:
        raise ValueError("Experiment config must reference the snapshot-v1 canonical path")
    contract = _read_object(project_root / contract_relative)
    snapshot = _read_object(project_root / snapshot_relative)
    validate_contract(contract)
    validate_snapshot_shape(snapshot, contract)
    return ContractBundle(
        project_root=project_root,
        experiment=experiment,
        contract=contract,
        snapshot=snapshot,
        contract_sha256=canonical_json_sha256(contract),
        snapshot_sha256=canonical_json_sha256(snapshot),
    )


def load_contract_snapshot(
    project_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = _read_object(project_root / contract_path)
    snapshot = _read_object(project_root / snapshot_path)
    validate_contract(contract)
    validate_snapshot_shape(snapshot, contract)
    return (
        contract,
        snapshot,
        canonical_json_sha256(contract),
        canonical_json_sha256(snapshot),
    )


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Unexpected benchmark contract version")
    if contract.get("data_schema_version") != "3.0.0":
        raise ValueError("The contract requires HDF5 schema 3.0.0")
    signal = contract.get("canonical_signal", {})
    if signal != {
        "sampling_rate_hz": 30,
        "channels": 6,
        "axis_frame": "sensor_local",
        "gravity": "retained",
    }:
        raise ValueError("Unexpected canonical signal contract")
    window = contract.get("window", {})
    if window.get("schema_version") != "causal_segment_2s_stride_0p5s_features_v2":
        raise ValueError("Unexpected training window schema version")
    if window.get("external_schema_version") != "causal_segment_2s_stride_0p5s_kfall_v2":
        raise ValueError("Unexpected external window schema version")
    if window.get("feature_schema_version") != "window_features_v1":
        raise ValueError("Unexpected engineered-feature schema version")
    if window.get("samples") != 60 or window.get("stride_samples") != 15:
        raise ValueError("The contract requires 60-sample windows with stride 15")
    if window.get("decision_sample") != "last_sample":
        raise ValueError("The contract requires causal last-sample decisions")
    supervision = contract.get("supervision", {})
    recording = supervision.get("recording", {})
    if recording != {"pooling": "top_fraction_mean", "mil_top_fraction": 0.1}:
        raise ValueError("Unexpected recording-level MIL contract")
    temporal = supervision.get("temporal", {})
    if temporal.get("positive_policy") != "decision_time_within_fall_activity_segment":
        raise ValueError("Unexpected temporal positive policy")
    if temporal.get("post_segment_overlap_policy") != "exclude":
        raise ValueError("Post-segment overlap windows must be excluded")
    if temporal.get("exclude_overlap_policy") != "exclude":
        raise ValueError("Explicit exclusion overlaps must be excluded")
    if temporal.get("event_detection_policy") != "single_window":
        raise ValueError("Contract v1 supports single-window event detection only")
    folds = contract.get("folds", {})
    if folds.get("count") != 5 or folds.get("seed") != 3888:
        raise ValueError("Unexpected participant-fold contract")
    if folds.get("evolution") != "preserve_existing_assign_new_deterministically":
        raise ValueError("Unexpected fold-evolution contract")
    resampling = contract.get("device_reference_resampling", {})
    if resampling != {
        "nominal_source_rate_hz": 25,
        "target_rate_hz": 30,
        "method": "monotonic_timestamp_linear_interpolation",
        "extrapolation": "forbidden",
        "ble_packet_timestamp_assignment": "out_of_scope",
    }:
        raise ValueError("Unexpected device reference-resampling contract")
    metrics = contract.get("metrics", {})
    if metrics.get("recording_primary") != "balanced_accuracy":
        raise ValueError("Unexpected recording-level primary metric")
    if metrics.get("temporal_primary") != [
        "event_sensitivity",
        "adl_false_positive_windows_per_hour",
        "onset_latency_median_s",
        "onset_latency_p95_s",
        "impact_offset_median_s",
    ]:
        raise ValueError("Unexpected temporal primary metrics")
    if metrics.get("threshold_selection") != "maximum_validation_balanced_accuracy":
        raise ValueError("Unexpected threshold-selection contract")


def validate_snapshot_shape(snapshot: dict[str, Any], contract: dict[str, Any]) -> None:
    if snapshot.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError("Unexpected data snapshot version")
    if snapshot.get("contract_version") != contract["contract_version"]:
        raise ValueError("Snapshot and contract versions do not match")
    collections = snapshot.get("collections")
    if not isinstance(collections, dict) or set(collections) != {"training", "external"}:
        raise ValueError("Snapshot must define training and external collections")
    for name, collection in collections.items():
        if not isinstance(collection, dict):
            raise ValueError(f"Invalid snapshot collection: {name}")
        datasets = collection.get("datasets")
        split = collection.get("split")
        if not isinstance(datasets, list) or not datasets or not isinstance(split, dict):
            raise ValueError(f"Incomplete snapshot collection: {name}")
        ids = [item.get("dataset_id") for item in datasets if isinstance(item, dict)]
        if len(ids) != len(datasets) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid dataset IDs in snapshot collection: {name}")
