from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "imu_benchmark_contract_v2"
SNAPSHOT_VERSION = "imu_25hz_snapshot_v2"
ACTIVE_SCHEMA_VERSION = "imu_benchmark_active_v2"
SUPPORTED_ACTIVE_SCHEMA_VERSIONS = {
    "imu_benchmark_active_v1",
    ACTIVE_SCHEMA_VERSION,
}
SUPPORTED_SNAPSHOT_VERSIONS = {
    "imu_25hz_snapshot_v1",
    SNAPSHOT_VERSION,
}
DEFAULT_CONTRACT_PATH = Path("configs/contracts/imu_benchmark_contract_v2.json")
DEFAULT_SNAPSHOT_PATH = Path("active.json")


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


def default_active_snapshot_path() -> Path:
    root = Path(os.environ.get("IMU_BENCH_WORK_ROOT", "~/imu-fall-work")).expanduser()
    if not root.is_absolute():
        raise ValueError("IMU_BENCH_WORK_ROOT must be an absolute path")
    return root.resolve() / "data" / DEFAULT_SNAPSHOT_PATH


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
        return {
            "contract_version": self.contract["contract_version"],
            "contract_sha256": self.contract_sha256,
            "snapshot_version": self.snapshot["snapshot_version"],
            "snapshot_sha256": self.snapshot_sha256,
            "data_schema_version": self.contract["data_schema_version"],
            "sampling_rate_hz": signal["sampling_rate_hz"],
            "window_schema_version": window["schema_version"],
            "feature_schema_version": window["feature_schema_version"],
            "window_samples": window["samples"],
            "stride_seconds": window["stride_seconds"],
            "mil_top_fraction": recording["mil_top_fraction"],
            "temporal_policy": temporal["positive_policy"],
            "post_segment_overlap_policy": temporal["post_segment_overlap_policy"],
            "alarm_policy": temporal["event_detection_policy"],
            "fold_seed": self.contract["folds"]["seed"],
        }


def load_contract_bundle(
    config_path: Path,
    *,
    snapshot_path: Path | None = None,
) -> ContractBundle:
    resolved = config_path.resolve()
    project_root = next(
        (parent.parent for parent in resolved.parents if parent.name == "configs"),
        None,
    )
    if project_root is None:
        raise ValueError("Experiment configurations must be stored under configs/")
    experiment = _read_object(resolved)
    contract, snapshot, contract_hash, snapshot_hash = load_contract_snapshot(
        project_root,
        snapshot_path=snapshot_path,
    )
    return ContractBundle(
        project_root=project_root,
        experiment=experiment,
        contract=contract,
        snapshot=snapshot,
        contract_sha256=contract_hash,
        snapshot_sha256=snapshot_hash,
    )


def load_contract_snapshot(
    project_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    snapshot_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = _read_object(project_root / contract_path)
    active_path = default_active_snapshot_path() if snapshot_path is None else snapshot_path
    snapshot = _read_object(active_path)
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
    if contract.get("data_schema_version") != "3.1.0":
        raise ValueError("The contract requires HDF5 schema 3.1.0")
    signal = contract.get("canonical_signal", {})
    if signal != {
        "sampling_rate_hz": 25,
        "channels": 6,
        "axis_frame": "sensor_local",
        "gravity": "retained",
    }:
        raise ValueError("Unexpected canonical signal contract")
    window = contract.get("window", {})
    if window != {
        "schema_version": "causal_decision_2s_stride_0p5s_v3",
        "feature_schema_version": "window_features_v2_25hz",
        "duration_seconds": 2.0,
        "samples": 50,
        "stride_seconds": 0.5,
        "stride_grid": "nearest_half_up_absolute_time",
        "decision_sample": "last_sample",
    }:
        raise ValueError("Unexpected 25 Hz window contract")
    supervision = contract.get("supervision", {})
    if supervision.get("recording") != {
        "pooling": "top_fraction_mean",
        "mil_top_fraction": 0.1,
    }:
        raise ValueError("Unexpected recording-level MIL contract")
    temporal = supervision.get("temporal", {})
    if temporal != {
        "positive_policy": "decision_time_within_fall_activity_segment",
        "post_segment_overlap_policy": "exclude",
        "exclude_overlap_policy": "exclude",
        "event_detection_policy": "single_window",
    }:
        raise ValueError("Unexpected temporal-supervision contract")
    folds = contract.get("folds", {})
    if folds != {
        "count": 5,
        "seed": 3888,
        "base_policy": "participant_grouped_0_to_4",
        "team_policy": "training_only_fold_minus_1",
        "protocol": "test_f_validation_next_train_remaining_three_plus_team",
    }:
        raise ValueError("Unexpected participant-fold contract")
    resampling = contract.get("device_reference_resampling", {})
    if resampling != {
        "nominal_source_rate_hz": 25,
        "target_rate_hz": 25,
        "method": "monotonic_timestamp_linear_interpolation",
        "extrapolation": "forbidden",
        "ble_packet_timestamp_assignment": "collector_contract",
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
    if metrics.get("threshold_selection") != "combined_validation_balanced_accuracy":
        raise ValueError("Unexpected threshold-selection contract")


def _validate_dataset_entries(
    entries: object,
    *,
    expected_role: str,
    collection_name: str,
) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Snapshot collection {collection_name} has no datasets")
    ids: set[str] = set()
    paths: set[str] = set()
    required = {
        "dataset_id",
        "path",
        "sha256",
        "logical_content_sha256",
        "size_bytes",
        "hdf5_schema_version",
        "sampling_rate_hz",
        "evaluation_role",
        "sequences",
        "rows",
        "annotations",
    }
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(f"Incomplete dataset entry in {collection_name}")
        dataset_id = entry["dataset_id"]
        relative = entry["path"]
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or dataset_id in ids
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in paths
        ):
            raise ValueError(f"Invalid dataset identity in {collection_name}")
        ids.add(dataset_id)
        paths.add(relative)
        if entry["hdf5_schema_version"] != "3.1.0":
            raise ValueError(f"Invalid HDF5 schema in {collection_name}")
        if float(entry["sampling_rate_hz"]) != 25.0:
            raise ValueError(f"Invalid sampling rate in {collection_name}")
        if entry["evaluation_role"] != expected_role:
            raise ValueError(f"Invalid evaluation role in {collection_name}")
        for digest_name in ("sha256", "logical_content_sha256"):
            digest = entry[digest_name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"Invalid {digest_name} in {collection_name}")
        for count_name in ("size_bytes", "sequences", "rows", "annotations"):
            if not isinstance(entry[count_name], int) or entry[count_name] < 0:
                raise ValueError(f"Invalid {count_name} in {collection_name}")


def validate_snapshot_shape(snapshot: dict[str, Any], contract: dict[str, Any]) -> None:
    schema_version = snapshot.get("schema_version")
    if schema_version not in SUPPORTED_ACTIVE_SCHEMA_VERSIONS:
        raise ValueError("Unexpected active-data manifest schema")
    if snapshot.get("snapshot_version") not in SUPPORTED_SNAPSHOT_VERSIONS:
        raise ValueError("Unexpected data snapshot version")
    if snapshot.get("contract_version") != contract["contract_version"]:
        raise ValueError("Snapshot and contract versions differ")
    if not isinstance(snapshot.get("bucket"), str) or not snapshot["bucket"].startswith("gs://"):
        raise ValueError("Active manifest has an invalid bucket")
    if not isinstance(snapshot.get("base_snapshot_id"), str):
        raise ValueError("Active manifest is missing the base snapshot ID")
    team_id = snapshot.get("team_snapshot_id")
    if team_id is not None and not isinstance(team_id, str):
        raise ValueError("Active manifest has an invalid team snapshot ID")
    collections = snapshot.get("collections")
    if not isinstance(collections, dict) or set(collections) not in ({"base"}, {"base", "team"}):
        raise ValueError("Active manifest requires base and optional team collections")
    base = collections["base"]
    if not isinstance(base, dict) or set(base) != {"data_path", "splits", "datasets"}:
        raise ValueError("Invalid base collection")
    splits = base["splits"]
    if not isinstance(splits, list) or not splits:
        raise ValueError("Base collection requires participant split manifests")
    for split in splits:
        required = {"path", "version", "sha256"}
        optional = {"size_bytes"} if schema_version == ACTIVE_SCHEMA_VERSION else set()
        if (
            not isinstance(split, dict)
            or not required.issubset(split)
            or not set(split).issubset(required | optional)
        ):
            raise ValueError("Invalid base split manifest")
        relative = split["path"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("Invalid base split path")
        digest = split["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid base split SHA-256")
        if "size_bytes" in split and (
            not isinstance(split["size_bytes"], int) or split["size_bytes"] < 0
        ):
            raise ValueError("Invalid base split size")
    _validate_dataset_entries(
        base["datasets"],
        expected_role="cross_validation",
        collection_name="base",
    )
    if "team" in collections:
        team = collections["team"]
        if not isinstance(team, dict) or set(team) != {"data_path", "fold_id", "datasets"}:
            raise ValueError("Invalid team collection")
        if team["fold_id"] != -1 or team_id is None:
            raise ValueError("Team data must use training-only fold -1")
        _validate_dataset_entries(
            team["datasets"],
            expected_role="training_only",
            collection_name="team",
        )
