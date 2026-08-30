"""Shared semantic validators for experiment and model publication metadata."""

from __future__ import annotations

import math
import re
from typing import Any

EXPERIMENT_SCHEMA_V1 = "imu_experiment_catalog_v1"
EXPERIMENT_CONTRACT_VERSION = "1.0.0"
EXPERIMENT_CATALOG_CONTRACT_VERSION = EXPERIMENT_CONTRACT_VERSION
MODEL_SCHEMA_V1 = "imu_model_release_v1"
MODEL_CONTRACT_VERSION = "1.0.0"

_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUS = {"PASS", "FAIL", "not_tested"}


def semver(value: object, *, name: str = "contract version") -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {name}")
    match = _SEMVER.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid {name}")
    return tuple(int(part) for part in match.groups())


def require_compatible_version(actual: object, expected: str, *, name: str) -> str:
    actual_parts = semver(actual, name=name)
    expected_parts = semver(expected, name=name)
    compatible = (
        actual_parts == expected_parts
        if expected_parts[0] == 0
        else actual_parts[0] == expected_parts[0]
    )
    if not compatible:
        raise ValueError(f"Incompatible {name}: {actual!r}")
    return str(actual)


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _status(value: object, name: str) -> dict[str, Any]:
    item = _object(value, name)
    if item.get("status") not in _STATUS:
        raise ValueError(f"{name}.status is invalid")
    return item


def validate_experiment_marker_v1(marker: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "contract_version",
        "publication_id",
        "run_id",
        "experiment_id",
        "evidence_level",
        "created_at_utc",
        "source",
        "data",
        "evaluation_fingerprint",
        "scheduled_jobs",
        "methods",
        "artifacts",
        "result_evidence",
        "known_limitations",
    }
    if not required.issubset(marker):
        raise ValueError("Experiment catalog is missing v1 required fields")
    if marker.get("schema_version") != EXPERIMENT_SCHEMA_V1:
        raise ValueError("Experiment catalog schema is invalid")
    require_compatible_version(
        marker.get("contract_version"),
        EXPERIMENT_CONTRACT_VERSION,
        name="experiment catalog contract version",
    )
    evidence_level = marker.get("evidence_level")
    if evidence_level not in {"formal_cv", "engineering"}:
        raise ValueError("Experiment evidence level is invalid")
    methods = marker.get("methods")
    artifacts = marker.get("artifacts")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Experiment catalog has no methods")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Experiment catalog has no artifacts")
    artifact_ids: set[str] = set()
    folds_by_method: dict[str, set[int]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Experiment artifact is invalid")
        artifact_id = artifact.get("artifact_id")
        method_id = artifact.get("method_id")
        fold = artifact.get("fold")
        metrics = artifact.get("metrics")
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in artifact_ids
            or not isinstance(method_id, str)
            or not isinstance(fold, int)
            or fold < 0
            or not isinstance(metrics, dict)
            or metrics.get("metric_split") != "test"
            or metrics.get("selection_eligible") is not False
        ):
            raise ValueError("Experiment artifact identity or metric scope is invalid")
        artifact_ids.add(artifact_id)
        folds_by_method.setdefault(method_id, set()).add(fold)
    method_ids: set[str] = set()
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError("Experiment method is invalid")
        method_id = method.get("method_id")
        member_ids = method.get("artifact_ids")
        fold_count = method.get("fold_count")
        if (
            not isinstance(method_id, str)
            or not method_id
            or method_id in method_ids
            or method.get("metric_split") != "test"
            or method.get("selection_eligible") is not False
            or not isinstance(member_ids, list)
            or not member_ids
            or not isinstance(fold_count, int)
            or fold_count != len(member_ids)
            or len(set(member_ids)) != len(member_ids)
            or not set(member_ids).issubset(artifact_ids)
        ):
            raise ValueError("Experiment method identity or metric scope is invalid")
        method_ids.add(method_id)
        folds = folds_by_method.get(method_id, set())
        if len(folds) != fold_count:
            raise ValueError("Experiment method fold set is incomplete")
        if evidence_level == "formal_cv" and folds != set(range(5)):
            raise ValueError("Formal cross-validation requires folds 0 through 4")
    if set(folds_by_method) != method_ids:
        raise ValueError("Experiment artifacts reference an unknown method")


def validate_selection_evidence(value: object) -> dict[str, Any]:
    evidence = _object(value, "selection evidence")
    required = {
        "source_run_id",
        "source_commit",
        "model_id",
        "training_recipe",
        "data_snapshot_fingerprint",
        "split_fingerprint",
        "selection_scope",
        "metric_split",
        "selection_eligible",
        "participant_once",
        "threshold_selection",
        "trigger_policy_selection",
    }
    if not required.issubset(evidence):
        raise ValueError("Selection evidence is incomplete")
    if (
        evidence.get("selection_scope") != "validation_only_oof"
        or evidence.get("metric_split") != "validation_oof"
        or evidence.get("selection_eligible") is not True
    ):
        raise ValueError("Selection evidence scope is invalid")
    _sha(evidence.get("data_snapshot_fingerprint"), "data snapshot fingerprint")
    _sha(evidence.get("split_fingerprint"), "split fingerprint")
    proof = _object(evidence.get("participant_once"), "participant-once proof")
    counts = proof.get("validation_fold_participant_counts")
    if (
        proof.get("status") != "PASS"
        or not isinstance(proof.get("participant_count"), int)
        or proof["participant_count"] <= 0
        or proof.get("appearances_per_participant") != 1
        or not isinstance(counts, list)
        or len(counts) != 5
        or any(not isinstance(count, int) or count <= 0 for count in counts)
        or sum(counts) != proof["participant_count"]
    ):
        raise ValueError("Participant-once proof is invalid")
    _sha(proof.get("assignment_sha256"), "participant assignment SHA-256")
    for name in ("threshold_selection", "trigger_policy_selection"):
        rule = _object(evidence.get(name), name.replace("_", " "))
        if not isinstance(rule.get("method"), str) or not isinstance(
            rule.get("tie_break"), str
        ):
            raise ValueError(f"{name} is incomplete")
    descriptor = evidence.get("artifact")
    if descriptor is not None:
        descriptor = _object(descriptor, "selection evidence artifact")
        _sha(descriptor.get("sha256"), "selection evidence artifact SHA-256")
        if (
            not isinstance(descriptor.get("object_key"), str)
            or not isinstance(descriptor.get("size_bytes"), int)
            or descriptor["size_bytes"] <= 0
        ):
            raise ValueError("Selection evidence artifact descriptor is invalid")
    return evidence


def validate_model_marker_v1(marker: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "contract_version",
        "release_id",
        "model_code",
        "name",
        "created_at_utc",
        "release_stage",
        "source",
        "data",
        "input",
        "output",
        "preprocessing",
        "windowing",
        "decision",
        "metrics",
        "verification",
        "validation",
        "known_limitations",
        "model",
    }
    if not required.issubset(marker):
        raise ValueError("Model release is missing v1 required fields")
    if marker.get("schema_version") != MODEL_SCHEMA_V1:
        raise ValueError("Model release schema is invalid")
    require_compatible_version(
        marker.get("contract_version"),
        MODEL_CONTRACT_VERSION,
        name="model release contract version",
    )
    if marker.get("release_stage") != "research_candidate":
        raise ValueError("Model release stage is invalid")
    source = _object(marker.get("source"), "model release source")
    selection = validate_selection_evidence(source.get("selection_evidence"))
    final = _object(source.get("final_training"), "final training evidence")
    if (
        final.get("dirty") is not False
        or not isinstance(final.get("commit"), str)
        or not isinstance(final.get("seed"), int)
        or not isinstance(final.get("fixed_epoch_source"), str)
        or not isinstance(final.get("training_scope"), str)
        or not isinstance(final.get("actual_epochs"), int)
        or final["actual_epochs"] <= 0
    ):
        raise ValueError("Final training evidence is incomplete")
    input_contract = _object(marker.get("input"), "model input")
    if (
        input_contract.get("semantic") != "si_window"
        or input_contract.get("dtype") != "float32"
        or input_contract.get("shape") != [None, 50, 6]
        or input_contract.get("axis_frame") != "sensor_local"
        or input_contract.get("gravity") != "retained"
        or input_contract.get("sampling_rate_hz") != 25
        or input_contract.get("channels") != ["ax", "ay", "az", "gx", "gy", "gz"]
    ):
        raise ValueError("Model input contract is invalid")
    output = _object(marker.get("output"), "model output")
    if (
        output.get("semantic") != "fall_score"
        or output.get("dtype") != "float32"
        or output.get("shape") != [None]
        or output.get("probability_calibrated") is not False
    ):
        raise ValueError("Model output contract is invalid")
    preprocessing = _object(marker.get("preprocessing"), "model preprocessing")
    normalization = _object(preprocessing.get("normalization"), "normalization")
    if preprocessing.get("location") != "onnx_graph" or normalization.get("embedded") is not True:
        raise ValueError("Model normalization is not embedded in ONNX")
    windowing = _object(marker.get("windowing"), "model windowing")
    interval = _finite(
        windowing.get("inference_interval_seconds"),
        "inference interval",
        positive=True,
    )
    if (
        _finite(windowing.get("window_seconds"), "window duration") != 2.0
        or windowing.get("anchor") != "window_end"
        or windowing.get("source_stride_seconds") != interval
        or windowing.get("sequence_boundary") != "reset"
        or windowing.get("timestamp_gap") != "reset"
    ):
        raise ValueError("Model windowing contract is invalid")
    decision = _object(marker.get("decision"), "model decision")
    threshold = _object(decision.get("score_threshold"), "score threshold")
    trigger = _object(decision.get("trigger_policy"), "trigger policy")
    if (
        decision.get("status") != "provisional_validation_derived"
        or threshold.get("comparison") != ">="
        or not isinstance(threshold.get("value"), (int, float))
        or not {
            "policy_id",
            "required_positive_windows",
            "lookback_windows",
            "consecutive",
            "cooldown_seconds",
        }.issubset(trigger)
    ):
        raise ValueError("Model decision contract is incomplete")
    metrics = _object(marker.get("metrics"), "model metrics")
    if (
        metrics.get("metric_split") != "validation_oof"
        or metrics.get("selection_eligible") is not True
        or metrics.get("final_model_independently_evaluated") is not False
    ):
        raise ValueError("Model metric scope is invalid")
    data = _object(marker.get("data"), "model data")
    if (
        data.get("snapshot_fingerprint") != selection["data_snapshot_fingerprint"]
        or data.get("split_fingerprint") != selection["split_fingerprint"]
    ):
        raise ValueError("Model data differs from selection evidence")
    verification = _object(marker.get("verification"), "model verification")
    fixtures = verification.get("golden_fixtures")
    if not isinstance(fixtures, list) or len(fixtures) < 3:
        raise ValueError("Model release needs at least three golden fixtures")
    fixture_ids: set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("Golden fixture is invalid")
        fixture_id = fixture.get("fixture_id")
        values = fixture.get("input_values")
        if (
            not isinstance(fixture_id, str)
            or not fixture_id
            or fixture_id in fixture_ids
            or not isinstance(values, list)
            or len(values) != 50
            or any(not isinstance(row, list) or len(row) != 6 for row in values)
        ):
            raise ValueError("Golden fixture input is invalid")
        fixture_ids.add(fixture_id)
        _finite(fixture.get("expected_fall_score"), "golden expected score")
        _finite(fixture.get("atol"), "golden absolute tolerance")
        _finite(fixture.get("rtol"), "golden relative tolerance")
    validation = _object(marker.get("validation"), "model validation")
    if _status(validation.get("onnx_checker"), "ONNX checker").get("status") != "PASS":
        raise ValueError("ONNX checker must pass")
    if (
        _status(validation.get("python_onnxruntime_parity"), "Python ONNX parity").get(
            "status"
        )
        != "PASS"
    ):
        raise ValueError("Python ONNX Runtime parity must pass")
    _status(validation.get("external_runtime"), "external runtime")
    _status(validation.get("device_replay"), "device replay")
    if not isinstance(marker.get("known_limitations"), list):
        raise ValueError("Model known limitations are invalid")
