import json
from pathlib import Path

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from imu_benchmark.cloud_models import package_model_release, validate_model_release


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release(tmp_path: Path) -> Path:
    source = tmp_path / "source.onnx"
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [1, 2])
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["imu", "axes"], ["fall_score"], keepdims=0)],
        "fixture",
        [helper.make_tensor_value_info("imu", TensorProto.FLOAT, [None, 50, 6])],
        [helper.make_tensor_value_info("fall_score", TensorProto.FLOAT, [None])],
        [axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, source)
    release_id = "identity-fixture-v1"
    fingerprint = "b" * 64
    spec = {
        "schema_version": "imu_model_package_spec_v1",
        "model_path": source.name,
        "metadata": {
            "schema_version": "imu_model_release_v1",
            "contract_version": "1.0.0",
            "release_id": release_id,
            "model_code": "identity-fixture",
            "name": "Identity fixture",
            "created_at_utc": "2026-08-30T00:00:00+00:00",
            "release_stage": "research_candidate",
            "source": {
                "selection_evidence": {
                    "source_run_id": "fixture-run",
                    "source_commit": "a" * 40,
                    "model_id": "identity-fixture",
                    "training_recipe": "natural",
                    "data_snapshot_fingerprint": fingerprint,
                    "split_fingerprint": "c" * 64,
                    "selection_scope": "validation_only_oof",
                    "metric_split": "validation_oof",
                    "selection_eligible": True,
                    "participant_once": {
                        "status": "PASS",
                        "participant_count": 5,
                        "appearances_per_participant": 1,
                        "validation_fold_participant_counts": [1, 1, 1, 1, 1],
                        "assignment_sha256": "d" * 64,
                    },
                    "threshold_selection": {
                        "method": "maximum_validation_balanced_accuracy",
                        "tie_break": "closest_to_0.5_then_lower",
                    },
                    "trigger_policy_selection": {
                        "method": "validation_pareto",
                        "tie_break": "policy_id_ascending",
                    },
                },
                "final_training": {
                    "commit": "a" * 40,
                    "dirty": False,
                    "seed": 3888,
                    "fixed_epoch_source": "validation_oof_median_best_epoch",
                    "training_scope": "all_development_participants_plus_training_only_team",
                    "actual_epochs": 4,
                },
            },
            "data": {
                "snapshot_fingerprint": fingerprint,
                "split_fingerprint": "c" * 64,
            },
            "input": {
                "semantic": "si_window",
                "name": "imu",
                "dtype": "float32",
                "shape": [None, 50, 6],
                "sampling_rate_hz": 25,
                "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
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
                "normalization": {"embedded": True},
            },
            "windowing": {
                "window_seconds": 2.0,
                "inference_interval_seconds": 0.5,
                "source_stride_seconds": 0.5,
                "anchor": "window_end",
                "sequence_boundary": "reset",
                "timestamp_gap": "reset",
            },
            "decision": {
                "score_threshold": {"value": 0.5, "comparison": ">="},
                "trigger_policy": {
                    "policy_id": "one_of_one",
                    "required_positive_windows": 1,
                    "lookback_windows": 1,
                    "consecutive": True,
                    "cooldown_seconds": 10.0,
                },
                "status": "provisional_validation_derived",
            },
            "metrics": {
                "metric_split": "validation_oof",
                "selection_eligible": True,
                "final_model_independently_evaluated": False,
                "window": {"balanced_accuracy": 0.8},
            },
            "validation": {
                "onnx_checker": {"status": "PASS"},
                "python_onnxruntime_parity": {"status": "PASS", "windows": 10},
                "external_runtime": {"status": "not_tested"},
                "device_replay": {"status": "not_tested"},
            },
            "known_limitations": ["test fixture"],
        },
    }
    spec_path = tmp_path / "package.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    root = tmp_path / "release"
    package_model_release(spec_path, root)
    return root


def test_two_file_model_release_checks_onnx_and_metadata(tmp_path: Path) -> None:
    metadata = validate_model_release(_release(tmp_path))

    assert metadata["release_id"] == "identity-fixture-v1"
    assert metadata["decision"]["score_threshold"]["comparison"] == ">="


def test_model_release_refuses_unverified_runtime_contract(tmp_path: Path) -> None:
    root = _release(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["validation"]["python_onnxruntime_parity"]["status"] = "not_tested"
    _write_json(root / "metadata.json", metadata)

    with pytest.raises(ValueError, match="parity must pass"):
        validate_model_release(root)


def test_model_release_refuses_extra_files(tmp_path: Path) -> None:
    root = _release(tmp_path)
    (root / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        validate_model_release(root)
