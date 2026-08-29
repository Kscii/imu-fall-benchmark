import hashlib
import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from imu_benchmark.cloud_models import validate_model_release


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    root.mkdir()
    graph = helper.make_graph(
        [helper.make_node("Identity", ["imu"], ["fall_score"])],
        "fixture",
        [helper.make_tensor_value_info("imu", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("fall_score", TensorProto.FLOAT, [None, 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    model_path = root / "model.onnx"
    onnx.save(model, model_path)
    release_id = "identity-fixture-v1"
    _write_json(
        root / "metadata.json",
        {
            "schema_version": "imu_model_release_v0",
            "contract_version": "0.1.0",
            "release_id": release_id,
            "model_code": "identity-fixture",
            "name": "Identity fixture",
            "created_at_utc": "2026-08-30T00:00:00+00:00",
            "source": {
                "commit": "a" * 40,
                "dirty": False,
                "run_id": "fixture-run",
                "experiment_publication_id": "fixture-run",
                "artifact_id": "fixture-artifact",
            },
            "data": {"snapshot_id": "fixture", "evaluation_fingerprint": "b" * 64},
            "input": {
                "semantic": "si_window",
                "name": "imu",
                "dtype": "float32",
                "shape": [None, 2],
            },
            "output": {
                "semantic": "fall_score",
                "name": "fall_score",
                "dtype": "float32",
                "shape": [None, 2],
            },
            "preprocessing": {"location": "none"},
            "decision": {
                "score_threshold": {"value": 0.5, "comparison": ">="},
                "trigger_policy": {
                    "policy_id": "one_of_one",
                    "required_positive_windows": 1,
                    "lookback_windows": 1,
                    "consecutive": True,
                    "cooldown_seconds": 10.0,
                },
                "anchor": "window_end",
            },
            "metrics": {"validation": {"balanced_accuracy": 0.8}},
            "validation": {
                "onnx_checker": "PASS",
                "python_onnxruntime_parity": "PASS",
                "external_runtime": "not_tested",
                "device_replay": "not_tested",
            },
            "known_limitations": ["test fixture"],
            "model": {
                "filename": "model.onnx",
                "object_key": (
                    "benchmark-model-catalog/models/identity-fixture-v1/model.onnx"
                ),
                "size_bytes": model_path.stat().st_size,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "content_type": "application/octet-stream",
            },
        },
    )
    return root


def test_two_file_model_release_checks_onnx_and_metadata(tmp_path: Path) -> None:
    metadata = validate_model_release(_release(tmp_path))

    assert metadata["release_id"] == "identity-fixture-v1"
    assert metadata["decision"]["score_threshold"]["comparison"] == ">="


def test_model_release_refuses_unverified_runtime_contract(tmp_path: Path) -> None:
    root = _release(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["validation"]["python_onnxruntime_parity"] = "not_tested"
    _write_json(root / "metadata.json", metadata)

    with pytest.raises(ValueError, match="validation status"):
        validate_model_release(root)


def test_model_release_refuses_extra_files(tmp_path: Path) -> None:
    root = _release(tmp_path)
    (root / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        validate_model_release(root)
