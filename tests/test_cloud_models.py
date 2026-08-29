import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from imu_benchmark.cloud_models import validate_model_package


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
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
    onnx.save(model, root / "model.onnx")
    _json(
        root / "manifest.json",
        {
            "schema_version": "imu_model_package_manifest_v1",
            "model_code": "identity-fixture",
            "display_name": "Identity fixture",
            "source": {"commit": "a" * 40, "dirty": False},
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
            },
            "preprocessing": {"location": "none"},
            "validation": {
                "onnx_checker": "PASS",
                "python_onnxruntime_parity": "PASS",
                "external_runtime": "not_tested",
                "device_replay": "not_tested",
            },
            "known_limitations": ["test fixture"],
        },
    )
    _json(root / "runtime_config.json", {"sampling_rate_hz": 25.0})
    _json(root / "metrics.json", {"evidence": "fixture"})
    values = np.asarray([[0.25, -0.5]], dtype=np.float32)
    np.savez(root / "golden_input.npz", input=values)
    np.savez(root / "golden_output.npz", output=values)
    filenames = (
        "model.onnx",
        "manifest.json",
        "runtime_config.json",
        "metrics.json",
        "golden_input.npz",
        "golden_output.npz",
    )
    (root / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in filenames
        ),
        encoding="utf-8",
    )
    return root


def test_final_model_package_checks_onnx_golden_and_checksums(tmp_path: Path) -> None:
    manifest, files = validate_model_package(_package(tmp_path))

    assert manifest["package_id"].startswith("identity-fixture-")
    assert len(manifest["logical_digest"]) == 64
    assert {item["filename"] for item in files} == {
        "model.onnx",
        "manifest.json",
        "runtime_config.json",
        "metrics.json",
        "golden_input.npz",
        "golden_output.npz",
        "checksums.sha256",
    }


def test_final_model_package_refuses_unverified_runtime_contract(tmp_path: Path) -> None:
    root = _package(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["validation"]["python_onnxruntime_parity"] = "not_tested"
    _json(root / "manifest.json", manifest)

    try:
        validate_model_package(root)
    except ValueError as error:
        assert "validation status" in str(error)
    else:
        raise AssertionError("an unverified package must not be publishable")
