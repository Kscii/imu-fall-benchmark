from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

ONNX_OPSET = 18
ONNX_RUNTIME_VERSION = "1.25.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_model() -> Any:
    import onnx
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Split", ["imu", "split_sizes"], ["acc", "gyro"], axis=2),
        helper.make_node("ReduceL2", ["acc", "axes_2"], ["acc_mag"], keepdims=0),
        helper.make_node("Sub", ["acc_mag", "gravity"], ["acc_delta"]),
        helper.make_node("Abs", ["acc_delta"], ["acc_abs"]),
        helper.make_node("ReduceMax", ["acc_abs", "axes_1"], ["acc_peak"], keepdims=0),
        helper.make_node("Div", ["acc_peak", "gravity"], ["acc_score"]),
        helper.make_node("ReduceL2", ["gyro", "axes_2"], ["gyro_mag"], keepdims=0),
        helper.make_node("ReduceMax", ["gyro_mag", "axes_1"], ["gyro_peak"], keepdims=0),
        helper.make_node("Div", ["gyro_peak", "gyro_scale"], ["gyro_score"]),
        helper.make_node("Add", ["acc_score", "gyro_score"], ["fall_score"]),
    ]
    initializers = [
        helper.make_tensor("split_sizes", TensorProto.INT64, [2], [3, 3]),
        helper.make_tensor("axes_2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("gravity", TensorProto.FLOAT, [], [9.80665]),
        helper.make_tensor("gyro_scale", TensorProto.FLOAT, [], [6.283185307179586]),
    ]
    graph = helper.make_graph(
        nodes,
        "threshold_impact",
        [helper.make_tensor_value_info("imu", TensorProto.FLOAT, ["batch", 50, 6])],
        [helper.make_tensor_value_info("fall_score", TensorProto.FLOAT, ["batch"])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="imu-fall-benchmark",
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _single_probability_output(model: Any) -> Any:
    from onnx import TensorProto, helper

    probability_name = model.graph.output[-1].name
    model.graph.initializer.append(
        helper.make_tensor("positive_class_index", TensorProto.INT64, [], [1])
    )
    model.graph.node.append(
        helper.make_node(
            "Gather",
            [probability_name, "positive_class_index"],
            ["fall_score"],
            axis=1,
        )
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("fall_score", TensorProto.FLOAT, ["batch"])
    )
    return model


def _tabular_model(adapter: Any, model_id: str, width: int) -> Any:
    if model_id in {"cuml_logistic_regression", "cuml_random_forest"}:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        estimator = adapter.estimator.as_sklearn()
        model = convert_sklearn(
            estimator,
            initial_types=[("features", FloatTensorType([None, width]))],
            target_opset=ONNX_OPSET,
            options={id(estimator): {"zipmap": False}},
        )
        return _single_probability_output(model)
    if model_id == "xgboost_cuda":
        import onnx
        from onnx import helper
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        # onnxmltools 1.16 supports XGBoost conversion through opset 15.
        # Convert at that supported boundary, normalize the output, then use
        # ONNX's official version converter for the repository-wide opset 18
        # deployment contract.
        model = convert_xgboost(
            adapter.estimator,
            initial_types=[("features", FloatTensorType([None, width]))],
            target_opset=15,
        )
        model = _single_probability_output(model)
        if not any(item.domain == "" for item in model.opset_import):
            # A tree-only graph may declare only ai.onnx.ml. The Gather node
            # added above belongs to the default domain and needs its own import.
            model.opset_import.append(helper.make_opsetid("", 15))
        return onnx.version_converter.convert_version(model, ONNX_OPSET)
    raise ValueError(f"Unsupported tabular ONNX model: {model_id}")


def _export_torch(
    adapter: Any, model_id: str, sample: np.ndarray, destination: Path
) -> None:
    import torch

    if adapter.model is None:
        raise RuntimeError("Cannot export an unfitted PyTorch model")

    class ProbabilityModel(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, values: Any) -> Any:
            return torch.sigmoid(self.model(values))

    # Export an isolated CPU copy so tracing never enters a CUDA/cuDNN custom
    # kernel and cannot mutate the trained model retained by the benchmark.
    wrapper = ProbabilityModel(copy.deepcopy(adapter.model).cpu()).eval()
    use_dynamo = model_id == "torch_1d_cnn"
    export_values = sample if use_dynamo else sample[:1]
    example = torch.as_tensor(export_values, dtype=torch.float32, device="cpu")
    torch.onnx.export(
        wrapper,
        (example,),
        destination,
        input_names=["imu"],
        output_names=["fall_score"],
        dynamic_axes={"imu": {0: "batch"}, "fall_score": {0: "batch"}},
        opset_version=ONNX_OPSET,
        # The legacy path is still the stable PyTorch 2.8 exporter for LSTM
        # modules; the dynamo path is retained for the convolution-only model.
        dynamo=use_dynamo,
    )


def export_and_validate_onnx(
    adapter: Any | None,
    model_id: str,
    sample: np.ndarray,
    native_scores: np.ndarray,
    destination: Path,
) -> dict[str, Any]:
    import onnx
    import onnxruntime

    values = np.ascontiguousarray(sample[:256], dtype=np.float32)
    expected = np.asarray(native_scores[: len(values)], dtype=np.float64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f"{destination.stem}.tmp-{os.getpid()}.onnx"
    if model_id == "threshold_impact":
        onnx.save_model(_threshold_model(), temporary)
        input_name = "imu"
    elif model_id.startswith("torch_"):
        _export_torch(adapter, model_id, values, temporary)
        input_name = "imu"
    else:
        onnx.save_model(_tabular_model(adapter, model_id, values.shape[1]), temporary)
        input_name = "features"
    model = onnx.load(temporary)
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        str(temporary),
        providers=["CPUExecutionProvider"],
    )
    tolerance = 1e-4
    batch_sizes = sorted({size for size in (1, min(7, len(values)), len(values)) if size})
    maximum_error = 0.0
    for batch_size in batch_sizes:
        observed = np.asarray(
            session.run(["fall_score"], {input_name: values[:batch_size]})[0]
        ).reshape(-1)
        batch_expected = expected[:batch_size]
        batch_error = (
            float(np.max(np.abs(observed - batch_expected))) if batch_size else 0.0
        )
        maximum_error = max(maximum_error, batch_error)
        if not np.allclose(
            observed, batch_expected, rtol=tolerance, atol=tolerance
        ):
            raise ValueError(
                f"ONNX parity failed for {model_id} at batch {batch_size}: "
                f"max_abs_error={batch_error:.8g}"
            )
    temporary.replace(destination)
    return {
        "status": "PASS",
        "path": str(destination),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "opset": ONNX_OPSET,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "provider": "CPUExecutionProvider",
        "input_name": input_name,
        "input_domain": "physical" if model_id == "threshold_impact" else "model_ready",
        "output_name": "fall_score",
        "samples": len(values),
        "validated_batch_sizes": batch_sizes,
        "maximum_absolute_error": maximum_error,
        "absolute_relative_tolerance": tolerance,
    }
