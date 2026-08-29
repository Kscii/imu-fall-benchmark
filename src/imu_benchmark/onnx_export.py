from __future__ import annotations

import copy
import hashlib
import os
import time
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
    *,
    additional_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    batch_size: int = 256,
    max_samples: int | None = 256,
    provider: str = "cpu",
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, Any]:
    import onnx
    import onnxruntime

    if batch_size <= 0:
        raise ValueError("ONNX parity batch size must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("ONNX parity max_samples must be null or positive")
    if provider != "cpu":
        raise ValueError("Only the CPU ONNX Runtime provider is supported")
    if rtol < 0 or atol < 0:
        raise ValueError("ONNX parity tolerances must be non-negative")
    if len(sample) != len(native_scores):
        raise ValueError("ONNX parity input and native-score lengths differ")
    if not len(sample):
        raise ValueError("ONNX export requires at least one validation sample")

    export_values = np.ascontiguousarray(sample[: min(256, len(sample))], dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f"{destination.stem}.tmp-{os.getpid()}.onnx"
    conversion_started = time.perf_counter()
    if model_id == "threshold_impact":
        onnx.save_model(_threshold_model(), temporary)
        input_name = "imu"
    elif model_id.startswith("torch_"):
        _export_torch(adapter, model_id, export_values, temporary)
        input_name = "imu"
    else:
        onnx.save_model(
            _tabular_model(adapter, model_id, export_values.shape[1]), temporary
        )
        input_name = "features"
    model = onnx.load(temporary)
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        str(temporary),
        providers=["CPUExecutionProvider"],
    )
    conversion_seconds = time.perf_counter() - conversion_started

    split_inputs = {"validation": (sample, native_scores)}
    if additional_splits:
        overlap = set(split_inputs) & set(additional_splits)
        if overlap:
            raise ValueError(f"Duplicate ONNX parity splits: {sorted(overlap)}")
        split_inputs.update(additional_splits)

    parity: dict[str, dict[str, Any]] = {}
    all_batch_sizes: set[int] = set()
    inference_seconds = 0.0
    comparison_seconds = 0.0
    maximum_error = 0.0
    total_samples = 0
    total_batches = 0
    for split_name, (split_values, split_scores) in split_inputs.items():
        if len(split_values) != len(split_scores):
            raise ValueError(
                f"ONNX parity input and native-score lengths differ for {split_name}"
            )
        limit = len(split_values) if max_samples is None else min(
            len(split_values), max_samples
        )
        if not limit:
            raise ValueError(f"ONNX parity split is empty: {split_name}")
        values = np.ascontiguousarray(split_values[:limit], dtype=np.float32)
        expected = np.asarray(split_scores[:limit], dtype=np.float64)
        errors = np.empty(limit, dtype=np.float64)

        # Retain the small dynamic-batch probes while also streaming the full split.
        probe_sizes = sorted({1, min(7, limit), min(batch_size, limit)})
        for probe_size in probe_sizes:
            started = time.perf_counter()
            observed = np.asarray(
                session.run(
                    ["fall_score"], {input_name: values[:probe_size]}
                )[0]
            ).reshape(-1)
            inference_seconds += time.perf_counter() - started
            started = time.perf_counter()
            probe_error = np.abs(observed - expected[:probe_size])
            comparison_seconds += time.perf_counter() - started
            if not np.allclose(
                observed, expected[:probe_size], rtol=rtol, atol=atol
            ):
                raise ValueError(
                    f"ONNX parity failed for {model_id}/{split_name} at batch "
                    f"{probe_size}: max_abs_error={float(np.max(probe_error)):.8g}, "
                    f"rtol={rtol:.8g}, atol={atol:.8g}"
                )
            all_batch_sizes.add(probe_size)

        batches = 0
        for start in range(0, limit, batch_size):
            stop = min(start + batch_size, limit)
            current = values[start:stop]
            started = time.perf_counter()
            observed = np.asarray(
                session.run(["fall_score"], {input_name: current})[0]
            ).reshape(-1)
            inference_seconds += time.perf_counter() - started
            started = time.perf_counter()
            batch_error = np.abs(observed - expected[start:stop])
            errors[start:stop] = batch_error
            comparison_seconds += time.perf_counter() - started
            if not np.allclose(
                observed, expected[start:stop], rtol=rtol, atol=atol
            ):
                raise ValueError(
                    f"ONNX parity failed for {model_id}/{split_name} at samples "
                    f"{start}:{stop}: max_abs_error={float(np.max(batch_error)):.8g}, "
                    f"rtol={rtol:.8g}, atol={atol:.8g}"
                )
            all_batch_sizes.add(len(current))
            batches += 1

        split_maximum = float(np.max(errors))
        maximum_error = max(maximum_error, split_maximum)
        total_samples += limit
        total_batches += batches
        parity[split_name] = {
            "samples": limit,
            "batches": batches,
            "maximum_absolute_error": split_maximum,
            "mean_absolute_error": float(np.mean(errors)),
            "p99_absolute_error": float(np.quantile(errors, 0.99)),
        }

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
        "samples": total_samples,
        "batches": total_batches,
        "batch_size": batch_size,
        "validated_batch_sizes": sorted(all_batch_sizes),
        "parity_splits": parity,
        "maximum_absolute_error": maximum_error,
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "timing": {
            "conversion_seconds": conversion_seconds,
            "runtime_inference_seconds": inference_seconds,
            "comparison_seconds": comparison_seconds,
            "total_seconds": (
                conversion_seconds + inference_seconds + comparison_seconds
            ),
        },
    }
