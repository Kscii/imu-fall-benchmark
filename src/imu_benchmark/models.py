from __future__ import annotations

import copy
import io
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .device import CudaUnavailable, require_cuda_modules, synchronize
from .specs import MODEL_SPECS

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class ModelConvergenceError(RuntimeError):
    """Raised when an iterative model reaches its configured limit."""


class CudaScaler:
    def __init__(self) -> None:
        self.mean: Any | None = None
        self.scale: Any | None = None

    def fit_transform(self, features: Any) -> Any:
        cp, _, _, _ = require_cuda_modules()
        self.mean = cp.mean(features, axis=0)
        scale = cp.std(features, axis=0)
        self.scale = cp.where(scale > 0, scale, 1.0)
        return (features - self.mean) / self.scale

    def transform(self, features: Any) -> Any:
        if self.mean is None or self.scale is None:
            raise RuntimeError("CUDA scaler has not been fitted")
        return (features - self.mean) / self.scale


class ModelAdapter(ABC):
    def __init__(self, model_id: str, params: dict[str, Any], *, random_seed: int) -> None:
        self.model_id = model_id
        self.params = copy.deepcopy(params)
        self.random_seed = random_seed
        self.scaler = CudaScaler() if MODEL_SPECS[model_id].standardize else None
        self.best_epoch: int | None = None

    def _training_arrays(self, features: np.ndarray, labels: np.ndarray) -> tuple[Any, Any]:
        cp, _, _, _ = require_cuda_modules()
        gpu_features = cp.asarray(features, dtype=cp.float32)
        gpu_labels = cp.asarray(labels, dtype=cp.int32)
        if self.scaler is not None:
            gpu_features = self.scaler.fit_transform(gpu_features)
        return gpu_features, gpu_labels

    def _prediction_array(self, features: np.ndarray) -> Any:
        cp, _, _, _ = require_cuda_modules()
        gpu_features = cp.asarray(features, dtype=cp.float32)
        return self.scaler.transform(gpu_features) if self.scaler is not None else gpu_features

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        final_epochs: int | None = None,
    ) -> None: ...

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def assert_cuda(self) -> None: ...

    def serialized_size(self) -> int | None:
        try:
            return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        except (TypeError, pickle.PickleError):
            return None

    def optimization_metadata(self) -> dict[str, int | bool] | None:
        return None


class CuMLAdapter(ModelAdapter):
    def __init__(self, model_id: str, params: dict[str, Any], *, random_seed: int) -> None:
        super().__init__(model_id, params, random_seed=random_seed)
        self.estimator: Any | None = None
        self.calibrator: Any | None = None

    def _make_estimator(self) -> Any:
        _, _, _, _ = require_cuda_modules()
        common = {"output_type": "cupy"}
        if self.model_id == "cuml_logistic_regression":
            from cuml.linear_model import LogisticRegression

            return LogisticRegression(**self.params, **common)
        if self.model_id == "cuml_random_forest":
            from cuml.ensemble import RandomForestClassifier

            return RandomForestClassifier(random_state=self.random_seed, **self.params, **common)
        if self.model_id == "cuml_rbf_svc":
            from cuml.svm import SVC

            return SVC(random_state=self.random_seed, **self.params, **common)
        if self.model_id == "cuml_knn":
            from cuml.neighbors import KNeighborsClassifier

            return KNeighborsClassifier(**self.params, **common)
        if self.model_id == "cuml_gaussian_nb":
            from cuml.naive_bayes import GaussianNB

            return GaussianNB(**self.params, **common)
        raise ValueError(f"Unsupported cuML model: {self.model_id}")

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        final_epochs: int | None = None,
    ) -> None:
        del validation, final_epochs
        gpu_features, gpu_labels = self._training_arrays(features, labels)
        self.estimator = self._make_estimator()
        self.estimator.fit(gpu_features, gpu_labels)
        self.assert_optimization_converged()
        if self.model_id == "cuml_rbf_svc":
            from cuml.linear_model import LogisticRegression

            margins = self.estimator.decision_function(gpu_features).reshape(-1, 1)
            self.calibrator = LogisticRegression(
                C=1.0,
                penalty="l2",
                max_iter=1000,
                output_type="cupy",
            )
            self.calibrator.fit(margins, gpu_labels)
        synchronize()
        self.assert_cuda()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model has not been fitted")
        cp, _, _, _ = require_cuda_modules()
        prediction_features = self._prediction_array(features)
        if self.model_id == "cuml_rbf_svc":
            if self.calibrator is None:
                raise RuntimeError("SVC probability calibrator has not been fitted")
            margins = self.estimator.decision_function(prediction_features).reshape(-1, 1)
            probabilities = cp.asarray(self.calibrator.predict_proba(margins))
        else:
            probabilities = cp.asarray(self.estimator.predict_proba(prediction_features))
        synchronize()
        result = cp.asnumpy(probabilities[:, 1]).astype(np.float64, copy=False)
        return _validated_probabilities(result)

    def assert_cuda(self) -> None:
        if self.estimator is None or not self.estimator.__class__.__module__.startswith("cuml."):
            raise CudaUnavailable(f"{self.model_id} is not a direct cuML estimator")
        if self.model_id == "cuml_rbf_svc" and (
            self.calibrator is None or not self.calibrator.__class__.__module__.startswith("cuml.")
        ):
            raise CudaUnavailable("RBF SVC probability calibration is not direct cuML")

    def optimization_metadata(self) -> dict[str, int | bool] | None:
        if self.model_id != "cuml_logistic_regression":
            return None
        if self.estimator is None or not hasattr(self.estimator, "n_iter_"):
            raise RuntimeError("Logistic Regression iteration state is unavailable")
        iterations = int(np.asarray(self.estimator.n_iter_).reshape(-1)[0])
        iteration_limit = int(self.params["max_iter"])
        return {
            "converged": iterations < iteration_limit,
            "iterations": iterations,
            "iteration_limit": iteration_limit,
        }

    def assert_optimization_converged(self) -> None:
        optimization = self.optimization_metadata()
        if optimization is not None and not optimization["converged"]:
            raise ModelConvergenceError(
                f"{self.model_id} reached its iteration limit: "
                f"{optimization['iterations']}/{optimization['iteration_limit']}"
            )


class XGBoostCudaAdapter(ModelAdapter):
    def __init__(self, model_id: str, params: dict[str, Any], *, random_seed: int) -> None:
        super().__init__(model_id, params, random_seed=random_seed)
        self.estimator: Any | None = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        final_epochs: int | None = None,
    ) -> None:
        del validation, final_epochs
        _, _, _, xgboost = require_cuda_modules()
        gpu_features, gpu_labels = self._training_arrays(features, labels)
        self.estimator = xgboost.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            random_state=self.random_seed,
            verbosity=0,
            **self.params,
        )
        self.estimator.fit(gpu_features, gpu_labels)
        synchronize()
        self.assert_cuda()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model has not been fitted")
        cp, _, _, _ = require_cuda_modules()
        output = self.estimator.predict_proba(self._prediction_array(features))
        probabilities = cp.asnumpy(cp.asarray(output)[:, 1]).astype(np.float64, copy=False)
        synchronize()
        return _validated_probabilities(probabilities)

    def assert_cuda(self) -> None:
        if self.estimator is None:
            raise CudaUnavailable("XGBoost estimator is missing")
        config = json.loads(self.estimator.get_booster().save_config())
        device = config["learner"]["generic_param"]["device"]
        if not str(device).startswith("cuda"):
            raise CudaUnavailable(f"XGBoost did not use CUDA: {device}")


class TorchMlpAdapter(ModelAdapter):
    def __init__(self, model_id: str, params: dict[str, Any], *, random_seed: int) -> None:
        super().__init__(model_id, params, random_seed=random_seed)
        self.model: Any | None = None

    def _build(self, width: int) -> Any:
        _, _, torch, _ = require_cuda_modules()
        layers: list[Any] = []
        current = width
        for hidden in self.params["hidden_layers"]:
            layers.extend((torch.nn.Linear(current, hidden), torch.nn.ReLU()))
            if self.params["dropout"]:
                layers.append(torch.nn.Dropout(float(self.params["dropout"])))
            current = hidden
        layers.append(torch.nn.Linear(current, 1))
        return torch.nn.Sequential(*layers).to("cuda")

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        final_epochs: int | None = None,
    ) -> None:
        cp, _, torch, _ = require_cuda_modules()
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.use_deterministic_algorithms(True)
        gpu_features, gpu_labels = self._training_arrays(features, labels)
        train_x = torch.from_dlpack(gpu_features).to(dtype=torch.float32)
        train_y = torch.from_dlpack(gpu_labels).to(dtype=torch.float32).reshape(-1, 1)
        self.model = self._build(train_x.shape[1])
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_function = torch.nn.BCEWithLogitsLoss()
        validation_tensors: tuple[Any, Any] | None = None
        if validation is not None:
            val_x = cp.asarray(validation[0], dtype=cp.float32)
            if self.scaler is not None:
                val_x = self.scaler.transform(val_x)
            validation_tensors = (
                torch.from_dlpack(val_x).to(dtype=torch.float32),
                torch.as_tensor(validation[1], dtype=torch.float32, device="cuda").reshape(-1, 1),
            )
        maximum = int(final_epochs or self.params["max_epochs"])
        patience = int(self.params["patience"])
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        remaining = patience
        generator = torch.Generator(device="cuda").manual_seed(self.random_seed)
        for epoch in range(1, maximum + 1):
            self.model.train()
            order = torch.randperm(len(train_x), generator=generator, device="cuda")
            batch_size = int(self.params["batch_size"])
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.model(train_x[indices]), train_y[indices])
                loss.backward()
                optimizer.step()
            if validation_tensors is None:
                self.best_epoch = epoch
                continue
            self.model.eval()
            with torch.inference_mode():
                val_loss = float(
                    loss_function(self.model(validation_tensors[0]), validation_tensors[1]).item()
                )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = {
                    key: value.detach().clone() for key, value in self.model.state_dict().items()
                }
                self.best_epoch = epoch
                remaining = patience
            else:
                remaining -= 1
                if remaining == 0:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        synchronize()
        self.assert_cuda()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        _, _, torch, _ = require_cuda_modules()
        gpu_features = self._prediction_array(features)
        tensor = torch.from_dlpack(gpu_features).to(dtype=torch.float32)
        self.model.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(tensor)).reshape(-1)
        synchronize()
        result = probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
        return _validated_probabilities(result)

    def assert_cuda(self) -> None:
        if self.model is None:
            raise CudaUnavailable("PyTorch MLP is missing")
        devices = {parameter.device.type for parameter in self.model.parameters()}
        if devices != {"cuda"}:
            raise CudaUnavailable(f"PyTorch model devices are not strict CUDA: {devices}")

    def serialized_size(self) -> int | None:
        if self.model is None:
            return None
        _, _, torch, _ = require_cuda_modules()
        destination = io.BytesIO()
        torch.save(self.model.state_dict(), destination)
        return destination.tell()


def create_adapter(
    model_id: str, params: dict[str, Any], *, random_seed: int
) -> ModelAdapter:
    spec = MODEL_SPECS[model_id]
    if spec.backend == "cuml":
        return CuMLAdapter(model_id, params, random_seed=random_seed)
    if spec.backend == "xgboost":
        return XGBoostCudaAdapter(model_id, params, random_seed=random_seed)
    if spec.backend == "pytorch":
        return TorchMlpAdapter(model_id, params, random_seed=random_seed)
    raise ValueError(f"Unknown model backend: {spec.backend}")


def release_gpu_memory() -> None:
    cp, _, torch, _ = require_cuda_modules()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _validated_probabilities(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Model returned invalid probability shape or values")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"Model probabilities are outside [0, 1]: min={minimum}, max={maximum}")
    return np.clip(values, 0.0, 1.0)
