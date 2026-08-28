from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .device import require_cuda_modules, synchronize
from .models import ModelAdapter, create_adapter
from .specs import MODEL_SPECS


def threshold_impact_scores(windows: np.ndarray) -> np.ndarray:
    """Return a unitless, interpretable impact score for each raw IMU window."""
    raw = np.asarray(windows, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (50, 6):
        raise ValueError(f"Expected (n, 50, 6) raw windows, got {raw.shape}")
    params = MODEL_SPECS["threshold_impact"].fixed_params
    acceleration = np.linalg.norm(raw[:, :, :3], axis=2)
    angular_velocity = np.linalg.norm(raw[:, :, 3:], axis=2)
    acceleration_impact = np.max(
        np.abs(acceleration - float(params["gravity_mps2"])), axis=1
    ) / float(params["gravity_mps2"])
    rotation_impact = np.max(angular_velocity, axis=1) / float(params["gyro_scale_rad_s"])
    return (acceleration_impact + rotation_impact).astype(np.float64)


def create_fixed_tabular_adapter(
    model_id: str, *, max_epochs: int, patience: int, random_seed: int
) -> ModelAdapter:
    params = copy.deepcopy(MODEL_SPECS[model_id].fixed_params)
    if model_id == "torch_mlp":
        params["max_epochs"] = max_epochs
        params["patience"] = patience
    return create_adapter(model_id, params, random_seed=random_seed)


def _network(model_id: str, params: dict[str, Any]) -> Any:
    _, _, torch, _ = require_cuda_modules()
    channels = int(params["channels"])
    hidden_size = int(params["hidden_size"])
    dropout = float(params["dropout"])

    class Cnn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv1d(6, channels // 2, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),
                torch.nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )
            self.output = torch.nn.Linear(channels, 1)

        def forward(self, values: Any) -> Any:
            encoded = self.features(values.transpose(1, 2)).squeeze(-1)
            return self.output(encoded).squeeze(-1)

    class Lstm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = torch.nn.LSTM(6, hidden_size, batch_first=True)
            self.dropout = torch.nn.Dropout(dropout)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            _, (hidden, _) = self.recurrent(values)
            return self.output(self.dropout(hidden[-1])).squeeze(-1)

    class CnnLstm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = torch.nn.Sequential(
                torch.nn.Conv1d(6, channels // 2, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2),
                torch.nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.recurrent = torch.nn.LSTM(channels, hidden_size, batch_first=True)
            self.dropout = torch.nn.Dropout(dropout)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            local = self.convolution(values.transpose(1, 2)).transpose(1, 2)
            _, (hidden, _) = self.recurrent(local)
            return self.output(self.dropout(hidden[-1])).squeeze(-1)

    constructors = {
        "torch_1d_cnn": Cnn,
        "torch_lstm": Lstm,
        "torch_cnn_lstm": CnnLstm,
    }
    if model_id not in constructors:
        raise ValueError(f"Unknown sequence architecture: {model_id}")
    return constructors[model_id]().to("cuda")


def aggregate_bag_scores(
    scores: np.ndarray, sequence_index: np.ndarray, top_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    sequence_ids = np.unique(sequence_index)
    aggregated = np.empty(len(sequence_ids), dtype=np.float64)
    for output_index, sequence_id in enumerate(sequence_ids):
        values = scores[sequence_index == sequence_id]
        count = max(1, int(np.ceil(len(values) * top_fraction)))
        aggregated[output_index] = float(np.mean(np.partition(values, -count)[-count:]))
    return sequence_ids, aggregated


class TorchSequenceAdapter:
    def __init__(
        self,
        model_id: str,
        *,
        max_epochs: int,
        patience: int,
        top_fraction: float,
        random_seed: int,
    ) -> None:
        self.model_id = model_id
        self.params = copy.deepcopy(MODEL_SPECS[model_id].fixed_params)
        self.max_epochs = max_epochs
        self.patience = patience
        self.top_fraction = top_fraction
        self.random_seed = random_seed
        self.model: Any | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.best_epoch: int | None = None

    def _normalize_fit(self, windows: np.ndarray) -> np.ndarray:
        self.mean = np.mean(windows, axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.scale = np.std(windows, axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        return self._normalize(windows)

    def _normalize(self, windows: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Sequence model normalization has not been fitted")
        return ((windows - self.mean[None, None, :]) / self.scale[None, None, :]).astype(np.float32)

    @staticmethod
    def _positive_weight(labels: np.ndarray) -> float:
        positive = int(np.count_nonzero(labels == 1))
        negative = int(np.count_nonzero(labels == 0))
        if not positive or not negative:
            raise ValueError("Training labels require both classes")
        return negative / positive

    def _optimizer_and_loss(self, labels: np.ndarray) -> tuple[Any, Any]:
        _, _, torch, _ = require_cuda_modules()
        if self.model is None:
            raise RuntimeError("Sequence model is missing")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        positive_weight = torch.tensor(
            self._positive_weight(labels), dtype=torch.float32, device="cuda"
        )
        return optimizer, torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    def _remember_best(
        self, loss: float, epoch: int, best_loss: float, remaining: int
    ) -> tuple[float, int, dict[str, Any] | None]:
        if self.model is None:
            raise RuntimeError("Sequence model is missing")
        if loss < best_loss - 1e-7:
            self.best_epoch = epoch
            return (
                loss,
                self.patience,
                {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                },
            )
        return best_loss, remaining - 1, None

    def fit_supervised(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        validation_windows: np.ndarray,
        validation_labels: np.ndarray,
    ) -> None:
        _, _, torch, _ = require_cuda_modules()
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.use_deterministic_algorithms(True)
        normalized_train = self._normalize_fit(train_windows)
        normalized_validation = self._normalize(validation_windows)
        self.model = _network(self.model_id, self.params)
        optimizer, loss_function = self._optimizer_and_loss(train_labels)
        generator = np.random.default_rng(self.random_seed)
        batch_size = int(self.params["batch_size"])
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        remaining = self.patience
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            order = generator.permutation(len(train_labels))
            for start in range(0, len(train_labels), batch_size):
                indices = order[start : start + batch_size]
                values = torch.as_tensor(normalized_train[indices], device="cuda")
                labels = torch.as_tensor(train_labels[indices], dtype=torch.float32, device="cuda")
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.model(values), labels)
                loss.backward()
                optimizer.step()
            validation_scores = self._predict_normalized(normalized_validation)
            probability = np.clip(validation_scores, 1e-7, 1.0 - 1e-7)
            validation_loss = float(
                -np.mean(
                    validation_labels * np.log(probability)
                    + (1 - validation_labels) * np.log(1 - probability)
                )
            )
            best_loss, remaining, state = self._remember_best(
                validation_loss, epoch, best_loss, remaining
            )
            if state is not None:
                best_state = state
            if remaining <= 0:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        synchronize()

    def _bag_logits(self, values: Any, sequence_index: np.ndarray) -> tuple[Any, np.ndarray]:
        _, _, torch, _ = require_cuda_modules()
        if self.model is None:
            raise RuntimeError("Sequence model is missing")
        window_logits = self.model(values)
        sequence_ids = np.unique(sequence_index)
        logits = []
        for sequence_id in sequence_ids:
            positions = np.flatnonzero(sequence_index == sequence_id)
            count = max(1, int(np.ceil(len(positions) * self.top_fraction)))
            index_tensor = torch.as_tensor(positions, dtype=torch.long, device="cuda")
            selected = window_logits.index_select(0, index_tensor)
            logits.append(torch.topk(selected, count).values.mean())
        return torch.stack(logits), sequence_ids

    def fit_mil(
        self,
        train_windows: np.ndarray,
        train_sequence_index: np.ndarray,
        train_bag_labels: dict[int, int],
        validation_windows: np.ndarray,
        validation_sequence_index: np.ndarray,
        validation_bag_labels: dict[int, int],
    ) -> None:
        _, _, torch, _ = require_cuda_modules()
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.use_deterministic_algorithms(True)
        normalized_train = self._normalize_fit(train_windows)
        normalized_validation = self._normalize(validation_windows)
        self.model = _network(self.model_id, self.params)
        bag_ids = np.unique(train_sequence_index)
        bag_labels = np.asarray([train_bag_labels[int(value)] for value in bag_ids], dtype=np.int8)
        optimizer, loss_function = self._optimizer_and_loss(bag_labels)
        generator = np.random.default_rng(self.random_seed)
        bag_batch_size = int(self.params["bag_batch_size"])
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        remaining = self.patience
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            shuffled = generator.permutation(bag_ids)
            for start in range(0, len(shuffled), bag_batch_size):
                selected_bags = shuffled[start : start + bag_batch_size]
                mask = np.isin(train_sequence_index, selected_bags)
                local_sequence_index = train_sequence_index[mask]
                values = torch.as_tensor(normalized_train[mask], device="cuda")
                logits, ordered_bags = self._bag_logits(values, local_sequence_index)
                labels = torch.as_tensor(
                    [train_bag_labels[int(value)] for value in ordered_bags],
                    dtype=torch.float32,
                    device="cuda",
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
            window_scores = self._predict_normalized(normalized_validation)
            ordered_bags, bag_scores = aggregate_bag_scores(
                window_scores, validation_sequence_index, self.top_fraction
            )
            labels = np.asarray(
                [validation_bag_labels[int(value)] for value in ordered_bags], dtype=np.int8
            )
            probability = np.clip(bag_scores, 1e-7, 1.0 - 1e-7)
            validation_loss = float(
                -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
            )
            best_loss, remaining, state = self._remember_best(
                validation_loss, epoch, best_loss, remaining
            )
            if state is not None:
                best_state = state
            if remaining <= 0:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        synchronize()

    def _predict_normalized(self, windows: np.ndarray) -> np.ndarray:
        _, _, torch, _ = require_cuda_modules()
        if self.model is None:
            raise RuntimeError("Sequence model has not been fitted")
        self.model.eval()
        result = np.empty(len(windows), dtype=np.float64)
        batch_size = int(self.params["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                end = min(start + batch_size, len(windows))
                values = torch.as_tensor(windows[start:end], device="cuda")
                result[start:end] = torch.sigmoid(self.model(values)).detach().cpu().numpy()
        return result

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        return self._predict_normalized(self._normalize(windows))

    def assert_cuda(self) -> None:
        if self.model is None:
            raise RuntimeError("Sequence model is missing")
        devices = {parameter.device.type for parameter in self.model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"Sequence model is not strict CUDA: {devices}")

    def serialized_size(self) -> int:
        if self.model is None:
            return 0
        return sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )
