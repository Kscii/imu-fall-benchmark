from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .device import CudaUnavailable, require_cuda_modules, synchronize
from .models import create_adapter
from .sequence_models import _network, aggregate_bag_scores


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    requested_mode: str
    effective_mode: str
    estimated_input_bytes: int
    required_device_bytes: int
    free_device_bytes: int
    total_device_bytes: int
    reserved_device_bytes: int
    reason: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "estimated_input_bytes": self.estimated_input_bytes,
            "required_device_bytes": self.required_device_bytes,
            "free_device_bytes": self.free_device_bytes,
            "total_device_bytes": self.total_device_bytes,
            "reserved_device_bytes": self.reserved_device_bytes,
            "reason": self.reason,
        }


def select_execution_mode(requested: str, arrays: tuple[np.ndarray, ...]) -> ExecutionDecision:
    if requested not in {"auto", "resident", "streaming"}:
        raise ValueError(f"Unknown GPU execution mode: {requested}")
    cp, _, _, _ = require_cuda_modules()
    free_bytes, total_bytes = (int(value) for value in cp.cuda.runtime.memGetInfo())
    input_bytes = sum(int(array.nbytes) for array in arrays)
    required = int(input_bytes * 1.35) + 512 * 1024**2
    reserved = max(4 * 1024**3, int(total_bytes * 0.40))
    resident_available = max(0, free_bytes - reserved)
    if requested == "resident":
        if required > resident_available:
            raise CudaUnavailable(
                "Forced resident mode cannot satisfy the VRAM reserve: "
                f"required={required}, available_after_reserve={resident_available}"
            )
        effective, reason = "resident", "forced_and_capacity_check_passed"
    elif requested == "streaming":
        effective, reason = "streaming", "forced"
    elif required <= resident_available:
        effective, reason = "resident", "auto_capacity_check_passed"
    else:
        effective, reason = "streaming", "auto_preserved_vram_reserve"
    return ExecutionDecision(
        requested_mode=requested,
        effective_mode=effective,
        estimated_input_bytes=input_bytes,
        required_device_bytes=required,
        free_device_bytes=free_bytes,
        total_device_bytes=total_bytes,
        reserved_device_bytes=reserved,
        reason=reason,
    )


def feature_normalization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def sequence_normalization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.std(train, axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    shape = (1,) * (values.ndim - 1) + (len(mean),)
    return ((values - mean.reshape(shape)) / scale.reshape(shape)).astype(np.float32)


class CudaSequenceTrainer:
    def __init__(
        self,
        model_id: str,
        params: dict[str, Any],
        *,
        max_epochs: int,
        patience: int,
        top_fraction: float,
        random_seed: int,
        precision: str,
        execution_mode: str,
        use_class_weight: bool,
        epoch_callback: Callable[[int, int, float, int], None] | None = None,
    ) -> None:
        if precision not in {"fp32", "bf16"}:
            raise ValueError(f"Unsupported PyTorch precision: {precision}")
        if execution_mode not in {"resident", "streaming"}:
            raise ValueError(f"Unsupported sequence execution mode: {execution_mode}")
        self.model_id = model_id
        self.params = copy.deepcopy(params)
        self.max_epochs = max_epochs
        self.patience = patience
        self.top_fraction = top_fraction
        self.random_seed = random_seed
        self.precision = precision
        self.execution_mode = execution_mode
        self.use_class_weight = use_class_weight
        self.epoch_callback = epoch_callback
        self.model: Any | None = None
        self.best_epoch: int | None = None
        self.training_history: list[dict[str, float | int]] = []

    @staticmethod
    def _positive_weight(labels: np.ndarray) -> float:
        positive = int(np.count_nonzero(labels == 1))
        negative = int(np.count_nonzero(labels == 0))
        if not positive or not negative:
            raise ValueError("Training labels require both classes")
        return negative / positive

    def _host_or_device(self, values: np.ndarray) -> Any:
        _, _, torch, _ = require_cuda_modules()
        contiguous = np.ascontiguousarray(values, dtype=np.float32)
        tensor = torch.from_numpy(contiguous)
        if self.execution_mode == "resident":
            return tensor.to(device="cuda", non_blocking=False)
        return tensor.pin_memory()

    def _batch(self, values: Any, indices: Any) -> Any:
        _, _, torch, _ = require_cuda_modules()
        batch = values[indices]
        if self.execution_mode == "streaming":
            batch = batch.to(device="cuda", non_blocking=True)
        return batch

    def _autocast(self) -> Any:
        _, _, torch, _ = require_cuda_modules()
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"
        )

    def _setup(self, labels: np.ndarray) -> tuple[Any, Any]:
        _, _, torch, _ = require_cuda_modules()
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.use_deterministic_algorithms(True)
        self.model = _network(self.model_id, self.params)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        positive_weight = torch.tensor(
            self._positive_weight(labels) if self.use_class_weight else 1.0,
            dtype=torch.float32,
            device="cuda",
        )
        return optimizer, torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    def _best(
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

    def _predict_prepared(self, values: Any) -> np.ndarray:
        _, _, torch, _ = require_cuda_modules()
        if self.model is None:
            raise RuntimeError("Sequence model has not been fitted")
        self.model.eval()
        result = np.empty(len(values), dtype=np.float64)
        batch_size = int(self.params["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                indices = slice(start, stop)
                batch = self._batch(values, indices)
                with self._autocast():
                    logits = self.model(batch)
                result[start:stop] = torch.sigmoid(logits.float()).detach().cpu().numpy()
        return result

    def fit_supervised(
        self,
        train: np.ndarray,
        train_labels: np.ndarray,
        validation: np.ndarray,
        validation_labels: np.ndarray,
    ) -> None:
        _, _, torch, _ = require_cuda_modules()
        train_values = self._host_or_device(train)
        validation_values = self._host_or_device(validation)
        optimizer, loss_function = self._setup(train_labels)
        generator = np.random.default_rng(self.random_seed)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        remaining = self.patience
        batch_size = int(self.params["batch_size"])
        for epoch in range(1, self.max_epochs + 1):
            if self.model is None:
                raise RuntimeError("Sequence model is missing")
            self.model.train()
            order = generator.permutation(len(train_labels))
            for start in range(0, len(order), batch_size):
                positions = order[start : start + batch_size]
                values = self._batch(train_values, positions)
                labels = torch.as_tensor(
                    train_labels[positions], dtype=torch.float32, device="cuda"
                )
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss = loss_function(self.model(values).float(), labels)
                loss.backward()
                optimizer.step()
            scores = self._predict_prepared(validation_values)
            probability = np.clip(scores, 1e-7, 1.0 - 1e-7)
            validation_loss = float(
                -np.mean(
                    validation_labels * np.log(probability)
                    + (1 - validation_labels) * np.log(1 - probability)
                )
            )
            best_loss, remaining, state = self._best(validation_loss, epoch, best_loss, remaining)
            self.training_history.append(
                {
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "patience_remaining": remaining,
                }
            )
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, self.max_epochs, validation_loss, remaining)
            if state is not None:
                best_state = state
            if remaining <= 0:
                break
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
        synchronize()

    def _bag_logits(self, values: Any, sequence_index: np.ndarray) -> tuple[Any, np.ndarray]:
        _, _, torch, _ = require_cuda_modules()
        if self.model is None:
            raise RuntimeError("Sequence model is missing")
        with self._autocast():
            window_logits = self.model(values).float()
        sequence_ids = np.unique(sequence_index)
        logits = []
        for sequence_id in sequence_ids:
            positions = np.flatnonzero(sequence_index == sequence_id)
            count = max(1, int(np.ceil(len(positions) * self.top_fraction)))
            index = torch.as_tensor(positions, dtype=torch.long, device="cuda")
            logits.append(torch.topk(window_logits.index_select(0, index), count).values.mean())
        return torch.stack(logits), sequence_ids

    def fit_mil(
        self,
        train: np.ndarray,
        train_sequence_index: np.ndarray,
        train_bag_labels: dict[int, int],
        validation: np.ndarray,
        validation_sequence_index: np.ndarray,
        validation_bag_labels: dict[int, int],
    ) -> None:
        _, _, torch, _ = require_cuda_modules()
        train_values = self._host_or_device(train)
        validation_values = self._host_or_device(validation)
        bag_ids = np.unique(train_sequence_index)
        labels = np.asarray([train_bag_labels[int(value)] for value in bag_ids], dtype=np.int8)
        optimizer, loss_function = self._setup(labels)
        generator = np.random.default_rng(self.random_seed)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        remaining = self.patience
        bag_batch_size = int(self.params["bag_batch_size"])
        for epoch in range(1, self.max_epochs + 1):
            if self.model is None:
                raise RuntimeError("Sequence model is missing")
            self.model.train()
            shuffled = generator.permutation(bag_ids)
            for start in range(0, len(shuffled), bag_batch_size):
                selected = shuffled[start : start + bag_batch_size]
                mask = np.isin(train_sequence_index, selected)
                values = self._batch(train_values, np.flatnonzero(mask))
                logits, ordered = self._bag_logits(values, train_sequence_index[mask])
                targets = torch.as_tensor(
                    [train_bag_labels[int(value)] for value in ordered],
                    dtype=torch.float32,
                    device="cuda",
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(logits, targets)
                loss.backward()
                optimizer.step()
            window_scores = self._predict_prepared(validation_values)
            ordered, bag_scores = aggregate_bag_scores(
                window_scores, validation_sequence_index, self.top_fraction
            )
            validation_labels = np.asarray(
                [validation_bag_labels[int(value)] for value in ordered], dtype=np.int8
            )
            probability = np.clip(bag_scores, 1e-7, 1.0 - 1e-7)
            validation_loss = float(
                -np.mean(
                    validation_labels * np.log(probability)
                    + (1 - validation_labels) * np.log(1 - probability)
                )
            )
            best_loss, remaining, state = self._best(validation_loss, epoch, best_loss, remaining)
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, self.max_epochs, validation_loss, remaining)
            if state is not None:
                best_state = state
            if remaining <= 0:
                break
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
        synchronize()

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self._predict_prepared(self._host_or_device(values))

    def serialized_size(self) -> int:
        if self.model is None:
            return 0
        return sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )


def fit_tabular_supervised(
    model_id: str,
    params: dict[str, Any],
    *,
    random_seed: int,
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    already_standardized: bool,
) -> Any:
    adapter = create_adapter(model_id, params, random_seed=random_seed)
    if already_standardized:
        adapter.scaler = None
    adapter.fit(train, train_labels, validation=(validation, validation_labels))
    adapter.assert_cuda()
    return adapter
