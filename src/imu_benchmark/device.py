from __future__ import annotations

import importlib
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any


class CudaUnavailable(RuntimeError):
    """Raised when strict CUDA execution cannot be proven."""


def require_cuda_modules() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name in ("cupy", "cuml", "torch", "xgboost"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise CudaUnavailable(f"Missing strict-GPU packages: {', '.join(missing)}")
    cp = modules["cupy"]
    torch = modules["torch"]
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:
        raise CudaUnavailable(f"CuPy cannot access the CUDA runtime: {error}") from error
    if device_count < 1:
        raise CudaUnavailable("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise CudaUnavailable("PyTorch reports CUDA unavailable")
    return cp, modules["cuml"], torch, modules["xgboost"]


def synchronize() -> None:
    cp, _, torch, _ = require_cuda_modules()
    cp.cuda.runtime.deviceSynchronize()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def cuda_environment() -> dict[str, Any]:
    cp, cuml, torch, xgboost = require_cuda_modules()
    device = cp.cuda.Device(0)
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "platform": platform.platform(),
        "gpu_name": str(name),
        "gpu_device_id": int(device.id),
        "gpu_total_bytes": int(total_bytes),
        "gpu_free_bytes_at_start": int(free_bytes),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cupy": cp.__version__,
        "cuml": cuml.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "torch_cuda": torch.version.cuda,
    }


@dataclass(slots=True)
class GpuMemoryMonitor:
    interval_seconds: float = 0.02
    peak_used_bytes: int = 0
    _stop: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _total_bytes: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._stop = threading.Event()

    def start(self) -> GpuMemoryMonitor:
        cp, _, _, _ = require_cuda_modules()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        self._total_bytes = int(total_bytes)
        self.peak_used_bytes = self._total_bytes - int(free_bytes)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        cp, _, _, _ = require_cuda_modules()
        while not self._stop.is_set():
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            self.peak_used_bytes = max(self.peak_used_bytes, self._total_bytes - int(free_bytes))
            time.sleep(self.interval_seconds)

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_used_bytes
