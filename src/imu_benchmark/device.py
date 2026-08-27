from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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
    start_used_bytes: int = 0
    _stop: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _total_bytes: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._stop = threading.Event()

    def start(self) -> GpuMemoryMonitor:
        cp, _, _, _ = require_cuda_modules()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        self._total_bytes = int(total_bytes)
        self.start_used_bytes = self._total_bytes - int(free_bytes)
        self.peak_used_bytes = self.start_used_bytes
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


def _parse_nvidia_smi_line(line: str) -> dict[str, float | None] | None:
    fields = [value.strip() for value in line.split(",")]
    if len(fields) != 4:
        return None

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    gpu_utilization, memory_utilization, memory_used_mib, power_watts = map(number, fields)
    if gpu_utilization is None or memory_used_mib is None:
        return None
    return {
        "gpu_utilization_percent": gpu_utilization,
        "memory_utilization_percent": memory_utilization,
        "memory_used_mib": memory_used_mib,
        "power_watts": power_watts,
    }


@dataclass(slots=True)
class NvidiaSmiMonitor:
    """Collect low-overhead run-level GPU telemetry from nvidia-smi."""

    interval_ms: int = 100
    _process: subprocess.Popen[str] | None = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _samples: list[tuple[float, dict[str, float | None]]] = field(
        init=False, default_factory=list, repr=False
    )
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)
    _error: str | None = field(init=False, default=None, repr=False)

    def start(self) -> NvidiaSmiMonitor:
        executable = shutil.which("nvidia-smi")
        fallback = Path("/usr/lib/wsl/lib/nvidia-smi")
        if executable is None and fallback.is_file():
            executable = str(fallback)
        if executable is None:
            self._error = "nvidia-smi_not_found"
            return self
        command = (
            executable,
            "--id=0",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        )
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self._error = f"nvidia-smi_start_failed:{error}"
            return self
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()
        return self

    def _collect(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            sample = _parse_nvidia_smi_line(line)
            if sample is None:
                continue
            with self._lock:
                self._samples.append((time.perf_counter(), sample))

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self, started: float, stopped: float) -> dict[str, Any]:
        with self._lock:
            samples = [
                sample for timestamp, sample in self._samples if started <= timestamp <= stopped
            ]
        if not samples:
            return {
                "status": "unavailable",
                "samples": 0,
                "reason": self._error or "no_samples_in_interval",
            }
        result: dict[str, Any] = {"status": "available", "samples": len(samples)}
        for name in (
            "gpu_utilization_percent",
            "memory_utilization_percent",
            "memory_used_mib",
            "power_watts",
        ):
            values = np.asarray(
                [sample[name] for sample in samples if sample[name] is not None],
                dtype=np.float64,
            )
            if not len(values):
                continue
            result[name] = {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        return result
