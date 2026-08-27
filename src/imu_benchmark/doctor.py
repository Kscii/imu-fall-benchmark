from __future__ import annotations

import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sklearn

from .device import cuda_environment
from .models import create_adapter, release_gpu_memory
from .runtime import is_wsl2, repository_is_on_linux_filesystem
from .sequence_models import _network, threshold_impact_scores
from .specs import MODEL_SPECS

PUBLIC_TABULAR_MODELS = (
    "cuml_logistic_regression",
    "cuml_random_forest",
    "xgboost_cuda",
)
PUBLIC_SEQUENCE_MODELS = ("torch_1d_cnn", "torch_lstm", "torch_cnn_lstm")

MINIMUM_RUNTIME_FREE_BYTES = 10 * 1024**3


def run_doctor(*, random_seed: int, project_root: Path, work_root: Path) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(work_root)
    if disk.free < MINIMUM_RUNTIME_FREE_BYTES:
        raise ValueError("At least 10 GiB free space is required under the work root")
    environment = {
        **cuda_environment(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scikit_learn": sklearn.__version__,
        "is_wsl2": is_wsl2(),
        "machine": platform.machine(),
        "repository_on_wsl_linux_filesystem": repository_is_on_linux_filesystem(project_root),
        "work_root": str(work_root),
        "work_root_free_bytes": int(disk.free),
    }
    rng = np.random.default_rng(random_seed)
    features = rng.normal(size=(64, 8)).astype(np.float32)
    labels = np.asarray([0, 1] * 32, dtype=np.int8)
    models: list[dict[str, Any]] = []
    threshold_windows = rng.normal(size=(8, 60, 6)).astype(np.float32)
    threshold_scores = threshold_impact_scores(threshold_windows)
    models.append(
        {
            "model_id": "threshold_impact",
            "status": "PASS",
            "seconds": 0.0,
            "fall_score_min": float(np.min(threshold_scores)),
            "fall_score_max": float(np.max(threshold_scores)),
            "strict_cuda": False,
        }
    )
    for model_id in PUBLIC_TABULAR_MODELS:
        params = dict(MODEL_SPECS[model_id].fixed_params)
        if model_id in {"cuml_random_forest", "xgboost_cuda"}:
            params["n_estimators"] = 10
        adapter = create_adapter(model_id, params, random_seed=random_seed)
        started = time.perf_counter()
        adapter.fit(features, labels)
        probabilities = adapter.predict_proba(features[:8])
        adapter.assert_cuda()
        models.append(
            {
                "model_id": model_id,
                "status": "PASS",
                "seconds": time.perf_counter() - started,
                "probability_min": float(np.min(probabilities)),
                "probability_max": float(np.max(probabilities)),
                "strict_cuda": True,
            }
        )
        del adapter
        release_gpu_memory()
    from .device import require_cuda_modules

    _, _, torch, _ = require_cuda_modules()
    for model_id in PUBLIC_SEQUENCE_MODELS:
        started = time.perf_counter()
        model = _network(model_id, dict(MODEL_SPECS[model_id].fixed_params))
        values = torch.as_tensor(threshold_windows, device="cuda")
        with torch.inference_mode():
            output = model(values)
        if output.shape != (len(threshold_windows),) or output.device.type != "cuda":
            raise ValueError(f"Invalid CUDA sequence-model output for {model_id}")
        models.append(
            {
                "model_id": model_id,
                "status": "PASS",
                "seconds": time.perf_counter() - started,
                "strict_cuda": True,
            }
        )
        del model, values, output
        release_gpu_memory()
    environment["torch_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return {
        "status": "PASS",
        "random_seed": random_seed,
        "determinism_policy": "fixed_seed_and_torch_deterministic_algorithms",
        "environment": environment,
        "models": models,
    }
