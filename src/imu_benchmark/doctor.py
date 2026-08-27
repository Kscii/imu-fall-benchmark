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
from .specs import MODEL_SPECS, TABULAR_MODEL_IDS

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
    for model_id in TABULAR_MODEL_IDS:
        params = dict(MODEL_SPECS[model_id].fixed_params)
        if model_id in {"cuml_random_forest", "xgboost_cuda"}:
            params["n_estimators"] = 10
        if model_id == "torch_mlp":
            params["max_epochs"] = 2
            params["patience"] = 1
        adapter = create_adapter(model_id, params, random_seed=random_seed)
        started = time.perf_counter()
        adapter.fit(features, labels, final_epochs=2 if model_id == "torch_mlp" else None)
        probabilities = adapter.predict_proba(features[:8])
        adapter.assert_cuda()
        models.append(
            {
                "model_id": model_id,
                "status": "PASS",
                "seconds": time.perf_counter() - started,
                "probability_min": float(np.min(probabilities)),
                "probability_max": float(np.max(probabilities)),
            }
        )
        del adapter
        release_gpu_memory()
    return {
        "status": "PASS",
        "random_seed": random_seed,
        "determinism_policy": "fixed_seed_and_torch_deterministic_algorithms",
        "environment": environment,
        "models": models,
    }
