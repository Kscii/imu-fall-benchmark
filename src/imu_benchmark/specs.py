from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

RANDOM_SEED = 3888

TABULAR_MODEL_IDS = (
    "cuml_logistic_regression",
    "cuml_random_forest",
    "cuml_rbf_svc",
    "cuml_knn",
    "cuml_gaussian_nb",
    "xgboost_cuda",
    "torch_mlp",
)
SEQUENCE_MODEL_IDS = ("torch_1d_cnn", "torch_lstm", "torch_cnn_lstm")
THRESHOLD_MODEL_ID = "threshold_impact"
MODEL_IDS = (THRESHOLD_MODEL_ID, *TABULAR_MODEL_IDS, *SEQUENCE_MODEL_IDS)

POSITION_SUITES = ("position_paired",)
MIL_SUITES = ("fall_universal", "fall_chest_only", "fall_waist_only")
SUITES = (*POSITION_SUITES, *MIL_SUITES)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    backend: str
    input_kind: str
    standardize: bool
    position: bool
    fall_mil: bool
    fixed_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JobSpec:
    suite: str
    model_id: str
    outer_fold: int
    target: str
    objective: str
    input_kind: str

    @property
    def run_key(self) -> str:
        return f"{self.suite}__{self.model_id}__fold_{self.outer_fold}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tabular(
    model_id: str, *, backend: str, standardize: bool, **params: Any
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        backend=backend,
        input_kind="features",
        standardize=standardize,
        position=True,
        fall_mil=False,
        fixed_params=params,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    THRESHOLD_MODEL_ID: ModelSpec(
        model_id=THRESHOLD_MODEL_ID,
        backend="rule",
        input_kind="raw",
        standardize=False,
        position=False,
        fall_mil=True,
        fixed_params={"gravity_mps2": 9.80665, "gyro_scale_rad_s": 6.283185307179586},
    ),
    "cuml_logistic_regression": _tabular(
        "cuml_logistic_regression",
        backend="cuml",
        standardize=True,
        C=1.0,
        penalty="l2",
        max_iter=500,
    ),
    "cuml_random_forest": _tabular(
        "cuml_random_forest",
        backend="cuml",
        standardize=False,
        n_estimators=200,
        max_depth=24,
        min_samples_leaf=1,
        max_features="sqrt",
        n_bins=128,
        n_streams=4,
        bootstrap=True,
    ),
    "cuml_rbf_svc": _tabular(
        "cuml_rbf_svc",
        backend="cuml",
        standardize=True,
        C=1.0,
        gamma="scale",
        kernel="rbf",
    ),
    "cuml_knn": _tabular(
        "cuml_knn",
        backend="cuml",
        standardize=True,
        n_neighbors=11,
        weights="distance",
        metric="euclidean",
    ),
    "cuml_gaussian_nb": _tabular(
        "cuml_gaussian_nb", backend="cuml", standardize=False, var_smoothing=1e-9
    ),
    "xgboost_cuda": _tabular(
        "xgboost_cuda",
        backend="xgboost",
        standardize=False,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    ),
    "torch_mlp": _tabular(
        "torch_mlp",
        backend="pytorch",
        standardize=True,
        hidden_layers=(64, 32),
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=256,
        max_epochs=100,
        patience=12,
    ),
    **{
        model_id: ModelSpec(
            model_id=model_id,
            backend="pytorch",
            input_kind="raw",
            standardize=True,
            position=True,
            fall_mil=True,
            fixed_params={
                "channels": 64,
                "hidden_size": 64,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 256,
                "bag_batch_size": 32,
            },
        )
        for model_id in SEQUENCE_MODEL_IDS
    },
}


def build_jobs(
    *, suites: tuple[str, ...], models: tuple[str, ...], folds: tuple[int, ...]
) -> list[JobSpec]:
    unknown_suites = set(suites) - set(SUITES)
    unknown_models = set(models) - set(MODEL_IDS)
    if unknown_suites:
        raise ValueError(f"Unknown suites: {sorted(unknown_suites)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    jobs: list[JobSpec] = []
    for suite in suites:
        for model_id in models:
            spec = MODEL_SPECS[model_id]
            supported = spec.position if suite in POSITION_SUITES else spec.fall_mil
            if not supported:
                continue
            target = "position" if suite in POSITION_SUITES else "fall"
            objective = "supervised" if suite in POSITION_SUITES else "mil"
            for fold in folds:
                jobs.append(
                    JobSpec(
                        suite=suite,
                        model_id=model_id,
                        outer_fold=fold,
                        target=target,
                        objective=objective,
                        input_kind=spec.input_kind,
                    )
                )
    return jobs
