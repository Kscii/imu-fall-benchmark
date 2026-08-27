from __future__ import annotations

from pathlib import Path

import numpy as np

from imu_benchmark.data import (
    extract_window_features,
    load_config,
    load_window_store,
    prepare_window_store,
)
from imu_benchmark.dataset import validate_data
from imu_benchmark.evaluation import best_threshold
from imu_benchmark.kfall_data import (
    load_kfall_config,
    load_kfall_window_store,
    prepare_kfall_window_store,
    strict_decision_time_labels,
)
from imu_benchmark.kfall_runner import _training_splits, plan_kfall_experiment
from imu_benchmark.runner import plan_experiment
from imu_benchmark.specs import MODEL_IDS, SUITES, build_jobs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_distributed_data_and_split_validate() -> None:
    result = validate_data(PROJECT_ROOT)
    assert result["status"] == "PASS"
    assert result["training"]["files"] == 5
    assert result["training"]["sequences"] == 12_318
    assert result["training"]["rows"] == 6_840_218
    assert result["training"]["split"]["participants"] == 106
    assert result["external"]["files"] == 1
    assert result["external"]["events"] == 2_346
    assert result["external"]["split"]["participants"] == 32


def test_reproduction_plan_contains_110_jobs() -> None:
    config = load_config(PROJECT_ROOT / "configs/initial_validation_recheck_v1.json")
    result = plan_experiment(
        config=config,
        profile_name="reproduce",
        suites=SUITES,
        models=MODEL_IDS,
    )
    assert result["scheduled_jobs"] == 110
    assert result["jobs_by_suite"] == {
        "position_paired": 50,
        "fall_universal": 20,
        "fall_chest_only": 20,
        "fall_waist_only": 20,
    }


def test_engineered_window_features_are_finite() -> None:
    values = np.random.default_rng(3888).normal(size=(4, 60, 6)).astype(np.float32)
    features = extract_window_features(values)
    assert features.shape == (4, 158)
    assert np.isfinite(features).all()


def test_threshold_tie_break_is_deterministic() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    first = best_threshold(labels, scores)
    second = best_threshold(labels, scores)
    assert first == second
    assert first[0] == 0.6
    assert first[1] == 1.0


def test_strict_kfall_decision_time_boundaries() -> None:
    starts = np.asarray([0, 15, 30, 45, 60], dtype=np.int32)
    ends = starts + 60
    labels, keep = strict_decision_time_labels(
        starts,
        ends,
        onset_sample=70,
        impact_sample=100,
    )
    assert labels.tolist() == [0, 1, 1, 0, 0]
    assert keep.tolist() == [True, True, True, False, False]


def test_kfall_workload_and_cache_contract() -> None:
    config = load_kfall_config(PROJECT_ROOT / "configs/kfall_external_v1_provisional.json")
    smoke = plan_kfall_experiment(config, "smoke")
    evaluate = plan_kfall_experiment(config, "evaluate")
    assert smoke["scheduled_jobs"] == 8
    assert evaluate["scheduled_jobs"] == 40
    path, manifest = prepare_kfall_window_store(
        project_root=PROJECT_ROOT,
        cache_root=PROJECT_ROOT / "cache",
        config=config,
    )
    store = load_kfall_window_store(path)
    assert manifest["windows"] == store.size == 53_260
    assert manifest["positive_windows"] == 3_489
    assert manifest["fall_events"] == 2_346
    assert manifest["events_without_positive_window"] == 33
    assert manifest["skipped_post_impact_overlap_windows"] == 9_225


def test_kfall_training_fold_is_participant_disjoint() -> None:
    config = load_kfall_config(PROJECT_ROOT / "configs/kfall_external_v1_provisional.json")
    path, _manifest = prepare_window_store(
        project_root=PROJECT_ROOT,
        cache_root=PROJECT_ROOT / "cache",
        config=config,
    )
    store = load_window_store(path)
    job = build_jobs(
        suites=("fall_universal",), models=("threshold_impact",), folds=(0,)
    )[0]
    splits = _training_splits(
        store,
        job,
        profile=config["profiles"]["smoke"],
        seed=int(config["random_seed"]),
    )
    train = set(store.window_participants()[splits["train"]])
    validation = set(store.window_participants()[splits["validation"]])
    assert train.isdisjoint(validation)
