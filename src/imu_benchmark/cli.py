from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from .data import load_config, prepare_window_store
from .dataset import validate_data
from .device import CudaUnavailable
from .doctor import run_doctor
from .kfall_data import load_kfall_config, prepare_kfall_window_store
from .kfall_runner import (
    plan_kfall_experiment,
    regenerate_kfall_report,
    run_kfall_experiment,
)
from .runner import plan_experiment, regenerate_report, run_experiment
from .specs import MODEL_IDS, SUITES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/initial_validation_recheck_v1.json"
DEFAULT_KFALL_CONFIG = PROJECT_ROOT / "configs/kfall_external_v1_provisional.json"
CACHE_ROOT = PROJECT_ROOT / "cache"


def _selection(value: str | None, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    if value is None:
        return allowed
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(selected) - set(allowed)
    if not selected or unknown:
        raise ValueError(f"Invalid {name}; unknown values: {sorted(unknown)}")
    return selected


def _folds(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(range(5)):
        raise ValueError("Folds must be unique values from 0 through 4")
    return selected


def _configured(args: argparse.Namespace) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    config = load_config(args.config.resolve())
    suites = _selection(getattr(args, "suites", None), SUITES, "suites")
    models = _selection(getattr(args, "models", None), MODEL_IDS, "models")
    selected_folds = _folds(getattr(args, "folds", None))
    if selected_folds is not None:
        config = copy.deepcopy(config)
        config["profiles"][args.profile]["outer_folds"] = list(selected_folds)
    return config, suites, models


def _run_fixed(profile: str, *, resume: bool, config_path: Path) -> dict:
    config = load_config(config_path.resolve())
    data_validation = validate_data(PROJECT_ROOT)
    doctor = run_doctor(random_seed=int(config["random_seed"]))
    result = run_experiment(
        project_root=PROJECT_ROOT,
        cache_root=CACHE_ROOT,
        run_root=PROJECT_ROOT / "runs" / str(config["experiment_version"]),
        config=config,
        profile_name=profile,
        suites=SUITES,
        models=MODEL_IDS,
        resume=resume,
        environment=doctor["environment"],
    )
    return {"data_validation": data_validation, "doctor": doctor, "benchmark": result}


def _run_kfall(profile: str, *, resume: bool, config_path: Path) -> dict:
    config = load_kfall_config(config_path.resolve())
    data_validation = validate_data(PROJECT_ROOT)
    doctor = run_doctor(random_seed=int(config["random_seed"]))
    result = run_kfall_experiment(
        project_root=PROJECT_ROOT,
        cache_root=CACHE_ROOT,
        run_root=PROJECT_ROOT / "runs" / str(config["experiment_version"]),
        config=config,
        profile_name=profile,
        resume=resume,
        environment=doctor["environment"],
    )
    return {"data_validation": data_validation, "doctor": doctor, "benchmark": result}


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("smoke", "reproduce"), default="smoke")
    parser.add_argument("--suites", help="Comma-separated suite IDs")
    parser.add_argument("--models", help="Comma-separated model IDs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the chest/waist IMU fall-detection benchmark on CUDA."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--kfall-config", type=Path, default=DEFAULT_KFALL_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Verify the CUDA environment and model backends")
    commands.add_parser("validate-data", help="Verify LFS data, HDF5 schemas, and folds")
    commands.add_parser("prepare", help="Build or reuse the derived window cache")
    commands.add_parser("smoke", help="Run the fixed 22-job smoke profile")

    reproduce = commands.add_parser("reproduce", help="Run the fixed 110-job recheck")
    reproduce.add_argument("--resume", action="store_true")

    plan = commands.add_parser("plan", help="Print a workload plan without training")
    _add_selection_arguments(plan)

    run = commands.add_parser("run", help="Run a selected advanced workload")
    _add_selection_arguments(run)
    run.add_argument("--folds", help="Comma-separated outer folds from 0 through 4")
    run.add_argument("--resume", action="store_true")

    report = commands.add_parser("report", help="Regenerate reports from complete checkpoints")
    _add_selection_arguments(report)

    kfall_plan = commands.add_parser(
        "kfall-plan", help="Print the provisional KFall workload without training"
    )
    kfall_plan.add_argument("--profile", choices=("smoke", "evaluate"), default="smoke")
    commands.add_parser("kfall-prepare", help="Build or reuse the strict KFall window cache")
    commands.add_parser("kfall-smoke", help="Run the fixed 8-job KFall CUDA smoke profile")
    kfall_evaluate = commands.add_parser(
        "kfall-evaluate", help="Run the fixed 40-job KFall external evaluation"
    )
    kfall_evaluate.add_argument("--resume", action="store_true")
    kfall_report = commands.add_parser(
        "kfall-report", help="Regenerate KFall reports from complete checkpoints"
    )
    kfall_report.add_argument("--profile", choices=("smoke", "evaluate"), default="evaluate")

    args = parser.parse_args()
    try:
        if args.command == "doctor":
            config = load_config(args.config.resolve())
            result = run_doctor(random_seed=int(config["random_seed"]))
        elif args.command == "validate-data":
            result = validate_data(PROJECT_ROOT)
        elif args.command == "prepare":
            config = load_config(args.config.resolve())
            validation = validate_data(PROJECT_ROOT)
            path, manifest = prepare_window_store(
                project_root=PROJECT_ROOT, cache_root=CACHE_ROOT, config=config
            )
            result = {
                "status": "PASS",
                "data_validation": validation,
                "path": str(path),
                **manifest,
            }
        elif args.command == "smoke":
            result = _run_fixed("smoke", resume=True, config_path=args.config)
        elif args.command == "reproduce":
            result = _run_fixed("reproduce", resume=args.resume, config_path=args.config)
        elif args.command == "plan":
            config, suites, models = _configured(args)
            result = plan_experiment(
                config=config,
                profile_name=args.profile,
                suites=suites,
                models=models,
            )
        elif args.command == "run":
            config, suites, models = _configured(args)
            validation = validate_data(PROJECT_ROOT)
            doctor = run_doctor(random_seed=int(config["random_seed"]))
            benchmark = run_experiment(
                project_root=PROJECT_ROOT,
                cache_root=CACHE_ROOT,
                run_root=PROJECT_ROOT / "runs" / str(config["experiment_version"]),
                config=config,
                profile_name=args.profile,
                suites=suites,
                models=models,
                resume=args.resume,
                environment=doctor["environment"],
            )
            result = {"data_validation": validation, "doctor": doctor, "benchmark": benchmark}
        elif args.command == "report":
            config, suites, models = _configured(args)
            result = regenerate_report(
                project_root=PROJECT_ROOT,
                cache_root=CACHE_ROOT,
                run_root=PROJECT_ROOT / "runs" / str(config["experiment_version"]),
                config=config,
                profile_name=args.profile,
                suites=suites,
                models=models,
            )
        elif args.command == "kfall-plan":
            config = load_kfall_config(args.kfall_config.resolve())
            result = plan_kfall_experiment(config, args.profile)
        elif args.command == "kfall-prepare":
            config = load_kfall_config(args.kfall_config.resolve())
            validation = validate_data(PROJECT_ROOT)
            path, manifest = prepare_kfall_window_store(
                project_root=PROJECT_ROOT,
                cache_root=CACHE_ROOT,
                config=config,
            )
            result = {
                "status": "PASS",
                "data_validation": validation,
                "path": str(path),
                **manifest,
            }
        elif args.command == "kfall-smoke":
            result = _run_kfall("smoke", resume=True, config_path=args.kfall_config)
        elif args.command == "kfall-evaluate":
            result = _run_kfall(
                "evaluate", resume=args.resume, config_path=args.kfall_config
            )
        elif args.command == "kfall-report":
            config = load_kfall_config(args.kfall_config.resolve())
            result = regenerate_kfall_report(
                project_root=PROJECT_ROOT,
                cache_root=CACHE_ROOT,
                run_root=PROJECT_ROOT / "runs" / str(config["experiment_version"]),
                config=config,
                profile_name=args.profile,
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(json.dumps(result, indent=2))
    except (CudaUnavailable, OSError, ValueError) as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": type(error).__name__, "detail": str(error)},
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
