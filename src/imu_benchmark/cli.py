from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .configuration import load_experiment
from .dataset import validate_data
from .device import CudaUnavailable
from .doctor import run_doctor
from .engine import plan_experiment, regenerate_report, run_experiment
from .runtime import (
    FORMAL_COMMANDS,
    WorkPaths,
    require_compute_runtime,
    resolve_work_paths,
    source_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOKE_CONFIG = PROJECT_ROOT / "configs/experiments/kfall_smoke_v1.yaml"


def _config(path: Path) -> dict:
    return load_experiment(PROJECT_ROOT, path.resolve())


def _doctor_result(config: dict, paths: WorkPaths) -> dict:
    source, warnings = source_provenance(PROJECT_ROOT)
    result = run_doctor(
        random_seed=int(config["seeds"][0]),
        project_root=PROJECT_ROOT,
        work_root=paths.root,
    )
    return {**result, "paths": paths.to_dict(), "source": source, "warnings": warnings}


def _execute(config_path: Path, *, resume: bool, paths: WorkPaths) -> dict:
    config = _config(config_path)
    validation = validate_data(PROJECT_ROOT)
    doctor = _doctor_result(config, paths)
    benchmark = run_experiment(
        project_root=PROJECT_ROOT,
        cache_root=paths.cache,
        runs_root=paths.runs,
        config=config,
        resume=resume,
        environment=doctor["environment"],
        source=doctor["source"],
        warnings=doctor["warnings"],
    )
    return {"data_validation": validation, "doctor": doctor, "benchmark": benchmark}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the WSL2/CUDA fall-detection benchmark from versioned YAML configs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Verify WSL2, CUDA, and public model backends")
    doctor.add_argument("config", nargs="?", type=Path, default=DEFAULT_SMOKE_CONFIG)
    commands.add_parser("validate-data", help="Verify LFS data, HDF5 v3, snapshot, and folds")
    commands.add_parser("smoke", help="Run the default seven-model KFall FP32 smoke config")
    plan = commands.add_parser("plan", help="Resolve a YAML experiment without training")
    plan.add_argument("config", type=Path)
    run = commands.add_parser("run", help="Run a versioned YAML experiment")
    run.add_argument("config", type=Path)
    run.add_argument("--resume", action="store_true")
    report = commands.add_parser("report", help="Regenerate a report from run checkpoints")
    report.add_argument("run_id")

    args = parser.parse_args()
    try:
        require_compute_runtime(args.command, PROJECT_ROOT)
        paths = resolve_work_paths()
        source, warnings = source_provenance(PROJECT_ROOT)
        if args.command in FORMAL_COMMANDS and warnings:
            print(
                f"WARNING: experiment source is not clean: {', '.join(warnings)}",
                file=sys.stderr,
            )
        if args.command == "doctor":
            result = _doctor_result(_config(args.config), paths)
        elif args.command == "validate-data":
            result = validate_data(PROJECT_ROOT)
        elif args.command == "smoke":
            result = _execute(DEFAULT_SMOKE_CONFIG, resume=True, paths=paths)
        elif args.command == "plan":
            result = plan_experiment(_config(args.config))
        elif args.command == "run":
            result = _execute(args.config, resume=args.resume, paths=paths)
        elif args.command == "report":
            result = regenerate_report(paths.runs, args.run_id)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    except (CudaUnavailable, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
