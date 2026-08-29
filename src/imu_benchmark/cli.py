from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .cloud_data import BASE_MANIFEST_PATH, activate_base, data_status, publish_base, pull_data
from .cloud_models import (
    publish_model_package,
    restore_published_model_package,
    verify_model_package,
)
from .cloud_results import (
    publish_result,
    pull_result,
    restore_published_result,
    verify_result,
)
from .configuration import load_experiment
from .dataset import validate_data
from .device import CudaUnavailable
from .doctor import run_doctor
from .engine import plan_experiment, regenerate_report, run_experiment
from .presentation import render_error, render_result
from .progress import (
    ProgressReporter,
    close_progress_reporter,
    create_progress_reporter,
)
from .runtime import (
    FORMAL_COMMANDS,
    WorkPaths,
    require_compute_runtime,
    resolve_work_paths,
    source_provenance,
)
from .splits import propose_from_active

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOKE_CONFIG = PROJECT_ROOT / "configs/experiments/temporal_smoke_v1.yaml"


def _config(path: Path, paths: WorkPaths) -> dict[str, Any]:
    return load_experiment(
        PROJECT_ROOT,
        path.resolve(),
        snapshot_path=paths.data / "active.json",
    )


def _doctor_result(
    config: dict[str, Any],
    paths: WorkPaths,
    reporter: ProgressReporter,
) -> dict[str, Any]:
    source, warnings = source_provenance(PROJECT_ROOT)
    result = run_doctor(
        random_seed=int(config["seeds"][0]),
        project_root=PROJECT_ROOT,
        work_root=paths.root,
        progress=reporter,
    )
    return {**result, "paths": paths.to_dict(), "source": source, "warnings": warnings}


def _execute(
    config_path: Path,
    *,
    resume: bool,
    paths: WorkPaths,
    reporter: ProgressReporter,
) -> dict[str, Any]:
    config = _config(config_path, paths)
    validation = validate_data(
        PROJECT_ROOT,
        snapshot_path=paths.data / "active.json",
        progress=reporter,
    )
    doctor = _doctor_result(config, paths, reporter)
    benchmark = run_experiment(
        project_root=PROJECT_ROOT,
        cache_root=paths.cache,
        runs_root=paths.runs,
        config=config,
        resume=resume,
        environment=doctor["environment"],
        source=doctor["source"],
        warnings=doctor["warnings"],
        progress=reporter,
    )
    return {"data_validation": validation, "doctor": doctor, "benchmark": benchmark}


def _version_text() -> str:
    payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(payload["project"]["version"])
    source, _ = source_provenance(PROJECT_ROOT)
    commit = source.get("commit")
    revision = str(commit)[:12] if commit else "unknown"
    if source.get("dirty"):
        revision += "+dirty"
    return f"imu-bench {version} | repository={PROJECT_ROOT} | commit={revision}"


def _run_check(
    description: str,
    command: Sequence[str],
    reporter: ProgressReporter,
) -> None:
    with reporter.task(description):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode == 0:
        return
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if len(output) > 4000:
        output = output[-4000:]
    raise RuntimeError(f"{description} failed (exit {completed.returncode})\n{output}")


def _run_tests(reporter: ProgressReporter) -> dict[str, Any]:
    _run_check(
        "Running Ruff",
        (sys.executable, "-m", "ruff", "check", "src", "tests"),
        reporter,
    )
    _run_check(
        "Running pytest",
        (sys.executable, "-m", "pytest", "-q"),
        reporter,
    )
    return {"status": "PASS", "ruff": "PASS", "pytest": "PASS"}


def _normalise_global_arguments(argv: Sequence[str]) -> list[str]:
    """Allow global output flags before or after a subcommand."""
    global_arguments: list[str] = []
    command_arguments: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in {"--json", "--version"}:
            global_arguments.append(argument)
        elif argument == "--progress":
            global_arguments.append(argument)
            index += 1
            if index >= len(argv):
                global_arguments.append("")
            else:
                global_arguments.append(argv[index])
        elif argument.startswith("--progress="):
            global_arguments.append(argument)
        else:
            command_arguments.append(argument)
        index += 1
    return [*global_arguments, *command_arguments]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WSL2/CUDA fall-detection benchmark from versioned YAML configs."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write the complete machine-readable result to stdout",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="auto",
        help="progress rendering mode (default: auto)",
    )
    parser.add_argument("--version", action="version", version=_version_text())
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Pull or inspect immutable GCS datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    pull = data_commands.add_parser(
        "pull", help="Install current or explicitly named immutable snapshots"
    )
    pull.add_argument("--base-snapshot")
    pull.add_argument("--team-snapshot")
    pull.add_argument("--no-team", action="store_true")
    data_commands.add_parser("status", help="Compare local active data with remote current")
    publish = data_commands.add_parser(
        "publish-base", help="Stage a reviewed immutable base snapshot without activating it"
    )
    publish.add_argument("--source-dir", type=Path, required=True)
    publish.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    publish.add_argument("--manifest", type=Path, default=PROJECT_ROOT / BASE_MANIFEST_PATH)
    activate = data_commands.add_parser(
        "activate-base", help="Switch current.json to an already staged base snapshot"
    )
    activate.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / BASE_MANIFEST_PATH
    )
    activate.add_argument("--expected-current", required=True)
    results = commands.add_parser(
        "results", help="Publish, pull, or verify immutable benchmark results"
    )
    result_commands = results.add_subparsers(dest="results_command", required=True)
    for action, help_text in (
        ("publish", "Publish one completed temporal-core run"),
        ("pull", "Download and validate one immutable result bundle"),
        ("verify", "Verify one remote result bundle and optional local run"),
    ):
        result_command = result_commands.add_parser(action, help=help_text)
        result_command.add_argument("run_id")
    result_restore = result_commands.add_parser(
        "restore", help="Admin-only restore of a deprecated result publication"
    )
    result_restore.add_argument("run_id")
    result_restore.add_argument("--expected-generation", type=int, required=True)
    models = commands.add_parser(
        "models", help="Publish or verify immutable final ONNX model packages"
    )
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_publish = model_commands.add_parser(
        "publish", help="Validate and publish one final model package directory"
    )
    model_publish.add_argument("package_dir", type=Path)
    model_verify = model_commands.add_parser(
        "verify", help="Verify one published final model package"
    )
    model_verify.add_argument("package_id")
    model_restore = model_commands.add_parser(
        "restore", help="Admin-only restore of a deprecated final model package"
    )
    model_restore.add_argument("package_id")
    model_restore.add_argument("--expected-generation", type=int, required=True)
    split = commands.add_parser("split", help="Propose sticky participant folds")
    split_commands = split.add_subparsers(dest="split_command", required=True)
    propose = split_commands.add_parser(
        "propose", help="Keep existing folds and assign only new public participants"
    )
    propose.add_argument("--version", required=True)
    propose.add_argument("--output-dir", type=Path)
    doctor = commands.add_parser("doctor", help="Verify WSL2, CUDA, and public model backends")
    doctor.add_argument("config", nargs="?", type=Path, default=DEFAULT_SMOKE_CONFIG)
    commands.add_parser("validate-data", help="Verify HDF5 v3.1 data, hashes, and folds")
    commands.add_parser("test", help="Run the repository Ruff and pytest checks")
    commands.add_parser(
        "smoke", help="Run the default seven-model temporal FP32 smoke config"
    )
    plan = commands.add_parser("plan", help="Resolve a YAML experiment without training")
    plan.add_argument("config", type=Path)
    run = commands.add_parser("run", help="Run a versioned YAML experiment")
    run.add_argument("config", type=Path)
    run.add_argument("--resume", action="store_true")
    report = commands.add_parser("report", help="Regenerate a report from run checkpoints")
    report.add_argument("run_id")
    return parser


def _dispatch(args: argparse.Namespace, reporter: ProgressReporter) -> dict[str, Any]:
    require_compute_runtime(args.command, PROJECT_ROOT)
    paths = resolve_work_paths()
    _, warnings = source_provenance(PROJECT_ROOT)
    if args.command in FORMAL_COMMANDS and warnings:
        print(
            f"WARNING: experiment source is not clean: {', '.join(warnings)}",
            file=sys.stderr,
        )
    if args.command == "doctor":
        return _doctor_result(_config(args.config, paths), paths, reporter)
    if args.command == "data":
        if args.data_command == "pull":
            if args.no_team and args.team_snapshot is not None:
                raise ValueError("--no-team and --team-snapshot cannot be used together")
            return pull_data(
                PROJECT_ROOT,
                paths.data,
                base_snapshot_id=args.base_snapshot,
                team_snapshot_id=args.team_snapshot,
                include_team=not args.no_team,
                progress=reporter,
            )
        if args.data_command == "status":
            return data_status(PROJECT_ROOT, paths.data, progress=reporter)
        if args.data_command == "publish-base":
            return publish_base(
                PROJECT_ROOT,
                args.source_dir.resolve(),
                manifest_path=args.manifest.resolve(),
                split_dir=args.split_dir.resolve(),
                progress=reporter,
            )
        if args.data_command == "activate-base":
            return activate_base(
                args.manifest.resolve(),
                expected_current_snapshot_id=args.expected_current,
                progress=reporter,
            )
        raise AssertionError(f"Unhandled data command: {args.data_command}")
    if args.command == "validate-data":
        return validate_data(
            PROJECT_ROOT,
            snapshot_path=paths.data / "active.json",
            progress=reporter,
        )
    if args.command == "results":
        if args.results_command == "publish":
            return publish_result(paths.runs, args.run_id, progress=reporter)
        if args.results_command == "pull":
            return pull_result(paths.runs, args.run_id, progress=reporter)
        if args.results_command == "verify":
            return verify_result(paths.runs, args.run_id, progress=reporter)
        if args.results_command == "restore":
            return restore_published_result(
                args.run_id, expected_generation=args.expected_generation
            )
        raise AssertionError(f"Unhandled results command: {args.results_command}")
    if args.command == "models":
        if args.models_command == "publish":
            return publish_model_package(args.package_dir, progress=reporter)
        if args.models_command == "verify":
            return verify_model_package(args.package_id)
        if args.models_command == "restore":
            return restore_published_model_package(
                args.package_id, expected_generation=args.expected_generation
            )
        raise AssertionError(f"Unhandled models command: {args.models_command}")
    if args.command == "split":
        if args.split_command == "propose":
            output_dir = (
                paths.root / "split-proposals"
                if args.output_dir is None
                else args.output_dir.resolve()
            )
            return propose_from_active(
                PROJECT_ROOT,
                paths.data / "active.json",
                output_dir,
                version=args.version,
            )
        raise AssertionError(f"Unhandled split command: {args.split_command}")
    if args.command == "test":
        return _run_tests(reporter)
    if args.command == "smoke":
        return _execute(DEFAULT_SMOKE_CONFIG, resume=True, paths=paths, reporter=reporter)
    if args.command == "plan":
        return plan_experiment(_config(args.config, paths))
    if args.command == "run":
        return _execute(args.config, resume=args.resume, paths=paths, reporter=reporter)
    if args.command == "report":
        return regenerate_report(paths.runs, args.run_id)
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    raw_arguments = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(_normalise_global_arguments(raw_arguments))
    reporter = create_progress_reporter(args.progress)
    try:
        result = _dispatch(args, reporter)
    except (CudaUnavailable, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        close_progress_reporter(reporter)
        render_error(str(error), as_json=args.json)
        raise SystemExit(1) from error
    close_progress_reporter(reporter)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    else:
        render_result(
            args.command,
            result,
            data_action=getattr(args, "data_command", None),
            results_action=getattr(args, "results_command", None),
        )


if __name__ == "__main__":
    main()
