from __future__ import annotations

import json
import sys
from typing import IO, Any

from rich import box
from rich.console import Console
from rich.table import Table


def _console(stream: IO[str]) -> Console:
    return Console(
        file=stream,
        force_terminal=bool(getattr(stream, "isatty", lambda: False)()),
        color_system="auto" if getattr(stream, "isatty", lambda: False)() else None,
        width=120,
    )


def _status(console: Console, name: str, result: dict[str, Any]) -> None:
    status = str(result.get("status", "UNKNOWN"))
    style = "green" if status == "PASS" else "yellow"
    console.print(f"[{style}]{status}[/{style}] {name}")


def _render_doctor(console: Console, result: dict[str, Any]) -> None:
    _status(console, "runtime and CUDA backends", result)
    environment = result["environment"]
    console.print(
        f"GPU: {environment['gpu_name']} | CUDA: {environment['torch_cuda']} | "
        f"PyTorch: {environment['torch']} | cuML: {environment['cuml']}"
    )
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Seconds", justify="right")
    table.add_column("Strict CUDA")
    for model in result["models"]:
        table.add_row(
            str(model["model_id"]),
            str(model["status"]),
            f"{float(model['seconds']):.3f}",
            "yes" if model["strict_cuda"] else "no",
        )
    console.print(table)
    source = result["source"]
    console.print(f"Source: {source.get('commit') or 'unknown'} | dirty={source.get('dirty')}")


def _render_data(console: Console, action: str, result: dict[str, Any]) -> None:
    _status(console, f"data {action}", result)
    if action == "pull":
        console.print(
            f"Base: {result['base_snapshot_id']} | Team: {result['team_snapshot_id'] or 'none'}"
        )
        console.print(f"Active manifest: {result['active_manifest']}")
    elif action == "status":
        local = result.get("local_active") or {}
        console.print(
            f"Local base: {local.get('base_snapshot_id', 'none')} | "
            f"Remote base: {result['remote_base_snapshot_id']}"
        )
        console.print(
            f"Local team: {local.get('team_snapshot_id') or 'none'} | "
            f"Remote team: {result['remote_team_snapshot_id'] or 'none'} | "
            f"Update available: {result['update_available']}"
        )
    else:
        console.print(
            f"Snapshot: {result['snapshot_id']} | Files: {result['files']} | "
            f"Manifest SHA-256: {result['manifest_sha256']}"
        )


def _render_validation(console: Console, result: dict[str, Any]) -> None:
    _status(console, "HDF5 v3.1 data validation", result)
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Collection")
    for name in ("Files", "Sequences", "Rows", "Events", "Participants"):
        table.add_column(name, justify="right")
    base = result["base"]
    table.add_row(
        "base",
        str(base["files"]),
        str(base["sequences"]),
        str(base["rows"]),
        str(base["events"]),
        str(base["split"]["participants"]),
    )
    team = result.get("team")
    if team is not None:
        table.add_row(
            "team",
            str(team["files"]),
            str(team["sequences"]),
            str(team["rows"]),
            str(team["events"]),
            str(team["participants"]),
        )
    console.print(table)
    console.print(
        f"Base snapshot: {result['base_snapshot_id']} | Team snapshot: "
        f"{result.get('team_snapshot_id') or 'none'}"
    )


def _render_benchmark(console: Console, result: dict[str, Any]) -> None:
    benchmark = result.get("benchmark", result)
    _status(console, "benchmark run", benchmark)
    console.print(
        f"Run ID: {benchmark['run_id']} | Jobs: {benchmark['completed_jobs']}/"
        f"{benchmark['scheduled_jobs']} | Computed: "
        f"{benchmark['computed_jobs_this_invocation']} | Cached: "
        f"{benchmark['cached_jobs_this_invocation']} | "
        f"Elapsed: {float(benchmark['elapsed_seconds']):.2f}s"
    )
    console.print(f"Report: {benchmark['report_path']}")
    if benchmark.get("failures"):
        for failure in benchmark["failures"]:
            console.print(f"[red]Failed:[/red] {failure.get('reason', 'unknown failure')}")


def _render_plan(console: Console, result: dict[str, Any]) -> None:
    console.print("[green]PASS[/green] experiment plan")
    console.print(
        f"Experiment: {result['experiment_id']} | Jobs: {result['scheduled_jobs']} | "
        f"Models: {len(result['models'])} | Folds: {len(result['folds'])} | "
        f"Seeds: {len(result['seeds'])}"
    )


def render_result(
    command: str,
    result: dict[str, Any],
    *,
    data_action: str | None = None,
    stream: IO[str] = sys.stdout,
) -> None:
    console = _console(stream)
    if command == "doctor":
        _render_doctor(console, result)
    elif command == "data":
        assert data_action is not None
        _render_data(console, data_action, result)
    elif command == "validate-data":
        _render_validation(console, result)
    elif command in {"smoke", "run"}:
        _render_benchmark(console, result)
    elif command == "plan":
        _render_plan(console, result)
    elif command == "report":
        _status(console, "report regeneration", result)
        console.print(f"Run ID: {result['run_id']} | Report: {result['report_path']}")
    elif command == "test":
        _status(console, "Ruff and pytest", result)
        console.print(f"Ruff: {result['ruff']} | pytest: {result['pytest']}")
    else:
        console.print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))


def render_error(reason: str, *, as_json: bool) -> None:
    payload = {"status": "FAIL", "reason": reason}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stdout)
    else:
        _console(sys.stderr).print(f"[red]ERROR:[/red] {reason}")
