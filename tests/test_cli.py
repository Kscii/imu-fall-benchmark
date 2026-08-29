from __future__ import annotations

import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path

import pytest

from imu_benchmark import cli
from imu_benchmark.progress import (
    PlainProgressReporter,
    RichProgressReporter,
    close_progress_reporter,
    create_progress_reporter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHELL_LIBRARY = PROJECT_ROOT / "scripts/benchmark-shell-lib.sh"


def _shell_digest(repository: Path, function: str = "dependency_digest") -> str:
    completed = subprocess.run(
        (
            "bash",
            "-c",
            'source "$1"; REPO_ROOT="$2"; "$3"',
            "digest-test",
            str(SHELL_LIBRARY),
            str(repository),
            function,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_dependency_digest_is_independent_of_checkout_path(tmp_path: Path) -> None:
    repositories = (tmp_path / "first", tmp_path / "second")
    for repository in repositories:
        repository.mkdir()
        for name in (
            "environment.yml",
            "requirements-torch-cu129.txt",
            "requirements-runtime.txt",
        ):
            shutil.copy2(PROJECT_ROOT / name, repository / name)
    assert _shell_digest(repositories[0]) == _shell_digest(repositories[1])


def test_legacy_digest_finds_the_existing_wsl_environment() -> None:
    old_checkout = Path("/home/kscii/projects/imu-fall-benchmark")
    assert _shell_digest(old_checkout, "legacy_dependency_digest") == "c5ec4b63123f81a7"


def test_compatible_legacy_environment_can_be_adopted_from_another_checkout(
    tmp_path: Path,
) -> None:
    envs = tmp_path / "envs"
    for name in ("bbbb", "aaaa"):
        environment = envs / name
        (environment / "bin").mkdir(parents=True)
        python = environment / "bin/python"
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        (environment / ".imu-benchmark-complete").write_text(name, encoding="utf-8")
    completed = subprocess.run(
        (
            "bash",
            "-c",
            'source "$1"; WORK_ROOT="$2"; REPO_ROOT="$3"; '
            "find_compatible_legacy_environment",
            "legacy-scan-test",
            str(SHELL_LIBRARY),
            str(tmp_path),
            str(PROJECT_ROOT),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == str(envs / "aaaa")


def test_plain_progress_is_line_oriented() -> None:
    stream = StringIO()
    reporter = PlainProgressReporter(stream)
    with reporter.task("Downloading", total=2, unit="files") as task:
        task.update(advance=1, detail="one.h5")
        task.update(advance=1, detail="two.h5")
    output = stream.getvalue()
    assert "[start] Downloading" in output
    assert "2/2 files (100%)" in output
    assert "[done] Downloading" in output


def test_rich_progress_supports_repeated_nested_tasks() -> None:
    stream = StringIO()
    reporter = RichProgressReporter(stream)
    with reporter.task("Parent", total=2) as parent:
        for index in range(2):
            with reporter.task(f"Child {index}", total=1) as child:
                child.update(advance=1)
            parent.update(advance=1)
    close_progress_reporter(reporter)
    output = stream.getvalue()
    assert "done Child 0" in output
    assert "done Child 1" in output
    assert "done Parent" in output


def test_auto_progress_falls_back_to_plain_without_a_tty() -> None:
    assert isinstance(create_progress_reporter("auto", stream=StringIO()), PlainProgressReporter)


def test_json_error_uses_stdout_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        raise RuntimeError("expected failure")

    monkeypatch.setattr(cli, "_dispatch", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(("plan", "example.yaml", "--json", "--progress", "off"))
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert json.loads(captured.out) == {"status": "FAIL", "reason": "expected failure"}
    assert captured.err == ""


def test_global_output_flags_are_accepted_after_the_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "experiment_id": "unit-test",
        "scheduled_jobs": 0,
        "models": [],
        "folds": [],
        "seeds": [],
    }
    monkeypatch.setattr(cli, "_dispatch", lambda *args, **kwargs: result)
    cli.main(("plan", "example.yaml", "--json", "--progress=off"))
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert captured.err == ""
