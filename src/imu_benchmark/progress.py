from __future__ import annotations

import os
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import IO, Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressTask(Protocol):
    def update(
        self,
        *,
        advance: float = 0,
        completed: float | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> None: ...


class ProgressReporter(Protocol):
    def task(
        self,
        description: str,
        *,
        total: float | None = None,
        unit: str = "items",
    ) -> AbstractContextManager[ProgressTask]: ...


@dataclass(slots=True)
class _NullTask:
    def update(
        self,
        *,
        advance: float = 0,
        completed: float | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> None:
        del advance, completed, description, detail


class _NullContext(AbstractContextManager[ProgressTask]):
    def __enter__(self) -> ProgressTask:
        return _NullTask()

    def __exit__(self, *args: object) -> None:
        del args


class NullProgressReporter:
    def task(
        self,
        description: str,
        *,
        total: float | None = None,
        unit: str = "items",
    ) -> AbstractContextManager[ProgressTask]:
        del description, total, unit
        return _NullContext()


@dataclass(slots=True)
class _PlainTask:
    stream: IO[str]
    description: str
    total: float | None
    unit: str
    started: float
    completed: float = 0
    detail: str | None = None
    last_reported_fraction: int = -1
    last_reported_time: float = 0

    def update(
        self,
        *,
        advance: float = 0,
        completed: float | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> None:
        if completed is not None:
            self.completed = completed
        else:
            self.completed += advance
        if description is not None:
            self.description = description
        if detail is not None:
            self.detail = detail
        now = time.monotonic()
        fraction = (
            int(min(100, self.completed * 100 / self.total))
            if self.total and self.total > 0
            else -1
        )
        milestone = fraction == 100 or fraction >= self.last_reported_fraction + 10
        timed = now - self.last_reported_time >= 30
        if not milestone and not timed:
            return
        if self.total is None:
            progress = f"{self.completed:g} {self.unit}" if self.completed else "working"
        else:
            progress = f"{self.completed:g}/{self.total:g} {self.unit} ({fraction}%)"
        suffix = f" - {self.detail}" if self.detail else ""
        print(f"[progress] {self.description}: {progress}{suffix}", file=self.stream, flush=True)
        self.last_reported_fraction = fraction
        self.last_reported_time = now


class _PlainContext(AbstractContextManager[ProgressTask]):
    def __init__(self, task: _PlainTask) -> None:
        self.task = task

    def __enter__(self) -> ProgressTask:
        print(f"[start] {self.task.description}", file=self.task.stream, flush=True)
        self.task.last_reported_time = self.task.started
        return self.task

    def __exit__(self, *args: object) -> None:
        failed = args[0] is not None
        elapsed = time.monotonic() - self.task.started
        state = "failed" if failed else "done"
        print(
            f"[{state}] {self.task.description} ({elapsed:.1f}s)",
            file=self.task.stream,
            flush=True,
        )


class PlainProgressReporter:
    def __init__(self, stream: IO[str] = sys.stderr) -> None:
        self.stream = stream

    def task(
        self,
        description: str,
        *,
        total: float | None = None,
        unit: str = "items",
    ) -> AbstractContextManager[ProgressTask]:
        return _PlainContext(
            _PlainTask(
                stream=self.stream,
                description=description,
                total=total,
                unit=unit,
                started=time.monotonic(),
            )
        )


class _RichTask:
    def __init__(self, reporter: RichProgressReporter, task_id: int) -> None:
        self.reporter = reporter
        self.task_id = task_id

    def update(
        self,
        *,
        advance: float = 0,
        completed: float | None = None,
        description: str | None = None,
        detail: str | None = None,
    ) -> None:
        fields: dict[str, object] = {}
        if completed is not None:
            fields["completed"] = completed
        if description is not None:
            fields["description"] = description
        if detail is not None:
            fields["detail"] = detail
        self.reporter.progress.update(self.task_id, advance=advance, **fields)


class _RichContext(AbstractContextManager[ProgressTask]):
    def __init__(
        self,
        reporter: RichProgressReporter,
        description: str,
        total: float | None,
        unit: str,
    ) -> None:
        self.reporter = reporter
        self.description = description
        self.total = total
        self.unit = unit
        self.task_id: int | None = None
        self.parent_id: int | None = None

    def __enter__(self) -> ProgressTask:
        if self.reporter.stack:
            self.parent_id = self.reporter.stack[-1]
            self.reporter.progress.update(self.parent_id, visible=False)
        self.task_id = self.reporter.progress.add_task(
            self.description,
            total=self.total,
            detail="",
            unit=self.unit,
        )
        self.reporter.stack.append(self.task_id)
        return _RichTask(self.reporter, self.task_id)

    def __exit__(self, *args: object) -> None:
        assert self.task_id is not None
        failed = args[0] is not None
        task = next(
            item for item in self.reporter.progress.tasks if item.id == self.task_id
        )
        if not failed and task.total is not None:
            self.reporter.progress.update(self.task_id, completed=task.total, refresh=True)
        description = task.description
        elapsed = task.elapsed or 0.0
        self.reporter.progress.remove_task(self.task_id)
        self.reporter.stack.pop()
        style = "red" if failed else "green"
        state = "failed" if failed else "done"
        self.reporter.console.print(
            f"[{style}]{state}[/{style}] {description} ({elapsed:.1f}s)"
        )
        if self.parent_id is not None:
            self.reporter.progress.update(self.parent_id, visible=True, refresh=True)


class RichProgressReporter:
    def __init__(self, stream: IO[str] = sys.stderr) -> None:
        self.console = Console(file=stream, stderr=stream is sys.stderr)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            console=self.console,
            transient=True,
            redirect_stdout=True,
            redirect_stderr=True,
            refresh_per_second=5,
        )
        self.stack: list[int] = []
        self.progress.start()

    def task(
        self,
        description: str,
        *,
        total: float | None = None,
        unit: str = "items",
    ) -> AbstractContextManager[ProgressTask]:
        return _RichContext(self, description, total, unit)

    def close(self) -> None:
        self.progress.stop()


def create_progress_reporter(
    mode: str,
    *,
    stream: IO[str] = sys.stderr,
) -> ProgressReporter:
    if mode == "off":
        return NullProgressReporter()
    if mode == "plain":
        return PlainProgressReporter(stream)
    if mode != "auto":
        raise ValueError(f"Unknown progress mode: {mode}")
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    if is_tty and os.environ.get("TERM") != "dumb" and not os.environ.get("CI"):
        return RichProgressReporter(stream)
    return PlainProgressReporter(stream)


def close_progress_reporter(reporter: ProgressReporter) -> None:
    close = getattr(reporter, "close", None)
    if callable(close):
        close()
