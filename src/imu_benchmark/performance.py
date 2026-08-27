from __future__ import annotations

import resource
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PERFORMANCE_SCHEMA_VERSION = 2


@dataclass(slots=True)
class PhaseTimer:
    """Accumulate wall-clock durations for named phases."""

    seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - started)

    def iterate(self, name: str, values: Iterable[Any]) -> Iterator[Any]:
        iterator = iter(values)
        while True:
            started = time.perf_counter()
            try:
                value = next(iterator)
            except StopIteration:
                return
            self.seconds[name] = self.seconds.get(name, 0.0) + (
                time.perf_counter() - started
            )
            yield value

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in sorted(self.seconds.items())}


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    user_seconds: float
    system_seconds: float
    max_rss_bytes: int
    input_blocks: int
    output_blocks: int
    voluntary_context_switches: int
    involuntary_context_switches: int


def process_snapshot() -> ProcessSnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ProcessSnapshot(
        user_seconds=float(usage.ru_utime),
        system_seconds=float(usage.ru_stime),
        max_rss_bytes=int(usage.ru_maxrss) * 1024,
        input_blocks=int(usage.ru_inblock),
        output_blocks=int(usage.ru_oublock),
        voluntary_context_switches=int(usage.ru_nvcsw),
        involuntary_context_switches=int(usage.ru_nivcsw),
    )


def process_delta(start: ProcessSnapshot, stop: ProcessSnapshot) -> dict[str, int | float]:
    return {
        "user_seconds": stop.user_seconds - start.user_seconds,
        "system_seconds": stop.system_seconds - start.system_seconds,
        "max_rss_bytes": stop.max_rss_bytes,
        "max_rss_growth_bytes": max(0, stop.max_rss_bytes - start.max_rss_bytes),
        "input_blocks": stop.input_blocks - start.input_blocks,
        "output_blocks": stop.output_blocks - start.output_blocks,
        "voluntary_context_switches": (
            stop.voluntary_context_switches - start.voluntary_context_switches
        ),
        "involuntary_context_switches": (
            stop.involuntary_context_switches - start.involuntary_context_switches
        ),
    }


def _phase_statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "total_seconds": float(np.sum(array)),
        "mean_seconds": float(np.mean(array)),
        "median_seconds": float(np.median(array)),
        "max_seconds": float(np.max(array)),
    }


def aggregate_job_performance(results: list[dict[str, Any]]) -> dict[str, Any]:
    phase_values: dict[str, list[float]] = defaultdict(list)
    model_phase_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    computed_jobs = 0
    cached_jobs = 0
    for result in results:
        status = result.get("status")
        if status == "cached":
            cached_jobs += 1
        elif status == "computed":
            computed_jobs += 1
        else:
            continue
        metadata = result.get("metadata", {})
        model_id = str(metadata.get("job", {}).get("model_id", "unknown"))
        phases = metadata.get("performance", {}).get("phase_seconds", {})
        for name, raw_value in phases.items():
            value = float(raw_value)
            phase_values[name].append(value)
            model_phase_values[model_id][name].append(value)
        if status == "computed":
            invocation = result.get("invocation_performance", {}).get("phase_seconds", {})
            for name, raw_value in invocation.items():
                value = float(raw_value)
                phase_values[name].append(value)
                model_phase_values[model_id][name].append(value)

    phases = {name: _phase_statistics(values) for name, values in sorted(phase_values.items())}
    by_model = {
        model_id: {
            name: _phase_statistics(values)
            for name, values in sorted(model_values.items())
        }
        for model_id, model_values in sorted(model_phase_values.items())
    }
    bottlenecks = sorted(
        (
            {"phase": name, "total_seconds": float(values["total_seconds"])}
            for name, values in phases.items()
        ),
        key=lambda item: item["total_seconds"],
        reverse=True,
    )
    return {
        "computed_jobs": computed_jobs,
        "cached_jobs": cached_jobs,
        "phases": phases,
        "by_model": by_model,
        "bottlenecks": bottlenecks,
    }


def build_performance_report(
    *,
    invocation_phases: dict[str, float],
    results: list[dict[str, Any]],
    process_usage: dict[str, int | float],
    gpu_telemetry: dict[str, Any],
    cache_manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "invocation_phase_seconds": {
            name: float(value) for name, value in sorted(invocation_phases.items())
        },
        "process_usage": process_usage,
        "gpu_telemetry": gpu_telemetry,
        "job_performance": aggregate_job_performance(results),
        "cache_manifests": cache_manifests,
    }
