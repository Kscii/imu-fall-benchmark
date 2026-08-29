from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from imu_benchmark.cloud_results import (
    _extract_bundle,
    _publication_manifest,
    _validate_publication_manifest,
    _validate_run,
    _write_bundle,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _formal_run(tmp_path: Path) -> Path:
    run_id = "onnx_full_parity_preflight_v1-test"
    run_dir = tmp_path / run_id
    jobs = run_dir / "jobs"
    jobs.mkdir(parents=True)
    model_ids = [
        "threshold_impact",
        "cuml_logistic_regression",
        "cuml_random_forest",
        "xgboost_cuda",
        "torch_1d_cnn",
        "torch_lstm",
        "torch_cnn_lstm",
    ]
    metrics = []
    events = []
    alarms = []
    parity = []
    for index, model_id in enumerate(model_ids):
        (jobs / f"{index:016x}.npz").write_bytes(f"job-{index}".encode())
        artifact_id = f"{index:016x}"
        artifact = run_dir / "models" / artifact_id
        artifact.mkdir(parents=True)
        onnx = artifact / "model.onnx"
        onnx.write_bytes(f"onnx-{model_id}".encode())
        import hashlib

        onnx_sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
        recipe = "not_applicable" if model_id == "threshold_impact" else "natural"
        input_kind = "features" if model_id.startswith(("cuml", "xgboost")) else "raw"
        spec = {
            "job": {
                "model_id": model_id,
                "training_recipe": recipe,
                "fold": 0,
                "seed": 3888,
            },
            "backend": "fixture",
            "input_kind": input_kind,
            "normalization": {"enabled": False, "mean": None, "scale": None},
        }
        (artifact / "model_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        base = {
            "model_id": model_id,
            "training_recipe": recipe,
            "fold": 0,
            "seed": 3888,
        }
        metrics.append(
            {
                **base,
                "threshold": 0.5,
                **{name: 0.8 + index / 100 for name in (
                    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
                    "precision", "f1", "mcc", "auroc", "auprc",
                )},
            }
        )
        events.append(
            {
                **base,
                "event_sensitivity": 0.9,
                "adl_recording_false_positive_rate": 0.1,
                "adl_false_positive_windows_per_hour": 2.0,
                "onset_latency_median_s": 0.2,
                "onset_latency_p95_s": 0.5,
                "impact_offset_median_s": -0.1,
            }
        )
        alarms.append(
            {
                **base,
                "alarm_policy_id": "reference",
                "reference_policy": True,
                "validation_pareto": True,
                "threshold": 0.5,
                "event_sensitivity": 0.85,
                "adl_recording_false_positive_rate": 0.08,
                "adl_alarm_episodes_per_hour": 1.5,
                "onset_latency_median_s": 0.25,
                "onset_latency_p95_s": 0.55,
                "impact_offset_median_s": -0.12,
            }
        )
        for split in ("validation", "test"):
            parity.append(
                {
                    **base,
                    "split": split,
                    "samples": 10,
                    "batches": 1,
                    "batch_size": 10,
                    "maximum_absolute_error": 0.0001,
                    "mean_absolute_error": 0.00001,
                    "p99_absolute_error": 0.00009,
                    "onnx_sha256": onnx_sha,
                }
            )
    _write_csv(run_dir / "metrics.csv", metrics)
    _write_csv(run_dir / "event_metrics.csv", events)
    _write_csv(run_dir / "alarm_metrics.csv", alarms)
    _write_csv(run_dir / "onnx_parity.csv", parity)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "cache_manifest": {
                    "contract_sha256": "d" * 64,
                    "data_split_fingerprint": "e" * 64,
                    "window_schema_version": "unified_fall_windows_v4_25hz",
                    "feature_schema_version": "window_features_v2_25hz",
                    "sampling_rate_hz": 25.0,
                    "window_samples": 50,
                    "stride_seconds": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.yaml").write_text(
        json.dumps(
            {
                "alarm_policy_sha256": "f" * 64,
                "alarm_policy": {
                    "schema_version": 1,
                    "reference_policy": "reference",
                    "selection": "validation_pareto_sensitivity_alarm_rate_latency",
                    "policies": [
                        {
                            "id": "reference",
                            "required_positive_windows": 1,
                            "lookback_windows": 1,
                            "consecutive": True,
                            "cooldown_seconds": 10.0,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps({"onnx": "fixture", "onnxruntime": "fixture"}),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "aggregate_metrics.csv").write_text("model,value\n", encoding="utf-8")
    (run_dir / "statistical_manifest.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "experiment_id": "onnx_full_parity_preflight_v1",
        "status": "PASS",
        "scheduled_jobs": 7,
        "completed_jobs": 7,
        "failures": [],
        "source": {
            "kind": "git",
            "commit": "a" * 40,
            "dirty": False,
            "snapshot_sha256": None,
        },
        "base_snapshot_id": "imu_25hz_snapshot_v2",
        "data_view_id": "temporal_core_v1",
        "snapshot_sha256": "b" * 64,
        "resolved_config_sha256": "c" * 64,
        "data_quality_status": "engineering_full_onnx_parity_preflight_only",
        "known_limitations": ["engineering fixture"],
        "statistical_analysis": None,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def test_result_bundle_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    run_manifest, entries = _validate_run(run_dir)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_bundle(run_dir, run_dir.name, entries, first)
    _write_bundle(run_dir, run_dir.name, entries, second)
    assert first.read_bytes() == second.read_bytes()
    from imu_benchmark.model_catalog import build_experiment_catalog

    catalog = build_experiment_catalog(run_dir, run_manifest)
    direct_files = [
        {
            "file_id": "onnx-0000000000000000",
            "artifact_id": "0000000000000000",
            "role": "onnx",
            "filename": "model.onnx",
            "object_key": (
                f"benchmark-results/engineering/{run_dir.name}/models/"
                "0000000000000000/model.onnx"
            ),
            "content_type": "application/octet-stream",
            "size_bytes": 10,
            "sha256": "f" * 64,
        }
    ]
    publication = _publication_manifest(
        run_manifest, entries, first, catalog, direct_files
    )
    _validate_publication_manifest(publication, run_id=run_dir.name)
    extracted = _extract_bundle(
        first, tmp_path / "extracted", run_dir.name, publication["files"]
    )
    assert (extracted / "run_manifest.json").is_file()
    assert len(list((extracted / "jobs").glob("*.npz"))) == 7


def test_result_publication_rejects_dirty_or_incomplete_runs(tmp_path: Path) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["dirty"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="clean, identified"):
        _validate_run(run_dir)

    manifest["source"]["dirty"] = False
    manifest["completed_jobs"] = 6
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="7/7"):
        _validate_run(run_dir)


def test_result_publication_rejects_onnx_different_from_parity_evidence(
    tmp_path: Path,
) -> None:
    run_dir = _formal_run(tmp_path / "runs")
    first_model = next((run_dir / "models").glob("*/model.onnx"))
    first_model.write_bytes(b"different-model")

    with pytest.raises(ValueError, match="differs from the parity evidence"):
        _validate_run(run_dir)


def test_existing_result_v1_manifest_remains_readable() -> None:
    run_id = "formal_baseline_temporal_core_onnx_v1-existing"
    payload = {
        "schema_version": "imu_benchmark_result_manifest_v1",
        "run_id": run_id,
        "experiment_id": "formal_baseline_temporal_core_onnx_v1",
        "scheduled_jobs": 65,
        "source": {"commit": "a" * 40, "dirty": False},
        "base_snapshot_id": "imu_25hz_snapshot_v2",
        "snapshot_sha256": "b" * 64,
        "resolved_config_sha256": "c" * 64,
        "bundle": {
            "filename": "run.tar.gz",
            "size_bytes": 10,
            "sha256": "d" * 64,
        },
        "files": [{"path": "report.md", "size_bytes": 1, "sha256": "e" * 64}],
        "quick_files": [],
    }

    _validate_publication_manifest(payload, run_id=run_id)
