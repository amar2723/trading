from __future__ import annotations

import os
from pathlib import Path


def configure_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install mlflow to use MLflow integration.") from exc

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "xauusd-trading-ai")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_training_run(metrics: dict, model_files: list[str] | None = None, report_dir: str = "reports") -> str:
    mlflow = configure_mlflow()
    with mlflow.start_run() as run:
        for key, value in _flatten(metrics).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            else:
                mlflow.log_param(key, str(value)[:250])
        for model_file in model_files or []:
            path = Path(model_file)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="models")
        report_path = Path(report_dir)
        if report_path.exists():
            mlflow.log_artifacts(str(report_path), artifact_path="reports")
        return run.info.run_id


def _flatten(payload: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat
