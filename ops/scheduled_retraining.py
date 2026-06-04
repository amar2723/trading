from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ops.mlflow_utils import log_training_run
from ops.model_registry import register_model
from src.realtime.alert_manager import AlertConfig, AlertManager
from src.training.train_pipeline import train_all_models


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    data_path = os.getenv("TRADING_AI_LABELED_DATA", "data/labeled/XAUUSD_M5_labeled.csv")
    model_dir = os.getenv("TRADING_AI_MODEL_DIR", "models")
    report_dir = os.getenv("TRADING_AI_REPORTS", "reports")
    alert = AlertManager(
        AlertConfig(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
            discord_webhook=os.getenv("DISCORD_WEBHOOK_URL") or None,
        )
    )

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Retraining data not found: {data_path}")

    logger.info("Starting scheduled retraining using %s", data_path)
    result = train_all_models(data_path, model_dir=model_dir, report_dir=report_dir, optimize=False)
    metrics = result.get("metrics", {})
    model_path = Path(model_dir) / "xgboost_model.pkl"
    registry_entry = register_model(model_path, metrics=metrics, stage="staging")
    run_id = log_training_run(metrics, [str(model_path)], report_dir)
    message = {
        "signal": "RETRAIN",
        "entry": registry_entry["version"],
        "sl": run_id,
        "tp1": result.get("reports"),
        "tp2": model_path,
        "confidence": 100,
    }
    alert.send(message, "XAUUSD")
    print(json.dumps({"result": result, "registry": registry_entry, "mlflow_run_id": run_id}, indent=2, default=str))


if __name__ == "__main__":
    main()
