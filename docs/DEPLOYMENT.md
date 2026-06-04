# Deployment Guide

This guide covers local Docker, cloud/VPS deployment, CI/CD, monitoring, retraining, alerting, crash recovery, and model versioning.

## 1. Prepare Environment

```bash
cd trading_ai
copy .env.example .env
```

Edit `.env` and set:

```text
MLFLOW_TRACKING_URI
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
DISCORD_WEBHOOK_URL
TRADING_AI_LABELED_DATA
```

## 2. Docker

Build:

```bash
docker build -t trading-ai .
```

Run dashboard:

```bash
docker run --env-file .env -p 8501:8501 -v %cd%/data:/app/data -v %cd%/models:/app/models -v %cd%/reports:/app/reports trading-ai
```

## 3. Docker Compose

Dashboard and MLflow:

```bash
docker compose up -d dashboard mlflow
```

Live engine:

```bash
docker compose --profile live up -d live
```

Scheduled retraining job:

```bash
docker compose --profile retrain run --rm retrainer
```

## 4. Health Checks

```bash
python ops/healthcheck.py --mode files
python ops/healthcheck.py --mode dashboard
python ops/healthcheck.py --mode mlflow
```

Docker Compose also runs health checks automatically.

## 5. MLflow

MLflow server:

```bash
docker compose up -d mlflow
```

Open:

```text
http://SERVER_IP:5000
```

Training and scheduled retraining log metrics, reports, and model artifacts through `ops/mlflow_utils.py`.

## 6. Model Versioning and Registry

Register a model after training:

```python
from ops.model_registry import register_model

register_model("models/xgboost_model.pkl", metrics={"win_rate": 55.2}, stage="staging")
```

Promote a model:

```python
from ops.model_registry import promote_model

promote_model("20260603_120000", stage="production")
```

Registry file:

```text
models/registry.json
```

Versioned models:

```text
models/versions/<version>/
```

## 7. Scheduled Retraining

Docker one-shot:

```bash
docker compose --profile retrain run --rm retrainer
```

Cron example:

```bash
crontab deploy/cron/retrain.cron
```

## 7.1 Strategy Optimization

Optimize for Profit Factor, Expectancy, Sharpe Ratio, Maximum Drawdown, and Risk of Ruin:

```bash
python optimize_strategy.py --data data/labeled/XAUUSD_M5_labeled.csv --model models/xgboost_model.pkl --trials 100
```

Outputs:

```text
reports/optimization/best_optimization.json
reports/optimization/optimization_trials.csv
```

## 8. Telegram and Discord Alerts

Set these in `.env`:

```text
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
```

Live signals and retraining completion can send alerts through `AlertManager`.

## 9. GitHub Actions CI/CD

Workflow:

```text
.github/workflows/ci.yml
```

It runs:

- dependency install
- compile check
- tests
- Docker build

For deployment, add repository secrets:

```text
VPS_HOST
VPS_USER
VPS_SSH_KEY
DEPLOY_PATH
```

Then replace the placeholder deploy step with SSH commands for your server.

## 10. Monitoring

Generated operational files:

```text
logs/app.log
logs/signals.csv
logs/retraining.log
reports/backtest/metrics.json
reports/metrics.json
```

Prometheus starter config:

```text
deploy/prometheus/prometheus.yml
```

## 11. Crash Recovery

Docker Compose services use:

```text
restart: unless-stopped
```

For Linux servers, install systemd services:

```bash
sudo cp deploy/systemd/trading-ai-dashboard.service /etc/systemd/system/
sudo cp deploy/systemd/trading-ai-live.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-ai-dashboard
sudo systemctl start trading-ai-dashboard
```

## 12. Security

Never commit:

- `.env`
- broker credentials
- Telegram tokens
- Discord webhooks
- private keys
- production model/data artifacts
