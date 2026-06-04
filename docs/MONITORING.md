# Monitoring and Operations

## Health Checks

```bash
python ops/healthcheck.py --mode files
python ops/healthcheck.py --mode dashboard
python ops/healthcheck.py --mode mlflow
```

## Logs

Important logs:

```text
logs/app.log
logs/signals.csv
logs/data_ingestion.log
logs/retraining.log
```

## Reports

```text
reports/metrics.json
reports/training_report.html
reports/backtest/metrics.json
reports/backtest/performance_report.html
```

## Alert Channels

Supported:

- console
- log file
- desktop notification
- Telegram
- Discord

Set alert secrets in `.env`.

## Crash Recovery

Docker:

```bash
docker compose ps
docker compose logs -f dashboard
docker compose restart dashboard
```

Systemd:

```bash
sudo systemctl status trading-ai-dashboard
sudo journalctl -u trading-ai-dashboard -f
```

## Model Drift Checks

Recommended manual checks:

- compare current win rate vs previous model
- compare backtest max drawdown vs previous model
- inspect feature importance shifts
- review SHAP summary for changed drivers
- monitor live signal frequency and rejection reasons
