# VPS Deployment Guide

## Recommended VPS

- Ubuntu 22.04 for dashboard/retraining
- Windows VPS for MT5 live execution
- 2 vCPU
- 4 GB RAM minimum
- 30 GB disk minimum

## Linux VPS Setup

```bash
sudo apt update
sudo apt install -y git curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

Clone and run:

```bash
git clone <YOUR_REPO_URL> /opt/trading_ai
cd /opt/trading_ai/trading_ai
cp .env.example .env
nano .env
docker compose up -d dashboard mlflow
```

## Windows VPS for MT5

1. Install MetaTrader5.
2. Log in to your broker account.
3. Install Python 3.11.
4. Install project requirements.
5. Run:

```powershell
python run_live.py
```

For crash recovery on Windows, use Task Scheduler or NSSM to run `python run_live.py` as a service.

## Firewall

Allow:

- `8501` for dashboard
- `5000` for MLflow only from trusted IPs
- `22` SSH

## Backups

Back up:

```text
models/
models/registry.json
mlruns/
reports/
logs/signals.csv
data/labeled/
```
