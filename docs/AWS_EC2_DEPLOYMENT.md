# AWS EC2 Deployment Guide

## 1. Launch EC2

Recommended:

- Ubuntu 22.04 LTS
- t3.medium or larger
- 30 GB gp3 disk minimum
- security group ports:
  - `22` SSH
  - `8501` Streamlit dashboard
  - `5000` MLflow, restrict to your IP

## 2. Install Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker
```

## 3. Deploy Project

```bash
sudo mkdir -p /opt/trading_ai
sudo chown ubuntu:ubuntu /opt/trading_ai
git clone <YOUR_REPO_URL> /opt/trading_ai
cd /opt/trading_ai/trading_ai
cp .env.example .env
nano .env
```

## 4. Start Services

```bash
docker compose up -d dashboard mlflow
```

Open:

```text
http://EC2_PUBLIC_IP:8501
http://EC2_PUBLIC_IP:5000
```

## 5. Enable Crash Recovery

```bash
sudo cp deploy/systemd/trading-ai-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-ai-dashboard
sudo systemctl start trading-ai-dashboard
```

## 6. Scheduled Retraining

```bash
crontab deploy/cron/retrain.cron
```

## 7. Notes for MT5

MetaTrader5 is Windows-terminal based. For true MT5 live trading, run the realtime service on a Windows VPS with MT5 installed, or connect through a broker/API bridge. Use AWS EC2 Linux primarily for dashboard, MLflow, reports, and retraining.
