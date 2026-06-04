from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def check_dashboard() -> bool:
    try:
        response = requests.get("http://127.0.0.1:8501/_stcore/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_mlflow() -> bool:
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        return response.status_code in {200, 404}
    except Exception:
        return False


def check_files() -> bool:
    required_dirs = ["data", "models", "logs", "reports"]
    return all(Path(path).exists() for path in required_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Container and deployment health checks.")
    parser.add_argument("--mode", choices=["dashboard", "mlflow", "files"], default="files")
    args = parser.parse_args()
    checks = {"dashboard": check_dashboard, "mlflow": check_mlflow, "files": check_files}
    ok = checks[args.mode]()
    print("ok" if ok else "failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
