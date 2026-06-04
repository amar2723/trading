from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for column in ["timestamp", "time", "entry_time", "exit_time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def load_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def latest_signal(log_path: str | Path = "logs/signals.csv") -> dict:
    signals = load_csv(log_path)
    if signals.empty:
        return {}
    return signals.iloc[-1].to_dict()


def available_files(base_dir: str | Path, pattern: str = "*.csv") -> list[str]:
    path = Path(base_dir)
    if not path.exists():
        return []
    return [str(item) for item in sorted(path.glob(pattern))]
