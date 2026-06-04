from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.labeling.entry_generator import generate_entries
from src.labeling.sl_tp_generator import generate_sl_tp
from src.labeling.trade_simulator import simulate_trades


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create the complete labeled training dataset from pattern data."""
    out = generate_entries(df)
    out = generate_sl_tp(out)
    out = simulate_trades(out)
    out["profitable_trade"] = out["trade_result"].isin(["WIN", "PARTIAL"]).astype(int)
    out["losing_trade"] = out["trade_result"].eq("LOSS").astype(int)
    return out


def split_time_series(df: pd.DataFrame, output_dir: str | Path, train_ratio: float = 0.7, validation_ratio: float = 0.15) -> dict[str, Path]:
    """Save chronological train/validation/test CSVs. Never shuffles rows."""
    data = df.sort_values(_time_col(df)).reset_index(drop=True) if _time_col(df) else df.reset_index(drop=True)
    n = len(data)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output / "train.csv",
        "validation": output / "validation.csv",
        "test": output / "test.csv",
    }
    data.iloc[:train_end].to_csv(paths["train"], index=False)
    data.iloc[train_end:validation_end].to_csv(paths["validation"], index=False)
    data.iloc[validation_end:].to_csv(paths["test"], index=False)
    return paths


def build_report(df: pd.DataFrame) -> dict:
    trades = df[df["entry_type"].isin(["BUY", "SELL"])].copy()
    wins = trades[trades["trade_result"].isin(["WIN", "PARTIAL"])]
    losses = trades[trades["trade_result"].eq("LOSS")]
    total = len(trades)

    profit_points = _profit_points(trades)
    return {
        "total_trades": int(total),
        "win_rate": float(len(wins) / total * 100) if total else 0.0,
        "loss_rate": float(len(losses) / total * 100) if total else 0.0,
        "average_rr": float(trades["risk_reward_ratio"].mean()) if total else 0.0,
        "average_profit": float(profit_points[profit_points > 0].mean()) if (profit_points > 0).any() else 0.0,
        "average_loss": float(profit_points[profit_points < 0].mean()) if (profit_points < 0).any() else 0.0,
    }


def save_labeled_dataset(df: pd.DataFrame, output_path: str | Path, create_splits: bool = True) -> dict[str, Path]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    paths = {"labeled": path}
    if create_splits:
        paths.update(split_time_series(df, path.parent))
    return paths


def _profit_points(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    buy = trades["entry_type"].eq("BUY")
    sell = trades["entry_type"].eq("SELL")
    exit_price = trades["exit_price"].fillna(trades["entry_price"])
    return pd.Series(
        np.select(
            [buy, sell],
            [exit_price - trades["entry_price"], trades["entry_price"] - exit_price],
            default=0.0,
        ),
        index=trades.index,
    )


def _time_col(df: pd.DataFrame) -> str | None:
    if "timestamp" in df.columns:
        return "timestamp"
    if "time" in df.columns:
        return "time"
    return None
