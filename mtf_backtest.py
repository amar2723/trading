from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from src.multi_timeframe.entry_confirmation import confirm_mtf_row, prepare_m5
from src.multi_timeframe.structure_validator import prepare_structure, structure_state_from_prepared
from src.multi_timeframe.trend_analyzer import trend_state_from_prepared
from trade_evaluator import build_metrics, evaluate_trades, simulate_trade


PERIODS = [1000, 5000, 10000]


def mtf_backtest(symbol: str, m5_bars: int, min_confidence: float = 75.0, lookahead: int = 20) -> tuple[pd.DataFrame, dict]:
    m5 = fetch_live_candles(symbol, "M5", m5_bars)
    m15 = fetch_live_candles(symbol, "M15", max(300, m5_bars // 3 + 100))
    h1 = fetch_live_candles(symbol, "H1", max(300, m5_bars // 12 + 100))
    m5_prepared = prepare_m5(m5)
    m15_prepared = prepare_structure(m15)
    from src.advanced_smc.features import add_core_features

    h1_prepared = add_core_features(h1)
    m15_times = pd.to_datetime(m15_prepared["timestamp"]).reset_index(drop=True)
    h1_times = pd.to_datetime(h1_prepared["timestamp"]).reset_index(drop=True)
    trades = []
    for idx in range(200, len(m5) - lookahead):
        ts = pd.Timestamp(m5.iloc[idx]["timestamp"])
        h1_idx = h1_times.searchsorted(ts, side="right") - 1
        m15_idx = m15_times.searchsorted(ts, side="right") - 1
        if h1_idx < 100 or m15_idx < 100:
            continue
        trend = trend_state_from_prepared(h1_prepared, int(h1_idx))
        structure = structure_state_from_prepared(m15_prepared, int(m15_idx))
        signal = confirm_mtf_row(trend, structure, m5_prepared.iloc[idx], min_confidence)
        if signal.signal not in {"BUY", "SELL"}:
            continue
        future = m5.iloc[idx + 1 : idx + 1 + lookahead]
        outcome, exit_price, exit_time, rr = simulate_trade(signal.signal, future, signal.stop_loss, signal.tp1, signal.tp2)
        trades.append({**signal.to_dict(), "outcome": outcome, "exit_price": exit_price, "exit_time": exit_time, "rr": rr})
    trade_df = pd.DataFrame(trades)
    return trade_df, build_metrics(trade_df) if not trade_df.empty else build_metrics(pd.DataFrame())


def compare_current_vs_mtf(symbol: str, output_dir: str | Path = "reports") -> pd.DataFrame:
    rows = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for bars in PERIODS:
        m5 = fetch_live_candles(symbol, "M5", bars)
        _, _, _, current_metrics = evaluate_trades(m5)
        mtf_trades, mtf_metrics = mtf_backtest(symbol, bars)
        mtf_trades.to_csv(output / f"mtf_trades_{bars}.csv", index=False)
        rows.append({"strategy": "Current", "candles": bars, **current_metrics})
        rows.append({"strategy": "Multi-Timeframe", "candles": bars, **mtf_metrics})
    results = pd.DataFrame(rows)
    results.to_csv(output / "mtf_results.csv", index=False)
    (output / "mtf_comparison.html").write_text(results.to_html(index=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current strategy vs MTF strategy.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    results = compare_current_vs_mtf(args.symbol, args.output_dir)
    print(json.dumps(results.to_dict(orient="records"), indent=2, default=str))


if __name__ == "__main__":
    main()
