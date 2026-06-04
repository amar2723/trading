from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from src.market_filters import add_filter_features, passes_chop_filter, passes_session_filter, passes_trend_filter
from src.pattern_detector import detect_pattern
from config import LOOKBACK


def evaluate_trades(
    candles: pd.DataFrame,
    lookahead: int = 20,
    min_score: int | None = None,
    body_multiplier: float | None = None,
    close_near_ratio: float | None = None,
    require_sweep: bool | None = None,
    session: str = "ALL",
    trend_alignment: bool = False,
    avoid_chop: bool = False,
    min_chop_ratio: float = 4.0,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    candles = add_filter_features(candles)
    trades = []

    for end in range(LOOKBACK + 2, len(candles) - lookahead + 1):
        window = candles.iloc[:end]
        kwargs = {"use_closed_candle": False}
        if min_score is not None:
            kwargs["min_score"] = min_score
        if body_multiplier is not None:
            kwargs["body_multiplier"] = body_multiplier
        if close_near_ratio is not None:
            kwargs["close_near_ratio"] = close_near_ratio
        if require_sweep is not None:
            kwargs["require_sweep"] = require_sweep
        signal = detect_pattern(window, **kwargs)
        if signal.signal not in {"BUY", "SELL"}:
            continue

        signal_index = end - 1
        signal_candle = candles.iloc[signal_index]
        if not passes_session_filter(signal.time, session):
            continue
        if not passes_trend_filter(signal_candle, signal.signal, trend_alignment):
            continue
        if not passes_chop_filter(signal_candle, avoid_chop, min_chop_ratio):
            continue
        future = candles.iloc[signal_index + 1 : signal_index + 1 + lookahead]
        plan = build_trade_plan(signal.signal, signal_candle)
        if plan is None:
            continue

        outcome, exit_price, exit_time, rr = simulate_trade(signal.signal, future, plan["sl"], plan["tp1"], plan["tp2"])
        pnl_r = rr

        trades.append(
            {
                "time": signal.time,
                "signal": signal.signal,
                "entry": plan["entry"],
                "sl": plan["sl"],
                "tp1": plan["tp1"],
                "tp2": plan["tp2"],
                "risk": plan["risk"],
                "outcome": outcome,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "rr": pnl_r,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "session": session,
                "trend_direction": signal_candle.get("trend_direction"),
                "chop_ratio": signal_candle.get("chop_ratio"),
            }
        )

    trade_df = pd.DataFrame(trades)
    buy_metrics = build_metrics(trade_df[trade_df["signal"].eq("BUY")]) if not trade_df.empty else build_metrics(trade_df)
    sell_metrics = build_metrics(trade_df[trade_df["signal"].eq("SELL")]) if not trade_df.empty else build_metrics(trade_df)
    combined_metrics = build_metrics(trade_df)
    return trade_df, buy_metrics, sell_metrics, combined_metrics


def evaluate_buy_trades(candles: pd.DataFrame, lookahead: int = 20) -> tuple[pd.DataFrame, dict]:
    trades, _, _, combined = evaluate_trades(candles, lookahead)
    buys = trades[trades["signal"].eq("BUY")] if not trades.empty else trades
    return buys, build_metrics(buys)


def build_trade_plan(signal: str, candle: pd.Series) -> dict | None:
    entry = float(candle["close"])
    if signal == "BUY":
        sl = float(candle["low"])
        risk = entry - sl
        if risk <= 0:
            return None
        return {"entry": entry, "sl": sl, "risk": risk, "tp1": entry + 1.5 * risk, "tp2": entry + 3.0 * risk}
    if signal == "SELL":
        sl = float(candle["high"])
        risk = sl - entry
        if risk <= 0:
            return None
        return {"entry": entry, "sl": sl, "risk": risk, "tp1": entry - 1.5 * risk, "tp2": entry - 3.0 * risk}
    return None


def simulate_trade(signal: str, future: pd.DataFrame, sl: float, tp1: float, tp2: float) -> tuple[str, float | None, str | None, float]:
    for _, candle in future.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        timestamp = str(candle.get("timestamp", candle.get("time", "")))

        if signal == "BUY":
            if low <= sl:
                return "SL", sl, timestamp, -1.0
            if high >= tp2:
                return "TP2", tp2, timestamp, 3.0
            if high >= tp1:
                return "TP1", tp1, timestamp, 1.5
        else:
            if high >= sl:
                return "SL", sl, timestamp, -1.0
            if low <= tp2:
                return "TP2", tp2, timestamp, 3.0
            if low <= tp1:
                return "TP1", tp1, timestamp, 1.5

    return "NO_HIT", None, None, 0.0


def simulate_buy_trade(future: pd.DataFrame, sl: float, tp1: float, tp2: float) -> tuple[str, float | None, str | None, float]:
    return simulate_trade("BUY", future, sl, tp1, tp2)


def build_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "average_rr": 0.0,
            "profit_factor": None,
            "max_drawdown": 0.0,
        }

    wins = trades[trades["rr"] > 0]
    losses = trades[trades["rr"] < 0]
    gross_profit = float(wins["rr"].sum())
    gross_loss = abs(float(losses["rr"].sum()))
    equity = trades["rr"].cumsum()
    drawdown = equity - equity.cummax()

    return {
        "total_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades) * 100),
        "loss_rate": float(len(losses) / len(trades) * 100),
        "average_rr": float(trades["rr"].mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def save_outputs(trades: pd.DataFrame, buy_metrics: dict, sell_metrics: dict, combined_metrics: dict, output_dir: str | Path = "reports") -> dict[str, Path]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "trade_outcomes.csv"
    buy_path = path / "buy_report.html"
    sell_path = path / "sell_report.html"
    combined_path = path / "combined_report.html"

    trades.to_csv(csv_path, index=False)
    _write_report(buy_path, "BUY Evaluation Report", buy_metrics, trades[trades["signal"].eq("BUY")] if not trades.empty else trades)
    _write_report(sell_path, "SELL Evaluation Report", sell_metrics, trades[trades["signal"].eq("SELL")] if not trades.empty else trades)
    _write_report(combined_path, "Combined Evaluation Report", combined_metrics, trades)
    return {"csv": csv_path, "buy": buy_path, "sell": sell_path, "combined": combined_path}


def _write_report(path: Path, title: str, metrics: dict, trades: pd.DataFrame) -> None:
    rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in metrics.items())
    sample = trades.tail(50).to_html(index=False) if not trades.empty else "<p>No trades found.</p>"
    path.write_text(
        f"<html><head><title>{title}</title></head><body><h1>{title}</h1><h2>Metrics</h2><table>{rows}</table><h2>Recent Trade Outcomes</h2>{sample}</body></html>",
        encoding="utf-8",
    )


def load_candles(path: str | None, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    if path:
        candles = pd.read_csv(path)
        if "timestamp" in candles.columns:
            candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
        return candles
    return fetch_live_candles(symbol, timeframe, bars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 BUY signals over historical candles.")
    parser.add_argument("--data", help="Optional candle CSV. If omitted, fetches from MT5.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--min-score", type=int)
    parser.add_argument("--body-multiplier", type=float)
    parser.add_argument("--close-near-ratio", type=float)
    parser.add_argument("--require-sweep", action="store_true", default=None)
    parser.add_argument("--allow-no-sweep", action="store_false", dest="require_sweep")
    parser.add_argument("--session", default="ALL", choices=["ALL", "LONDON", "NEW_YORK", "OVERLAP"])
    parser.add_argument("--trend-alignment", action="store_true")
    parser.add_argument("--avoid-chop", action="store_true")
    parser.add_argument("--min-chop-ratio", type=float, default=4.0)
    args = parser.parse_args()

    candles = load_candles(args.data, args.symbol, args.timeframe, args.bars)
    trades, buy_metrics, sell_metrics, combined_metrics = evaluate_trades(
        candles,
        args.lookahead,
        min_score=args.min_score,
        body_multiplier=args.body_multiplier,
        close_near_ratio=args.close_near_ratio,
        require_sweep=args.require_sweep,
        session=args.session,
        trend_alignment=args.trend_alignment,
        avoid_chop=args.avoid_chop,
        min_chop_ratio=args.min_chop_ratio,
    )
    paths = save_outputs(trades, buy_metrics, sell_metrics, combined_metrics, args.output_dir)

    print("Total BUY Trades:", buy_metrics["total_trades"])
    print("BUY Win Rate:", f"{buy_metrics['win_rate']:.2f}%")
    print("BUY Profit Factor:", buy_metrics["profit_factor"])
    print("Total SELL Trades:", sell_metrics["total_trades"])
    print("SELL Win Rate:", f"{sell_metrics['win_rate']:.2f}%")
    print("SELL Profit Factor:", sell_metrics["profit_factor"])
    print("Combined Win Rate:", f"{combined_metrics['win_rate']:.2f}%")
    print("Combined Profit Factor:", combined_metrics["profit_factor"])
    print("Combined Drawdown:", f"{combined_metrics['max_drawdown']:.2f}R")
    print("Trade CSV:", paths["csv"])
    print("BUY Report:", paths["buy"])
    print("SELL Report:", paths["sell"])
    print("Combined Report:", paths["combined"])


if __name__ == "__main__":
    main()
