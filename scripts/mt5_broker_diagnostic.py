from __future__ import annotations

"""
MetaTrader 5 & Broker Connection Read-Only Diagnostic Utility for XAUUSD.
Safely queries:
1. MetaTrader 5 library availability and terminal executable detection
2. Process state detection (terminal64.exe running vs inactive)
3. Account type verification (DEMO vs REAL) and margin metrics
4. Symbol discovery and verification (XAUUSD, GOLD, XAUUSDm, etc.)
5. Exact contract specification (point size, contract size, volume limits)
6. Live quote freshness, bid/ask spread (points, pips, USD), and latency
7. Market session state and trading permissions

Guaranteed 100% READ-ONLY: Never places, modifies, or cancels any trade.
Never blocks or hangs if the desktop terminal is inactive.
"""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from src.broker_adapter import SpreadQuote

COMMON_TERMINAL_PATHS = [
    Path(r"C:\Program Files\MetaTrader 5\terminal64.exe"),
    Path(r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe"),
    Path(os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal")),
]


def is_terminal_process_active() -> bool:
    try:
        out = subprocess.check_output("tasklist", shell=True).decode(errors="ignore")
        return "terminal64.exe" in out.lower()
    except Exception:
        return False


def find_terminal_executable() -> str | None:
    for p in COMMON_TERMINAL_PATHS:
        if p.is_file():
            return str(p)
    return None


def run_broker_diagnostic(symbol: str = "XAUUSD", timeout_seconds: float = 4.0, mt5_lib=None) -> dict:
    _mt5 = mt5_lib if mt5_lib is not None else mt5
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    term_path = find_terminal_executable()
    process_active = is_terminal_process_active() if mt5_lib is None else True

    result = {
        "diagnostic_time_utc": timestamp_utc,
        "symbol_requested": symbol,
        "mt5_package_installed": _mt5 is not None,
        "terminal_executable_path": term_path,
        "terminal_process_active": process_active,
        "terminal_connected": False,
        "account_info": {},
        "terminal_info": {},
        "symbol_specs": {},
        "live_quote": {},
        "status": "DISCONNECTED",
        "notes": [],
    }

    if _mt5 is None:
        result["status"] = "MT5_PACKAGE_NOT_INSTALLED"
        result["notes"].append("MetaTrader5 python package is not installed in the active environment.")
        return result

    if not process_active:
        result["status"] = "TERMINAL_NOT_RUNNING"
        if term_path:
            result["notes"].append(f"MetaTrader 5 executable found at: '{term_path}'.")
        result["notes"].append(
            "The desktop terminal process ('terminal64.exe') is currently NOT running in your Windows session."
        )
        result["notes"].append(
            "To connect: Open MetaTrader 5 on your desktop, log into your Octa Demo account, and ensure Algo Trading is enabled."
        )
        return result

    # When process is active, attempt safe initialization with non-blocking timeout
    def _do_init():
        if term_path and mt5_lib is None:
            return _mt5.initialize(path=term_path, timeout=int(timeout_seconds * 1000))
        return _mt5.initialize(timeout=int(timeout_seconds * 1000))

    init_ok = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_do_init)
    try:
        init_ok = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        result["status"] = "INITIALIZE_TIMEOUT"
        result["notes"].append(f"mt5.initialize() timed out after {timeout_seconds} seconds.")
        return result
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        result["status"] = "INITIALIZE_EXCEPTION"
        result["notes"].append(f"mt5.initialize() raised exception: {exc}")
        return result
    finally:
        executor.shutdown(wait=False)

    if not init_ok:
        err = _mt5.last_error()
        result["status"] = "TERMINAL_CONNECT_FAILED"
        result["notes"].append(f"Failed to connect to active MT5 terminal. Last MT5 error: {err}")
        return result

    result["terminal_connected"] = True

    # Terminal info
    t_info = _mt5.terminal_info()
    if t_info is not None:
        t_dict = t_info._asdict() if hasattr(t_info, "_asdict") else dict(t_info)
        result["terminal_info"] = {
            "name": t_dict.get("name"),
            "path": t_dict.get("path"),
            "connected": bool(t_dict.get("connected", False)),
            "trade_allowed": bool(t_dict.get("trade_allowed", False)),
            "ping_last": t_dict.get("ping_last"),
        }

    # Account info
    acc_info = _mt5.account_info()
    if acc_info is not None:
        acc_dict = acc_info._asdict() if hasattr(acc_info, "_asdict") else dict(acc_info)
        trade_mode_val = acc_dict.get("trade_mode")
        # MT5 trade modes: 0 = DEMO, 1 = CONTEST, 2 = REAL
        trade_mode_str = "DEMO" if trade_mode_val == 0 else ("CONTEST" if trade_mode_val == 1 else "REAL")
        result["account_info"] = {
            "login": acc_dict.get("login"),
            "trade_mode": trade_mode_str,
            "trade_mode_code": trade_mode_val,
            "server": acc_dict.get("server"),
            "currency": acc_dict.get("currency"),
            "balance": acc_dict.get("balance"),
            "equity": acc_dict.get("equity"),
            "margin_free": acc_dict.get("margin_free"),
            "leverage": acc_dict.get("leverage"),
            "name": acc_dict.get("name"),
            "company": acc_dict.get("company"),
        }
    else:
        result["notes"].append("account_info() returned None. Not logged into an account.")

    # Symbol discovery
    symbols_to_try = [symbol, "GOLD", "XAUUSDm", "XAUUSD.a", "XAUUSD_i"]
    selected_symbol = None
    for sym in symbols_to_try:
        s_info = _mt5.symbol_info(sym)
        if s_info is not None:
            selected_symbol = sym
            break

    if selected_symbol is None:
        result["status"] = "SYMBOL_NOT_FOUND"
        result["notes"].append(f"Could not find {symbol} or common aliases among broker symbols.")
        _mt5.shutdown()
        return result

    _mt5.symbol_select(selected_symbol, True)
    sym_info = _mt5.symbol_info(selected_symbol)
    if sym_info is not None:
        sd = sym_info._asdict() if hasattr(sym_info, "_asdict") else dict(sym_info)
        result["symbol_specs"] = {
            "symbol": selected_symbol,
            "digits": sd.get("digits"),
            "point": sd.get("point"),
            "contract_size": sd.get("trade_contract_size"),
            "volume_min": sd.get("volume_min"),
            "volume_step": sd.get("volume_step"),
            "volume_max": sd.get("volume_max"),
            "spread_points": sd.get("spread"),
            "trade_mode": sd.get("trade_mode"),
            "currency_base": sd.get("currency_base"),
            "currency_profit": sd.get("currency_profit"),
            "currency_margin": sd.get("currency_margin"),
        }

    # Live Quote & Freshness
    tick = _mt5.symbol_info_tick(selected_symbol)
    if tick is not None and getattr(tick, "ask", 0) > getattr(tick, "bid", 0) > 0:
        quote = SpreadQuote.from_bid_ask(
            bid=tick.bid,
            ask=tick.ask,
            time_str=str(datetime.fromtimestamp(tick.time, tz=timezone.utc)),
            is_broker_confirmed=True,
            quote_source="BROKER_MT5_SPOT",
        )
        tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        latency_sec = round((now_utc - tick_time).total_seconds(), 2)

        result["live_quote"] = {
            "bid": quote.bid,
            "ask": quote.ask,
            "spread_price_usd": quote.spread_price_usd,
            "spread_usd": quote.spread_price_usd,
            "spread_points": quote.spread_points,
            "spread_pips": quote.spread_pips,
            "tick_time_utc": quote.time,
            "quote_time": quote.time,
            "latency_seconds": latency_sec,
            "is_fresh": (0.0 <= latency_sec <= 30.0),
        }
    else:
        result["notes"].append("symbol_info_tick returned invalid or zero tick data.")

    # Strict Validation Gate for CONNECTED_DEMO_VERIFIED
    if not result["account_info"]:
        result["status"] = "NO_ACCOUNT_LOGGED_IN"
        result["notes"].append("No trading account is logged in on the MT5 terminal.")
    elif result["account_info"].get("trade_mode") == "REAL":
        result["status"] = "BLOCKED_REAL_ACCOUNT"
        result["notes"].append("SAFETY VETO: Real-money live trading account detected. Automated execution is strictly prohibited on real accounts.")
    elif not result["terminal_info"].get("connected"):
        result["status"] = "TERMINAL_NOT_CONNECTED_TO_SERVER"
        result["notes"].append("MT5 terminal reports no active connection to broker trade server.")
    elif not result["symbol_specs"] or (result["symbol_specs"].get("contract_size") or 0) <= 0:
        result["status"] = "INVALID_SYMBOL_SPECS"
        result["notes"].append(f"Symbol specifications for {selected_symbol} are missing or invalid.")
    elif not result["live_quote"]:
        result["status"] = "NO_LIVE_TICK"
        result["notes"].append(f"No valid tick quote received for {selected_symbol}.")
    elif not result["live_quote"].get("is_fresh"):
        result["status"] = "STALE_BROKER_QUOTE"
        result["notes"].append(f"Broker quote latency ({result['live_quote']['latency_seconds']:.1f}s) exceeds 30.0s freshness limit.")
    else:
        result["status"] = "CONNECTED_DEMO_VERIFIED"
        result["notes"].append(f"All pre-flight checks verified: Connected to DEMO account {result['account_info']['login']} with fresh quotes.")

    _mt5.shutdown()
    return result


def print_diagnostic_report(diag: dict):
    print("=" * 70)
    print("         XAUUSD / MT5 BROKER READ-ONLY DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Timestamp (UTC):       {diag.get('diagnostic_time_utc')}")
    print(f"Diagnostic Status:     {diag.get('status')}")
    print(f"MT5 Package Installed: {diag.get('mt5_package_installed')}")
    print(f"Executable Located:    {diag.get('terminal_executable_path')}")
    print(f"Desktop Process Open:  {diag.get('terminal_process_active')}")
    print(f"Terminal Connected:    {diag.get('terminal_connected')}")
    print("-" * 70)

    acc = diag.get("account_info", {})
    if acc:
        print("ACCOUNT DETAILS:")
        print(f"  Login:         {acc.get('login')}")
        print(f"  Account Type:  {acc.get('trade_mode')} (Code: {acc.get('trade_mode_code')})")
        print(f"  Server:        {acc.get('server')}")
        print(f"  Company:       {acc.get('company')}")
        print(f"  Balance:       ${acc.get('balance', 0.0):,.2f} {acc.get('currency', 'USD')}")
        print(f"  Equity:        ${acc.get('equity', 0.0):,.2f} {acc.get('currency', 'USD')}")
        print(f"  Free Margin:   ${acc.get('margin_free', 0.0):,.2f}")
        print(f"  Leverage:      1:{acc.get('leverage')}")
    else:
        print("ACCOUNT DETAILS: Inactive (MT5 desktop terminal is not running or logged in)")

    print("-" * 70)
    specs = diag.get("symbol_specs", {})
    if specs:
        print(f"SYMBOL SPECIFICATIONS ({specs.get('symbol')}):")
        print(f"  Digits:        {specs.get('digits')} (Point = {specs.get('point')})")
        print(f"  Contract Size: {specs.get('contract_size')} oz")
        print(f"  Volume Limits: Min: {specs.get('volume_min')} | Step: {specs.get('volume_step')} | Max: {specs.get('volume_max')}")
        print(f"  Current Spread:{specs.get('spread_points')} points")
    else:
        print("SYMBOL SPECIFICATIONS: High-Fidelity Shadow Model (OctaTrader 100 oz / 0.01 lot step)")

    print("-" * 70)
    quote = diag.get("live_quote", {})
    if quote:
        print("LIVE BROKER QUOTE:")
        print(f"  Bid:           ${quote.get('bid'):.2f}")
        print(f"  Ask:           ${quote.get('ask'):.2f}")
        print(f"  Spread:        ${quote.get('spread_usd'):.2f} / {quote.get('spread_pips'):.1f} pips / {quote.get('spread_points'):.0f} points")
        print(f"  Quote Time:    {quote.get('tick_time_utc')}")
        print(f"  Quote Latency: {quote.get('latency_seconds')} seconds (Fresh: {quote.get('is_fresh')})")
    else:
        print("LIVE BROKER QUOTE: None (Standby)")

    notes = diag.get("notes", [])
    if notes:
        print("-" * 70)
        print("DIAGNOSTIC GUIDANCE & NOTES:")
        for note in notes:
            print(f"  • {note}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only MT5 Broker Diagnostic Tool")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to inspect (default: XAUUSD)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of text report")
    args = parser.parse_args()

    diag_result = run_broker_diagnostic(symbol=args.symbol)
    if args.json:
        print(json.dumps(diag_result, indent=2))
    else:
        print_diagnostic_report(diag_result)
