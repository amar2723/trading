from __future__ import annotations

"""
Immutable Signal Journal for XAUUSD Trading.
Persists every signal candidate, filter decision, and risk verdict
chronologically BEFORE the outcome occurs.
Logs to reports/signal_journal.jsonl
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_PATH = ROOT / "reports/signal_journal.jsonl"


class SignalJournal:
    def __init__(self, journal_file: Path = JOURNAL_PATH):
        self.journal_file = journal_file
        self.journal_file.parent.mkdir(parents=True, exist_ok=True)

    def log_signal_decision(
        self,
        strategy_name: str,
        action: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        risk_pips: float,
        predicted_win_prob: float,
        expected_R: float,
        ml_decision: str,
        risk_approved: bool,
        risk_lot_size: float,
        risk_usd: float,
        veto_reason: str | None = None,
        bar_timestamp: str = "",
    ) -> str:
        signal_id = f"sig_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:6]}"
        record = {
            "signal_id": signal_id,
            "decision_timestamp": datetime.now(timezone.utc).isoformat(),
            "bar_timestamp": bar_timestamp,
            "strategy_name": strategy_name,
            "intended_side": action,
            "intended_entry": entry,
            "intended_sl": sl,
            "intended_tp1": tp1,
            "intended_tp2": tp2,
            "risk_pips": risk_pips,
            "predicted_win_prob": predicted_win_prob,
            "predicted_expected_R": expected_R,
            "ml_filter_decision": ml_decision,
            "risk_governor_approved": risk_approved,
            "approved_lot_size": risk_lot_size,
            "risk_usd": risk_usd,
            "veto_reason": veto_reason,
            "final_action": "TRADE" if (ml_decision == "TAKE" and risk_approved and action != "HOLD") else "HOLD",
            "resolution_status": "PENDING" if (ml_decision == "TAKE" and risk_approved and action != "HOLD") else "NOT_EXECUTED",
        }
        with open(self.journal_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return signal_id

    def log_trade_resolution(
        self,
        signal_id: str,
        outcome: str,  # "TP1_WIN", "SL_LOSS", "BE_EXIT", "TIMEOUT"
        exit_price: float,
        net_pnl_usd: float,
        net_R: float,
        mae_pips: float = 0.0,
        mfe_pips: float = 0.0,
    ):
        record = {
            "event": "TRADE_RESOLVED",
            "signal_id": signal_id,
            "resolution_timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "exit_price": exit_price,
            "net_pnl_usd": net_pnl_usd,
            "net_R": net_R,
            "mae_pips": mae_pips,
            "mfe_pips": mfe_pips,
        }
        with open(self.journal_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
