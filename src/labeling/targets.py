from __future__ import annotations

import numpy as np
import pandas as pd


def add_tp_before_sl_target(
    df: pd.DataFrame,
    horizon: int = 60,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    targets: list[float] = []
    for i, row in out.iterrows():
        if pd.isna(row.get("atr")):
            targets.append(np.nan)
            continue
        entry = row["close"]
        side = 1 if row.get("bullish_mss", False) or row.get("bullish_bos", False) else -1
        tp = entry + side * row["atr"] * tp_atr_mult
        sl = entry - side * row["atr"] * sl_atr_mult
        future = out.iloc[i + 1 : i + 1 + horizon]
        label = np.nan
        for _, f in future.iterrows():
            if side == 1:
                if f["low"] <= sl:
                    label = 0
                    break
                if f["high"] >= tp:
                    label = 1
                    break
            else:
                if f["high"] >= sl:
                    label = 0
                    break
                if f["low"] <= tp:
                    label = 1
                    break
        targets.append(label)
    out["target_tp_before_sl"] = targets
    return out
