from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.advanced_smc.features import add_core_features
from src.advanced_smc.liquidity import add_liquidity
from src.advanced_smc.patterns import add_displacement, add_fvg, add_mss_bos, add_order_blocks


@dataclass
class StructureState:
    bullish_structure_valid: bool
    bearish_structure_valid: bool
    bullish_mss: bool
    bearish_mss: bool
    bullish_bos: bool
    bearish_bos: bool
    bullish_ob: bool
    bearish_ob: bool
    bullish_fvg: bool
    bearish_fvg: bool


def prepare_structure(candles: pd.DataFrame) -> pd.DataFrame:
    data = add_core_features(candles)
    data = add_liquidity(data)
    data = add_displacement(data)
    data = add_mss_bos(data)
    data = add_order_blocks(data)
    data = add_fvg(data)
    return data


def validate_m15_structure(m15_candles: pd.DataFrame) -> StructureState:
    data = prepare_structure(m15_candles)
    return structure_state_from_prepared(data, len(data) - 1)


def structure_state_from_prepared(data: pd.DataFrame, idx: int) -> StructureState:
    recent = data.iloc[max(0, idx - 20): idx + 1]
    bullish_mss = bool(recent["bullish_mss"].any())
    bearish_mss = bool(recent["bearish_mss"].any())
    bullish_bos = bool(recent["bullish_bos"].any())
    bearish_bos = bool(recent["bearish_bos"].any())
    bullish_ob = bool(recent["bullish_ob"].any())
    bearish_ob = bool(recent["bearish_ob"].any())
    bullish_fvg = bool(recent["bullish_fvg"].any())
    bearish_fvg = bool(recent["bearish_fvg"].any())
    return StructureState(
        bullish_structure_valid=bullish_mss and bullish_bos,
        bearish_structure_valid=bearish_mss and bearish_bos,
        bullish_mss=bullish_mss,
        bearish_mss=bearish_mss,
        bullish_bos=bullish_bos,
        bearish_bos=bearish_bos,
        bullish_ob=bullish_ob,
        bearish_ob=bearish_ob,
        bullish_fvg=bullish_fvg,
        bearish_fvg=bearish_fvg,
    )
