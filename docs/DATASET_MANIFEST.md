# Dataset Manifest & Data Provenance Specification

**Last Updated**: 2026-11-18  
**Repository**: `amar2723/trading`  
**Working Branch**: `research/quant-optimization`

---

## 1. Historical Dataset Inventory & Identity

| File Path | Rows | Columns | SHA-256 Hash | Provenance & Instrument Identity |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/XAUUSD_M5_60d_max.csv` | 13,854 | `timestamp, open, high, low, close, volume` | `99dba735fcce892396acc0417f25be81657127b0527cb8b3ab98e2b836ae1cfb` | **Yahoo Finance COMEX Gold Futures (`GC=F`) 5-Minute Proxy**. <br>• Date Range: 2026-09-08 04:00 to 2026-11-17 10:45 UTC.<br>• Price levels: \$3,860 – \$4,487.<br>• **CRITICAL**: This is NOT broker spot gold (`XAUUSD`). It reflects CME futures contract quotes with rollover and basis differentials. |
| `data/raw/XAUUSD_1H_2years.csv` | 13,755 | `timestamp, open, high, low, close, volume` | `e53f22817a7fb1c158fca586c910f4594813121dee902bafda0300baccc1c89b` | **Yahoo Finance COMEX Gold Futures (`GC=F`) 1-Hour Proxy**. <br>• Date Range: 2024-06-26 04:00 to 2026-11-17 10:00 UTC.<br>• Price levels: \$2,323 – \$3,872.<br>• **CRITICAL**: This is NOT broker spot gold (`XAUUSD`). |

---

## 2. Broker Spot Data vs Futures Proxy Distinction

1. **Broker Spot Data (`XAUUSD`)**:
   - Trading Venue: OctaTrader / MetaTrader 5 Spot Gold OTC market.
   - Quoting: Spot price per troy ounce against USD (e.g., \$3,771.50 – \$3,775.00).
   - Contract Specifications:
     - 1.00 Lot = 100 troy ounces.
     - Tick size = \$0.01 (1 point).
     - Tick value = \$1.00 per standard lot (\$0.01 per 0.01 lot).
     - 1 pip = 10 points = \$0.10 price move = \$10.00 / lot.
     - Live floating spread: ~4.2 points (~\$0.42 / 4.2 pips).
   - Execution: Requires active broker connection (OctaTrader API / MT5 Terminal).

2. **Futures Proxy Data (`GC=F`)**:
   - Trading Venue: CME Group (COMEX) Gold Futures.
   - Purpose in Repository: Used strictly for quantitative research, backtester regression verification, and feature pipeline mathematical parity tests.
   - **Operational Restriction**: Execution-qualified trading signals from `GC=F` data are strictly **prohibited**. If only futures proxy data is available, all order routing modules must return `HOLD`.

---

## 3. Data Integrity & Validation Rules

Any candle dataset consumed by `compute_indicator_dataframe()` or the live trading engine must satisfy:
1. **Fully Closed Bars**: The current in-progress candle must be dropped before feature calculation.
2. **Strict Chronological Monotonicity**: Timestamps must be sorted in strictly ascending order with zero duplicate timestamps.
3. **UTC Normalization**: All timestamps must be normalized to UTC without timezone ambiguity.
4. **Freshness Gating**: In live mode, candles older than 30 minutes trigger an automatic `HOLD` failsafe.
