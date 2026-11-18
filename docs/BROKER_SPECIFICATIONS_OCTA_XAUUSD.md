# OctaTrader XAUUSD Broker Specifications & Contract Parameters

**Audit Timestamp**: 2026-11-18 07:05 UTC  
**Environment**: OctaTrader Live Web Terminal (`https://my.octabroker.com/webterminal/`)  
**Account**: Real Account `125726194` (Inspection mode: strictly read-only, 0 trades placed)

---

## 1. Verified Symbol Specifications

| Specification Field | Live Broker Value | Quantitative Definition & Impact on Sizing |
| :--- | :--- | :--- |
| **Symbol Name** | `XAUUSD` | Gold Spot traded against US Dollar (XAU/USD). |
| **Category** | `Metals` | Grouped under Metals (alongside XAGUSD). |
| **Contract Size** | `100.00` | 1.00 standard lot represents **100 troy ounces** of gold. |
| **Digits / Precision** | `2` decimal places | Quotes are to 2 decimal places (e.g., \$3,771.50). Smallest price increment is \$0.01. |
| **Tick Size** | `$0.01` | Minimum price movement per ounce = \$0.01. |
| **Tick Value (1.00 lot)**| `$1.00` | Moving \$0.01 on 100 oz = \$1.00 USD PnL per standard lot. |
| **Pip Definition** | `$0.10` price move | Standard MT4/MT5 convention for 2-digit gold: 1 pip = 10 points = \$0.10 move. |
| **Pip Price (1.00 lot)** | `$10.00` | Moving 1 pip (\$0.10) on 100 oz = **\$10.00 USD** PnL per standard lot. |
| **Pip Price (0.01 lot)** | `$0.10` | Moving 1 pip (\$0.10) on 1 oz = **\$0.10 USD** PnL per micro lot. |
| **Minimum Volume** | `0.01 lot` | Minimum allowable position size = **0.01 lot** (1 troy ounce). |
| **Volume Step** | `0.01 lot` | Increments are in 0.01 lot steps. |
| **Maximum Volume** | `50.00 lots` | Maximum single order volume = 50.00 lots (5,000 ounces). |
| **Margin Percentage** | `100.00%` | Baseline margin requirement without leverage adjustments. |
| **Order Execution Mode**| `Market` | Market execution; fills execute at current available broker liquidity. |
| **Live Spread** | `~4.2 points` (`~42 pips`) | Floating spread typically oscillates between 38 and 50 pips (\$0.38 - \$0.50/oz). |

---

## 2. Trading Session & Schedule (UTC)

| Day | Active Trading Hours (UTC) | Maintenance / Daily Break |
| :--- | :--- | :--- |
| **Monday** | 04:30 – 00:00 (next day) | Market open at 04:30 UTC |
| **Tuesday** | 00:00 – 03:30 & 04:30 – 00:00 | Daily maintenance break: 03:30 – 04:30 UTC |
| **Wednesday** | 00:00 – 03:30 & 04:30 – 00:00 | Daily maintenance break: 03:30 – 04:30 UTC |
| **Thursday** | 00:00 – 03:30 & 04:30 – 00:00 | Daily maintenance break: 03:30 – 04:30 UTC |
| **Friday** | 00:00 – 03:30 & 04:30 – 00:00 | Daily maintenance break: 03:30 – 04:30 UTC |
| **Saturday** | 00:00 – 03:30 | Weekly market close at 03:30 UTC Saturday |
| **Sunday** | **Market Closed** | No weekend quotes or execution |

---

## 3. Mathematical Implications for Risk Governance

1. **Monetary Risk Formula**:
   $$\text{Monetary Risk (\$)} = \text{Volume (lots)} \times 100 \times |\text{Fill Price} - \text{Stop Price}| + \text{Slippage \& Commissions}$$
   For a micro lot ($0.01\text{ lot} = 1\text{ oz}$):
   $$\text{Risk per 0.01 lot} = 1.0 \times |\text{Fill Price} - \text{Stop Price}|$$
   Therefore, a $\$4.50$ stop distance costs **exactly \$4.50** of risk per $0.01$ lot.

2. **Equity Budget Constraints**:
   With a $1.0\%$ maximum risk ceiling:
   - On a $\$100$ account: $1.0\% = \$1.00$. Any stop wider than $\$1.00$ **exceeds the risk budget** and must be strictly **REJECTED** (no auto-bumping to 0.01 lot).
   - On a $\$500$ account: $1.0\% = \$5.00$. A $\$4.50$ stop is permitted at $0.01$ lot (\$4.50 risk = 0.90% equity).
   - On a $\$1,000$ account: $1.0\% = \$10.00$. Sizing allows $0.02$ lots for a $\$4.50$ stop (\$9.00 risk = 0.90% equity).

3. **Spread Drag on Scalping / M5**:
   With a floating spread of $\sim \$0.42$ per ounce, entering and exiting immediately incurs $\$0.42$ per $0.01$ lot. For an average take-profit of $\$1.50$ to $\$3.00$, the spread alone consumes $14\%$ to $28\%$ of gross trade expectancy. Strategies must clear a minimum gross edge of at least $2.0 \times \text{spread}$ ($>\$0.85$) to break even after slippage and commissions.

4. **Yahoo Futures (`GC=F`) vs Spot XAUUSD Separation**:
   Yahoo `GC=F` reflects CME Comex Gold Futures (often trading at a basis premium/discount, differing holiday calendars, and different tick regimes). In `src/live_data.py`, `GC=F` must never be conflated with spot broker feed without explicit labeling.
