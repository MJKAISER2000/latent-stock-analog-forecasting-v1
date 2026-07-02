# LTSAF Live Paper Trading — Results Log

**Strategy:** LTSAF_live_v1 — LightGBM LambdaRank stock selector (original ranker + stock latent-neighbor ranker, 70/30 blend), inverse-volatility weighting, technology-relative-strength / drawdown regime filter.

**Universe:** ~500 U.S. large-cap equities
**Starting capital:** $100,000 (fully simulated / paper)
**Inception:** 2026-05-31
**Rebalance:** monthly, on the first month-end signal after each month close
**Execution:** whole shares only, at live prices on the execution date; leftover held as cash

> This is a paper-trading experiment, not investment advice. Results are simulated, use current-constituent universes (survivorship bias), and cover a very short live window — treat them as a research log, not a track record.

---

## Month 1 — May 31 → June 30, 2026

The inception portfolio was defensive and utility-heavy (the regime/ranking blend favored low-volatility names at the May month-end).

| Metric | Value |
|---|---|
| Starting value (May 31) | $100,000.00 |
| Ending value (Jun 30) | $102,312.35 |
| **Return** | **+2.31%** |
| SPY return (same period) | −1.03% |
| **Excess vs SPY** | **+3.34 pts** |
| Beat SPY | ✅ Yes |

*Return is the realized month-end mark of the actually-held portfolio (30 positions + cash), valued at June-30 close prices, vs. a same-start SPY buy-and-hold.*

### Month 1 holdings (executed at May 31)

| Ticker | Shares | Entry $ | Market Value $ | Weight |
|---|--:|--:|--:|--:|
| TRMB | 113 | 56.41 | 6,374.33 | 6.37% |
| DUK | 39 | 122.73 | 4,786.47 | 4.79% |
| AWK | 37 | 123.27 | 4,560.99 | 4.56% |
| SO | 49 | 92.05 | 4,510.45 | 4.51% |
| PPL | 127 | 35.39 | 4,494.53 | 4.49% |
| EVRG | 52 | 82.04 | 4,266.08 | 4.27% |
| ED | 40 | 105.63 | 4,225.20 | 4.23% |
| CNP | 98 | 42.26 | 4,141.48 | 4.14% |
| NI | 89 | 46.22 | 4,113.58 | 4.11% |
| WEC | 35 | 111.05 | 3,886.75 | 3.89% |
| GEV | 4 | 968.32 | 3,873.28 | 3.87% |
| LNT | 53 | 71.61 | 3,795.33 | 3.80% |
| CMS | 51 | 72.57 | 3,701.07 | 3.70% |
| EXC | 81 | 45.64 | 3,696.84 | 3.70% |
| PEG | 47 | 78.65 | 3,696.55 | 3.70% |
| FE | 73 | 46.39 | 3,386.47 | 3.39% |
| KEYS | 10 | 338.33 | 3,383.30 | 3.38% |
| XEL | 35 | 79.50 | 2,782.50 | 2.78% |
| ES | 40 | 68.27 | 2,730.80 | 2.73% |
| FICO | 2 | 1,250.59 | 2,501.18 | 2.50% |
| CIEN | 4 | 580.23 | 2,320.92 | 2.32% |
| WDC | 4 | 531.21 | 2,124.84 | 2.12% |
| PCG | 110 | 16.34 | 1,797.40 | 1.80% |
| VST | 11 | 160.23 | 1,762.53 | 1.76% |
| DDOG | 7 | 247.35 | 1,731.45 | 1.73% |
| LITE | 2 | 854.96 | 1,709.92 | 1.71% |
| AES | 114 | 14.67 | 1,672.38 | 1.67% |
| SMCI | 36 | 46.09 | 1,659.24 | 1.66% |
| DELL | 3 | 420.91 | 1,262.73 | 1.26% |
| CEG | 4 | 287.75 | 1,151.00 | 1.15% |
| **CASH** | — | — | 3,900.41 | 3.90% |

---

## Month 2 — buys for the June 30 signal (executed July 2, 2026)

At the June month-end the model rotated fully out of utilities into growth, financials/REITs, and tech. The portfolio was rebalanced at live July-2 prices, so each position's cost basis is its actual purchase price.

- **Signal date:** 2026-06-30
- **Execution date:** 2026-07-02
- **Redeployed value:** $101,563.23
- **Positions:** 29 stocks + cash
- **Cash weight:** 10.55%

| Ticker | Shares | Entry $ | Market Value $ | Weight |
|---|--:|--:|--:|--:|
| TYL | 21 | 318.09 | 6,679.89 | 6.58% |
| DIS | 62 | 98.18 | 6,087.47 | 5.99% |
| ADSK | 27 | 208.78 | 5,637.06 | 5.55% |
| AMT | 33 | 166.29 | 5,487.57 | 5.40% |
| SPGI | 12 | 437.07 | 5,244.78 | 5.16% |
| PYPL | 110 | 45.01 | 4,951.65 | 4.88% |
| META | 7 | 584.02 | 4,088.14 | 4.03% |
| A | 29 | 132.13 | 3,831.77 | 3.77% |
| ACN | 25 | 137.76 | 3,444.12 | 3.39% |
| KKR | 35 | 93.49 | 3,272.15 | 3.22% |
| CTSH | 77 | 42.03 | 3,236.69 | 3.19% |
| ARES | 28 | 114.87 | 3,216.36 | 3.17% |
| AZO | 1 | 3,180.88 | 3,180.88 | 3.13% |
| SBAC | 16 | 184.18 | 2,946.88 | 2.90% |
| PLTR | 21 | 131.28 | 2,756.88 | 2.71% |
| EPAM | 30 | 87.70 | 2,631.00 | 2.59% |
| WAT | 7 | 374.70 | 2,622.90 | 2.58% |
| HPQ | 118 | 21.88 | 2,581.84 | 2.54% |
| FICO | 2 | 1,258.06 | 2,516.12 | 2.48% |
| COHR | 7 | 330.34 | 2,312.41 | 2.28% |
| ANET | 14 | 159.61 | 2,234.54 | 2.20% |
| HPE | 52 | 41.02 | 2,133.04 | 2.10% |
| PSKY | 181 | 10.14 | 1,835.34 | 1.81% |
| FIX | 1 | 1,720.73 | 1,720.73 | 1.69% |
| APP | 3 | 531.27 | 1,593.81 | 1.57% |
| CRWD | 8 | 194.17 | 1,553.36 | 1.53% |
| SMCI | 48 | 26.85 | 1,289.04 | 1.27% |
| DDOG | 4 | 260.61 | 1,042.44 | 1.03% |
| LITE | 1 | 720.52 | 720.52 | 0.71% |
| **CASH** | — | — | 10,713.85 | 10.55% |

---

## Notes & methodology

- **Whole-share paper execution.** Targets are sized to portfolio value, then floored to whole shares; the remainder is held as cash. This is why cash weight drifts month to month.
- **Cost basis = actual purchase price.** Positions are entered at live prices on the execution date, so per-position P&L starts at zero on the day it's bought (not at the prior month-end signal price).
- **Benchmark.** SPY comparison is a same-starting-value buy-and-hold from inception (shares bought at the inception SPY price, held and marked at the current price).
- **Turnover.** The strategy fully rotated from a defensive Month-1 book to a growth/tech Month-2 book, reflecting the regime and cross-sectional ranking shift at the June month-end.

_Generated 2026-07-02._
