# Latent Market Twin v2 (internal codename: v3) — Experiment With Everything

> Published on GitHub as **latent_market_twin_v2**. Inside this monorepo the folder is `v3/` because it came after the v2 overlay lab; the public name is v2 since it's the second published generation of the system.

v3 asks the big structural questions v1 and v2 couldn't: does *more data* help the model, does the latent space have a better size than the 16 dims v1 picked, is the top-20 portfolio the right size, and which hedge actually earns its keep? One walk-forward CV per model branch + a 1,060-variant construction grid answers all of them on the same 114 out-of-sample months (Dec 2016 – May 2026).

## How it works, end to end

The system predicts which U.S. large caps will outperform next month, using the idea that a stock's current *state* has historical analogs whose outcomes are informative. The pipeline has five stages:

**1. Data (`download_v3_data.py`).** Daily OHLCV for ~504 current large caps plus 12 macro tickers (VIX, treasury yields, TLT, GLD, HYG, LQD, QQQ, IWM, SPY) from yfinance, 2013 to present.

**2. Features (`build_v3_features.py`).** Everything is resampled to one row per stock per month, with only trailing (backward-looking) windows: momentum, moving-average ratios, drawdowns, volatilities, plus the new v3 groups — liquidity, range-based volatility, return shape, 52-week position, market beta/correlation, and month-level macro context shared by all stocks.

**3. The "market twin" (`build_v3_latents.py`).** All ~200 state features per stock-month are standardized and compressed by PCA into a small latent vector — a compact description of "what kind of situation this stock is in right now." For each stock-month, the 50 nearest neighbors in latent space are found **among earlier months only, with already-realized outcomes**, and their subsequent returns are summarized into features like `neighbor_avg_future_1m_return` ("stocks that looked like this went on to do X"). That's the analog-forecasting core of the project.

**4. Ranking model (`run_v3_cv.py`).** Two LightGBM LambdaRank models are trained fresh every month on all history strictly before that month: an *original* branch (all engineered features) and a *neighbor* branch (only the analog features). Each scores every stock for the coming month. Because training never sees the scored month, every month is a true out-of-sample test — expanding-window walk-forward cross-validation. All 503 scores are saved every month.

**5. Portfolio construction (`run_v3_grid.py`).** From the saved rankings: take the top N from each branch, weight positions by inverse 12-month volatility, blend the two branches, optionally apply a hedge that cuts exposure to 40% in stress regimes, charge 10 bps on all turnover, and realize next month's returns. Because construction runs on saved scores, a 1,060-variant sweep over sizes/blends/hedges costs minutes, and every variant is evaluated on identical out-of-sample months — so differences between variants are attributable to construction, not luck-of-the-retrain.

## What's new in v3

### 1. Expanded dataset (33 new features, all from yfinance)

`download_v3_data.py` pulls full daily **OHLCV** for the 504-ticker universe (v1 only kept adjusted closes) plus 12 macro/hedge tickers. `build_v3_features.py` turns that into:

| Group | Features (prefix `v3_`) |
|---|---|
| Liquidity | log dollar volume, 3m/12m volume trend, Amihud illiquidity |
| Range volatility | Parkinson vol (1m/3m), intramonth high-low range, downside vol |
| Return shape | daily skew & kurtosis (3m), MAX/MIN daily move (lottery demand) |
| Price position | % from 52w high/low, price vs 50d/200d MA, MA-cross state |
| Market relation | rolling 12m beta to SPY, 6m correlation, 6m idiosyncratic vol |
| Macro context (`v3m_`) | VIX level/change/z-score, 10y & 13w yields, term spread, TLT/GLD/HYG returns, credit appetite (HYG−LQD), HYG drawdown, QQQ−IWM, SPY vs 10m MA |

Macro columns are constant within a month; the ranker uses them as split *context* (regime-conditional stock picking).

### 2. Latent dimension sweep

The stock-state PCA (now over 203 state features) was rebuilt at **4 / 8 / 16 / 32** dims (explained variance 16% / 22% / 31% / 41%), with point-in-time neighbor features rebuilt for each (`build_v3_latents.py`).

### 3. Rankings-first CV → construction becomes free

`run_v3_cv.py` retrains monthly exactly like the live system but saves **every stock's score every month** for six branches (2 feature sets + 4 neighbor dims). `run_v3_grid.py` then sweeps portfolio construction as pure post-processing: sizes (10/20/35/50 × 5/10/20), blends (70/30, 50/50, 100/0, 0/100), and five hedges (none, SPY-trend, VIX>30, HYG-credit-drawdown, trend-or-credit) — 1,060 variants in ~3 minutes.

## Results

Baselines on the same months: **v1 = 17.4% ann / 0.92 Sharpe / -31.7% DD; SPY = 15.3% / 0.99 / -23.9%.**

### The marginal analysis (trust this, not the single best cell)

Averaged over all other grid settings, per dimension:

| Question | Answer | Evidence (mean net Sharpe) |
|---|---|---|
| Do the new features help? | **Yes, modestly but consistently** | v3feats 1.010 vs v1feats 0.992, and worst-fold Sharpe 0.44 vs 0.36 |
| Best latent dimension? | **8 (v1's 16 was the worst choice)** | dim8 1.034, dim32 1.021, dim4 0.945, dim16 0.956 |
| Best portfolio size? | **~35 stocks, not 20** | top35 1.033 vs top20 0.985; also shallower drawdowns |
| Is the neighbor branch worth it? | **Only as a 30% satellite** | 70/30 blend 1.018 ≈ pure-original 1.018; pure-neighbor 0.80 — the latent sleeve alone is weak, but a 70/30 blend keeps its diversification without its drag |
| Which hedge earns its keep? | **Credit (HYG drawdown < -5%)** | credit 1.021 with mean maxDD -26%; none 1.017 but maxDD -30%; SPY-trend and VIX hedges *reduce* Sharpe (0.95, 0.97) |

### Recommended v3 config (picked by marginal winners, then looked up — not cherry-picked; it ranks #72/1060)

`original_v3feats · top 35 · neighbor dim8 top 10 · 70/30 · credit hedge`

| | v3 recommended | v3 same, unhedged | v1 | SPY |
|---|---|---|---|---|
| Annualized return | **19.7%** | 25.2% | 17.4% | 15.3% |
| Sharpe | **1.10** | 1.12 | 0.92 | 0.99 |
| Max drawdown | **-24.8%** | -27.6% | -31.7% | -23.9% |
| Worst-fold Sharpe | **0.56** | 0.69 | ~0.40 | — |

The out-of-sample Sharpe finally clears SPY, with a drawdown profile close to the index. The single best grid cell (v3feats/35/dim32-10/70-30/credit) shows 19.6% at 1.16 Sharpe — treat that as an optimistic bound, not an expectation.

### Honest caveats

- **Selection pressure**: even reading only marginals, this grid saw the test data 1,060 times. The recommended config's true forward Sharpe is likely lower than 1.10. The dim8-beats-dim16 and 35-beats-20 findings are backed by 250+ variants each, so they're the most defensible takeaways.
- **All v1 caveats still apply** — survivorship-biased universe (the biggest flatterer), full-panel PCA fit, cash at 0%, rf=0 Sharpe, yfinance data quality.
- **Macro history is short**: HYG/VIX-based hedges were only tested through two real stress events (2020, 2022).

## Reproduce

```powershell
cd C:\latent_market_twin
.\.venv312\Scripts\python.exe v3\download_v3_data.py     # ~3 min, needs internet
.\.venv312\Scripts\python.exe v3\build_v3_features.py    # ~1 min
.\.venv312\Scripts\python.exe v3\build_v3_latents.py     # ~7 min
.\.venv312\Scripts\python.exe v3\run_v3_cv.py            # ~15 min, checkpointed
.\.venv312\Scripts\python.exe v3\run_v3_grid.py          # ~3 min
```

Outputs in `outputs/v3_backtest/`: `rankings_*.csv` (full monthly scores per branch), `v3_grid_results.csv` (all 1,060 variants), `v3_marginal_analysis.csv`, `v3_report.txt`.

## What v4 should do

1. **Freeze the v3 recommended config and paper-trade it live** next to LTSAF_live_v1 — the only test that escapes the selection-pressure caveat.
2. **Point-in-time universe** (still the biggest lie in every number here).
3. **Rank buffer** for turnover (v2's roadmap #1, now trivial: full rankings are already saved).
4. Walk-forward PCA + neighbor rebuild per month, now that dim8 makes it 2× cheaper.
