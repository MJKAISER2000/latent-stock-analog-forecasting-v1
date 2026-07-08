# Latent Market Twin (LTSAF)

A monthly-rebalanced U.S. large-cap equity strategy built on latent representation learning, run as a live paper-trading system with a real cash simulation. The core idea: compress each stock's engineered state each month into a PCA latent space (the "market twin"), find each stock-month's nearest historical analogs in that space, and use how those analogs actually performed as features alongside conventional momentum/volatility features in a two-branch LightGBM ranking model.

The live model is `LTSAF_live_v1` (`configs/live_model_config.yaml`). This README covers the strategy, its cross-validated backtest, and the live paper-trading system. The week-by-week research history that led here is in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Strategy at a glance

| Component | Choice |
|---|---|
| Universe | ~503 current U.S. large caps (`data/external/week15_500_stock_universe.csv`) |
| Rebalance | Monthly, at month-end signal dates |
| Model | Two independent LightGBM LambdaRank rankers, retrained monthly on all history |
| Original branch | All engineered features except `neighbor_*` — top 20 stocks, 70% of book |
| Latent-neighbor branch | Only `neighbor_*` latent-analog features — top 10 stocks, 30% of book |
| Position weighting | Inverse 12-month volatility within each branch (weights summed if a ticker is in both) |
| Regime filter | If Information Technology sector drawdown < -20%, go 100% cash |
| Transaction cost | 10 bps per dollar traded (config `portfolio.transaction_cost: 0.001`) |

## Cross-Validated Backtest Results

Expanding-window walk-forward cross-validation over **114 out-of-sample monthly signals (Dec 2016 – May 2026)**, retraining both branches every month, net of 10 bps transaction costs on realized turnover (avg 1.51x two-way per month). Full details in `outputs/cv_backtest/cv_backtest_report.txt`.

### Headline (net of costs)

| Metric | Strategy (net) | Strategy (gross) | SPY buy & hold |
|---|---|---|---|
| Annualized return | **17.4%** | 19.6% | 15.3% |
| Sharpe ratio (rf=0) | **0.92** | 1.01 | 0.99 |
| Annualized volatility | 19.8% | 19.8% | 15.7% |
| Sortino ratio | 0.80 | 0.83 | 0.88 |
| Max drawdown | **-31.7%** | -30.5% | -23.9% |
| Total return (9.5y) | 360% | 446% | 288% |
| Positive months | 65% | 67% | 70% |
| Information ratio vs SPY | 0.24 | 0.42 | — |
| Monthly hit rate vs SPY | 51% | 54% | — |

The honest read: the strategy compounds faster than SPY (+2.1%/yr net) but does it with more volatility and a deeper worst drawdown, so **on a risk-adjusted basis it is roughly a coin flip with the index** (Sharpe 0.92 vs 0.99, and that's *before* accounting for the survivorship bias below). Costs matter: 10 bps on ~1.5x monthly turnover eats ~2.1%/yr, which is most of the edge over SPY.

### Cross-validation folds (5 contiguous ~23-month folds, net)

| Fold | Window | Ann. return | Sharpe | Max DD | SPY ann. return | SPY Sharpe |
|---|---|---|---|---|---|---|
| 1 | Dec 2016 – Oct 2018 | 27.3% | 2.01 | -9.4% | 13.4% | 1.43 |
| 2 | Nov 2018 – Sep 2020 | 7.4% | 0.40 | -27.7% | 11.5% | 0.63 |
| 3 | Oct 2020 – Aug 2022 | 8.4% | 0.47 | -30.8% | 6.4% | 0.42 |
| 4 | Sep 2022 – Jul 2024 | 19.2% | 1.09 | -16.5% | 28.6% | 1.86 |
| 5 | Aug 2024 – May 2026 | 26.8% | 1.39 | -14.5% | 18.1% | 1.34 |

**Fold Sharpe: 1.07 ± 0.67.** That dispersion is the real finding of the cross-validation: performance is regime-dependent, not steady. The strategy shines in trending bull tapes (folds 1, 5) and struggles through the 2018–2022 chop — 2022 alone was **-24.2% net vs SPY's -8.2%**, and the tech-drawdown regime filter (risk-off in only 7 of 114 months) fired too late to save it.

### By calendar year (net total return vs SPY)

| Year | Strategy | SPY | | Year | Strategy | SPY |
|---|---|---|---|---|---|---|
| 2017 | +35.1% | +26.3% | | 2022 | **-24.2%** | -8.2% |
| 2018 | +12.0% | -2.4% | | 2023 | +22.3% | +20.6% |
| 2019 | +18.4% | +21.4% | | 2024 | +22.2% | +26.2% |
| 2020 | +16.7% | +17.2% | | 2025 | +27.3% | +16.3% |
| 2021 | +26.8% | +23.2% | | 2026 (5m) | +13.9% | +8.5% |

Beat SPY in 6 of 10 calendar years; the losses concentrate in one year (2022). Remember these numbers are **flattered by survivorship bias** (current-constituent universe — see limitations below), so treat the SPY-relative edge as an upper bound.

### What later cross-validation found (v2/v3 updates)

Follow-up experiments on the same out-of-sample months revised several of v1's design choices — the full evidence is in [v3/README.md](v3/README.md) (published separately as *latent_market_twin_v2*):

- **v1's 16 latent dimensions was the worst of the tested options.** 8 dims ranks best (mean net Sharpe 1.03 vs 0.96 for 16, averaged over 250+ construction variants).
- **20 stocks is too concentrated.** ~35 holdings improves both Sharpe and drawdowns.
- **The tech-drawdown-20 regime filter fired only 7 of 114 months and missed 2022.** A credit-stress hedge (HYG drawdown < -5% → cut exposure to 40%) protects better without costing Sharpe; SPY-trend and VIX hedges tested worse. The v2 overlay lab ([v2/README.md](v2/README.md)) also showed a dip-buying tilt *hurts* this strategy and vol-targeting reacts too slowly on monthly data.
- **More yfinance data helps modestly**: adding volume/liquidity, range-volatility, return-shape, and macro-context features lifts mean Sharpe from 0.99 to 1.01 and improves the worst CV fold noticeably.
- The best defensible v3 configuration reached **19.7% annualized / 1.10 Sharpe / -24.8% max drawdown** net — the first configuration whose out-of-sample Sharpe clears SPY (0.99) — with the caveat that the selection process saw the test window many times, so forward performance should be expected to be lower.

## Backtest methodology

The backtest lives in [backtests/run_cv_backtest.py](backtests/run_cv_backtest.py) and is an **expanding-window walk-forward cross-validation** — the standard out-of-sample validation for time-series strategies. For every signal month `t` in the test window it re-runs the live pipeline exactly as it would have run at `t`:

1. **Visibility cutoff.** Only rows with `date <= t` are used — feature imputation medians are computed on that visible slice only, never on future months.
2. **Train.** Both branches are trained on all months strictly before `t` using the same `src/models/rankers.py` code path the live system calls, including its 80/20 time-ordered train/validation split and early stopping.
3. **Score & build.** Month `t` is scored out-of-sample, and the portfolio is assembled by the same `src/paper_trading/portfolio.py` code the live system uses (branch blend, inverse-vol weights, tech-drawdown regime filter evaluated on prices up to `t` only).
4. **Realize.** The next month's return is computed from each holding's realized `future_1m_return`; cash earns 0%.
5. **Costs.** 10 bps is charged on two-way turnover against the drifted prior-month book, including the initial buy-in.

Because the model is retrained at every step and every scored month is strictly out-of-sample, all ~9.5 years of test months are honest out-of-sample observations. The test months are additionally split into 5 contiguous folds to show how Sharpe and annualized return vary across sub-periods — a single full-period Sharpe can hide the fact that all the performance came from one regime.

Run it yourself (checkpointed per month — safe to interrupt and resume):

```powershell
cd C:\latent_market_twin
.\.venv312\Scripts\python.exe backtests\run_cv_backtest.py            # full run
.\.venv312\Scripts\python.exe backtests\run_cv_backtest.py --fresh    # recompute from scratch
.\.venv312\Scripts\python.exe backtests\run_cv_backtest.py --limit-months 3   # smoke test
```

Outputs land in `outputs/cv_backtest/`: `cv_monthly_results.csv` (per-month gross/net/SPY returns, turnover, regime state), `cv_holdings.csv` (every position every month), `cv_overall_summary.csv`, `cv_fold_summary.csv`, `cv_yearly_summary.csv`, and a human-readable `cv_backtest_report.txt`.

## Known limitations (read before trusting the numbers)

These are real biases in the backtest, listed so nobody (including future-you) mistakes the numbers for tradeable truth:

- **Survivorship bias — the big one.** The universe is a *current-constituent* list of ~503 large caps. Every stock in the backtest is known, today, to have survived and stayed large. This inflates returns, especially in the earlier years. A point-in-time constituent universe would give lower, more honest numbers.
- **PCA latents are fit on the full panel.** `build_live_stock_state_pca.py` fits the scaler and PCA rotation once on all months, including future ones. The neighbor *outcomes* are point-in-time (each month only sees earlier months' realized returns), but the latent axes themselves embed some full-sample structure.
- **Signal-date execution.** The backtest assumes you trade at the month-end price the signal is computed from. The live system intentionally executes at live prices a day or two later, so live results will drift from backtest assumptions.
- **Cash earns 0%.** Risk-off months (100% cash) are credited nothing; using T-bill yields would slightly raise the risk-off periods' contribution.
- **Sharpe uses rf = 0** and monthly returns annualized by sqrt(12).
- **yfinance adjusted closes** — dividend adjustments are baked into returns, but data quality is retail-grade.

## How the live system works

The live system runs the same strategy forward with real order sheets and a $100,000 paper-cash simulation. All live artifacts live under `outputs/paper_trading_live/`; the older "frozen" research pipeline (`outputs/paper_trading/`, `run_monthly_update.py`) is a static baseline kept for comparison.

### 1. Universe and price data

- `scripts/refresh_live_monthly_prices.py` downloads daily adjusted closes for the whole universe (plus SPY) via `yfinance`, resamples to month-end, and computes monthly returns. Outputs: `data/processed/live_500_daily_prices.parquet`, `live_500_monthly_prices.parquet`, `live_500_monthly_returns.parquet`.
- `scripts/compare_live_vs_research_prices.py` is an optional sanity check that diffs the freshly downloaded live prices against the frozen research price history.

### 2. Feature engineering

`scripts/build_live_base_features.py` turns the raw price panel into one row per stock per month with:

- trailing return / moving-average / volatility / drawdown features per stock
- SPY market return and volatility features
- sector and industry one-hot features, plus sector-relative return/vol/drawdown features
- a per-month ranking label: stocks are bucketed by percentile of realized next-month return within each date, which is what the ranker is trained to reproduce

Output: `data/processed/live_full500_modeling_dataset.parquet`.

### 3. Latent stock-state model (the "market twin")

This is the actual latent-representation piece the project is named for:

- `scripts/build_live_stock_state_pca.py` takes the non-leakage engineered features for every stock-month, standardizes them, and compresses them into a handful of PCA latent dimensions describing each stock's "state" that month. Output: `data/processed/live_stock_state_pca_latents_with_metadata.parquet`.
- `scripts/build_live_latent_neighbors.py` then finds, for every stock-month, its nearest neighbors in that latent space among all *earlier* stock-months with already-realized outcomes, and derives neighbor-based features (e.g. `neighbor_avg_future_1m_excess_return`, `neighbor_outperform_spy_1m_rate`) summarizing how similar-latent-state stocks actually performed afterward. Output merged into `data/processed/live_full500_with_stock_latent_neighbors.parquet`.

### 4. Two-branch LightGBM ranker

`scripts/run_live_final_pipeline.py` trains two independent LightGBM LambdaRank models on all available history and blends them:

| Branch | Feature set | Selection | Weight |
|---|---|---|---|
| Original | every engineered feature except the `neighbor_*` columns | top 20 by score | 70% |
| Latent-neighbor | only the `neighbor_*` latent-neighbor features | top 10 by score | 30% |

Each branch is inverse-volatility weighted within itself (`vol_12m`), the two branch portfolios are combined (weights summed if a ticker appears in both), and then a **regime filter** is applied: if Information Technology sector drawdown is worse than -20%, stock weights are scaled down and the freed weight moves to cash (100% cash when the filter trips, per `regime_filter.cash_weight_when_off` in the config).

This step appends to `live_portfolio_signals.csv` (per-ticker target weight for the month), `live_run_summary.csv`, and `live_order_ledger.csv`.

### 5. Rebalance orders vs. execution — two separate steps

This distinction matters and was the source of a real bug: **generating a new signal does not move the portfolio.**

- `scripts/generate_live_rebalance_orders.py` compares `current_live_holdings.csv` against the newest target signal and writes proposed BUY/SELL orders (`latest_live_rebalance_orders.csv`) — it only computes what *should* change, it does not touch holdings.
- `scripts/apply_live_rebalance.py` is the step that actually executes the rebalance. It fetches **live prices right now** for every held and target ticker, values the current book at those live prices (the true dollar amount available to redeploy), buys the new target weights in whole shares at those same live prices, and puts the remainder in cash. The prior holdings file is backed up (`current_live_holdings.pre_rebalance_<timestamp>.csv`) before being overwritten. Executing at live "now" prices (not the signal's month-end price) matters so each position's cost basis is its real purchase price and since-entry P&L starts at zero, instead of silently including a few days of market movement between the signal date and the day you actually rebalance.

### 6. Daily valuation and the SPY benchmark

`scripts/check_ltsaf_live_value.py` is the daily mark-to-market job:

- Fetches a live quote (current price + previous close) for every held ticker and SPY.
- Computes day P&L / day return, and appends a row to `live_value_snapshots.csv`.
- Computes a **true buy-and-hold SPY benchmark anchored at inception**: the first time this runs it records the portfolio's starting value and SPY's price on the inception date to `spy_benchmark_anchor.csv`, then every subsequent run values "what if you'd bought SPY instead" purely from `inception_value * (spy_price_today / spy_price_at_inception)`. Because it only depends on the inception anchor and the current price, the cumulative SPY gap stays correct even across days where no snapshot was logged at all.
- If every stock quote fails (e.g. the machine had no internet when the scheduled task fired), the run is skipped entirely rather than writing a misleading cash-only snapshot that would show up as a value crash.

`scripts/track_ltsaf_live_performance.py` separately evaluates each completed signal-to-next-signal holding period against realized monthly prices and SPY, and maintains cumulative return/drawdown stats in `live_performance_ledger.csv`.

### 7. Dashboard

`dashboard/ltsaf_live_dashboard.py` (Streamlit) reads all of the CSV/parquet artifacts above and renders a Robinhood-style local UI: portfolio value vs. the inception-anchored SPY benchmark, every current position (not just the top few) with live Day % / Since-Entry % colored green/red, per-branch signal detail, rebalance orders, and a system-health page. Data is cached per-session — use the "Refresh dashboard cache" button after running any of the pipeline scripts.

### 8. Scheduling

Three Windows Task Scheduler jobs (`schtasks.exe`, set up via `setup_daily_value_task.ps1`, `setup_live_monthly_task.ps1`, `setup_monthly_task.ps1`) drive the live system. Their batch/PowerShell launchers resolve their own folder (`%~dp0` / `$PSScriptRoot`) so they keep working if the project directory moves.

| Task | Schedule | Runs | Does |
|---|---|---|---|
| `LatentMarketTwinDailyValueCheck` | Mon-Fri, 4:30 PM | `run_daily_value_check.bat` -> `check_ltsaf_live_value.py` | Daily mark-to-market + SPY gap |
| `LatentMarketTwinLiveMonthlyRebuild` | 2nd of month, 9:30 AM | `run_live_rebuild.bat` -> `run_live_rebuild_pipeline.py` | Full refresh + retrain + new signal + rebalance orders (does **not** execute the trade — see step 5) |
| `LatentMarketTwinMonthlyUpdate` | 1st of month, 9:00 AM | `run_monthly_update.bat` -> `run_monthly_update.py` | The older frozen research-baseline pipeline, kept separate from the live system |

All three are configured to catch up on a missed run and wake the machine from sleep, but still need an internet connection at run time and (unless the logon type has been switched to S4U) need the user to be logged in.

### 9. Running it manually

```powershell
cd C:\latent_market_twin

# Daily value refresh
.\.venv312\Scripts\python.exe scripts\check_ltsaf_live_value.py

# Full monthly rebuild: refresh prices, rebuild features/latents, retrain, generate new signal + orders
.\.venv312\Scripts\python.exe scripts\run_live_rebuild_pipeline.py

# Execute the newest signal into actual holdings (whole shares, live prices)
.\.venv312\Scripts\python.exe scripts\apply_live_rebalance.py

# Walk-forward CV backtest
.\.venv312\Scripts\python.exe backtests\run_cv_backtest.py

# Dashboard
.\.venv312\Scripts\python.exe -m streamlit run dashboard\ltsaf_live_dashboard.py
```

Key live outputs, all under `outputs/paper_trading_live/`: `current_live_holdings.csv` (what you actually hold), `live_portfolio_signals.csv` (monthly target weights), `latest_live_rebalance_orders.csv` (proposed trades), `live_value_snapshots.csv` and `spy_benchmark_anchor.csv` (daily value history + SPY benchmark), `live_performance_ledger.csv` (realized month-over-month performance).

## Repository layout

```
backtests/          Walk-forward CV backtest (run_cv_backtest.py, cv_metrics.py)
v2/                 v2 overlay lab: hedging, dip-tilt, vol targeting (see v2/README.md)
v3/                 v3 experiment grid: expanded yfinance features, latent-dim &
                    portfolio-size sweeps, hedge comparison (see v3/README.md)
configs/            Model configs (live_model_config.yaml is the live one)
dashboard/          Streamlit live dashboard
data/external/      Universe definition
data/processed/     Price panels, feature datasets, PCA latents, neighbor features
models/             Saved model artifacts
outputs/cv_backtest/         CV backtest results
outputs/paper_trading_live/  Live paper-trading state and ledgers
outputs/paper_trading/       Frozen research baseline
scripts/            Live pipeline steps + automation entry points
src/                Shared library code (loaders, features, rankers, portfolio, paper trading)
```

## Further reading

- [v3/README.md](v3/README.md) — the v3 experiment grid: 33 new yfinance features, latent-dimension and portfolio-size sweeps, hedge comparison — best out-of-sample config found so far
- [v2/README.md](v2/README.md) — the v2 overlay lab: hedging, buy-dips/sell-high tilt, vol targeting, results and roadmap
- [RESEARCH_LOG.md](RESEARCH_LOG.md) — the full week-by-week research history (universe sweeps, horizon pivot, latent experiments, model selection)
- [PAPER_TRADING_RESULTS.md](PAPER_TRADING_RESULTS.md) — actual month-by-month live paper-trading results and buy lists
- [LTSAF_LIVE_V1_FREEZE.md](LTSAF_LIVE_V1_FREEZE.md) — the frozen live model definition
- [PAPER_TRADING_RUNBOOK.md](PAPER_TRADING_RUNBOOK.md) — operational runbook
