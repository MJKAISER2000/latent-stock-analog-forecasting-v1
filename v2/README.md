# LTSAF v2 — Overlay Lab

v2 keeps the v1 model's signals untouched and experiments with everything that happens *after* the model speaks: hedging, buy-the-dip / sell-the-high tilts, volatility targeting, and trade filtering. The v1 walk-forward CV showed the model's stock picks add value in trending markets but the *portfolio construction* leaks money three ways — no fast market hedge (2022: -24% vs SPY -8%), no risk scaling, and ~2.1%/yr of transaction costs. v2 attacks those directly.

## Why overlays instead of a new model

The v1 CV backtest (`backtests/run_cv_backtest.py`) produced 114 months of strictly out-of-sample portfolio books (`outputs/cv_backtest/cv_holdings.csv`). Every v2 overlay transforms one month's book using only information available on that signal date, so:

- v2 results are **exactly as out-of-sample as v1's** — no retraining, no new leakage surface
- the whole variant grid runs in seconds, not minutes
- any improvement is attributable to construction, not to a different model getting lucky

## What's implemented

| Overlay | Idea | Config knobs |
|---|---|---|
| `dip_tilt` | **Buy dips, sell high, automatically.** Within the sleeve, overweight holdings below their own 6-month MA, trim extended ones. Exposure unchanged — only redistributes. | `strength`, `min_mult`, `max_mult` |
| `trend_hedge` | **Generic market hedge.** SPY below its 10-month MA → cut the stock sleeve to 40%, rest in cash. Much faster trigger than v1's tech-drawdown-20 filter (which went risk-off only 7/114 months). | `ma_months`, `exposure_when_off` |
| `vol_target` | Scale exposure so the strategy's own trailing 6-month realized vol approaches 15% annualized. No leverage. | `target_annual_vol`, `window_months`, `max_exposure` |
| `no_trade_band` | Lazy rebalance: skip trades smaller than 0.5% of the book vs the drifted position, cutting cost churn. | `band` |

Code: [overlays.py](overlays.py) (pure per-month functions) + [run_v2_backtest.py](run_v2_backtest.py) (sequential engine, variant grid, fold stats). Config: [v2_config.yaml](v2_config.yaml).

## Results (114 months, Dec 2016 – May 2026, net of 10 bps costs)

| Variant | Ann. return | Sharpe | Max DD | Vol | Fold Sharpe (mean ± std) |
|---|---|---|---|---|---|
| v1 baseline | 17.4% | 0.92 | -31.7% | 19.8% | 1.07 ± 0.67 |
| dip_tilt | 17.1% | 0.90 | -31.7% | 19.8% | 1.06 ± 0.67 |
| trend_hedge | 14.3% | 0.91 | **-26.6%** | 16.3% | 0.98 ± 0.69 |
| vol_target | 12.7% | 0.80 | -26.6% | 16.9% | 0.93 ± 0.75 |
| no_trade_band | **17.5%** | **0.92** | -31.8% | 19.8% | 1.08 ± 0.68 |
| v2 combined | 12.1% | 0.84 | **-25.2%** | **15.0%** | 0.90 ± 0.74 |
| SPY | 15.3% | 0.99 | -23.9% | 15.7% | 1.13 ± 0.60 |

### Honest read

- **The baseline row reproduces v1 exactly** (17.43% / 0.92), which validates the overlay engine end-to-end.
- **`dip_tilt` doesn't help.** Buying dips within an already momentum-flavored top-20 slightly fights the model's signal. Not worth it at any strength before parameter-tuning turns into overfitting.
- **`trend_hedge` is insurance, not alpha**: it cuts the worst drawdown by 5 points (-31.7% → -26.6%) and vol by 3.5 points, at a price of ~3.1%/yr of return. Sharpe is a wash (0.91). Take it if you care about 2022-style years; skip it if you only care about compounding.
- **`vol_target` hurts here** — with monthly data, a 6-month vol window reacts after the damage and then de-risks into recoveries.
- **`no_trade_band` is free money but only a little** (+0.1%/yr). The real turnover isn't small weight tweaks — it's the model replacing most of the top-20 every month. Fixing that needs a *rank buffer* (see roadmap), not a weight band.
- **The combined stack lowers risk, not risk-adjusted return.** Nothing in this overlay set beats SPY's 0.99 Sharpe. The construction layer wasn't v1's real problem — the model's regime-dependence is.

## Run it

```powershell
cd C:\latent_market_twin
.\.venv312\Scripts\python.exe backtests\run_cv_backtest.py   # v1 books (only needed once)
.\.venv312\Scripts\python.exe v2\run_v2_backtest.py          # the whole variant grid, ~5s
```

Outputs in `outputs/v2_backtest/`: `v2_variant_comparison.csv`, `v2_monthly_returns.csv` (every variant, every month, with exposure/turnover/trend state), `v2_report.txt`. Toggle or tune overlays in `v2_config.yaml` and rerun.

## Roadmap — ranked by expected impact

1. **Rank buffer / holding inertia (turnover killer).** Buy into the top 20, but don't sell until a holding falls out of the top ~40. This attacks the actual cost source (monthly top-20 replacement, ~1.5x two-way turnover). Needs per-month *full rankings* saved, so it requires a small change to the v1 backtest to persist scores for all 503 names, then it slots in as an overlay.
2. **Point-in-time universe.** The current-constituent list flatters every number in this project. Rebuilding with historical S&P constituents would give the first trustworthy absolute numbers — and might change which ideas look good.
3. **Regime-aware branch weights.** The 70/30 original/neighbor blend is static. Let the blend shift with the trend state (e.g. neighbor branch weight up in choppy tapes) — cheap to test as an overlay since branch membership is in the v1 branch detail.
4. **Walk-forward PCA latents.** Refit the scaler+PCA only on data before each signal date to remove the last structural lookahead. Expensive (rebuild neighbors per month) but makes the latent branch defensible.
5. **Cash earning T-bill yield.** Free realism, matters for the hedged variants that sit in cash.
6. **Better hedge instrument.** Instead of cash, hedge risk-off months with long TLT/GLD sleeves or partial SPY short — needs those price histories added to the data layer.

A warning to future-us: the overlay grid makes it very easy to try 50 parameter combos and pick the best. That's in-sample selection on the test set. Keep the discipline: pick parameters on priors (like the 10-month MA — a 100-year-old rule), run once, report whatever comes out.
