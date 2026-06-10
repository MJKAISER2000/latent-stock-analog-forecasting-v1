# Week 22 Live-Test Readiness Checklist

## Goal

Make the Latent Twin Stock Analog Forecasting system ready for forward paper testing.

The model is already built. Week 22 is about making the process repeatable, auditable, and safe from silent errors.

---

## Final Model

Current final model:

70% original_only top20  
30% neighbor_only top10  
inverse-volatility weighting  
tech drawdown 20% → cash regime filter

Main command:

python scripts\run_final_pipeline.py

---

## Required before serious forward testing

### 1. Data Refresh Pipeline

Need a reliable way to update:

- monthly adjusted close prices
- stock universe
- processed modeling dataset
- stock latent-neighbor features
- latest signal date

Target future command:

python scripts\refresh_data_pipeline.py

or eventually:

python scripts\run_monthly_update.py

Status: not built yet.

---

### 2. Holdings Ledger

Need a persistent file that stores current paper holdings.

Target file:

outputs/paper_trading/current_paper_holdings.csv

Should include:

- ticker
- shares
- average entry price
- current price
- market value
- current weight
- cash
- last_updated

Status: not built yet.

---

### 3. Rebalance Orders

Current order sheet assumes starting from cash.

Need a rebalance sheet that compares:

current holdings vs target holdings

and outputs:

- buy shares
- sell shares
- hold shares
- target shares
- current shares
- trade dollar amount
- estimated cash after rebalance

Target file:

outputs/paper_trading/paper_rebalance_orders_<timestamp>.csv

Status: not built yet.

---

### 4. Performance Ledger

Need to track whether the signal worked after each month.

Target file:

outputs/paper_trading/paper_performance_ledger.csv

Should include:

- signal date
- evaluation date
- portfolio return
- SPY return
- excess return
- portfolio value
- drawdown
- best holding
- worst holding
- win/loss month

Status: not built yet.

---

### 5. Sanity Report

Every run should print and save:

- signal date
- risk-on status
- tech drawdown
- number of selected tickers
- largest weight
- cash weight
- sector concentration
- overlap between original and latent-neighbor branches
- missing prices
- leftover cash
- top holdings

Target file:

outputs/paper_trading/latest_sanity_report_<timestamp>.txt

Status: partially built through summary files, but not complete.

---

### 6. One-Command Monthly Update

Eventually want:

python scripts\run_monthly_update.py

This should:

1. refresh or verify data
2. run final pipeline
3. update target portfolio
4. create rebalance orders
5. update performance if a prior signal can be evaluated
6. save sanity report

Status: not built yet.

---

## Ready-to-paper-test definition

The system is ready for serious paper testing when one command produces:

- latest signal
- final target weights
- rebalance orders
- updated holdings
- performance ledger
- sanity report

without manual editing.

---

## Week 22 Build Order

1. Add holdings ledger.
2. Add rebalance order generator.
3. Add performance tracker.
4. Add sanity report.
5. Add monthly runner.
6. Add data refresh plan.