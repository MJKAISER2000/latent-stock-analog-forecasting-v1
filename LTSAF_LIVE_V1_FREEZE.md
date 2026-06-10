\# LTSAF\_live\_v1 Freeze Note



\## Model Name



LTSAF\_live\_v1



\## Freeze Date



2026-06-03



\## Purpose



This document freezes the current live version of the Latent Twin Stock Analog Forecasting model for forward paper testing.



The goal is to prevent accidental strategy drift.



\---



\## Frozen Strategy Logic



The live strategy is:



70% original branch  

30% latent-neighbor branch  

inverse-volatility weighting  

tech drawdown 20% cash filter  

monthly rebalance  

completed-month data only  



\---



\## Original Branch



Feature set:



original\_only



Model:



LightGBM LambdaRank



Portfolio:



top20 stocks



Role:



Captures conventional stock features such as momentum, volatility, drawdown, sector, and industry patterns.



\---



\## Latent-Neighbor Branch



Feature set:



neighbor\_only



Model:



LightGBM LambdaRank



Portfolio:



top10 stocks



Role:



Captures analog information from similar historical stock states.



Mechanism:



stock state → PCA latent state → nearest historical latent states → future outcome summaries → ranker



\---



\## Final Ensemble



Branch weights:



original branch: 70%  

latent-neighbor branch: 30%  



Weighting method:



inverse volatility



Risk overlay:



If technology drawdown is below -20%, move to cash according to the configured cash weight.



\---



\## Data Policy



Use only completed monthly data.



Do not treat incomplete current-month prices as month-end data.



Current live data source:



yfinance daily adjusted close prices, resampled manually to completed month-end.



\---



\## Allowed Changes During Forward Test



Allowed:



\- bug fixes

\- logging improvements

\- reporting improvements

\- typo fixes

\- better error messages

\- data-quality warnings

\- file organization improvements



Not allowed:



\- changing branch weights

\- changing top\_n values

\- changing LightGBM objective

\- changing ranking label construction

\- changing latent-neighbor features

\- changing PCA dimensionality

\- changing regime filter rule

\- changing the stock universe to chase performance

\- changing the strategy after a bad month without logging it as a new version



\---



\## If Strategy Logic Changes



If any strategy logic changes, create a new model version:



LTSAF\_live\_v2



and document:



\- what changed

\- why it changed

\- when it changed

\- whether the change was based on forward-test performance

\- whether previous live results remain comparable



\---



\## Forward-Test Minimum



Minimum forward-test period:



3 months



Preferred forward-test period:



6 months



Primary benchmark:



SPY



Primary evaluation metrics:



\- monthly return

\- SPY return

\- excess return

\- cumulative value

\- drawdown

\- win rate vs SPY

\- best and worst holdings

\- regime status



\---



\## Current Status



LTSAF\_live\_v1 is frozen for live paper testing.



The model should now be monitored, not optimized.

