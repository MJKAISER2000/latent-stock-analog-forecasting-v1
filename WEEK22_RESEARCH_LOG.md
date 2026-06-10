\# Week 22 Research Log — Live Latent Twin Paper-Trading System



\## Main Goal



Week 22 moved the Latent Twin Stock Analog Forecasting project from a backtest/paper-trading prototype into a partially automated live forward-testing system.



The main objective was not to improve the model itself. The objective was to make the model operational:



\- repeatable

\- auditable

\- automated

\- separated between frozen research files and live rebuilt files

\- ready for forward paper testing



\---



\## Final Live Model Name



LTSAF\_live\_v1



This stands for:



Latent Twin Stock Analog Forecasting — Live Version 1



\---



\## High-Level System



The project now has two separate systems.



\### 1. Frozen Research / Paper-Trading Baseline



Config:



configs/final\_model\_config.yaml



Output directory:



outputs/paper\_trading/



Purpose:



This is the original validated research/paper-trading version. It uses the frozen Week 15 / Week 20 processed data files.



This system is useful as the reference baseline.



\---



\### 2. Live Rebuilt Pipeline



Config:



configs/live\_model\_config.yaml



Output directory:



outputs/paper\_trading\_live/



Live processed files:



data/processed/live\_500\_daily\_prices.parquet  

data/processed/live\_500\_monthly\_prices.parquet  

data/processed/live\_500\_monthly\_returns.parquet  

data/processed/live\_full500\_modeling\_dataset.parquet  

data/processed/live\_stock\_state\_pca\_latents\_with\_metadata.parquet  

data/processed/live\_stock\_latent\_neighbor\_features.parquet  

data/processed/live\_full500\_with\_stock\_latent\_neighbors.parquet  



Purpose:



This is the live forward-testing candidate. It rebuilds the feature stack from refreshed market data and generates a live monthly signal.



\---



\## Current Live Rebuild Chain



The live chain is:



1\. Download daily adjusted prices from yfinance.

2\. Resample daily adjusted closes to completed month-end prices.

3\. Build live monthly returns.

4\. Build live base stock feature dataset.

5\. Build live stock-state PCA coordinates.

6\. Build live latent-neighbor analog features.

7\. Train the live original ranker branch.

8\. Train the live latent-neighbor ranker branch.

9\. Blend both branches into a final live portfolio.

10\. Apply the tech drawdown regime filter.

11\. Save live target weights, orders, and ledgers.



\---



\## Why Monthly Prices Use Completed Months Only



The live price downloader excludes the incomplete current month.



This is important because yfinance can provide partial current-month data. If that partial month is labeled as month-end data, the model would accidentally treat an unfinished month as complete.



The live system now uses:



include\_incomplete\_current\_month = False



This means, for example:



If today is inside June, the latest completed month is May 31.



This prevents lookahead / partial-period contamination.



\---



\## Live Price Validation



We compared live rebuilt prices against the frozen research monthly prices.



After switching from yfinance monthly bars to daily adjusted prices resampled manually to month-end, the comparison improved sharply.



Final result:



\- Median latest absolute percent difference: 0.0

\- Mean latest absolute percent difference: about 0.064%

\- Tickers with latest difference above 5%: 1



The main outlier was FX, likely due to a vendor/ticker adjustment issue.



Conclusion:



The live price refresh is good enough to use as the live data foundation, while still keeping the frozen research data separate.



\---



\## Live Base Feature Dataset



Created:



data/processed/live\_full500\_modeling\_dataset.parquet



Observed shape:



72683 rows  

182 columns  



Date range:



2013-12-31 to 2026-05-31



Ticker count:



503



Important note:



The latest month has NaN values for:



\- future\_1m\_return

\- future\_1m\_spy\_return

\- future\_1m\_excess\_return

\- ranking\_label



This is expected and correct. The latest month is the month being scored, so future returns are not known yet.



\---



\## Live Stock-State PCA



Created:



data/processed/live\_stock\_state\_pca\_latents\_with\_metadata.parquet



The stock-state PCA maps each stock-month into a lower-dimensional latent state.



The PCA representation is used for nearest-neighbor analog search.



PCA result:



16 components explained about 27.4% cumulative variance.



This is acceptable because the PCA is not meant to perfectly reconstruct the full feature table. It is used as a compact stock-state coordinate system for analog matching.



\---



\## Live Latent-Neighbor Features



Created:



data/processed/live\_stock\_latent\_neighbor\_features.parquet  

data/processed/live\_full500\_with\_stock\_latent\_neighbors.parquet  



For each stock-month, the system finds similar historical stock states and summarizes their future outcomes.



Main live neighbor features:



\- neighbor\_count

\- neighbor\_distance\_mean

\- neighbor\_distance\_median

\- neighbor\_distance\_min

\- neighbor\_avg\_future\_1m\_return

\- neighbor\_median\_future\_1m\_return

\- neighbor\_avg\_future\_1m\_excess\_return

\- neighbor\_outperform\_spy\_1m\_rate

\- neighbor\_positive\_1m\_return\_rate



Latest date rows had:



neighbor\_count = 50



This means the live analog system is working and providing full neighbor sets.



\---



\## Live Final Pipeline



Created:



scripts/run\_live\_final\_pipeline.py



This uses:



configs/live\_model\_config.yaml



and outputs to:



outputs/paper\_trading\_live/



The live final pipeline trains:



Original branch:



original\_only features → LightGBM LambdaRank → top20



Latent-neighbor branch:



neighbor\_only features → LightGBM LambdaRank → top10



Final portfolio:



70% original branch  

30% latent-neighbor branch  

inverse-volatility weighting  

tech drawdown 20% risk filter  



\---



\## Latest Live Signal



Latest live signal date:



2026-05-31



This differs from the frozen research signal date because the live system only uses completed months.



Top live names from the latest generated portfolio included:



\- ROP

\- TYL

\- PPL

\- CNP

\- NI

\- WEC

\- AEE

\- PEG

\- CMS

\- EXC

\- TRMB

\- FE

\- SRE

\- ATO

\- EP

\- KLAC

\- XEL

\- ES

\- ETX

\- KEYS



\---



\## Live Persistent Ledgers



Added persistent ledgers for the live pipeline:



outputs/paper\_trading\_live/live\_portfolio\_signals.csv  

outputs/paper\_trading\_live/live\_run\_summary.csv  

outputs/paper\_trading\_live/live\_order\_ledger.csv  



These are separate from the frozen system ledgers.



This is important because the live system should be forward-tested independently.



\---



\## Live Rebuild Wrapper



Created:



scripts/run\_live\_rebuild\_pipeline.py



This runs the full live rebuild chain:



1\. make live config

2\. refresh live monthly prices

3\. compare live vs research prices

4\. build live base features

5\. build live stock-state PCA

6\. build live latent-neighbor features

7\. compare live vs research dataset

8\. run live final pipeline



One-command live rebuild:



python scripts\\run\_live\_rebuild\_pipeline.py



Batch file:



run\_live\_rebuild.bat



\---



\## Scheduled Automation



Created scheduled task:



LatentMarketTwinLiveMonthlyRebuild



This runs:



run\_live\_rebuild.bat



Schedule:



Monthly on the 2nd day of the month at 9:30 AM.



Reason:



Running on the 2nd gives market data time to settle after month-end.



\---



\## Frozen vs Live Signal Comparison



Created:



scripts/compare\_frozen\_vs\_live\_signals.py



This compares:



outputs/paper\_trading/paper\_portfolio\_signals.csv



against:



outputs/paper\_trading\_live/live\_portfolio\_signals.csv



The frozen signal date was:



2026-06-30



The live signal date was:



2026-05-31



Because these dates differ, the portfolios are not expected to match exactly.



Overlap names included:



\- DELL

\- SMCI

\- LITE

\- DDOG

\- CIEN

\- WDC



Conclusion:



The live signal is structurally working and not detached from the frozen system, but it is different enough to treat as its own forward-test model.



\---



\## Daily Live Value Tracking



Created daily live value checking through:



scripts/check\_live\_portfolio\_value.py



Batch file:



run\_daily\_value\_check.bat



Scheduled task:



LatentMarketTwinDailyValueCheck



This checks live holdings value and compares against SPY.



Outputs:



outputs/paper\_trading/live\_portfolio\_value\_snapshots.csv  

outputs/paper\_trading/latest\_live\_portfolio\_value\_summary.txt  

outputs/paper\_trading/latest\_live\_portfolio\_value\_detail.csv  

outputs/figures/live\_portfolio\_vs\_spy.png  



\---



\## Monthly Frozen Paper-Trading Automation



The original frozen monthly automation remains:



LatentMarketTwinMonthlyUpdate



This runs the frozen research/paper system.



It is still useful as a baseline, but it should not be confused with the live rebuilt model.



\---



\## Current Automation Layers



\### Frozen Monthly Update



Task:



LatentMarketTwinMonthlyUpdate



Purpose:



Runs the frozen paper-trading pipeline.



\---



\### Daily Live Value Check



Task:



LatentMarketTwinDailyValueCheck



Purpose:



Marks holdings to live prices and updates portfolio-vs-SPY chart.



\---



\### Live Monthly Rebuild



Task:



LatentMarketTwinLiveMonthlyRebuild



Purpose:



Refreshes live data, rebuilds features, rebuilds latent neighbors, and generates a live monthly signal.



\---



\## Important Research Decision



The live model should now be frozen as:



LTSAF\_live\_v1



Rule:



Do not keep changing model logic during the forward test.



Allowed changes:



\- bug fixes

\- logging improvements

\- reporting improvements

\- data-quality fixes



Not allowed during forward test:



\- changing the portfolio weights

\- changing the ranker objective

\- changing the neighbor construction

\- changing the regime filter

\- changing feature sets to chase performance



Reason:



If the model keeps changing, we will never know if the live system actually works.



\---



\## Forward-Test Rule



The live system should run for at least 3 to 6 months before making major strategy changes.



Track:



\- daily portfolio value

\- monthly returns

\- SPY comparison

\- drawdown

\- live signal overlap

\- best and worst holdings

\- regime status



The purpose is to determine whether the latent-neighbor analog signal has real forward predictive value.



\---



\## Current Status



Week 22 successfully created a live forward-testing infrastructure.



The project now has:



\- frozen research pipeline

\- live data refresh

\- live feature rebuild

\- live PCA latent state rebuild

\- live latent-neighbor rebuild

\- live final signal generation

\- live scheduled rebuild

\- live persistent ledgers

\- daily live value tracking

\- SPY comparison

\- frozen-vs-live comparison



The system is not production trading infrastructure, but it is now a real research-grade paper-trading system.



\---



\## Next Recommended Phase



Week 23 should focus on monitoring and reporting, not changing the model.



Recommended Week 23 tasks:



1\. Create a live dashboard summary.

2\. Add live portfolio performance ledger.

3\. Add live rebalance order generator.

4\. Add live holdings ledger separate from frozen holdings.

5\. Add automatic email/log summary after scheduled runs.

6\. Keep LTSAF\_live\_v1 frozen.

