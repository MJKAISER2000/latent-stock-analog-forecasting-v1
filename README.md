# Latent Market Twin

This project tests whether latent representation learning can improve long-term stock prediction and portfolio construction.

## Core Research Question

Can latent twin or autoencoder-style models learn lower-dimensional market structures that improve one-year stock direction and market outperformance prediction?

## First Experiment

The first experiment uses monthly large-cap stock data from 2000 onward.

For each stock and month, the model predicts whether the stock outperforms SPY over the next 12 months.

## Main Target

target_outperform_spy = 1 if the stock's next 12-month return is greater than SPY's next 12-month return.

## Data Sources

Initial version uses free data:
- Yahoo Finance via yfinance
- FRED macroeconomic data from public CSV links

## Week 1 Completed

Built the project folder structure, virtual environment, config file, data download script, target construction script, market regime labeling script, and first processed datasets.

## Project Phases

1. Dataset skeleton
2. Feature engineering
3. Baseline models
4. Autoencoder latent representation
5. Latent space analysis
6. Backtesting engine
7. Latent dynamics / latent twins version
8. Final evaluation and research write-up

## Week 2 Completed

Created the first engineered feature dataset for the latent market twin experiment.

The dataset now includes:
- trailing stock returns
- moving average ratios
- stock-level volatility
- stock-level drawdown
- SPY market return features
- SPY volatility features
- bull/bear/correction/crash regime labels
- macroeconomic variables
- yield curve feature
- CPI year-over-year change
- sector one-hot features
- one-year SPY outperformance target

Main modeling file:

data/processed/modeling_dataset.parquet
## Week 3 Completed

Built baseline machine learning models to predict one-year SPY outperformance.

Models tested:
- logistic regression
- random forest
- gradient boosting

Main target:
- target_outperform_spy

Key findings:
- Random forest and gradient boosting produced test AUC values above random guessing.
- The model-ranked top stocks had much higher average future 12-month returns than SPY in the first test period.
- The ranking diagnostic showed: top-ranked stocks outperformed the average stock, and average stocks outperformed bottom-ranked stocks.

Important caution:
- This is still a small starter universe of about 20 large-cap stocks.
- The result is promising but not yet evidence of a real trading edge.
- Next steps should test whether latent autoencoder features can match or beat these raw-feature baselines.
## Week 4 Completed

Built the first autoencoder-based latent market model.

The autoencoder compressed the engineered stock and market feature space into lower-dimensional latent representations with dimensions:
- 2
- 5
- 10
- 20

Each latent representation was then tested using:
- logistic regression
- gradient boosting

Key findings:
- Reconstruction loss improved as latent dimension increased.
- The 20-dimensional latent representation preserved most of the useful predictive information from the raw feature set.
- The best latent model, latent_dim=20 with gradient boosting, achieved a top-5 average future 12-month return roughly comparable to or slightly better than the best raw-feature baseline.
- The latent model's top-ranked stocks outperformed the average stock, bottom-ranked stocks, and SPY in the test period.

Important conclusion:
The Week 4 result supports the core latent market twin hypothesis: high-dimensional stock and macroeconomic feature data can be compressed into a lower-dimensional latent representation while preserving meaningful stock-ranking structure.

Important caution:
This is still a small starter universe of large-cap stocks. The next step is to analyze the latent space visually and test whether it organizes stocks by sector, regime, risk, or future return.

## Week 5 Completed

Analyzed the 20-dimensional autoencoder latent space using PCA and t-SNE visualizations.

Generated figures:
- latent space PCA colored by sector
- latent space PCA colored by future return
- latent space PCA colored by return bucket
- latent space t-SNE colored by sector
- latent space t-SNE colored by future return
- latent space t-SNE colored by return bucket

Key findings:
- The latent space is mixed overall, but t-SNE shows clear sector clustering.
- Technology, communication, financials, healthcare, energy, consumer staples, and consumer discretionary stocks occupy visibly different latent regions.
- Future return separation is weaker than sector separation, but there are still visible pockets of high-return and low-return behavior.
- Several latent coordinates show measurable correlation with future 12-month return.
- The strongest observed correlations with future return were around latent dimensions such as z_18, z_7, and z_2.

Interpretation:
The autoencoder is not merely compressing noise. It appears to learn economically meaningful structure, especially sector-level market organization, while preserving some information related to future returns and SPY outperformance.

Important caution:
Because the current universe is small and large-cap-heavy, some of the apparent predictive structure may come from sector effects, especially technology and communication outperformance during the test period.
## Week 6 Completed

Built the first simplified backtesting engine for the latent market twin project.

Backtest setup:
- model used: latent_dim=20 autoencoder + gradient boosting classifier
- monthly rebalance
- top 5 and top 10 model-ranked stocks
- 0.1% transaction cost approximation per rebalance
- compared against SPY and an equal-weight all-stock benchmark

Key findings:
- The latent top-5 strategy produced the strongest simplified backtest result.
- The latent top-5 portfolio outperformed SPY and the equal-weight benchmark in the test period.
- The equity curve showed that the latent top-5 portfolio began separating strongly after mid-2022 and continued outperforming through 2023–2024.
- The top holdings were concentrated in a small group of major winners including NFLX, META, NVDA, TSLA, LLY, GOOGL, and AMZN.

Important interpretation:
The simplified backtest supports the idea that the latent model learned a useful stock-ranking signal. However, much of the performance came from repeated exposure to a concentrated set of large-cap growth and AI-related winners.

Important limitation:
This backtest is still simplified because it converts 12-month forward returns into approximate monthly returns. The next rigorous version should use true realized next-month returns after each rebalance date.
## Week 7 Completed

Built the first time-aware sequence model.

Model:
- 12-month feature sequence
- GRU neural network
- binary prediction target: one-year SPY outperformance

Key findings:
- The GRU achieved strong training performance but weak test performance, indicating overfitting.
- The GRU sequence model did not outperform the Week 4 static autoencoder latent model.
- The GRU portfolio beat SPY but failed to beat the equal-weight all-stock benchmark.
- The GRU selected more defensive and old-economy stocks such as XOM, WMT, PG, KO, UNH, and V.
- It underselected major 2023–2024 winners such as NVDA, which hurt performance.

Interpretation:
The first time-aware model was too flexible for the small starter universe and did not generalize well. The simpler static latent autoencoder model remains the strongest version so far.

Important conclusion:
More model complexity does not automatically improve market prediction. The current evidence favors simpler latent compression plus tree-based prediction over a GRU sequence model.
## Week 9 Completed

Expanded the latent market twin project beyond the original 20-stock starter universe.

New files created:
- data/external/expanded_ticker_universe.csv
- data/raw/expanded_stock_prices_raw.parquet
- data/processed/expanded_monthly_prices.parquet
- data/processed/expanded_targets.parquet
- data/processed/expanded_market_regimes.parquet
- data/processed/expanded_features.parquet
- data/processed/expanded_modeling_dataset.parquet
- outputs/reports/week9_universe_expansion_summary.txt

Key upgrade:
The project now has a larger NASDAQ-style stock universe, making the next experiments more reflective of real stock selection and less dependent on the original hand-picked 20-stock universe.

Important caution:
This expanded universe is still not fully survivorship-bias-free and remains tilted toward large NASDAQ-style companies. Future work should use historical index membership or survivorship-bias-aware datasets.
## Week 10 Completed

Added sector and industry metadata to the expanded stock universe.

New files created:
- data/external/expanded_ticker_metadata.csv
- data/processed/expanded_modeling_dataset_with_metadata.parquet
- outputs/reports/week10_sector_metadata_summary.txt
- outputs/figures/week10_sector_distribution.png
- outputs/reports/week10_sector_balance_diagnostic.txt
- outputs/tables/week10_sector_counts.csv

Key upgrades:
- Replaced manual sector labels with real ticker-level metadata where available.
- Added sector and industry labels.
- Added market-cap buckets.
- Added sector-relative features, including:
  - return minus sector average
  - return sector z-score
  - volatility minus sector average
  - volatility sector z-score
  - drawdown minus sector average
  - drawdown sector z-score

Key finding:
The expanded NASDAQ-style universe is heavily tilted toward Technology and related growth sectors. This is expected, but it means later performance must be tested against sector-neutral and sector-balanced benchmarks.

Important research implication:
Future results may partly come from sector allocation rather than stock-specific selection. Week 13 should explicitly test sector-neutral portfolios.
## Week 11 Completed

Built the first true next-month realized-return backtester.

New files created:
- data/processed/expanded_realized_monthly_returns.parquet
- outputs/reports/week11_realized_monthly_returns_summary.txt
- outputs/tables/week11_expanded_autoencoder_metrics.csv
- outputs/tables/week11_expanded_autoencoder_portfolio_signal.csv
- outputs/tables/week11_expanded_predictions_latent20_gb.csv
- outputs/tables/week11_realized_backtest_curves.csv
- outputs/tables/week11_realized_backtest_stats.csv
- outputs/tables/week11_predictions_with_realized_returns.csv
- outputs/figures/week11_realized_backtest_equity_curves.png
- outputs/reports/week11_realized_backtest_summary.txt

Key result:
In the expanded NASDAQ-style universe, the latent autoencoder model did not clearly outperform SPY or the equal-weight universe in a true next-month realized-return backtest.

Main backtest finding:
The top 5 and top 10 model portfolios produced decent returns but underperformed SPY and equal-weight on a risk-adjusted basis. Drawdowns were also larger than SPY.

Important interpretation:
The model may contain weak one-year ranking signal, but that signal does not translate cleanly into a monthly trading strategy.

Diagnostic finding:
The top-ranked portfolio was concentrated in names like TTD, TSLA, PYPL, MAR, and HON. The model sometimes picked good stocks, but the signal was inconsistent and heavily name-dependent.

Drawdown filter test:
A simple filter excluding stocks with drawdown worse than -30% did not improve results. This suggests the problem is not just excessive drawdown exposure.

Next research direction:
Train a model directly on next-month realized outperformance instead of using a 12-month outperformance model for one-month trading.

## Week 12 Progress: Horizon Alignment Pivot

Tested whether aligning the prediction target with the trading horizon improves performance.

Key pivot:
- The previous 12-month target did not translate well into monthly trading.
- A new next-month target was created: target_outperform_spy_1m.
- Gradient boosting trained directly on target_outperform_spy_1m produced the strongest current trading result.

Best current model:
- Model: gradient boosting
- Target: next-month SPY outperformance
- Portfolio: top 5 ranked stocks each month
- Result: top-5 strategy outperformed SPY and the equal-weight universe in the realized monthly backtest.

Important finding:
- The signal is concentrated in the highest-ranked stocks.
- Top 10, Top 25, and Top 50 portfolios were much less impressive.
- This suggests the model is useful mainly as a high-conviction selector, not as a broad market-wide classifier.

Additional tests:
- Momentum and drawdown filters did not improve the strategy.
- Removing PYPL slightly improved results, but this is ticker-specific and not a general rule.
- A top-quintile target was tested, but it did not outperform the next-month SPY-outperformance target.

Current conclusion:
The most promising working-model direction is a next-month gradient boosting ranking model using top-5 portfolio selection.
## Week 13 Completed

Ran a full horizon sweep to test which prediction horizon is most useful for the latent market twin project.

Horizons tested:
- 1 month
- 3 months
- 6 months
- 12 months
- 24 months
- 36 months

Key files created:
- data/processed/week13_horizon_sweep_targets.parquet
- outputs/tables/week13_horizon_model_metrics.csv
- outputs/tables/week13_horizon_portfolio_signal.csv
- outputs/tables/week13_overlapping_horizon_backtest_stats.csv
- outputs/tables/week13_overlapping_horizon_backtest_curves.csv
- outputs/figures/week13_overlapping_horizon_equity_curves.png
- outputs/reports/week13_overlapping_horizon_backtest_summary.txt
- outputs/tables/week13_two_system_backtest_stats.csv
- outputs/tables/week13_gated_two_system_backtest_stats.csv

Main finding:
The strongest current strategy is the 36-month structural model with a top-10 portfolio and 36-month overlapping holding period.

Best current core model:
- Model: gradient boosting
- Target: 36-month SPY outperformance
- Portfolio: top 10 stocks by predicted 36-month outperformance probability
- Holding method: 36-month overlapping monthly baskets
- Result: strongest annualized return among tested strategies

Important interpretation:
The horizon sweep suggests that market structure is more predictable at longer horizons than at short horizons. The 1-month model remains useful as a tactical high-conviction signal, but the 36-month model appears stronger as a core structural stock-selection model.

Two-system model tests:
- Weighted blend of 36-month structural score and 1-month tactical score did not beat the pure 36-month model.
- Gated two-system model also did not beat the pure 36-month model.
- The best architecture for now is to keep the 36-month structural model as the core model and treat the 1-month model as a separate tactical strategy.

Current leaderboard:
1. pure_36m_top10 overlapping portfolio
2. 1-month top-5 tactical strategy
3. gated_pool20_top5
4. weighted structural/tactical blend
5. SPY
6. equal-weight universe

Important caution:
The 36-month model beats SPY on return, but has larger drawdowns and does not clearly beat SPY on Sharpe ratio. The next research step should focus on risk control.
## Week 14 Completed

Ran risk-control experiments on the best horizon models.

Main goal:
Test whether risk controls could preserve strong returns while reducing drawdown.

Risk controls tested:
- 12-month volatility filter
- inverse-volatility weighting
- bear-market cash overlay
- combined volatility filter + inverse-volatility weighting
- cross-horizon risk comparison

Key finding:
Risk controls changed the model leaderboard.

Before risk control:
- The best model by raw return was the 36-month structural model, especially h36_top10.

After risk control:
- The 36-month top-10 model remained the strongest long-horizon return engine.
- The 1-month top-5 model with inverse-volatility weighting became the strongest return/drawdown efficiency strategy.

Current champion models:
1. Long-horizon return engine:
   - h36_top10_vol20
   - 36-month prediction horizon
   - top 10 portfolio
   - 36-month overlapping holding period
   - volatility filter: vol_12m < 0.20

2. Risk-adjusted tactical engine:
   - h1_top5_inverse_vol or h1_top5_vol20_inverse_vol
   - 1-month prediction horizon
   - top 5 portfolio
   - inverse-volatility weighting

Important interpretation:
The best model depends on the metric. If the goal is maximum long-term return, the 36-month model remains strongest. If the goal is return relative to drawdown, the 1-month inverse-volatility strategy becomes more attractive.

Important caution:
Risk controls improved the strategies but did not fully eliminate large drawdowns. More sophisticated risk controls may still be needed.

Next research direction:
Week 15 will expand the stock universe into higher-volatility names and test whether autoencoder latent features become more useful in a noisier, more nonlinear universe.
## Week 15 Completed

Expanded the project from the previous NASDAQ-style universe into a broader 500-stock universe and volatility-ranked subsets.

Main goal:
Test whether the model behaves differently on a broader and more volatile stock universe, and whether autoencoder latent features become more useful in noisier, more nonlinear data.

Universes created:
- full500: current S&P 500-style broad universe
- highvol100: top 100 most volatile stocks from the broad universe
- highvol200: top 200 most volatile stocks from the broad universe

Important limitation:
The broad universe uses current S&P 500 constituents, so it is not survivorship-bias-free. It is useful as a broad stress-test universe, but future research should use historical index membership.

Key files created:
- data/external/week15_500_stock_universe.csv
- data/processed/week15_500_monthly_prices.parquet
- outputs/tables/week15_500_volatility_rankings.csv
- data/external/week15_high_vol_100_universe.csv
- data/external/week15_high_vol_200_universe.csv
- data/processed/week15_full500_modeling_dataset.parquet
- data/processed/week15_highvol100_modeling_dataset.parquet
- data/processed/week15_highvol200_modeling_dataset.parquet
- outputs/tables/week15_raw_gb_model_metrics.csv
- outputs/tables/week15_raw_gb_portfolio_signal.csv
- outputs/tables/week15_ae_gb_model_metrics.csv
- outputs/tables/week15_ae_gb_portfolio_signal.csv
- outputs/tables/week15_raw_vs_ae_backtest_stats.csv
- outputs/tables/week15_raw_vs_ae_backtest_curves.csv
- outputs/figures/week15_raw_vs_ae_backtest_equity_curves.png
- outputs/reports/week15_raw_vs_ae_backtest_summary.txt

Main finding:
Expanding into a high-volatility universe improved upside potential, especially for short-horizon raw gradient boosting models.

Best raw-return model:
- highvol100_raw_gb_h1_top5_vol20_inverse_vol
- Universe: highvol100
- Model: raw gradient boosting
- Target: 1-month SPY outperformance
- Strategy: top 5 portfolio
- Risk control: 12-month volatility filter plus inverse-volatility weighting
- Approximate result: strongest annualized return among tested Week 15 strategies

Best risk-adjusted / structural model:
- full500_ae_gb_h36_top10_base
- Universe: full500
- Model: autoencoder latent features + gradient boosting
- Target: 36-month SPY outperformance
- Strategy: top 10 structural portfolio
- Important result: strong Sharpe and much lower drawdown than the high-volatility tactical strategies

Important interpretation:
Raw gradient boosting performed best in the high-volatility tactical setting, but the autoencoder became more interesting in the broad full500 universe. This suggests that latent representations may be more useful when the model sees a representative mix of market types rather than only high-volatility stocks.

Updated model view:
- Raw GB is currently strongest for high-volatility short-term tactical selection.
- AE + GB is promising for broad-market long-horizon structural selection.
- The autoencoder did not universally outperform raw GB, but it produced one of the strongest risk-adjusted structural results.

Next research direction:
Week 16 will build a more representative market universe by explicitly combining low-volatility, mid-volatility, and high-volatility stocks. The goal is to test whether autoencoder latent features improve when the universe contains contrast across volatility regimes, sectors, and market styles.
## Week 16 Completed

Ran a large-universe latent structure stress test using a 1000+ ticker broad U.S. stock universe.

Main goal:
Test whether autoencoder latent features improve when the universe contains a representative mix of low-volatility, mid-volatility, and high-volatility stocks.

Updated universe construction:
- Built a 1000+ ticker broad U.S. stock universe using currently listed U.S. equities.
- Downloaded monthly price data from 2014 onward.
- Ranked stocks by realized volatility.
- Created large volatility buckets:
  - lowvol300
  - midvol300
  - highvol300
  - balanced450
  - balanced900

Important limitation:
The 1000+ universe is not survivorship-bias-free and does not yet have full sector-quality metadata. It is a large stress-test universe, not a final production universe.

Key files created:
- data/external/week16_1000_stock_universe.csv
- data/processed/week16_1000_monthly_prices.parquet
- outputs/tables/week16_1000_volatility_rankings.csv
- data/external/week16_lowvol300_universe.csv
- data/external/week16_midvol300_universe.csv
- data/external/week16_highvol300_universe.csv
- data/external/week16_balanced450_universe.csv
- data/external/week16_balanced900_universe.csv
- data/processed/week16_balanced450_modeling_dataset.parquet
- data/processed/week16_balanced900_modeling_dataset.parquet
- data/processed/week16_lowvol300_modeling_dataset.parquet
- data/processed/week16_midvol300_modeling_dataset.parquet
- data/processed/week16_highvol300_modeling_dataset.parquet
- outputs/tables/week16_raw_gb_model_metrics.csv
- outputs/tables/week16_raw_gb_portfolio_signal.csv
- outputs/tables/week16_ae_gb_model_metrics.csv
- outputs/tables/week16_ae_gb_portfolio_signal.csv
- outputs/tables/week16_raw_vs_ae_backtest_stats.csv
- outputs/tables/week16_raw_vs_ae_backtest_curves.csv
- outputs/figures/week16_raw_vs_ae_backtest_equity_curves.png
- outputs/reports/week16_raw_vs_ae_backtest_summary.txt

Models tested:
- Raw gradient boosting
- Autoencoder latent features + gradient boosting

Horizons tested:
- 1-month SPY outperformance
- 36-month SPY outperformance

Main finding:
The larger 1000+ ticker universe did not improve performance compared with the cleaner Week 15 universes. Bigger universe size did not automatically create a better model.

Raw GB finding:
Raw gradient boosting struggled in the large balanced universes. The 1-month models did not produce strong top-ranked portfolios, and the 36-month models were not strong enough to replace the Week 15 champions.

Autoencoder finding:
The autoencoder showed some useful internal ranking structure in balanced and lower-volatility universes, especially for 36-month targets. However, the full Week 16 backtest did not beat the best Week 15 models.

Important interpretation:
The autoencoder appears to help most when the universe is broad but still clean. It does not automatically fix a noisy universe with weaker metadata, lower-quality names, shorter histories, and more unstable stock behavior.

Main conclusion:
Universe quality matters as much as model architecture. A larger universe can make the learning problem harder if the added stocks are noisy, illiquid, poorly labeled, or structurally different from the rest of the dataset.

Best current models after Week 16 remain from Week 15:
1. Raw-return tactical champion:
   - highvol100_raw_gb_h1_top5_vol20_inverse_vol
   - Universe: highvol100
   - Model: raw gradient boosting
   - Horizon: 1 month
   - Portfolio: top 5
   - Risk control: vol_12m < 0.20 plus inverse-volatility weighting

2. Risk-adjusted structural champion:
   - full500_ae_gb_h36_top10_base
   - Universe: full500
   - Model: autoencoder latent features + gradient boosting
   - Horizon: 36 months
   - Portfolio: top 10
   - Best evidence so far that latent features help with broad structural ranking

Week 16 takeaway:
A carefully filtered 500-stock universe worked better than a noisy 1000+ ticker universe. The next improvement should focus less on adding more tickers and more on improving the learning objective.

Next research direction:
Week 17 will replace binary classification with a ranking model. Since the actual portfolio task is to rank stocks and select the top names, a ranking objective should better match the trading problem than predicting a simple yes/no SPY outperformance label.
## Week 17 Completed

Ran the ranking-model upgrade using LightGBM LambdaRank.

Main goal:
Replace binary classification with a ranking objective that better matches the actual portfolio task.

Previous modeling objective:
- Predict whether each stock beats SPY: yes/no.

Week 17 modeling objective:
- Rank stocks within each month so the strongest future performers are placed near the top.

Reason for the change:
The portfolio only buys the top-ranked stocks. Therefore, the model does not need to classify every stock correctly. It mainly needs to order the best stocks above the weaker stocks.

Model tested:
- LightGBM LGBMRanker
- Objective: LambdaRank
- Ranking groups: each month/date
- Ranking label: future return percentile bucket within each month
- Horizons tested:
  - 1-month ranking
  - 36-month ranking

Datasets tested:
- week15_full500
- week15_highvol100
- week15_highvol200
- week16_balanced450
- week16_balanced900

Key files created:
- outputs/tables/week17_lgbm_ranker_portfolio_signal.csv
- outputs/tables/week17_lgbm_ranker_predictions_week15_full500_1m.csv
- outputs/tables/week17_lgbm_ranker_predictions_week15_full500_36m.csv
- outputs/tables/week17_lgbm_ranker_predictions_week15_highvol100_1m.csv
- outputs/tables/week17_lgbm_ranker_predictions_week15_highvol100_36m.csv
- outputs/tables/week17_lgbm_ranker_backtest_stats.csv
- outputs/tables/week17_lgbm_ranker_backtest_curves.csv
- outputs/tables/week17_lgbm_ranker_backtest_holdings.csv
- outputs/figures/week17_lgbm_ranker_backtest_equity_curves.png
- outputs/reports/week17_lgbm_ranker_backtest_summary.txt
- outputs/tables/week17_ranker_diagnostics_year_stats.csv
- outputs/tables/week17_ranker_diagnostics_monthly_returns.csv
- outputs/tables/week17_ranker_diagnostics_ticker_frequency.csv
- outputs/reports/week17_ranker_diagnostics_summary.txt
- outputs/tables/week17_ranker_transaction_cost_stress_stats.csv
- outputs/tables/week17_ranker_topn_stress_stats.csv
- outputs/tables/week17_ranker_exclude_2026_stress_stats.csv
- outputs/tables/week17_ranker_remove_top_tickers_stress_stats.csv
- outputs/tables/week17_sector_regime_selected_vs_leaders.csv
- outputs/reports/week17_sector_regime_diagnostic_summary.txt

Main finding:
The LightGBM ranking model produced the strongest results of the project so far.

Best aggressive model:
- week15_full500_lgbm_ranker_h1_top5_inverse_vol
- Universe: week15_full500
- Model: LightGBM LambdaRank
- Horizon: 1 month
- Portfolio: top 5 stocks
- Risk control: inverse-volatility weighting

Approximate full-period result:
- Annualized return: about 61%
- Sharpe: about 1.55
- Max drawdown: about -26.5%
- Win rate: about 70.8%

Important interpretation:
The ranking objective was a major upgrade because it directly matches the stock-selection problem. Instead of asking whether every stock beats SPY, the model learns which stocks should be ranked above others each month.

Backtest behavior:
The equity curve was ahead for most of the test period and then accelerated sharply in 2025–2026. This suggests the ranker captured the AI / semiconductor / growth-stock surge, but the final headline return is likely boosted by that regime.

Robustness tests completed:

1. Transaction cost stress test:
The strategy remained strong under higher transaction cost assumptions.
Approximate results:
- 0.10% cost: about 61% annualized
- 0.25% cost: about 58% annualized
- 0.50% cost: about 54% annualized
- 1.00% cost: about 45% annualized
- 2.00% cost: about 29% annualized

Conclusion:
The ranker signal does not disappear under reasonable transaction-cost stress.

2. Top-N diversification stress test:
The ranker was tested with top 5, top 10, top 15, top 20, top 25, and top 50 portfolios.

Approximate results:
- Top 5: about 61% annualized
- Top 10: about 39% annualized
- Top 15: about 34% annualized
- Top 20: about 29% annualized
- Top 25: about 28% annualized
- Top 50: about 21% annualized

Conclusion:
The edge decays naturally as more lower-ranked stocks are included. This supports the idea that the model has a real ranking gradient.

3. Exclude-2026 stress test:
The model was tested without the explosive 2026 period.

Approximate top-5 results:
- Full period: about 61% annualized
- Excluding 2026: about 42% annualized
- 2021–2024 only: about 31% annualized
- 2025–2026 only: about 176% annualized

Conclusion:
The model is boosted by 2026, but it does not depend entirely on 2026. The pre-2026 result is still strong.

4. Remove top-selected tickers stress test:
The strategy was rerun after banning the most frequently selected tickers.

Approximate results:
- Remove top 0: about 61% annualized
- Remove top 1: about 52% annualized
- Remove top 3: about 43% annualized
- Remove top 5: about 35% annualized
- Remove top 10: about 29% annualized
- Remove top 15: about 25% annualized

Conclusion:
The model benefits heavily from its biggest winners, but it does not collapse when they are removed. This suggests the ranker is finding a broader class of winners, not just memorizing one or two names.

5. Sector/theme concentration diagnostic:
The ranker is extremely concentrated in Information Technology.

Selected sectors by year:
- 2021: 100% Information Technology
- 2022: 85% Information Technology, 15% Health Care
- 2023: about 82% Information Technology, 18% Health Care
- 2024: about 92% Information Technology, 8% Health Care
- 2025: about 92% Information Technology, 8% Health Care
- 2026: 96% Information Technology, 4% Health Care

Top industries selected:
- Semiconductors
- Application Software
- Technology Hardware / Storage / Peripherals
- Communications Equipment
- System Software

Important interpretation:
The ranker is not a broad sector-rotation model yet. It is mostly a technology / growth / semiconductor / AI infrastructure ranking model.

Corrected yearly sector leadership showed:
- Information Technology was strong in most years.
- Information Technology ranked poorly in 2022.
- The strategy also struggled in 2022.

Conclusion:
The model appears very good at exploiting tech/growth leadership, but it may be vulnerable when technology is not the leading sector.

Current model hierarchy after Week 17:

1. Best aggressive model:
   - week15_full500_lgbm_ranker_h1_top5_inverse_vol
   - Highest return and strongest breakthrough result

2. More defensible diversified model:
   - week15_full500_lgbm_ranker_h1_top10_inverse_vol
   - Lower return than top 5 but more diversified

3. Previous high-vol raw GB champion:
   - highvol100_raw_gb_h1_top5_vol20_inverse_vol

4. Previous structural AE champion:
   - full500_ae_gb_h36_top10_base

Main Week 17 conclusion:
LightGBM LambdaRank is the most important model improvement so far. The ranking objective better matches the portfolio task and produced a major performance increase.

Important caution:
The strongest ranker is highly concentrated in technology and growth stocks. The headline 61% annualized return should not be treated as a stable expectation. A more conservative expected range after robustness haircuts is closer to 20–30% annualized, assuming the signal survives stricter testing.

Next research direction:
Week 18 will focus on sector-aware ranking and regime controls.

Week 18 goals:
- Test sector caps.
- Test top-10 ranker with sector constraints.
- Test whether sector-aware portfolios reduce technology concentration.
- Explore regime filters for when technology is weak.
- Try to preserve most of the ranker upside while reducing dependence on a single market theme.
## Week 18 Completed

Ran the sector-aware and regime-control phase of the latent market twin project.

Main goal:
Test whether the strong Week 17 LightGBM ranker could be made more robust to technology / AI / semiconductor concentration.

Starting point:
Week 17 showed that the LightGBM LambdaRank model was the strongest model so far, especially:

- week15_full500_lgbm_ranker_h1_top5_inverse_vol
- week15_full500_lgbm_ranker_h1_top10_inverse_vol

However, diagnostics showed that the strategy was extremely concentrated in Information Technology, especially semiconductors, application software, hardware/storage, and AI infrastructure names. Week 18 tested whether we could reduce this theme risk without destroying the edge.

Main question:
Can we preserve most of the ranker upside while reducing dependence on Information Technology / AI / semiconductor leadership?

Key files created:
- outputs/tables/week18_sector_cap_backtest_stats.csv
- outputs/tables/week18_sector_cap_exposure_summary.csv
- outputs/reports/week18_sector_cap_backtest_summary.txt
- data/processed/week18_full500_leadership_modeling_dataset.parquet
- outputs/tables/week18_leadership_ranker_portfolio_signal.csv
- outputs/tables/week18_leadership_ranker_backtest_stats.csv
- outputs/figures/week18_leadership_ranker_backtest_equity_curves.png
- outputs/reports/week18_leadership_ranker_backtest_summary.txt
- outputs/tables/week18_tech_regime_filter_stats.csv
- outputs/tables/week18_top10_tech_regime_filter_stats.csv
- outputs/tables/week18_final_model_comparison.csv
- outputs/reports/week18_final_model_comparison_summary.txt

Tests completed:

1. Sector-cap backtest

Tested:
- top5 uncapped
- top5 max 2 stocks per sector
- top10 uncapped
- top10 max 3 stocks per sector
- top10 max 2 stocks per sector

Result:
Sector caps did not work well.

Approximate results:
- top5 uncapped: about 61% annualized, Sharpe about 1.55
- top5 max 2 per sector: about 16% annualized, Sharpe about 0.62
- top10 uncapped: about 39% annualized, Sharpe about 1.29
- top10 max 3 per sector: about 19% annualized, Sharpe about 0.77
- top10 max 2 per sector: about 18% annualized, Sharpe about 0.68

Interpretation:
The sector cap reduced technology concentration, but it also destroyed much of the model’s edge. This suggests that the ranker was correctly identifying the dominant winning theme during the test period. Forcing sector diversification made the portfolio less aligned with the strongest opportunities.

Conclusion:
Sector caps are too crude for this model. They lower returns without meaningfully improving drawdown enough to justify the sacrifice.

2. Sector / industry leadership-feature dataset

Built a new Week 18 leadership-feature dataset by adding features designed to teach the model about sector and industry leadership.

Added features included:
- sector_ret_1m, sector_ret_3m, sector_ret_6m, sector_ret_12m
- industry_ret_1m, industry_ret_3m, industry_ret_6m, industry_ret_12m
- sector_minus_spy_1m, sector_minus_spy_3m, sector_minus_spy_6m, sector_minus_spy_12m
- industry_minus_spy_1m, industry_minus_spy_3m, industry_minus_spy_6m, industry_minus_spy_12m
- stock_minus_sector_1m, stock_minus_sector_3m, stock_minus_sector_6m, stock_minus_sector_12m
- stock_minus_industry_1m, stock_minus_industry_3m, stock_minus_industry_6m, stock_minus_industry_12m
- sector_rank_by_date
- industry_rank_by_date
- sector/industry cross-sectional z-score features

Goal:
Teach the model to recognize industry leadership instead of simply overfitting to the recent technology trend.

3. Leadership-feature LightGBM ranker

Tested a new LightGBM LambdaRank model trained on the leadership-feature dataset.

Result:
The leadership features did not improve the short-term 1-month ranker.

The 1-month leadership ranker performed poorly:
- top5 average future return was below SPY
- top10 average future return was also weak

The 36-month leadership ranker was somewhat interesting:
- top5 future return was above SPY
- top10 future return was above SPY
- but the backtest did not beat the best Week 17/Week 18 models

Backtest result:
The best leadership-feature strategy was roughly:
- week18_leadership_h36_top5_base
- annualized return: about 20.5%
- Sharpe: about 0.81
- max drawdown: about -37.5%

Interpretation:
Leadership features may have some value for long-horizon structural models, but they did not improve the short-term tactical ranker. For the 1-month system, the original Week 17 ranker remained much stronger.

Conclusion:
Do not replace the Week 17 ranker with the leadership-feature ranker.

4. Tech regime filter on top5 ranker

Since sector caps and leadership features did not improve the model, Week 18 shifted to a regime overlay.

Instead of forcing diversification, the regime filter keeps the ranker active when technology is strong and reduces exposure when technology is weak.

Tested rules:
- no filter
- if tech 6-month return < SPY 6-month return, move partly or fully to cash
- if tech 12-month return < SPY 12-month return, move partly or fully to cash
- if tech 6-month return is negative, move partly or fully to cash
- if tech drawdown is worse than -20%, move partly to cash

Best top5 regime-filtered result:
- top5 ranker + tech 6m under SPY 100% cash filter

Approximate result:
- annualized return: about 60.2%
- Sharpe: about 1.65
- max drawdown: about -18.2%
- return/drawdown: about 3.31

Compared with unfiltered top5:
- annualized return: about 61.3%
- Sharpe: about 1.55
- max drawdown: about -26.5%

Interpretation:
The tech regime filter gave up almost no annualized return while sharply reducing drawdown. This was the first Week 18 test that clearly improved the model.

Conclusion:
A regime overlay works better than a sector cap.

5. Tech regime filter on top10 ranker

Then tested the same idea on the more diversified top10 ranker.

Best top10 regime-filtered result:
- top10 ranker + tech 12m under SPY 100% cash filter

Approximate result:
- annualized return: about 43.7%
- Sharpe: about 1.61
- max drawdown: about -16.3%
- return/drawdown: about 2.67

Compared with unfiltered top10:
- annualized return: about 39.4%
- Sharpe: about 1.29
- max drawdown: about -28.6%

Interpretation:
The top10 regime filter improved return, Sharpe, and drawdown. This is probably the best “defensible” final candidate so far because it is more diversified than top5 and has a much cleaner drawdown profile.

6. Final model comparison

Built a final comparison table across major champion models from Weeks 15–18.

Models compared:
- SPY benchmark
- equal-weight benchmark
- Week 15 raw GB highvol100 tactical champion
- Week 15 AE full500 structural champion
- Week 17 ranker top5 aggressive champion
- Week 17 ranker top10 diversified candidate
- Week 17 ranker top20 robustness reference
- Week 18 top5 ranker + tech 6m under SPY cash filter
- Week 18 top5 ranker + tech drawdown filter
- Week 18 top10 ranker + tech 12m under SPY cash filter
- Week 18 top10 ranker + tech 6m under SPY cash filter

Final leaderboard interpretation:

Best aggressive risk-adjusted system:
- Week 18 top5 ranker + tech 6m under SPY cash filter
- annualized return: about 60.2%
- Sharpe: about 1.65
- max drawdown: about -18.2%

Highest-return filtered system:
- Week 18 top5 ranker + tech drawdown filter
- annualized return: about 64.2%
- Sharpe: about 1.67
- max drawdown: about -20.9%

Best diversified final candidate:
- Week 18 top10 ranker + tech 12m under SPY cash filter
- annualized return: about 43.7%
- Sharpe: about 1.61
- max drawdown: about -16.3%

Main Week 18 conclusion:
Sector caps and leadership features did not improve the short-term ranker. The best improvement came from keeping the Week 17 LightGBM ranker unchanged and adding a technology-relative-strength regime overlay.

Final architecture after Week 18:
- LightGBM LambdaRank stock selector
- Full500 clean universe
- 1-month ranking target
- Top-N portfolio construction
- Inverse-volatility weighting
- Technology-relative-strength regime filter

Best aggressive model:
- top5 LightGBM ranker
- inverse-volatility weighted
- tech 6-month under SPY 100% cash filter

Best defensible model:
- top10 LightGBM ranker
- inverse-volatility weighted
- tech 12-month under SPY 100% cash filter

Important caution:
The model is still highly connected to technology/growth leadership. The regime filter reduces exposure when technology weakens, but the model still needs stricter validation before being treated as reliable.

Current conservative interpretation:
The headline backtests are very strong, but likely optimistic because of survivorship bias, limited test window, current-constituent universe construction, and possible dependence on the recent AI/semiconductor cycle.

Practical expected range after haircuts:
- Aggressive version: possibly 25–35% annualized if the signal survives stricter validation
- More defensible version: possibly 20–30% annualized if the signal survives stricter validation

Next research direction:
Week 19 will focus on stricter validation and moving toward a live/paper-trading-ready system.

Week 19 goals:
- Walk-forward retraining
- Feature importance analysis
- Check whether the model is using sensible features
- Test stability across retraining windows
- Add more realistic turnover/slippage assumptions
- Build a live monthly paper-trading simulation
- Generate current-month ranked picks from the final model
## Week 19 Completed

Ran stricter validation on the latent market twin / stock ranking project and returned the project back toward the original Latent Twins idea.

Main goal:
Move beyond strong static backtests and test whether the model survives more realistic validation. Then investigate whether latent market-state representations can improve the ranker by giving it regime context.

Starting point:
After Week 18, the strongest static models were:

- Aggressive static candidate:
  - top5 LightGBM LambdaRank
  - inverse-volatility weighting
  - tech 6-month under SPY cash filter
  - about 60% annualized in static backtest

- Defensible static candidate:
  - top10 LightGBM LambdaRank
  - inverse-volatility weighting
  - tech 12-month under SPY cash filter
  - about 44% annualized in static backtest

Concern:
These results were strong, but possibly too optimistic because the model was trained once and evaluated over the full test period. Week 19 tested whether the signal survives stricter validation.

Key files created:
- outputs/tables/week19_ranker_feature_importance.csv
- outputs/tables/week19_no_sector_ranker_portfolio_signal.csv
- outputs/tables/week19_no_sector_ranker_backtest_stats.csv
- outputs/tables/week19_walk_forward_ranker_stats.csv
- outputs/tables/week19_walk_forward_ranker_curves.csv
- outputs/tables/week19_walk_forward_yearly_stats.csv
- outputs/tables/week19_walk_forward_regime_filter_stats.csv
- data/processed/week19_market_state_dataset.parquet
- data/processed/week19_market_state_pca_latents.parquet
- data/processed/week19_full500_with_market_pca_latents.parquet
- outputs/tables/week19_walk_forward_pca_latent_ranker_stats.csv
- outputs/tables/week19_pca_latent_regime_filter_stats.csv
- outputs/tables/week19_pca_latent_diagnostics_full_stats.csv
- outputs/tables/week19_pca_latent_ablation_stats.csv
- data/processed/week19_full500_with_market_pca_interactions.parquet
- outputs/tables/week19_pca_interaction_ablation_stats.csv

Main tests completed:

1. Feature importance on the Week 17 ranker

Purpose:
Check whether the ranker was using reasonable features or leaking future information.

Result:
No obvious target leakage was detected by feature-name scan.

Important feature groups:
- sector metadata
- stock return momentum
- industry metadata
- volatility
- drawdown
- moving average features

Important finding:
The model relied heavily on Information Technology sector identity. This confirmed that the Week 17 model was strongly tied to technology/growth leadership.

Interpretation:
The model was not just a pure universal stock-ranking system. It was heavily using tech-sector context plus stock-level momentum, volatility, and drawdown features.

2. No-sector / no-industry ranker validation

Purpose:
Test whether the model still works after removing all sector and industry identity features.

Result:
The no-sector model did not collapse.

Approximate no-sector backtest:
- top5 inverse-vol: about 25% annualized
- top10 inverse-vol: about 23% annualized
- top20 inverse-vol: about 20% annualized

Interpretation:
Sector and industry features help a lot, but the signal is not entirely caused by knowing which stocks are technology stocks. Stock-level behavior features still contain real ranking signal.

Conclusion:
The ranker is not purely a static tech-label overfit.

3. Walk-forward retraining validation

Purpose:
Run a stricter validation where the model retrains through time.

Method:
- Train on all data before a given year
- Predict that year
- Move forward year by year
- Test 2021–2026

Result:
The walk-forward model was much weaker than the static model.

Approximate walk-forward results:
- top5: about 11% annualized, Sharpe about 0.40, max drawdown about -44%
- top10: about 12% annualized, Sharpe about 0.48, max drawdown about -37%
- top20: about 17% annualized, Sharpe about 0.74, max drawdown about -29%

Interpretation:
The huge static results were too optimistic. Under walk-forward retraining, the signal survived, but it was much weaker and much less stable.

Important finding:
Top20 performed better than top5 in walk-forward testing. This suggests that the model has broad ranking signal but is not precise enough to consistently identify only the best 5 stocks under stricter validation.

4. Walk-forward year-by-year diagnostics

Purpose:
Find which years caused the walk-forward model to struggle.

Important result:
The model was very year-dependent.

For walk-forward top5:
- 2021: strong positive
- 2022: major loss
- 2023: strong positive
- 2024: weak positive
- 2025: very strong positive
- 2026: strong positive

Main failure year:
2022.

Interpretation:
The model works in growth/tech-friendly regimes but gets hurt badly when that style breaks down.

Conclusion:
The ranker needs a regime-aware exposure layer.

5. Walk-forward + tech regime filter

Purpose:
Test whether the regime filter also improves strict walk-forward results.

Best result:
- top20 + tech drawdown 20% → 100% cash

Approximate result:
- annualized return: about 24%
- Sharpe: about 1.20
- max drawdown: about -19.5%

Compared with base walk-forward top20:
- annualized return improved from about 17% to about 24%
- Sharpe improved from about 0.74 to about 1.20
- max drawdown improved from about -29% to about -19.5%

Interpretation:
The regime filter worked under stricter validation. It reduced bad-regime damage and improved the model’s risk-adjusted profile.

6. Return to latent market twin idea

After the walk-forward diagnostics, the project shifted back toward the original Latent Twins idea.

Reason:
The core issue was regime dependence. The ranker worked in some market states and failed in others. This suggested that we needed a learned market-state representation.

Goal:
Build a latent market-state model so the ranker can condition stock selection on the current market regime.

Concept:
Instead of recursively forecasting future time steps, encode the current market into a latent state and use that state to guide ranking and risk.

7. Market-state dataset

Built a one-row-per-month market-state dataset.

Included features such as:
- SPY returns
- SPY volatility
- SPY drawdown
- equal-weight returns
- equal-weight drawdown
- average stock momentum
- cross-sectional return dispersion
- average stock volatility
- average stock drawdown
- market breadth
- percent of stocks above moving averages
- sector returns
- sector return dispersion
- technology sector relative strength

Output:
- data/processed/week19_market_state_dataset.parquet

Purpose:
This dataset represents the full market state at each month and becomes the input to the latent market twin.

8. PCA latent market-state model

Built a PCA-based latent market twin.

Output features:
- market_pca_z1
- market_pca_z2
- market_pca_z3
- market_pca_z4
- market_pca_z5
- market_pca_z6
- market_pca_reconstruction_error
- market_pca_regime_cluster
- distance to PCA regime clusters

Merged these latent market-state features into the stock modeling dataset.

Output:
- data/processed/week19_full500_with_market_pca_latents.parquet

Initial result:
The PCA latent walk-forward model appeared to improve the ranker substantially.

Approximate initial PCA latent walk-forward:
- top20 annualized return: about 37%
- Sharpe: about 1.51
- max drawdown: about -29%

Then with regime filter:
- top20 + tech drawdown 20% → 100% cash
- annualized return: about 42%
- Sharpe: about 1.91
- max drawdown: about -13.6%

This looked like the strongest result so far.

9. PCA latent diagnostics

Purpose:
Compare:
- nonlatent top20
- PCA latent top20
- PCA latent top20 + regime filter

Full-period comparison:
- nonlatent top20: about 16.7% annualized, Sharpe about 0.74
- PCA latent top20: about 37.0% annualized, Sharpe about 1.51
- PCA latent top20 + regime: about 42.2% annualized, Sharpe about 1.91

Year-by-year result:
The PCA latent model improved several years, especially:
- 2021
- 2023
- 2025

The regime overlay helped most in 2022:
- nonlatent 2022: about -25%
- PCA latent 2022: about -16%
- PCA latent + regime 2022: slightly positive

Initial interpretation:
The latent market-state idea looked very promising.

10. PCA latent feature importance

Purpose:
Check whether the ranker was actually using the PCA latent variables.

Result:
The static feature-importance run showed that market PCA latent features had zero gain importance.

Interpretation:
The initial PCA improvement could not yet be confidently attributed to the PCA latent variables. It may have been caused by training randomness, feature sampling, or altered model path.

Important correction:
We could not honestly claim that the PCA latent variables directly caused the improvement without controlled ablation.

11. Controlled PCA latent ablation

Purpose:
Test:
- original features only
- PCA features only
- original + PCA features

Important bug found:
The first ablation accidentally included `ranking_label` as a feature, which caused impossible results. This was identified as leakage and corrected.

Corrected ablation result:
- original_only top20: about 33% annualized, Sharpe about 1.33
- original_plus_pca top20: about 32% annualized, Sharpe about 1.25
- pca_only top20: about 10% annualized, Sharpe about 0.70

Conclusion:
The PCA latent features did not clearly improve the model in controlled testing. Original stock features alone performed slightly better.

Interpretation:
The initial PCA latent improvement was likely not a clean causal improvement from the latent variables themselves.

12. PCA interaction dataset

Purpose:
Test a more realistic latent-twin idea:

stock state × latent market state → regime-conditioned ranking

Created explicit interaction features such as:
- ret_3m × market_pca_z1
- ret_6m × market_pca_z1
- vol_12m × market_pca_z1
- stock_drawdown × market_pca_z1
- price_to_ma_3m × market_pca_z1
- stock features × PCA regime cluster flags

Output:
- data/processed/week19_full500_with_market_pca_interactions.parquet

13. PCA interaction ablation

Tested:
- original_only
- original_plus_pca
- original_plus_pca_interactions
- pca_interactions_only

Result:
The explicit PCA interaction features did not improve the model.

Best controlled result remained:
- original_only top20
- annualized return: about 33%
- Sharpe: about 1.33
- max drawdown: about -29%

The interaction model was worse:
- original_plus_pca_interactions top20: about 17.5% annualized
- original_plus_pca_interactions top10: about 21.9% annualized

Conclusion:
Simple PCA latent market features and simple PCA interaction features are not the final version of the Latent Twin idea.

Main Week 19 conclusions:

1. The static top5/top10 models were too optimistic.
2. Walk-forward validation is much stricter and more realistic.
3. The stock ranker still has real signal, especially at top20.
4. Sector/industry features help, but the model does not completely depend on them.
5. Regime filters are consistently useful.
6. PCA market-state latents looked promising initially but did not survive controlled ablation as a clear causal improvement.
7. Simple date-level market PCA features are probably too weak because they are the same for every stock within a monthly ranking group.
8. Explicit PCA interactions also did not improve the model.
9. The true Latent Twin direction should shift from market-level PCA to stock-level latent similarity.

Best current controlled model:
- Walk-forward LightGBM LambdaRank
- original stock features
- top20 inverse-volatility basket
- about 33% annualized
- Sharpe about 1.33
- max drawdown about -29%

Best validated risk-control idea:
- Apply tech/growth regime overlay
- Especially tech drawdown 20% → cash
- This improved earlier walk-forward results and remains important for final system design.

Current honest final model direction:
- LightGBM LambdaRank remains the stock-selection engine.
- Top20 is more reliable than top5 under walk-forward validation.
- Regime filters are necessary.
- The next true latent-twin layer should use stock-level latent similarity, not only market-level PCA.

Why stock-level latent similarity is the next step:
The original Latent Twins idea is about comparing current system states to similar historical states without recursively forecasting each time step.

For stocks, that means:
- encode each stock-month into a latent vector
- find historical stock-months with similar latent states
- examine what happened after those similar states
- use neighbor future returns, win rates, and distances as features

This is closer to the true thesis:
Current stock state resembles these historical latent states, so its future outcome distribution may resemble theirs.

Week 20 direction:
Build the stock-level latent twin.

Week 20 goals:
- Train a stock-feature autoencoder or PCA baseline.
- Encode every stock-month into a latent vector.
- Build nearest-neighbor similarity features.
- Compute historical-neighbor future return statistics.
- Add those latent-neighbor features to the ranker.
- Test whether stock-level latent similarity improves walk-forward performance.

## Week 20 Completed

Built and tested the first real stock-level Latent Twin system.

Main goal:
Move beyond market-level PCA features and build a true stock-level latent similarity model.

Core idea:
Instead of recursively forecasting future prices step by step, encode each stock/month into a latent state and compare it to similar historical stock states.

Main hypothesis:
Stocks in similar latent states may have similar future return distributions.

Starting point:
Week 19 showed that the LightGBM ranker works, but the market-level PCA latent features did not clearly survive controlled ablation. The next direction was to build a stock-level latent twin:

stock/month feature vector → latent stock state → nearest historical latent neighbors → future outcome statistics

Key files created:
- data/processed/week20_stock_state_matrix.parquet
- data/processed/week20_stock_state_matrix_scaled.parquet
- data/processed/week20_stock_state_metadata.parquet
- outputs/tables/week20_stock_state_feature_list.csv
- outputs/tables/week20_stock_state_scaler_stats.csv
- data/processed/week20_stock_state_pca_latents.parquet
- data/processed/week20_stock_state_pca_latents_with_metadata.parquet
- outputs/tables/week20_stock_state_pca_explained_variance.csv
- outputs/tables/week20_stock_state_pca_cluster_summary.csv
- data/processed/week20_stock_latent_neighbor_features.parquet
- data/processed/week20_full500_with_stock_latent_neighbors.parquet
- outputs/tables/week20_neighbor_feature_ablation_stats.csv
- outputs/tables/week20_neighbor_regime_filter_stats.csv
- outputs/tables/week20_neighbor_ensemble_stats.csv
- outputs/tables/week20_ensemble_diagnostics_full_stats.csv
- outputs/tables/week20_ensemble_diagnostics_yearly_stats.csv

Main tests completed:

1. Built clean stock-state matrix

Created a clean stock/month feature matrix using only information available at time t.

Excluded:
- future returns
- target columns
- ranking labels
- date/ticker/company metadata

Scaled the feature matrix for PCA / latent encoding.

Purpose:
Prepare stock-level states for latent representation.

2. Built stock-state PCA latent embeddings

Compressed each stock/month feature vector into stock-level PCA latent coordinates.

Concept:
Each row now has a latent representation of that stock’s current state.

stock/month features → stock_pca_z1 ... stock_pca_z16

Also added:
- stock_pca_reconstruction_error
- stock_pca_cluster
- distance to stock PCA clusters

Purpose:
Create the stock-level latent space needed for latent twin similarity.

3. Built historical latent-neighbor features

For each stock/month, searched only historical rows before that month.

For each current stock state:
- find nearest historical stock states in latent space
- summarize what happened to those similar states afterward

Neighbor features included:
- neighbor_count
- neighbor_distance_mean
- neighbor_distance_median
- neighbor_distance_min
- neighbor_avg_future_1m_return
- neighbor_median_future_1m_return
- neighbor_avg_future_1m_excess_return
- neighbor_outperform_spy_1m_rate
- neighbor_positive_1m_return_rate

Important:
Neighbor search used only past data. No future rows were allowed as neighbors.

This is the first direct implementation of the Latent Twin idea:
current stock state → similar historical states → future outcome distribution

4. Merged latent-neighbor features into the modeling dataset

Created:
data/processed/week20_full500_with_stock_latent_neighbors.parquet

This dataset contains the original stock features plus the latent-neighbor outcome features.

5. Latent-neighbor feature ablation

Tested:
- original_only
- neighbor_only
- original_plus_neighbors

Results:

Original-only branch:
- original_only top20: about 33.3% annualized
- Sharpe: about 1.33
- max drawdown: about -28.6%

Latent-neighbor-only branch:
- neighbor_only top10: about 33.6% annualized
- Sharpe: about 1.23
- max drawdown: about -25.1%

Lower-drawdown latent-neighbor branch:
- neighbor_only top5: about 27.2% annualized
- Sharpe: about 1.03
- max drawdown: about -18.7%

Combined feature model:
- original_plus_neighbors performed poorly
- adding neighbor features into the same LightGBM model hurt performance

Interpretation:
The latent-neighbor features work as their own standalone signal, but they should not be forced into the same feature model as the original ranker.

Main conclusion:
The latent-neighbor branch is a real independent signal.

This was the first clean evidence that the stock-level Latent Twin idea works.

6. Regime filter on latent-neighbor model

Applied the same tech/growth regime filters to the latent-neighbor branch.

Best latent-neighbor filtered result:
- neighbor_only top10
- tech drawdown 20% → 100% cash

Approximate result:
- annualized return: about 39.1%
- Sharpe: about 1.62
- max drawdown: about -17.9%

Another strong result:
- neighbor_only top10
- tech 12-month under SPY → 100% cash
- annualized return: about 34.7%
- Sharpe: about 1.57
- max drawdown: about -14.5%

Interpretation:
The latent-neighbor branch remains strong after regime filtering. The regime filter improves drawdown and risk-adjusted performance.

7. Original + latent-neighbor ensemble

Because original_plus_neighbors failed inside one model, tested portfolio-level blending instead.

Main ensemble:
- 70% original_only_top20
- 30% neighbor_only_top10

Then applied regime filters.

Best ensemble result:
- 70% original_only_top20
- 30% neighbor_only_top10
- tech drawdown 20% → 100% cash

Approximate performance:
- annualized return: about 41.1%
- Sharpe: about 1.97
- max drawdown: about -11.2%
- win rate: about 63%
- return/drawdown: about 3.45

This became the strongest final candidate so far.

8. Ensemble diagnostics

Year-by-year performance for the best 70/30 filtered ensemble:

- 2021: about +36.7%
- 2022: about +7.7%
- 2023: about +50.5%
- 2024: about +36.9%
- 2025: about +57.4%
- 2026: about +111.8%

Most important result:
2022 became positive.

Earlier unfiltered models were badly hurt in 2022. The final filtered ensemble survived the bad growth/tech regime and preserved strong upside in later years.

Best final Week 20 architecture:

Original branch:
LightGBM LambdaRank using original stock features

Latent twin branch:
Stock-state PCA latent space → historical nearest neighbors → neighbor outcome features → LightGBM LambdaRank

Portfolio ensemble:
70% original_only_top20
30% neighbor_only_top10

Risk overlay:
If technology drawdown is worse than 20%, move to cash.

Best current final candidate:
70% original_only_top20 + 30% neighbor_only_top10 + tech drawdown 20% cash filter

Approximate result:
- annualized return: about 41.1%
- Sharpe: about 1.97
- max drawdown: about -11.2%

Main Week 20 conclusions:

1. The stock-level latent-neighbor model works as an independent signal.
2. The latent-neighbor model should be blended at the portfolio level, not merged into the same LightGBM feature set.
3. The original ranker remains very strong.
4. The latent-neighbor branch improves diversification of signal.
5. The tech drawdown regime filter remains very important.
6. The best model is now an ensemble of conventional ranking and latent-twin similarity ranking.
7. This is the strongest and most coherent Latent Twins result so far.

Final research interpretation:
The project now has a real Latent Twin component.

The model is no longer just:
stock features → rank future return

It is now:
stock features → ranker branch
stock latent state → similar historical states → neighbor outcome branch
ensemble → risk overlay → portfolio

This matches the original thesis much better:
Use latent similarity to compare current system states to historical analogs, rather than recursively forecasting each future time step.

Week 21 direction:
Move from research backtest to paper-trading pipeline.

Week 21 goals:
- Build a clean final model runner
- Generate current-month picks
- Save final model signals
- Save final portfolio weights
- Track paper-trading performance going forward
- Create a monthly update workflow
## Week 21 Completed

Week 21 moved the Latent Twin Stock Analog Forecasting project from a research backtest into a usable paper-trading pipeline.

Main goal:
Stop adding model complexity and turn the best Week 20 model into a repeatable, auditable monthly signal-generation system.

Final model selected:

70% original LightGBM ranker top20  
30% stock latent-neighbor LightGBM ranker top10  
inverse-volatility weighting  
tech drawdown 20% → cash regime filter

Best validation result from Week 20:
- Annualized return: about 41.1%
- Sharpe: about 1.97
- Max drawdown: about -11.2%
- 2022 return: about +7.7%

This is still a research/paper-trading model, not a real-money trading system.

---

### Main idea

The project now has two signal branches.

Original branch:

stock features → LightGBM LambdaRank → top20 stocks

Latent Twin branch:

stock state → PCA latent state → nearest historical latent neighbors → neighbor future outcome features → LightGBM LambdaRank → top10 stocks

Final portfolio:

70% original branch + 30% latent-neighbor branch

Risk overlay:

If technology sector drawdown is worse than -20%, move to cash. Otherwise remain invested.

---

## Week 21 buildout

### Step 1: Final model config

Created:

configs/final_model_config.yaml

This file stores the final model settings, including:
- dataset paths
- model parameters
- original branch settings
- latent-neighbor branch settings
- portfolio weights
- inverse-volatility weighting
- transaction cost
- tech drawdown regime filter
- paper-trading starting cash

This made the pipeline configurable instead of hard-coded.

---

### Step 2: Config utility module

Created:

src/utils/config.py

Added reusable functions:
- load_config
- ensure_output_dirs
- check_required_files
- print_config_summary

This cleaned up the runner and made the project easier to maintain.

---

### Step 3: Data loading module

Created:

src/data/loaders.py

Added reusable loaders for:
- base modeling dataset
- neighbor-enhanced dataset
- monthly prices
- stock universe
- monthly returns
- latest available signal date

This separated data loading from model logic.

---

### Step 4: Feature selection module

Created:

src/features/feature_sets.py

Added feature set logic for:
- original_only
- neighbor_only
- original_plus_neighbors

Also added leakage checks to exclude:
- future return columns
- target columns
- ranking labels
- row_id/date/ticker/company metadata

This ensures the model branches use the correct features.

---

### Step 5: Ranker training module

Created:

src/models/rankers.py

Added reusable LightGBM LambdaRank logic:
- make_ranking_label
- group_sizes_by_date
- train_lambdarank_model
- predict_ranker_scores
- train_predict_latest

Important bug fixed:
Some rows had missing future_1m_return, causing ranking-label creation to fail. The ranker module now drops rows with missing future labels during training while still allowing the latest signal date to be scored.

---

### Step 6: Final portfolio builder

Created:

src/paper_trading/portfolio.py

This module builds the actual final portfolio:
- selects top20 from original branch
- selects top10 from latent-neighbor branch
- applies inverse-volatility weights inside each branch
- combines duplicated tickers
- applies 70/30 branch weights
- applies tech drawdown regime filter
- adds CASH if risk-off

Current latest signal date:
2026-06-30

Regime status:
- tech_drawdown: 0.0
- risk_on: True
- cash_weight_when_off: 1.0

Because risk_on was True, the final portfolio was fully invested.

---

### Step 7: Persistent signal ledger

Created:

src/paper_trading/ledger.py

Added persistent paper-trading ledgers:

outputs/paper_trading/paper_portfolio_signals.csv  
outputs/paper_trading/paper_trading_run_summary.csv

These store:
- run timestamp
- model name
- signal date
- ticker
- final weight
- branch source
- regime status
- tech drawdown
- top ticker
- largest weight
- cash weight

---

### Step 8: Duplicate-safe ledger behavior

Improved ledger logic so rerunning the pipeline on the same model and signal date overwrites the prior signal instead of appending duplicate rows.

This fixed the issue where repeated test runs kept growing the signal ledger.

Also fixed mixed date parsing in the ledger, since previous rows had inconsistent date formats like:

2026-06-30  
2026-06-30 00:00:00

The ledger now safely normalizes signal dates.

---

### Step 9: Paper-trading order sheet

Created:

src/paper_trading/orders.py

This converts portfolio weights into paper-trading orders using the configured starting cash.

Current starting cash:

$100,000

The order sheet includes:
- ticker
- target weight
- target dollars
- latest price
- target shares
- rounded whole shares
- rounded dollar amount
- leftover cash from rounding

Persistent order ledger:

outputs/paper_trading/paper_trade_orders_ledger.csv

Latest run produced about:
- invested rounded total: about $94,312
- leftover cash from rounding: about $5,688
- explicit cash target: $0

The leftover cash is caused by whole-share rounding.

---

## Latest generated portfolio

Latest signal date:

2026-06-30

Regime:

risk_on = True

Top holdings included:
- DHR
- MTD
- TECH
- TMO
- FOX
- ZTS
- SWKS
- EPAM
- FIX
- AMD
- DELL
- LYB
- NXPI
- NTAP
- CIEN
- ON
- PANW
- FNT
- WDC
- CRWD
- STX
- LITE
- DDOG
- CNC
- SATS
- SMCI
- MU
- INTC

Some names came from both branches, including:
- AMD
- DELL
- CRWD
- SMCI

These are higher-conviction overlap names because both the original ranker and latent-neighbor ranker selected them.

---

## Week 21 final pipeline command

From the project root:

C:\ResearchCode\latent_market_twin

Run:

python scripts\run_final_pipeline.py

This now produces:
- original branch predictions
- latent-neighbor branch predictions
- branch portfolio detail
- final portfolio weights
- regime status
- signal generation summary
- persistent signal ledger
- persistent run summary
- paper-trade order sheet
- persistent order ledger

---

## Important output files

Timestamped audit files:
- outputs/paper_trading/latest_original_branch_predictions_<timestamp>.csv
- outputs/paper_trading/latest_neighbor_branch_predictions_<timestamp>.csv
- outputs/paper_trading/latest_branch_portfolio_detail_<timestamp>.csv
- outputs/paper_trading/latest_final_portfolio_weights_<timestamp>.csv
- outputs/paper_trading/latest_regime_status_<timestamp>.csv
- outputs/paper_trading/latest_signal_generation_summary_<timestamp>.txt
- outputs/paper_trading/paper_trade_orders_<timestamp>.csv

Persistent ledger files:
- outputs/paper_trading/paper_portfolio_signals.csv
- outputs/paper_trading/paper_trading_run_summary.csv
- outputs/paper_trading/paper_trade_orders_ledger.csv

Documentation created:
- PAPER_TRADING_RUNBOOK.md

---

## Week 21 conclusion

Week 21 successfully turned the research model into a working paper-trading signal generator.

The project can now:
1. train the two model branches,
2. generate current ranked picks,
3. build a final blended portfolio,
4. apply a regime filter,
5. convert weights into paper orders,
6. save persistent ledgers,
7. avoid duplicate ledger rows,
8. produce an auditable run summary.

This is now a real paper-trading pipeline, not just a backtest.

The model is not ready for real money yet, but it is ready to start being tested forward in a structured way.

---

## Current status

The best model is:

70% original ranker top20  
30% latent-neighbor ranker top10  
inverse-vol weighting  
tech drawdown 20% cash filter

The pipeline is operational.

Next phase:
Week 22 should make the system live-test ready by adding data refresh, holdings tracking, rebalance logic, and performance tracking.

# Week 22 Research Log — Live Latent Twin Paper-Trading System

## Main Goal

Week 22 moved the Latent Twin Stock Analog Forecasting project from a backtest/paper-trading prototype into a partially automated live forward-testing system.

The main objective was not to improve the model itself. The objective was to make the model operational:

- repeatable
- auditable
- automated
- separated between frozen research files and live rebuilt files
- ready for forward paper testing

---

## Final Live Model Name

LTSAF_live_v1

This stands for:

Latent Twin Stock Analog Forecasting — Live Version 1

---

## High-Level System

The project now has two separate systems.

### 1. Frozen Research / Paper-Trading Baseline

Config:

configs/final_model_config.yaml

Output directory:

outputs/paper_trading/

Purpose:

This is the original validated research/paper-trading version. It uses the frozen Week 15 / Week 20 processed data files.

This system is useful as the reference baseline.

---

### 2. Live Rebuilt Pipeline

Config:

configs/live_model_config.yaml

Output directory:

outputs/paper_trading_live/

Live processed files:

data/processed/live_500_daily_prices.parquet  
data/processed/live_500_monthly_prices.parquet  
data/processed/live_500_monthly_returns.parquet  
data/processed/live_full500_modeling_dataset.parquet  
data/processed/live_stock_state_pca_latents_with_metadata.parquet  
data/processed/live_stock_latent_neighbor_features.parquet  
data/processed/live_full500_with_stock_latent_neighbors.parquet  

Purpose:

This is the live forward-testing candidate. It rebuilds the feature stack from refreshed market data and generates a live monthly signal.

---

## Current Live Rebuild Chain

The live chain is:

1. Download daily adjusted prices from yfinance.
2. Resample daily adjusted closes to completed month-end prices.
3. Build live monthly returns.
4. Build live base stock feature dataset.
5. Build live stock-state PCA coordinates.
6. Build live latent-neighbor analog features.
7. Train the live original ranker branch.
8. Train the live latent-neighbor ranker branch.
9. Blend both branches into a final live portfolio.
10. Apply the tech drawdown regime filter.
11. Save live target weights, orders, and ledgers.

---

## Why Monthly Prices Use Completed Months Only

The live price downloader excludes the incomplete current month.

This is important because yfinance can provide partial current-month data. If that partial month is labeled as month-end data, the model would accidentally treat an unfinished month as complete.

The live system now uses:

include_incomplete_current_month = False

This means, for example:

If today is inside June, the latest completed month is May 31.

This prevents lookahead / partial-period contamination.

---

## Live Price Validation

We compared live rebuilt prices against the frozen research monthly prices.

After switching from yfinance monthly bars to daily adjusted prices resampled manually to month-end, the comparison improved sharply.

Final result:

- Median latest absolute percent difference: 0.0
- Mean latest absolute percent difference: about 0.064%
- Tickers with latest difference above 5%: 1

The main outlier was FX, likely due to a vendor/ticker adjustment issue.

Conclusion:

The live price refresh is good enough to use as the live data foundation, while still keeping the frozen research data separate.

---

## Live Base Feature Dataset

Created:

data/processed/live_full500_modeling_dataset.parquet

Observed shape:

72683 rows  
182 columns  

Date range:

2013-12-31 to 2026-05-31

Ticker count:

503

Important note:

The latest month has NaN values for:

- future_1m_return
- future_1m_spy_return
- future_1m_excess_return
- ranking_label

This is expected and correct. The latest month is the month being scored, so future returns are not known yet.

---

## Live Stock-State PCA

Created:

data/processed/live_stock_state_pca_latents_with_metadata.parquet

The stock-state PCA maps each stock-month into a lower-dimensional latent state.

The PCA representation is used for nearest-neighbor analog search.

PCA result:

16 components explained about 27.4% cumulative variance.

This is acceptable because the PCA is not meant to perfectly reconstruct the full feature table. It is used as a compact stock-state coordinate system for analog matching.

---

## Live Latent-Neighbor Features

Created:

data/processed/live_stock_latent_neighbor_features.parquet  
data/processed/live_full500_with_stock_latent_neighbors.parquet  

For each stock-month, the system finds similar historical stock states and summarizes their future outcomes.

Main live neighbor features:

- neighbor_count
- neighbor_distance_mean
- neighbor_distance_median
- neighbor_distance_min
- neighbor_avg_future_1m_return
- neighbor_median_future_1m_return
- neighbor_avg_future_1m_excess_return
- neighbor_outperform_spy_1m_rate
- neighbor_positive_1m_return_rate

Latest date rows had:

neighbor_count = 50

This means the live analog system is working and providing full neighbor sets.

---

## Live Final Pipeline

Created:

scripts/run_live_final_pipeline.py

This uses:

configs/live_model_config.yaml

and outputs to:

outputs/paper_trading_live/

The live final pipeline trains:

Original branch:

original_only features → LightGBM LambdaRank → top20

Latent-neighbor branch:

neighbor_only features → LightGBM LambdaRank → top10

Final portfolio:

70% original branch  
30% latent-neighbor branch  
inverse-volatility weighting  
tech drawdown 20% risk filter  

---

## Latest Live Signal

Latest live signal date:

2026-05-31

This differs from the frozen research signal date because the live system only uses completed months.

Top live names from the latest generated portfolio included:

- ROP
- TYL
- PPL
- CNP
- NI
- WEC
- AEE
- PEG
- CMS
- EXC
- TRMB
- FE
- SRE
- ATO
- EP
- KLAC
- XEL
- ES
- ETX
- KEYS

---

## Live Persistent Ledgers

Added persistent ledgers for the live pipeline:

outputs/paper_trading_live/live_portfolio_signals.csv  
outputs/paper_trading_live/live_run_summary.csv  
outputs/paper_trading_live/live_order_ledger.csv  

These are separate from the frozen system ledgers.

This is important because the live system should be forward-tested independently.

---

## Live Rebuild Wrapper

Created:

scripts/run_live_rebuild_pipeline.py

This runs the full live rebuild chain:

1. make live config
2. refresh live monthly prices
3. compare live vs research prices
4. build live base features
5. build live stock-state PCA
6. build live latent-neighbor features
7. compare live vs research dataset
8. run live final pipeline

One-command live rebuild:

python scripts\run_live_rebuild_pipeline.py

Batch file:

run_live_rebuild.bat

---

## Scheduled Automation

Created scheduled task:

LatentMarketTwinLiveMonthlyRebuild

This runs:

run_live_rebuild.bat

Schedule:

Monthly on the 2nd day of the month at 9:30 AM.

Reason:

Running on the 2nd gives market data time to settle after month-end.

---

## Frozen vs Live Signal Comparison

Created:

scripts/compare_frozen_vs_live_signals.py

This compares:

outputs/paper_trading/paper_portfolio_signals.csv

against:

outputs/paper_trading_live/live_portfolio_signals.csv

The frozen signal date was:

2026-06-30

The live signal date was:

2026-05-31

Because these dates differ, the portfolios are not expected to match exactly.

Overlap names included:

- DELL
- SMCI
- LITE
- DDOG
- CIEN
- WDC

Conclusion:

The live signal is structurally working and not detached from the frozen system, but it is different enough to treat as its own forward-test model.

---

## Daily Live Value Tracking

Created daily live value checking through:

scripts/check_live_portfolio_value.py

Batch file:

run_daily_value_check.bat

Scheduled task:

LatentMarketTwinDailyValueCheck

This checks live holdings value and compares against SPY.

Outputs:

outputs/paper_trading/live_portfolio_value_snapshots.csv  
outputs/paper_trading/latest_live_portfolio_value_summary.txt  
outputs/paper_trading/latest_live_portfolio_value_detail.csv  
outputs/figures/live_portfolio_vs_spy.png  

---

## Monthly Frozen Paper-Trading Automation

The original frozen monthly automation remains:

LatentMarketTwinMonthlyUpdate

This runs the frozen research/paper system.

It is still useful as a baseline, but it should not be confused with the live rebuilt model.

---

## Current Automation Layers

### Frozen Monthly Update

Task:

LatentMarketTwinMonthlyUpdate

Purpose:

Runs the frozen paper-trading pipeline.

---

### Daily Live Value Check

Task:

LatentMarketTwinDailyValueCheck

Purpose:

Marks holdings to live prices and updates portfolio-vs-SPY chart.

---

### Live Monthly Rebuild

Task:

LatentMarketTwinLiveMonthlyRebuild

Purpose:

Refreshes live data, rebuilds features, rebuilds latent neighbors, and generates a live monthly signal.

---

## Important Research Decision

The live model should now be frozen as:

LTSAF_live_v1

Rule:

Do not keep changing model logic during the forward test.

Allowed changes:

- bug fixes
- logging improvements
- reporting improvements
- data-quality fixes

Not allowed during forward test:

- changing the portfolio weights
- changing the ranker objective
- changing the neighbor construction
- changing the regime filter
- changing feature sets to chase performance

Reason:

If the model keeps changing, we will never know if the live system actually works.

---

## Forward-Test Rule

The live system should run for at least 3 to 6 months before making major strategy changes.

Track:

- daily portfolio value
- monthly returns
- SPY comparison
- drawdown
- live signal overlap
- best and worst holdings
- regime status

The purpose is to determine whether the latent-neighbor analog signal has real forward predictive value.

---

## Current Status

Week 22 successfully created a live forward-testing infrastructure.

The project now has:

- frozen research pipeline
- live data refresh
- live feature rebuild
- live PCA latent state rebuild
- live latent-neighbor rebuild
- live final signal generation
- live scheduled rebuild
- live persistent ledgers
- daily live value tracking
- SPY comparison
- frozen-vs-live comparison

The system is not production trading infrastructure, but it is now a real research-grade paper-trading system.

---

## Next Recommended Phase

Week 23 should focus on monitoring and reporting, not changing the model.

Recommended Week 23 tasks:

1. Create a live dashboard summary.
2. Add live portfolio performance ledger.
3. Add live rebalance order generator.
4. Add live holdings ledger separate from frozen holdings.
5. Add automatic email/log summary after scheduled runs.
6. Keep LTSAF_live_v1 frozen.

# Week 23 Research Log — LTSAF Live Monitoring

## Goal

Week 23 focuses on monitoring and usability, not changing the LTSAF_live_v1 model logic.

The model remains frozen. Work this week is limited to:

- live holdings tracking
- live value checking
- live rebalance orders
- live performance tracking
- live dashboard improvements
- reporting and monitoring tools

---

## Model Freeze Reminder

Current frozen model:

LTSAF_live_v1

Do not change:

- branch weights
- top_n values
- ranking objective
- latent-neighbor construction
- PCA dimensionality
- regime filter
- strategy logic

Allowed changes:

- bug fixes
- dashboard improvements
- logging improvements
- file organization
- reporting tools
- data-quality warnings

---

## Completed So Far

### Step 1 — Live Holdings Ledger

Created:

outputs/paper_trading_live/current_live_holdings.csv

This initializes the live paper portfolio from the latest LTSAF_live_v1 order sheet.

---

### Step 2 — Live Value Checker

Created:

scripts/check_ltsaf_live_value.py

This marks the live portfolio to market using current quotes and compares daily performance against SPY.

Outputs:

outputs/paper_trading_live/live_value_snapshots.csv  
outputs/paper_trading_live/latest_live_value_detail.csv  
outputs/paper_trading_live/latest_live_value_summary.txt  
outputs/figures/ltsaf_live_value_vs_spy.png  

---

### Step 3 — Daily Live Value Automation

Updated:

run_daily_value_check.bat

The daily scheduled task now tracks LTSAF live holdings instead of the older frozen-paper holdings.

Task:

LatentMarketTwinDailyValueCheck

---

### Step 4 — Live Rebalance Orders

Created:

scripts/generate_live_rebalance_orders.py

This compares:

current live holdings vs latest live target portfolio

and outputs buy/sell/hold orders.

Outputs:

outputs/paper_trading_live/latest_live_rebalance_orders.csv  
outputs/paper_trading_live/live_rebalance_orders_ledger.csv  
outputs/paper_trading_live/latest_live_rebalance_summary.txt  

---

### Step 5 — Live Rebalance in Monthly Rebuild

Updated:

scripts/run_live_rebuild_pipeline.py

The live monthly rebuild now ends by generating rebalance orders.

---

### Step 6 — Live Performance Tracker

Created:

scripts/track_ltsaf_live_performance.py

This evaluates completed live holding periods against SPY once a later completed monthly price exists.

Outputs:

outputs/paper_trading_live/live_performance_ledger.csv  
outputs/paper_trading_live/latest_live_performance_summary.txt  

---

### Step 7 — Live Performance in Monthly Rebuild

Updated:

scripts/run_live_rebuild_pipeline.py

The live monthly rebuild now attempts to track completed live performance.

---

### Step 8 — Interactive Dashboard

Created:

dashboard/ltsaf_live_dashboard.py

The dashboard provides a Robinhood-style local UI with:

- portfolio value
- daily P&L
- holdings percentages
- individual stock explorer
- SPY comparison
- original branch view
- latent-neighbor branch view
- final portfolio view
- rebalance table
- frozen vs live comparison
- system health page

Launch command:

streamlit run dashboard\ltsaf_live_dashboard.py

Batch launcher:

run_ltsaf_dashboard.bat

---

## Weekly UI Review Rule

At the end of every research week, review the dashboard UI.

Questions to ask:

1. Is the dashboard easy to understand in under 30 seconds?
2. Can I immediately see portfolio value, day return, and SPY comparison?
3. Can I inspect each holding clearly?
4. Can I see why the model picked each stock?
5. Can I compare original branch, latent-neighbor branch, final portfolio, and SPY?
6. Are rebalance orders obvious?
7. Are errors or stale data obvious?
8. What would make this feel more like a real portfolio app?

UI changes are allowed during the forward test because they do not change model logic.

---

## Next Steps

1. Test the dashboard.
2. Improve the Robinhood-style visual layout.
3. Add individual stock detail cards.
4. Add branch-level portfolio charts.
5. Add sector and industry exposure improvements.
6. Add stale-data warnings.
7. End the week with a UI review.