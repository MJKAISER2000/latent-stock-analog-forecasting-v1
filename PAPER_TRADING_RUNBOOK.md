# Latent Market Twin — Paper Trading Runbook

## Current Final Model

The current paper-trading model is:

```text
70% original LightGBM ranker top20
30% stock latent-neighbor LightGBM ranker top10
inverse-volatility weighting
tech drawdown 20% risk filter