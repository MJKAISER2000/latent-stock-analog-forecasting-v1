import os
import pandas as pd


def read_csv_if_exists(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def format_table(df: pd.DataFrame) -> str:
    if df is None:
        return "File not found.\n"
    return df.to_string(index=False)


def main():
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    week3_metrics = read_csv_if_exists("outputs/tables/week3_baseline_metrics.csv")
    week3_portfolio = read_csv_if_exists("outputs/tables/week3_portfolio_signal.csv")

    week4_metrics = read_csv_if_exists("outputs/tables/week4_autoencoder_metrics.csv")
    week4_portfolio = read_csv_if_exists("outputs/tables/week4_autoencoder_portfolio_signal.csv")

    week6_stats = read_csv_if_exists("outputs/tables/week6_backtest_stats.csv")

    week7_metrics = read_csv_if_exists("outputs/tables/week7_sequence_model_metrics.csv")
    week7_portfolio = read_csv_if_exists("outputs/tables/week7_sequence_portfolio_signal.csv")

    report_path = "outputs/reports/final_latent_market_twin_report.md"

    lines = []

    lines.append("# Final Latent Market Twin Report")
    lines.append("")
    lines.append("## Project Goal")
    lines.append("")
    lines.append(
        "This project tested whether latent representation learning can compress high-dimensional "
        "stock, market, sector, and macroeconomic data into a lower-dimensional market representation "
        "while preserving useful predictive structure for one-year stock outperformance."
    )
    lines.append("")
    lines.append("The main target was:")
    lines.append("")
    lines.append("`target_outperform_spy = 1` if a stock's next 12-month return exceeded SPY's next 12-month return.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        "The first prototype used a small starter universe of large-cap stocks with monthly data. "
        "Features included trailing returns, moving average ratios, volatility, drawdown, SPY market features, "
        "macro variables, regime labels, yield curve features, CPI change, and sector one-hot features."
    )
    lines.append("")
    lines.append("## Week 3: Raw-Feature Baselines")
    lines.append("")
    lines.append("Baseline models tested:")
    lines.append("")
    lines.append("- logistic regression")
    lines.append("- random forest")
    lines.append("- gradient boosting")
    lines.append("")
    lines.append("### Week 3 Classification Metrics")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week3_metrics))
    lines.append("```")
    lines.append("")
    lines.append("### Week 3 Portfolio Signal")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week3_portfolio))
    lines.append("```")
    lines.append("")
    lines.append(
        "The raw-feature models produced promising first-pass results. Random forest and gradient boosting "
        "had test AUC values above random guessing, and the model-ranked top stocks had much higher average "
        "future 12-month returns than SPY."
    )
    lines.append("")
    lines.append("## Week 4: Autoencoder Latent Market Model")
    lines.append("")
    lines.append(
        "An autoencoder compressed the engineered feature space into latent dimensions of 2, 5, 10, and 20. "
        "The latent vectors were then used as inputs to logistic regression and gradient boosting classifiers."
    )
    lines.append("")
    lines.append("### Week 4 Classification Metrics")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week4_metrics))
    lines.append("```")
    lines.append("")
    lines.append("### Week 4 Portfolio Signal")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week4_portfolio))
    lines.append("```")
    lines.append("")
    lines.append(
        "The best latent model was the 20-dimensional autoencoder representation with gradient boosting. "
        "It produced a top-5 average future 12-month return comparable to or slightly better than the best raw-feature baseline, "
        "while compressing the original feature set into a much smaller representation."
    )
    lines.append("")
    lines.append("## Week 5: Latent Space Analysis")
    lines.append("")
    lines.append(
        "The latent space was analyzed using PCA and t-SNE. The t-SNE visualization showed clear sector clustering, "
        "suggesting that the autoencoder learned economically meaningful organization. Future return separation was weaker, "
        "but several latent dimensions showed measurable correlation with future 12-month returns and SPY outperformance."
    )
    lines.append("")
    lines.append("Main interpretation:")
    lines.append("")
    lines.append(
        "The autoencoder was not merely compressing noise. It learned sector-level structure and preserved information "
        "related to future stock behavior."
    )
    lines.append("")
    lines.append("## Week 6: Simplified Backtest")
    lines.append("")
    lines.append(
        "The best Week 4 model was used to build a simplified monthly-rebalanced portfolio backtest. "
        "The strategy selected the top 5 or top 10 model-ranked stocks each month and compared performance against SPY "
        "and an equal-weight all-stock benchmark."
    )
    lines.append("")
    lines.append("### Week 6 Backtest Stats")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week6_stats))
    lines.append("```")
    lines.append("")
    lines.append(
        "The latent top-5 portfolio strongly outperformed SPY and the equal-weight benchmark in the simplified backtest. "
        "However, performance was concentrated in repeated exposure to a small group of major winners such as NFLX, META, NVDA, "
        "TSLA, LLY, GOOGL, and AMZN."
    )
    lines.append("")
    lines.append("Important limitation:")
    lines.append("")
    lines.append(
        "The Week 6 backtest was simplified because it converted 12-month forward returns into approximate monthly returns. "
        "A more rigorous next version should use true realized next-month returns after each rebalance date."
    )
    lines.append("")
    lines.append("## Week 7: GRU Sequence Model")
    lines.append("")
    lines.append(
        "A time-aware model was tested using a 12-month sequence of features as input to a GRU neural network. "
        "The goal was to test whether movement through feature space improved prediction."
    )
    lines.append("")
    lines.append("### Week 7 Classification Metrics")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week7_metrics))
    lines.append("```")
    lines.append("")
    lines.append("### Week 7 Portfolio Signal")
    lines.append("")
    lines.append("```text")
    lines.append(format_table(week7_portfolio))
    lines.append("```")
    lines.append("")
    lines.append(
        "The GRU sequence model overfit heavily. It achieved very strong training performance but weak test performance. "
        "It beat SPY but failed to beat the equal-weight stock universe. The simpler static autoencoder model remained stronger."
    )
    lines.append("")
    lines.append("## Final Model Ranking")
    lines.append("")
    lines.append("1. **Best overall:** Week 4 autoencoder latent_dim=20 + gradient boosting")
    lines.append("2. **Strong baseline:** Week 3 raw-feature gradient boosting / random forest")
    lines.append("3. **Useful but simplified:** Week 6 latent top-5 backtest")
    lines.append("4. **Underperformed:** Week 7 GRU sequence model")
    lines.append("")
    lines.append("## Main Conclusion")
    lines.append("")
    lines.append(
        "The project provides early evidence that high-dimensional stock and market features can be compressed into a lower-dimensional "
        "latent representation while preserving meaningful stock-ranking information. The autoencoder latent representation matched or slightly "
        "improved the top-5 portfolio signal of the raw-feature baseline, and the latent space showed clear sector organization."
    )
    lines.append("")
    lines.append(
        "The strongest current evidence supports the static latent compression approach rather than the more complex GRU sequence model."
    )
    lines.append("")
    lines.append("## Key Limitations")
    lines.append("")
    lines.append("- The stock universe is small and large-cap-heavy.")
    lines.append("- The test period heavily rewards mega-cap growth and AI-related winners.")
    lines.append("- The Week 6 backtest is simplified and not yet a fully realistic trading simulation.")
    lines.append("- There may be sector-driven effects rather than stock-specific predictive skill.")
    lines.append("- Survivorship bias has not yet been fully handled.")
    lines.append("- Fundamentals and paid datasets have not yet been added.")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Expand the universe to S&P 500 or Russell 1000 stocks.")
    lines.append("2. Add true next-month realized return backtesting.")
    lines.append("3. Add sector-neutral portfolio testing.")
    lines.append("4. Add walk-forward retraining.")
    lines.append("5. Add survivorship-bias-aware data if available.")
    lines.append("6. Add fundamentals and valuation metrics more systematically.")
    lines.append("7. Compare latent models against stronger baselines like XGBoost or LightGBM.")
    lines.append("8. Test whether the latent model works in bear markets, not only growth-led bull regimes.")
    lines.append("")
    lines.append("## Research Log Summary")
    lines.append("")
    lines.append(
        "Built a full prototype latent market twin pipeline: data collection, feature engineering, raw-feature baselines, "
        "autoencoder latent compression, latent space visualization, simplified backtesting, and a first sequence-model attempt. "
        "The best result came from a 20-dimensional autoencoder latent representation combined with gradient boosting."
    )
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved final report to {report_path}")
    print("")
    print("Final report created successfully.")


if __name__ == "__main__":
    main()