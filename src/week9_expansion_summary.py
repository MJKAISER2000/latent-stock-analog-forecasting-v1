import os
import pandas as pd


def main():
    os.makedirs("outputs/reports", exist_ok=True)

    universe_path = "data/external/expanded_ticker_universe.csv"
    prices_path = "data/processed/expanded_monthly_prices.parquet"
    targets_path = "data/processed/expanded_targets.parquet"
    modeling_path = "data/processed/expanded_modeling_dataset.parquet"
    output_path = "outputs/reports/week9_universe_expansion_summary.txt"

    universe = pd.read_csv(universe_path)
    prices = pd.read_parquet(prices_path)
    targets = pd.read_parquet(targets_path)
    modeling = pd.read_parquet(modeling_path)

    targets["date"] = pd.to_datetime(targets["date"])
    modeling["date"] = pd.to_datetime(modeling["date"])

    lines = []

    lines.append("Week 9 Universe Expansion Summary")
    lines.append("=================================")
    lines.append("")
    lines.append("Goal:")
    lines.append(
        "Expand the latent market twin project beyond the original 20-stock starter universe "
        "into a larger NASDAQ-style stock universe."
    )
    lines.append("")
    lines.append("Files created:")
    lines.append("- data/external/expanded_ticker_universe.csv")
    lines.append("- data/raw/expanded_stock_prices_raw.parquet")
    lines.append("- data/processed/expanded_monthly_prices.parquet")
    lines.append("- data/processed/expanded_targets.parquet")
    lines.append("- data/processed/expanded_market_regimes.parquet")
    lines.append("- data/processed/expanded_features.parquet")
    lines.append("- data/processed/expanded_modeling_dataset.parquet")
    lines.append("")
    lines.append("Universe summary:")
    lines.append(f"Tickers listed in universe file: {universe['ticker'].nunique()}")
    lines.append(f"Tickers with usable target rows: {targets['ticker'].nunique()}")
    lines.append(f"Tickers in final modeling dataset: {modeling['ticker'].nunique()}")
    lines.append("")
    lines.append("Monthly price data:")
    lines.append(f"Monthly prices shape: {prices.shape}")
    lines.append(f"Monthly price date range: {prices.index.min()} to {prices.index.max()}")
    lines.append("")
    lines.append("Expanded targets:")
    lines.append(f"Targets shape: {targets.shape}")
    lines.append(f"Targets date range: {targets['date'].min()} to {targets['date'].max()}")
    lines.append("Target outperform SPY balance:")
    lines.append(str(targets["target_outperform_spy"].value_counts(normalize=True)))
    lines.append("")
    lines.append("Expanded modeling dataset:")
    lines.append(f"Modeling shape: {modeling.shape}")
    lines.append(f"Missing values: {modeling.isna().sum().sum()}")
    lines.append(f"Modeling date range: {modeling['date'].min()} to {modeling['date'].max()}")
    lines.append("Modeling target balance:")
    lines.append(str(modeling["target_outperform_spy"].value_counts(normalize=True)))
    lines.append("")
    lines.append("Rows per ticker, top 20:")
    lines.append(str(modeling["ticker"].value_counts().head(20)))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "Week 9 successfully expanded the project from a small hand-picked starter universe "
        "to a larger NASDAQ-style universe. This makes the next experiments more reflective "
        "of real stock selection and reduces dependence on a small set of mega-cap names."
    )
    lines.append("")
    lines.append("Important caution:")
    lines.append(
        "This expanded universe is still not fully survivorship-bias-free and is still tilted "
        "toward large NASDAQ-style companies. Future upgrades should use historical index membership "
        "or a survivorship-bias-aware dataset."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print(f"Saved Week 9 summary to {output_path}")


if __name__ == "__main__":
    main()