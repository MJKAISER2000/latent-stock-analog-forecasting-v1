import os
import pandas as pd
import numpy as np


BASE_DATASET_PATH = "data/processed/week15_full500_modeling_dataset.parquet"
UNIVERSE_PATH = "data/external/week15_500_stock_universe.csv"
OUTPUT_PATH = "data/processed/week18_full500_leadership_modeling_dataset.parquet"
REPORT_PATH = "outputs/reports/week18_full500_leadership_dataset_summary.txt"

MOMENTUM_WINDOWS = [1, 3, 6, 12]


def safe_zscore(x: pd.Series) -> pd.Series:
    std = x.std()

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=x.index)

    return (x - x.mean()) / std


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading base Week 15 full500 modeling dataset...")
    df = pd.read_parquet(BASE_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    print("Loading universe metadata...")
    universe = pd.read_csv(UNIVERSE_PATH)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()

    keep_cols = ["ticker", "sector", "industry"]

    meta = universe[keep_cols].copy()
    meta["sector"] = meta["sector"].fillna("Unknown").astype(str)
    meta["industry"] = meta["industry"].fillna("Unknown").astype(str)

    # The base dataset may already have one-hot encoded sector columns,
    # but we need raw sector/industry labels for group leadership features.
    df = df.merge(meta, on="ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    df["industry"] = df["industry"].fillna("Unknown").astype(str)

    print("Building sector and industry leadership features...")

    for window in MOMENTUM_WINDOWS:
        ret_col = f"ret_{window}m"
        spy_col = f"spy_ret_{window}m"

        if ret_col not in df.columns:
            print(f"Skipping window {window}: missing {ret_col}")
            continue

        # Sector average momentum at each date.
        sector_ret = df.groupby(["date", "sector"])[ret_col].transform("mean")
        industry_ret = df.groupby(["date", "industry"])[ret_col].transform("mean")

        df[f"sector_ret_{window}m"] = sector_ret
        df[f"industry_ret_{window}m"] = industry_ret

        # Leadership versus SPY.
        if spy_col in df.columns:
            df[f"sector_minus_spy_{window}m"] = df[f"sector_ret_{window}m"] - df[spy_col]
            df[f"industry_minus_spy_{window}m"] = df[f"industry_ret_{window}m"] - df[spy_col]

        # Stock relative to its group.
        df[f"stock_minus_sector_{window}m"] = df[ret_col] - df[f"sector_ret_{window}m"]
        df[f"stock_minus_industry_{window}m"] = df[ret_col] - df[f"industry_ret_{window}m"]

        # Group ranks by date.
        sector_rank_table = (
            df[["date", "sector", f"sector_ret_{window}m"]]
            .drop_duplicates()
            .copy()
        )

        sector_rank_table[f"sector_rank_{window}m"] = sector_rank_table.groupby("date")[
            f"sector_ret_{window}m"
        ].rank(ascending=False, method="dense")

        industry_rank_table = (
            df[["date", "industry", f"industry_ret_{window}m"]]
            .drop_duplicates()
            .copy()
        )

        industry_rank_table[f"industry_rank_{window}m"] = industry_rank_table.groupby("date")[
            f"industry_ret_{window}m"
        ].rank(ascending=False, method="dense")

        df = df.merge(
            sector_rank_table[["date", "sector", f"sector_rank_{window}m"]],
            on=["date", "sector"],
            how="left",
        )

        df = df.merge(
            industry_rank_table[["date", "industry", f"industry_rank_{window}m"]],
            on=["date", "industry"],
            how="left",
        )

        # Cross-sectional z-scores of leadership features.
        for col in [
            f"sector_ret_{window}m",
            f"industry_ret_{window}m",
            f"sector_minus_spy_{window}m",
            f"industry_minus_spy_{window}m",
            f"stock_minus_sector_{window}m",
            f"stock_minus_industry_{window}m",
        ]:
            if col in df.columns:
                df[f"{col}_date_z"] = df.groupby("date")[col].transform(safe_zscore)

    # One-hot encode raw sector/industry labels.
    # This keeps metadata usable but also gives model explicit categorical structure.
    print("One-hot encoding raw sector/industry labels...")
    df = pd.get_dummies(
        df,
        columns=["sector", "industry"],
        prefix=["raw_sector", "raw_industry"],
        drop_first=False,
    )

    # Clean numeric columns.
    non_numeric_keep = ["date", "ticker", "company"]

    for col in df.columns:
        if col not in non_numeric_keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    target_cols = [
        "future_1m_return",
        "future_1m_spy_return",
        "future_1m_excess_return",
        "target_outperform_spy_1m",
        "target_top_quintile_1m",
        "future_36m_return",
        "future_36m_spy_return",
        "future_36m_excess_return",
        "target_outperform_spy_36m",
        "target_top_quintile_36m",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in non_numeric_keep and c not in target_cols
    ]

    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH)

    leadership_cols = [
        c for c in df.columns
        if (
            "sector_ret_" in c
            or "industry_ret_" in c
            or "sector_minus_spy_" in c
            or "industry_minus_spy_" in c
            or "stock_minus_sector_" in c
            or "stock_minus_industry_" in c
            or "sector_rank_" in c
            or "industry_rank_" in c
        )
    ]

    lines = []
    lines.append("Week 18 Full500 Leadership Feature Dataset Summary")
    lines.append("=================================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Add sector and industry leadership features to the Week 15 full500 modeling dataset.")
    lines.append("")
    lines.append(f"Output shape: {df.shape}")
    lines.append(f"Ticker count: {df['ticker'].nunique()}")
    lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    lines.append(f"Leadership feature count: {len(leadership_cols)}")
    lines.append("")
    lines.append("Example leadership columns:")
    lines.append(", ".join(leadership_cols[:60]))
    lines.append("")
    lines.append("Output file:")
    lines.append(OUTPUT_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", OUTPUT_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()