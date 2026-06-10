import os
import pandas as pd
import numpy as np


def add_market_cap_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "market_cap" not in df.columns:
        df["market_cap_bucket"] = "Unknown"
        return df

    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    df["market_cap_bucket"] = pd.cut(
        df["market_cap"],
        bins=[-np.inf, 2e9, 10e9, 50e9, 200e9, np.inf],
        labels=[
            "Small",
            "Mid",
            "Large",
            "Mega",
            "UltraMega",
        ],
    )

    df["market_cap_bucket"] = df["market_cap_bucket"].astype(str)
    df.loc[df["market_cap"].isna(), "market_cap_bucket"] = "Unknown"

    return df


def add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sector_group = df.groupby(["date", "sector"])

    base_features = [
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "vol_3m",
        "vol_6m",
        "vol_12m",
        "stock_drawdown",
    ]

    for col in base_features:
        if col not in df.columns:
            continue

        sector_mean = sector_group[col].transform("mean")
        sector_std = sector_group[col].transform("std")

        df[f"{col}_sector_mean"] = sector_mean
        df[f"{col}_minus_sector"] = df[col] - sector_mean

        # z-score inside sector, with protection against zero std
        df[f"{col}_sector_z"] = (df[col] - sector_mean) / sector_std.replace(0, np.nan)
        df[f"{col}_sector_z"] = df[f"{col}_sector_z"].replace([np.inf, -np.inf], np.nan)
        df[f"{col}_sector_z"] = df[f"{col}_sector_z"].fillna(0.0)

    return df


def clean_after_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
        "sector",
        "industry",
        "long_name",
        "market_cap_bucket",
    ]

    # Keep market_cap as a numeric feature if present
    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    categorical_cols = ["sector", "industry", "market_cap_bucket"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    df = pd.get_dummies(
        df,
        columns=[col for col in categorical_cols if col in df.columns],
        prefix=[col for col in categorical_cols if col in df.columns],
        drop_first=False,
    )

    final_non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
        "long_name",
    ]

    feature_cols = [col for col in df.columns if col not in final_non_feature_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    missing_fraction = df[feature_cols].isna().mean()
    keep_feature_cols = missing_fraction[missing_fraction < 0.40].index.tolist()

    dropped = sorted(set(feature_cols) - set(keep_feature_cols))
    if dropped:
        print("Dropped mostly-missing columns:")
        print(dropped)

    feature_cols = keep_feature_cols

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    final_cols = final_non_feature_cols + feature_cols
    df = df[final_cols]

    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    modeling_path = "data/processed/expanded_modeling_dataset.parquet"
    metadata_path = "data/external/expanded_ticker_metadata.csv"

    output_path = "data/processed/expanded_modeling_dataset_with_metadata.parquet"
    report_path = "outputs/reports/week10_sector_metadata_summary.txt"

    print("Loading expanded modeling dataset...")
    df = pd.read_parquet(modeling_path)
    df["date"] = pd.to_datetime(df["date"])

    print("Loading metadata...")
    meta = pd.read_csv(metadata_path)

    meta["ticker"] = meta["ticker"].astype(str).str.strip()
    meta["sector"] = meta["sector"].fillna("Unknown")
    meta["industry"] = meta["industry"].fillna("Unknown")
    meta["long_name"] = meta["long_name"].fillna("")

    print("Merging metadata...")
    merged = df.merge(meta, on="ticker", how="left")

    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown")
    merged["long_name"] = merged["long_name"].fillna("")

    print("Adding market cap buckets...")
    merged = add_market_cap_bucket(merged)

    print("Adding sector-relative features...")
    merged = add_sector_relative_features(merged)

    print("Cleaning and one-hot encoding metadata...")
    clean = clean_after_metadata(merged)

    clean.to_parquet(output_path)

    lines = []
    lines.append("Week 10 Sector and Industry Metadata Summary")
    lines.append("===========================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Add real sector/industry metadata and sector-relative features to the expanded modeling dataset.")
    lines.append("")
    lines.append("Input dataset:")
    lines.append(str(df.shape))
    lines.append("")
    lines.append("Output dataset:")
    lines.append(str(clean.shape))
    lines.append("")
    lines.append("Missing values:")
    lines.append(str(clean.isna().sum().sum()))
    lines.append("")
    lines.append("Ticker count:")
    lines.append(str(clean["ticker"].nunique()))
    lines.append("")
    lines.append("Sector counts:")
    lines.append(str(merged[["ticker", "sector"]].drop_duplicates()["sector"].value_counts()))
    lines.append("")
    lines.append("Industry counts, top 25:")
    lines.append(str(merged[["ticker", "industry"]].drop_duplicates()["industry"].value_counts().head(25)))
    lines.append("")
    lines.append("Market cap bucket counts:")
    lines.append(str(merged[["ticker", "market_cap_bucket"]].drop_duplicates()["market_cap_bucket"].value_counts()))
    lines.append("")
    lines.append("New sector-relative features include:")
    lines.append("- return minus sector average")
    lines.append("- return sector z-score")
    lines.append("- volatility minus sector average")
    lines.append("- volatility sector z-score")
    lines.append("- drawdown minus sector average")
    lines.append("- drawdown sector z-score")
    lines.append("")
    lines.append("Output file:")
    lines.append(output_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))
    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()