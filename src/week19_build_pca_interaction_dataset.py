import os
import pandas as pd
import numpy as np


INPUT_PATH = "data/processed/week19_full500_with_market_pca_latents.parquet"
OUTPUT_PATH = "data/processed/week19_full500_with_market_pca_interactions.parquet"
REPORT_PATH = "outputs/reports/week19_pca_interaction_dataset_summary.txt"


MARKET_LATENT_COLS = [
    "market_pca_z1",
    "market_pca_z2",
    "market_pca_z3",
    "market_pca_z4",
    "market_pca_z5",
    "market_pca_z6",
    "market_pca_reconstruction_error",
]

BASE_STOCK_INTERACTION_COLS = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",

    "ret_1m_minus_sector",
    "ret_3m_minus_sector",
    "ret_6m_minus_sector",
    "ret_12m_minus_sector",

    "ret_1m_sector_z",
    "ret_3m_sector_z",
    "ret_6m_sector_z",
    "ret_12m_sector_z",

    "vol_3m",
    "vol_6m",
    "vol_12m",

    "vol_3m_minus_sector",
    "vol_6m_minus_sector",
    "vol_12m_minus_sector",

    "vol_3m_sector_z",
    "vol_6m_sector_z",
    "vol_12m_sector_z",

    "stock_drawdown",
    "stock_drawdown_minus_sector",
    "stock_drawdown_sector_z",

    "price_to_ma_3m",
    "price_to_ma_6m",
    "price_to_ma_12m",
]


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("Loading PCA latent dataset...")
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    print("Input shape:", df.shape)

    available_latent_cols = [c for c in MARKET_LATENT_COLS if c in df.columns]
    available_stock_cols = [c for c in BASE_STOCK_INTERACTION_COLS if c in df.columns]

    missing_latent = [c for c in MARKET_LATENT_COLS if c not in df.columns]
    missing_stock = [c for c in BASE_STOCK_INTERACTION_COLS if c not in df.columns]

    print("Available latent columns:", available_latent_cols)
    print("Available stock columns:", available_stock_cols)

    if missing_latent:
        print("Missing latent columns:", missing_latent)

    if missing_stock:
        print("Missing stock interaction columns:", missing_stock)

    if len(available_latent_cols) == 0:
        raise ValueError("No market PCA latent columns found.")

    if len(available_stock_cols) == 0:
        raise ValueError("No stock columns found for interactions.")

    # Clean the source columns before creating interactions.
    for col in available_latent_cols + available_stock_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].fillna(0.0)

    interaction_cols = []

    print("Creating interaction features...")

    for stock_col in available_stock_cols:
        for latent_col in available_latent_cols:
            new_col = f"interaction__{stock_col}__x__{latent_col}"
            df[new_col] = df[stock_col] * df[latent_col]
            interaction_cols.append(new_col)

    # Optional compact regime cluster interactions.
    # Treat cluster as categorical-ish by creating simple flags if present.
    if "market_pca_regime_cluster" in df.columns:
        df["market_pca_regime_cluster"] = pd.to_numeric(
            df["market_pca_regime_cluster"],
            errors="coerce",
        ).fillna(-1).astype(int)

        clusters = sorted(df["market_pca_regime_cluster"].dropna().unique().tolist())

        for cluster in clusters:
            flag_col = f"market_pca_cluster_{cluster}_flag"
            df[flag_col] = (df["market_pca_regime_cluster"] == cluster).astype(int)

            for stock_col in available_stock_cols:
                new_col = f"interaction__{stock_col}__x__{flag_col}"
                df[new_col] = df[stock_col] * df[flag_col]
                interaction_cols.append(new_col)

    # Clean interactions.
    for col in interaction_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH)

    lines = []
    lines.append("Week 19 PCA Interaction Dataset Summary")
    lines.append("======================================")
    lines.append("")
    lines.append("Goal:")
    lines.append(
        "Create explicit stock-feature x latent-market-state interaction features."
    )
    lines.append("")
    lines.append(f"Input path: {INPUT_PATH}")
    lines.append(f"Output path: {OUTPUT_PATH}")
    lines.append(f"Input/output shape: {df.shape}")
    lines.append(f"Latent columns used: {len(available_latent_cols)}")
    lines.append(f"Stock columns used: {len(available_stock_cols)}")
    lines.append(f"Interaction feature count: {len(interaction_cols)}")
    lines.append("")
    lines.append("Latent columns:")
    lines.append(", ".join(available_latent_cols))
    lines.append("")
    lines.append("Stock interaction columns:")
    lines.append(", ".join(available_stock_cols))
    lines.append("")
    lines.append("Example interaction columns:")
    lines.append(", ".join(interaction_cols[:80]))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    print("\n".join(lines))
    print("")
    print("Saved:", OUTPUT_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()