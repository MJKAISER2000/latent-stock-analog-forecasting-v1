import os
import pandas as pd


def main():
    os.makedirs("data/external", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    vol_path = "outputs/tables/week15_500_volatility_rankings.csv"

    low100_path = "data/external/week16_lowvol100_universe.csv"
    mid100_path = "data/external/week16_midvol100_universe.csv"
    high100_path = "data/external/week16_highvol100_universe.csv"

    balanced300_path = "data/external/week16_balanced300_universe.csv"
    balanced150_path = "data/external/week16_balanced150_universe.csv"

    report_path = "outputs/reports/week16_balanced_universe_summary.txt"

    print("Loading Week 15 volatility rankings...")
    vol = pd.read_csv(vol_path)

    vol["ticker"] = vol["ticker"].astype(str).str.strip().str.upper()
    vol = vol.sort_values("full_sample_vol", ascending=True).reset_index(drop=True)

    n = len(vol)

    low100 = vol.head(100).copy()
    high100 = vol.tail(100).sort_values("full_sample_vol", ascending=False).copy()

    mid_start = max((n // 2) - 50, 0)
    mid100 = vol.iloc[mid_start:mid_start + 100].copy()

    low50 = vol.head(50).copy()
    high50 = vol.tail(50).sort_values("full_sample_vol", ascending=False).copy()

    mid_start_50 = max((n // 2) - 25, 0)
    mid50 = vol.iloc[mid_start_50:mid_start_50 + 50].copy()

    low100["vol_bucket"] = "low"
    mid100["vol_bucket"] = "mid"
    high100["vol_bucket"] = "high"

    low50["vol_bucket"] = "low"
    mid50["vol_bucket"] = "mid"
    high50["vol_bucket"] = "high"

    balanced300 = pd.concat([low100, mid100, high100], ignore_index=True)
    balanced300 = balanced300.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    balanced150 = pd.concat([low50, mid50, high50], ignore_index=True)
    balanced150 = balanced150.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    low100.to_csv(low100_path, index=False)
    mid100.to_csv(mid100_path, index=False)
    high100.to_csv(high100_path, index=False)
    balanced300.to_csv(balanced300_path, index=False)
    balanced150.to_csv(balanced150_path, index=False)

    lines = []
    lines.append("Week 16 Balanced Universe Summary")
    lines.append("=================================")
    lines.append("")
    lines.append("Goal:")
    lines.append("Create representative universes that mix low-volatility, mid-volatility, and high-volatility stocks.")
    lines.append("")
    lines.append(f"Total ranked universe size: {n}")
    lines.append("")
    lines.append(f"lowvol100 shape: {low100.shape}")
    lines.append(f"midvol100 shape: {mid100.shape}")
    lines.append(f"highvol100 shape: {high100.shape}")
    lines.append(f"balanced300 shape: {balanced300.shape}")
    lines.append(f"balanced150 shape: {balanced150.shape}")
    lines.append("")
    lines.append("Balanced300 bucket counts:")
    lines.append(str(balanced300["vol_bucket"].value_counts()))
    lines.append("")
    lines.append("Balanced150 bucket counts:")
    lines.append(str(balanced150["vol_bucket"].value_counts()))
    lines.append("")
    lines.append("Lowest volatility sample:")
    lines.append(low100[["ticker", "company", "sector", "full_sample_vol", "max_drawdown"]].head(20).to_string(index=False))
    lines.append("")
    lines.append("Middle volatility sample:")
    lines.append(mid100[["ticker", "company", "sector", "full_sample_vol", "max_drawdown"]].head(20).to_string(index=False))
    lines.append("")
    lines.append("Highest volatility sample:")
    lines.append(high100[["ticker", "company", "sector", "full_sample_vol", "max_drawdown"]].head(20).to_string(index=False))
    lines.append("")
    lines.append("Output files:")
    lines.append(low100_path)
    lines.append(mid100_path)
    lines.append(high100_path)
    lines.append(balanced300_path)
    lines.append(balanced150_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", low100_path)
    print("Saved:", mid100_path)
    print("Saved:", high100_path)
    print("Saved:", balanced300_path)
    print("Saved:", balanced150_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()