import os
import pandas as pd


def main():
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    metadata_path = "data/external/expanded_ticker_metadata.csv"
    output_report = "outputs/reports/week10_sector_balance_diagnostic.txt"
    output_table = "outputs/tables/week10_sector_counts.csv"

    meta = pd.read_csv(metadata_path)

    sector_counts = (
        meta["sector"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("sector")
        .reset_index(name="ticker_count")
    )

    sector_counts["percentage"] = sector_counts["ticker_count"] / sector_counts["ticker_count"].sum()

    sector_counts.to_csv(output_table, index=False)

    top_sector = sector_counts.iloc[0]["sector"]
    top_sector_pct = sector_counts.iloc[0]["percentage"]

    lines = []
    lines.append("Week 10 Sector Balance Diagnostic")
    lines.append("=================================")
    lines.append("")
    lines.append("Sector counts:")
    lines.append(sector_counts.to_string(index=False))
    lines.append("")
    lines.append(f"Most represented sector: {top_sector}")
    lines.append(f"Most represented sector percentage: {top_sector_pct:.2%}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "The expanded NASDAQ-style universe is not sector-balanced. Technology and related growth sectors "
        "are likely overrepresented. This is expected for a NASDAQ-style universe, but it means later model "
        "results must be tested against sector-neutral and sector-balanced benchmarks."
    )
    lines.append("")
    lines.append("Research implication:")
    lines.append(
        "If the latent model performs well in this universe, some of the signal may come from sector allocation "
        "rather than stock-specific selection. Week 13 should explicitly test sector-neutral portfolios."
    )
    lines.append("")
    lines.append("Next dataset upgrade:")
    lines.append(
        "Add a broader S&P-style universe or create a sector-balanced sample so the model is not dominated by Technology."
    )

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print("Saved:", output_report)
    print("Saved:", output_table)


if __name__ == "__main__":
    main()