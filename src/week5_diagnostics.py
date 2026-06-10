import os
import pandas as pd


def main():
    input_path = "outputs/tables/week5_latent_test_coordinates.csv"
    output_path = "outputs/reports/week5_diagnostics.txt"

    df = pd.read_csv(input_path)

    zcols = [c for c in df.columns if c.startswith("z_")]

    sector_return = (
        df.groupby("sector_label")["future_12m_return"]
        .mean()
        .sort_values(ascending=False)
    )

    sector_outperform = (
        df.groupby("sector_label")["target_outperform_spy"]
        .mean()
        .sort_values(ascending=False)
    )

    sector_counts = df["sector_label"].value_counts()

    corrs = df[zcols + ["future_12m_return", "target_outperform_spy"]].corr()

    return_corr = (
        corrs["future_12m_return"]
        .loc[zcols]
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

    outperform_corr = (
        corrs["target_outperform_spy"]
        .loc[zcols]
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

    lines = []

    lines.append("Week 5 Latent Space Diagnostics")
    lines.append("===============================")
    lines.append("")
    lines.append("Main qualitative observation:")
    lines.append(
        "The latent space is mixed overall, but t-SNE visualizations show clear sector clustering "
        "and partial structure by future return bucket."
    )
    lines.append("")
    lines.append("Average future 12-month return by sector:")
    lines.append(str(sector_return))
    lines.append("")
    lines.append("SPY outperformance rate by sector:")
    lines.append(str(sector_outperform))
    lines.append("")
    lines.append("Sector row counts:")
    lines.append(str(sector_counts))
    lines.append("")
    lines.append("Latent coordinate correlations with future 12-month return:")
    lines.append(str(return_corr))
    lines.append("")
    lines.append("Latent coordinate correlations with SPY outperformance target:")
    lines.append(str(outperform_corr))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "The autoencoder latent representation appears to encode economically meaningful structure. "
        "Sector clustering is strong in t-SNE space, and several latent dimensions show measurable "
        "correlation with future 12-month returns and SPY outperformance. This supports the idea "
        "that the high-dimensional market feature space can be compressed while preserving useful "
        "market organization."
    )
    lines.append("")
    lines.append("Caution:")
    lines.append(
        "The test universe is still small and large-cap-heavy. Some signal may come from sector effects, "
        "especially technology and communication outperformance during the test period. Future weeks "
        "should test whether the model remains useful after expanding the universe and controlling for sector."
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("")
    print(f"Saved diagnostics report to {output_path}")


if __name__ == "__main__":
    main()