import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("outputs/figures", exist_ok=True)

    metadata_path = "data/external/expanded_ticker_metadata.csv"
    output_path = "outputs/figures/week10_sector_distribution.png"

    meta = pd.read_csv(metadata_path)

    counts = meta["sector"].fillna("Unknown").value_counts().sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    counts.plot(kind="barh")
    plt.title("Week 10 Expanded Universe Sector Distribution")
    plt.xlabel("Number of Tickers")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved sector distribution chart to:", output_path)
    print("")
    print(counts.sort_values(ascending=False))


if __name__ == "__main__":
    main()