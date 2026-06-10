import os
import yaml
import pandas as pd


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/experiment_01.yaml")
    processed_dir = config["processed_data_dir"]

    input_path = os.path.join(processed_dir, "modeling_dataset.parquet")
    output_path = os.path.join("outputs", "reports", "week2_dataset_summary.txt")

    df = pd.read_parquet(input_path)

    non_feature_cols = [
        "date",
        "ticker",
        "future_12m_return",
        "future_12m_spy_return",
        "target_abs_direction",
        "target_outperform_spy",
    ]

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    summary_lines = []

    summary_lines.append("Week 2 Dataset Summary")
    summary_lines.append("======================")
    summary_lines.append("")
    summary_lines.append(f"Rows: {df.shape[0]}")
    summary_lines.append(f"Columns: {df.shape[1]}")
    summary_lines.append(f"Number of feature columns: {len(feature_cols)}")
    summary_lines.append(f"Start date: {df['date'].min()}")
    summary_lines.append(f"End date: {df['date'].max()}")
    summary_lines.append(f"Number of tickers: {df['ticker'].nunique()}")
    summary_lines.append("")
    summary_lines.append("Tickers:")
    summary_lines.append(", ".join(sorted(df["ticker"].unique())))
    summary_lines.append("")
    summary_lines.append("Target outperform SPY counts:")
    summary_lines.append(str(df["target_outperform_spy"].value_counts()))
    summary_lines.append("")
    summary_lines.append("Target outperform SPY proportions:")
    summary_lines.append(str(df["target_outperform_spy"].value_counts(normalize=True)))
    summary_lines.append("")
    summary_lines.append("Absolute direction counts:")
    summary_lines.append(str(df["target_abs_direction"].value_counts()))
    summary_lines.append("")
    summary_lines.append("Average future 12-month return:")
    summary_lines.append(str(df["future_12m_return"].mean()))
    summary_lines.append("")
    summary_lines.append("Average future 12-month SPY return:")
    summary_lines.append(str(df["future_12m_spy_return"].mean()))
    summary_lines.append("")
    summary_lines.append("Feature columns:")
    summary_lines.append(", ".join(feature_cols))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print("")
    print(f"Saved summary report to {output_path}")


if __name__ == "__main__":
    main()