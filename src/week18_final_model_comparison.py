import os
import pandas as pd
import numpy as np


def load_first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"None found: {paths}")


def clean_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    rows = []

    metric_cols = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_no_risk_free",
        "max_drawdown",
        "win_rate",
        "return_over_abs_drawdown",
    ]

    # -----------------------------
    # Week 15 raw / AE comparison
    # -----------------------------
    week15_stats_path = "outputs/tables/week15_raw_vs_ae_backtest_stats.csv"

    if os.path.exists(week15_stats_path):
        w15 = pd.read_csv(week15_stats_path)
        w15 = clean_numeric(w15, metric_cols)

        candidates = [
            {
                "label": "Week 15 raw GB highvol100 tactical champion",
                "strategy_match": "highvol100_raw_gb_h1_top5_vol20_inverse_vol",
            },
            {
                "label": "Week 15 AE full500 structural champion",
                "strategy_match": "full500_ae_gb_h36_top10_base",
            },
            {
                "label": "SPY benchmark",
                "strategy_match": "spy",
            },
            {
                "label": "Equal weight benchmark",
                "strategy_match": "equal_weight",
            },
        ]

        for c in candidates:
            temp = w15[w15["strategy"].astype(str) == c["strategy_match"]].copy()

            if len(temp) == 0:
                continue

            r = temp.iloc[0].to_dict()

            rows.append(
                {
                    "model_label": c["label"],
                    "source_week": "Week 15",
                    "strategy": r.get("strategy"),
                    "model_type": r.get("model_type"),
                    "universe": r.get("dataset"),
                    "config": r.get("strategy_config"),
                    "notes": "From Week 15 raw-vs-AE backtest",
                    **{col: r.get(col, np.nan) for col in metric_cols},
                }
            )

    # -----------------------------
    # Week 17 ranker backtest
    # -----------------------------
    week17_stats_path = "outputs/tables/week17_lgbm_ranker_backtest_stats.csv"

    if os.path.exists(week17_stats_path):
        w17 = pd.read_csv(week17_stats_path)
        w17 = clean_numeric(w17, metric_cols)

        candidates = [
            {
                "label": "Week 17 ranker top5 aggressive champion",
                "strategy_match": "week15_full500_lgbm_ranker_h1_top5_inverse_vol",
            },
            {
                "label": "Week 17 ranker top10 diversified candidate",
                "strategy_match": None,
            },
        ]

        for c in candidates:
            if c["strategy_match"] is not None:
                temp = w17[w17["strategy"].astype(str) == c["strategy_match"]].copy()
            else:
                # The top10 ranker was created in top-N stress rather than this original backtest.
                temp = pd.DataFrame()

            if len(temp) == 0:
                continue

            r = temp.iloc[0].to_dict()

            rows.append(
                {
                    "model_label": c["label"],
                    "source_week": "Week 17",
                    "strategy": r.get("strategy"),
                    "model_type": r.get("model_type"),
                    "universe": r.get("dataset"),
                    "config": r.get("strategy_config"),
                    "notes": "From Week 17 ranker backtest",
                    **{col: r.get(col, np.nan) for col in metric_cols},
                }
            )

    # -----------------------------
    # Week 17 Top-N stress test
    # -----------------------------
    topn_path = "outputs/tables/week17_ranker_topn_stress_stats.csv"

    if os.path.exists(topn_path):
        topn = pd.read_csv(topn_path)
        topn = clean_numeric(topn, metric_cols + ["top_n"])

        topn_candidates = [
            {
                "label": "Week 17 ranker top5 aggressive champion",
                "top_n": 5,
            },
            {
                "label": "Week 17 ranker top10 diversified candidate",
                "top_n": 10,
            },
            {
                "label": "Week 17 ranker top20 robustness reference",
                "top_n": 20,
            },
        ]

        for c in topn_candidates:
            temp = topn[topn["top_n"] == c["top_n"]].copy()

            if len(temp) == 0:
                continue

            r = temp.iloc[0].to_dict()

            rows.append(
                {
                    "model_label": c["label"],
                    "source_week": "Week 17",
                    "strategy": r.get("strategy"),
                    "model_type": "lgbm_ranker",
                    "universe": "week15_full500",
                    "config": f"top{int(c['top_n'])}_inverse_vol",
                    "notes": "From Week 17 top-N stress test",
                    **{col: r.get(col, np.nan) for col in metric_cols},
                }
            )

    # -----------------------------
    # Week 18 top5 regime filter
    # -----------------------------
    top5_filter_path = "outputs/tables/week18_tech_regime_filter_stats.csv"

    if os.path.exists(top5_filter_path):
        top5 = pd.read_csv(top5_filter_path)
        top5 = clean_numeric(top5, metric_cols)

        candidates = [
            {
                "label": "Week 18 top5 ranker + tech 6m under SPY cash filter",
                "rule_name": "tech_6m_under_spy_100cash",
            },
            {
                "label": "Week 18 top5 ranker + tech drawdown filter",
                "rule_name": "tech_drawdown_20_50cash",
            },
        ]

        for c in candidates:
            temp = top5[top5["rule_name"].astype(str) == c["rule_name"]].copy()

            if len(temp) == 0:
                continue

            r = temp.iloc[0].to_dict()

            rows.append(
                {
                    "model_label": c["label"],
                    "source_week": "Week 18",
                    "strategy": c["rule_name"],
                    "model_type": "lgbm_ranker_regime_filter",
                    "universe": "week15_full500",
                    "config": "top5_inverse_vol",
                    "notes": "From Week 18 top5 tech-regime filter",
                    **{col: r.get(col, np.nan) for col in metric_cols},
                }
            )

    # -----------------------------
    # Week 18 top10 regime filter
    # -----------------------------
    top10_filter_path = "outputs/tables/week18_top10_tech_regime_filter_stats.csv"

    if os.path.exists(top10_filter_path):
        top10 = pd.read_csv(top10_filter_path)
        top10 = clean_numeric(top10, metric_cols)

        candidates = [
            {
                "label": "Week 18 top10 ranker + tech 12m under SPY cash filter",
                "rule_name": "tech_12m_under_spy_100cash",
            },
            {
                "label": "Week 18 top10 ranker + tech 6m under SPY cash filter",
                "rule_name": "tech_6m_under_spy_100cash",
            },
        ]

        for c in candidates:
            temp = top10[top10["rule_name"].astype(str) == c["rule_name"]].copy()

            if len(temp) == 0:
                continue

            r = temp.iloc[0].to_dict()

            rows.append(
                {
                    "model_label": c["label"],
                    "source_week": "Week 18",
                    "strategy": c["rule_name"],
                    "model_type": "lgbm_ranker_regime_filter",
                    "universe": "week15_full500",
                    "config": "top10_inverse_vol",
                    "notes": "From Week 18 top10 tech-regime filter",
                    **{col: r.get(col, np.nan) for col in metric_cols},
                }
            )

    comparison = pd.DataFrame(rows)

    if comparison.empty:
        raise ValueError("No comparison rows created. Check that output CSV files exist.")

    comparison = comparison.drop_duplicates(
        subset=["model_label", "strategy", "config"],
        keep="first",
    )

    comparison = comparison.sort_values("return_over_abs_drawdown", ascending=False).reset_index(drop=True)

    output_path = "outputs/tables/week18_final_model_comparison.csv"
    report_path = "outputs/reports/week18_final_model_comparison_summary.txt"

    comparison.to_csv(output_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Week 18 Final Model Comparison\n")
        f.write("==============================\n\n")
        f.write("Goal:\n")
        f.write("Compare the major champion models from Weeks 15-18 in one leaderboard.\n\n")
        f.write("Sorted by return over absolute drawdown:\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n")
        f.write("Top model by annualized return:\n")
        f.write(comparison.sort_values("annualized_return", ascending=False).head(5).to_string(index=False))
        f.write("\n\n")
        f.write("Top model by Sharpe:\n")
        f.write(comparison.sort_values("sharpe_no_risk_free", ascending=False).head(5).to_string(index=False))
        f.write("\n\n")
        f.write("Top model by drawdown-adjusted return:\n")
        f.write(comparison.sort_values("return_over_abs_drawdown", ascending=False).head(5).to_string(index=False))
        f.write("\n")

    print("")
    print("Saved:", output_path)
    print("Saved:", report_path)

    display_cols = [
        "model_label",
        "source_week",
        "config",
        "annualized_return",
        "annualized_volatility",
        "sharpe_no_risk_free",
        "max_drawdown",
        "win_rate",
        "return_over_abs_drawdown",
    ]

    print("")
    print("FINAL COMPARISON — SORTED BY RETURN / DRAWDOWN")
    print(comparison[display_cols].to_string(index=False))

    print("")
    print("TOP BY ANNUALIZED RETURN")
    print(
        comparison[display_cols]
        .sort_values("annualized_return", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("")
    print("TOP BY SHARPE")
    print(
        comparison[display_cols]
        .sort_values("sharpe_no_risk_free", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()