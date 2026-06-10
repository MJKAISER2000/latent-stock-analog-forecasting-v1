import os
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import (
    load_config,
    ensure_output_dirs,
    check_required_files,
    print_config_summary,
)

from src.data.loaders import (
    load_neighbor_dataset,
    load_monthly_prices,
    load_universe,
    get_latest_available_signal_date,
)

from src.features.feature_sets import (
    print_feature_set_summary,
    prepare_feature_matrix,
)

from src.models.rankers import train_predict_latest

from src.paper_trading.portfolio import build_final_portfolio

from src.paper_trading.orders import build_order_sheet


CONFIG_PATH = "configs/live_model_config.yaml"


def prepare_dataset_for_training(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    working = df.copy()

    X = prepare_feature_matrix(working, feature_cols)

    for col in feature_cols:
        working[col] = X[col]

    return working


def append_live_signal_ledger(
    final_portfolio: pd.DataFrame,
    regime_status: dict,
    config: dict,
    run_timestamp: str,
) -> str:
    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    ledger_path = os.path.join(outputs_dir, "live_portfolio_signals.csv")

    rows = final_portfolio.copy()

    model_name = config["final_model_name"]
    signal_date = pd.Timestamp(regime_status["signal_date"]).normalize()

    rows["run_timestamp"] = run_timestamp
    rows["model_name"] = model_name
    rows["signal_date"] = signal_date
    rows["regime_price_date"] = pd.Timestamp(regime_status["regime_price_date"]).normalize()
    rows["regime_rule"] = regime_status["rule"]
    rows["regime_risk_on"] = regime_status["risk_on"]
    rows["tech_drawdown"] = regime_status["tech_drawdown"]
    rows["cash_weight_when_off"] = regime_status["cash_weight_when_off"]

    preferred_cols = [
        "run_timestamp",
        "model_name",
        "signal_date",
        "regime_price_date",
        "regime_rule",
        "regime_risk_on",
        "tech_drawdown",
        "cash_weight_when_off",
        "ticker",
        "final_weight",
        "final_weight_before_regime",
        "branches",
        "avg_ranker_score",
        "best_rank",
        "date",
    ]

    remaining_cols = [c for c in rows.columns if c not in preferred_cols]
    rows = rows[[c for c in preferred_cols if c in rows.columns] + remaining_cols]

    if os.path.exists(ledger_path):
        existing = pd.read_csv(ledger_path)
        existing["signal_date"] = pd.to_datetime(
            existing["signal_date"],
            errors="coerce",
        ).dt.normalize()

        keep = ~(
            (existing["model_name"] == model_name)
            & (existing["signal_date"] == signal_date)
        )

        existing = existing[keep].copy()
        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows.copy()

    combined["signal_date"] = pd.to_datetime(
        combined["signal_date"],
        errors="coerce",
    ).dt.normalize()

    combined = combined.sort_values(
        ["signal_date", "model_name", "final_weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    combined.to_csv(ledger_path, index=False)

    return ledger_path


def append_live_run_summary(
    final_portfolio: pd.DataFrame,
    regime_status: dict,
    config: dict,
    run_timestamp: str,
) -> str:
    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    summary_path = os.path.join(outputs_dir, "live_run_summary.csv")

    model_name = config["final_model_name"]
    signal_date = pd.Timestamp(regime_status["signal_date"]).normalize()

    stock_portfolio = final_portfolio[final_portfolio["ticker"] != "CASH"].copy()

    cash_weight = float(
        final_portfolio.loc[
            final_portfolio["ticker"] == "CASH",
            "final_weight",
        ].sum()
    )

    row = {
        "run_timestamp": run_timestamp,
        "model_name": model_name,
        "signal_date": signal_date,
        "regime_price_date": pd.Timestamp(regime_status["regime_price_date"]).normalize(),
        "regime_rule": regime_status["rule"],
        "regime_risk_on": regime_status["risk_on"],
        "tech_drawdown": regime_status["tech_drawdown"],
        "cash_weight_when_off": regime_status["cash_weight_when_off"],
        "portfolio_name_count": len(stock_portfolio),
        "cash_weight": cash_weight,
        "largest_weight": float(final_portfolio["final_weight"].max()),
        "top_ticker": str(final_portfolio.iloc[0]["ticker"]),
    }

    new_row = pd.DataFrame([row])

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing["signal_date"] = pd.to_datetime(
            existing["signal_date"],
            errors="coerce",
        ).dt.normalize()

        keep = ~(
            (existing["model_name"] == model_name)
            & (existing["signal_date"] == signal_date)
        )

        existing = existing[keep].copy()
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row.copy()

    combined["signal_date"] = pd.to_datetime(
        combined["signal_date"],
        errors="coerce",
    ).dt.normalize()

    combined = combined.sort_values(["signal_date", "model_name"]).reset_index(drop=True)

    combined.to_csv(summary_path, index=False)

    return summary_path


def append_live_order_ledger(
    orders: pd.DataFrame,
    config: dict,
) -> str:
    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    ledger_path = os.path.join(outputs_dir, "live_order_ledger.csv")

    rows = orders.copy()
    model_name = config["final_model_name"]
    rows["model_name"] = model_name

    signal_date = pd.Timestamp(rows["signal_date"].iloc[0]).normalize()

    if os.path.exists(ledger_path):
        existing = pd.read_csv(ledger_path)
        existing["signal_date"] = pd.to_datetime(
            existing["signal_date"],
            errors="coerce",
        ).dt.normalize()

        keep = ~(
            (existing["model_name"] == model_name)
            & (existing["signal_date"] == signal_date)
        )

        existing = existing[keep].copy()
        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows.copy()

    combined["signal_date"] = pd.to_datetime(
        combined["signal_date"],
        errors="coerce",
    ).dt.normalize()

    combined = combined.sort_values(
        ["signal_date", "model_name", "final_weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    combined.to_csv(ledger_path, index=False)

    return ledger_path


def main():
    config = load_config(CONFIG_PATH)

    ensure_output_dirs(config)
    check_required_files(config)
    print_config_summary(config)

    dataset = load_neighbor_dataset(config)
    prices = load_monthly_prices(config)
    universe = load_universe(config)

    signal_date = get_latest_available_signal_date(dataset)

    print("")
    print("=" * 100)
    print("LIVE FINAL PIPELINE")
    print("=" * 100)
    print("Signal date:", signal_date)
    print("Dataset shape:", dataset.shape)
    print("Dataset date range:", dataset["date"].min(), "to", dataset["date"].max())
    print("Ticker count:", dataset["ticker"].nunique())
    print("Prices shape:", prices.shape)
    print("Price date range:", prices.index.min(), "to", prices.index.max())

    original_feature_set = config["portfolio"]["original_branch"]["feature_set"]
    neighbor_feature_set = config["portfolio"]["latent_neighbor_branch"]["feature_set"]

    original_features = print_feature_set_summary(dataset, original_feature_set)
    neighbor_features = print_feature_set_summary(dataset, neighbor_feature_set)

    working_original = prepare_dataset_for_training(dataset, original_features)
    working_neighbor = prepare_dataset_for_training(dataset, neighbor_features)

    print("")
    print("=" * 100)
    print("TRAINING LIVE ORIGINAL BRANCH")
    print("=" * 100)

    original_model, original_predictions = train_predict_latest(
        df=working_original,
        feature_cols=original_features,
        config=config,
        signal_date=signal_date,
    )

    original_predictions["branch"] = "original"
    original_predictions["feature_set"] = original_feature_set

    print("")
    print("=" * 100)
    print("TRAINING LIVE LATENT-NEIGHBOR BRANCH")
    print("=" * 100)

    neighbor_model, neighbor_predictions = train_predict_latest(
        df=working_neighbor,
        feature_cols=neighbor_features,
        config=config,
        signal_date=signal_date,
    )

    neighbor_predictions["branch"] = "latent_neighbor"
    neighbor_predictions["feature_set"] = neighbor_feature_set

    final_portfolio, branch_detail, regime_status = build_final_portfolio(
        original_predictions=original_predictions,
        neighbor_predictions=neighbor_predictions,
        dataset=dataset,
        prices=prices,
        universe=universe,
        signal_date=signal_date,
        config=config,
    )

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    orders = build_order_sheet(
        final_portfolio=final_portfolio,
        prices=prices,
        config=config,
        run_timestamp=timestamp_tag,
    )

    outputs_dir = config["paths"]["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)

    original_path = os.path.join(
        outputs_dir,
        f"live_original_branch_predictions_{timestamp_tag}.csv",
    )

    neighbor_path = os.path.join(
        outputs_dir,
        f"live_neighbor_branch_predictions_{timestamp_tag}.csv",
    )

    branch_detail_path = os.path.join(
        outputs_dir,
        f"live_branch_portfolio_detail_{timestamp_tag}.csv",
    )

    final_portfolio_path = os.path.join(
        outputs_dir,
        f"live_final_portfolio_weights_{timestamp_tag}.csv",
    )

    regime_path = os.path.join(
        outputs_dir,
        f"live_regime_status_{timestamp_tag}.csv",
    )

    orders_path = os.path.join(
        outputs_dir,
        f"live_paper_trade_orders_{timestamp_tag}.csv",
    )

    summary_path = os.path.join(
        outputs_dir,
        f"live_signal_generation_summary_{timestamp_tag}.txt",
    )

    original_predictions.to_csv(original_path, index=False)
    neighbor_predictions.to_csv(neighbor_path, index=False)
    branch_detail.to_csv(branch_detail_path, index=False)
    final_portfolio.to_csv(final_portfolio_path, index=False)
    pd.DataFrame([regime_status]).to_csv(regime_path, index=False)
    orders.to_csv(orders_path, index=False)

    signal_ledger_path = append_live_signal_ledger(
        final_portfolio=final_portfolio,
        regime_status=regime_status,
        config=config,
        run_timestamp=timestamp_tag,
    )

    run_summary_path = append_live_run_summary(
        final_portfolio=final_portfolio,
        regime_status=regime_status,
        config=config,
        run_timestamp=timestamp_tag,
    )

    order_ledger_path = append_live_order_ledger(
        orders=orders,
        config=config,
    )

    print("")
    print("=" * 100)
    print("LIVE TOP ORIGINAL BRANCH PICKS")
    print("=" * 100)
    print(
        original_predictions.sort_values("rank_by_date")
        .head(20)
        .to_string(index=False)
    )

    print("")
    print("=" * 100)
    print("LIVE TOP LATENT-NEIGHBOR BRANCH PICKS")
    print("=" * 100)
    print(
        neighbor_predictions.sort_values("rank_by_date")
        .head(10)
        .to_string(index=False)
    )

    print("")
    print("=" * 100)
    print("LIVE REGIME STATUS")
    print("=" * 100)
    for key, value in regime_status.items():
        print(f"{key}: {value}")

    print("")
    print("=" * 100)
    print("LIVE FINAL PORTFOLIO WEIGHTS")
    print("=" * 100)
    print(final_portfolio.to_string(index=False))

    print("")
    print("=" * 100)
    print("LIVE PAPER TRADE ORDER SHEET")
    print("=" * 100)
    print(orders.to_string(index=False))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Latent Market Twin Live Signal Generation Summary\n")
        f.write("===============================================\n\n")
        f.write(f"Timestamp: {timestamp_tag}\n")
        f.write(f"Signal date: {signal_date}\n")
        f.write(f"Model name: {config['final_model_name']}\n\n")

        f.write("Regime status:\n")
        for key, value in regime_status.items():
            f.write(f"{key}: {value}\n")

        f.write("\nLive top original branch picks:\n")
        f.write(
            original_predictions.sort_values("rank_by_date")
            .head(20)
            .to_string(index=False)
        )

        f.write("\n\nLive top latent-neighbor branch picks:\n")
        f.write(
            neighbor_predictions.sort_values("rank_by_date")
            .head(10)
            .to_string(index=False)
        )

        f.write("\n\nLive final portfolio weights:\n")
        f.write(final_portfolio.to_string(index=False))

        f.write("\n\nLive paper trade order sheet:\n")
        f.write(orders.to_string(index=False))

        f.write("\n\nSaved files:\n")
        f.write(f"Original predictions: {original_path}\n")
        f.write(f"Neighbor predictions: {neighbor_path}\n")
        f.write(f"Branch detail: {branch_detail_path}\n")
        f.write(f"Final portfolio: {final_portfolio_path}\n")
        f.write(f"Regime status: {regime_path}\n")
        f.write(f"Order sheet: {orders_path}\n")
        f.write(f"Signal ledger: {signal_ledger_path}\n")
        f.write(f"Run summary ledger: {run_summary_path}\n")
        f.write(f"Order ledger: {order_ledger_path}\n")

    print("")
    print("Saved live original predictions:", original_path)
    print("Saved live neighbor predictions:", neighbor_path)
    print("Saved live branch detail:", branch_detail_path)
    print("Saved live final portfolio:", final_portfolio_path)
    print("Saved live regime status:", regime_path)
    print("Saved live order sheet:", orders_path)
    print("Saved live signal ledger:", signal_ledger_path)
    print("Saved live run summary:", run_summary_path)
    print("Saved live order ledger:", order_ledger_path)
    print("Saved live summary:", summary_path)
    print("")
    print("Week 22 Step 11K complete.")


if __name__ == "__main__":
    main()