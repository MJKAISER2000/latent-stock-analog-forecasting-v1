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

from src.paper_trading.ledger import (
    append_signal_ledger,
    append_run_summary,
)

from src.paper_trading.orders import (
    build_order_sheet,
    save_order_sheet,
    append_order_ledger,
)

from src.paper_trading.holdings import (
    get_holdings_path,
    initialize_holdings_from_orders,
    load_current_holdings,
    mark_holdings_to_market,
    save_current_holdings,
    print_holdings_summary,
)


CONFIG_PATH = "configs/final_model_config.yaml"


def prepare_dataset_for_training(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    working = df.copy()

    X = prepare_feature_matrix(working, feature_cols)

    for col in feature_cols:
        working[col] = X[col]

    return working


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
    print("=" * 80)
    print("LATEST SIGNAL DATE")
    print("=" * 80)
    print(signal_date)

    original_feature_set = config["portfolio"]["original_branch"]["feature_set"]
    neighbor_feature_set = config["portfolio"]["latent_neighbor_branch"]["feature_set"]

    original_features = print_feature_set_summary(dataset, original_feature_set)
    neighbor_features = print_feature_set_summary(dataset, neighbor_feature_set)

    working_original = prepare_dataset_for_training(dataset, original_features)
    working_neighbor = prepare_dataset_for_training(dataset, neighbor_features)

    print("")
    print("=" * 80)
    print("TRAINING ORIGINAL BRANCH")
    print("=" * 80)

    original_model, original_predictions = train_predict_latest(
        df=working_original,
        feature_cols=original_features,
        config=config,
        signal_date=signal_date,
    )

    original_predictions["branch"] = "original"
    original_predictions["feature_set"] = original_feature_set

    print("")
    print("=" * 80)
    print("TRAINING LATENT NEIGHBOR BRANCH")
    print("=" * 80)

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

    outputs_dir = config["paths"]["outputs_dir"]
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_path = os.path.join(
        outputs_dir,
        f"latest_original_branch_predictions_{timestamp_tag}.csv",
    )

    neighbor_path = os.path.join(
        outputs_dir,
        f"latest_neighbor_branch_predictions_{timestamp_tag}.csv",
    )

    branch_detail_path = os.path.join(
        outputs_dir,
        f"latest_branch_portfolio_detail_{timestamp_tag}.csv",
    )

    final_portfolio_path = os.path.join(
        outputs_dir,
        f"latest_final_portfolio_weights_{timestamp_tag}.csv",
    )

    regime_path = os.path.join(
        outputs_dir,
        f"latest_regime_status_{timestamp_tag}.csv",
    )

    original_predictions.to_csv(original_path, index=False)
    neighbor_predictions.to_csv(neighbor_path, index=False)
    branch_detail.to_csv(branch_detail_path, index=False)
    final_portfolio.to_csv(final_portfolio_path, index=False)
    pd.DataFrame([regime_status]).to_csv(regime_path, index=False)

    signal_ledger_path = append_signal_ledger(
        final_portfolio=final_portfolio,
        regime_status=regime_status,
        config=config,
        run_timestamp=timestamp_tag,
        overwrite_same_signal_date=True,
    )

    run_summary_path = append_run_summary(
        config=config,
        run_timestamp=timestamp_tag,
        signal_date=signal_date,
        regime_status=regime_status,
        final_portfolio=final_portfolio,
        overwrite_same_signal_date=True,
    )

    orders = build_order_sheet(
        final_portfolio=final_portfolio,
        prices=prices,
        config=config,
        run_timestamp=timestamp_tag,
    )

    orders_path = save_order_sheet(
        orders=orders,
        config=config,
        run_timestamp=timestamp_tag,
    )

    orders_ledger_path = append_order_ledger(
        orders=orders,
        config=config,
        overwrite_same_signal_date=True,
    )

    holdings_path = get_holdings_path(config)

    if not os.path.exists(holdings_path):
        initialized_holdings_path = initialize_holdings_from_orders(
            orders=orders,
            config=config,
            overwrite=False,
        )
        current_holdings = load_current_holdings(config)
        print("")
        print("Initialized current holdings:", initialized_holdings_path)
    else:
        current_holdings = load_current_holdings(config)
        current_holdings = mark_holdings_to_market(
            holdings=current_holdings,
            prices=prices,
            as_of_date=signal_date,
        )
        saved_holdings_path = save_current_holdings(
            holdings=current_holdings,
            config=config,
        )
        print("")
        print("Updated current holdings mark-to-market:", saved_holdings_path)

    print("")
    print("=" * 80)
    print("TOP ORIGINAL BRANCH PICKS")
    print("=" * 80)
    print(
        original_predictions.sort_values("rank_by_date")
        .head(20)
        .to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("TOP LATENT NEIGHBOR BRANCH PICKS")
    print("=" * 80)
    print(
        neighbor_predictions.sort_values("rank_by_date")
        .head(10)
        .to_string(index=False)
    )

    print("")
    print("=" * 80)
    print("REGIME STATUS")
    print("=" * 80)
    for key, value in regime_status.items():
        print(f"{key}: {value}")

    print("")
    print("=" * 80)
    print("FINAL PORTFOLIO WEIGHTS")
    print("=" * 80)
    print(final_portfolio.to_string(index=False))

    print("")
    print("=" * 80)
    print("PAPER TRADE ORDER SHEET")
    print("=" * 80)
    print(orders.to_string(index=False))

    print_holdings_summary(current_holdings)

    summary_path = os.path.join(
        outputs_dir,
        f"latest_signal_generation_summary_{timestamp_tag}.txt",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Latent Market Twin Latest Signal Generation Summary\n")
        f.write("==================================================\n\n")
        f.write(f"Timestamp: {timestamp_tag}\n")
        f.write(f"Signal date: {signal_date}\n")
        f.write(f"Final model: {config['final_model_name']}\n\n")

        f.write("Regime status:\n")
        for key, value in regime_status.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTop original branch picks:\n")
        f.write(
            original_predictions.sort_values("rank_by_date")
            .head(20)
            .to_string(index=False)
        )

        f.write("\n\nTop latent neighbor branch picks:\n")
        f.write(
            neighbor_predictions.sort_values("rank_by_date")
            .head(10)
            .to_string(index=False)
        )

        f.write("\n\nFinal portfolio weights:\n")
        f.write(final_portfolio.to_string(index=False))

        f.write("\n\nPaper trade order sheet:\n")
        f.write(orders.to_string(index=False))

        f.write("\n\nCurrent paper holdings:\n")
        f.write(current_holdings.to_string(index=False))

        f.write("\n\nSaved files:\n")
        f.write(f"Original predictions: {original_path}\n")
        f.write(f"Neighbor predictions: {neighbor_path}\n")
        f.write(f"Branch detail: {branch_detail_path}\n")
        f.write(f"Final portfolio: {final_portfolio_path}\n")
        f.write(f"Regime status: {regime_path}\n")
        f.write(f"Signal ledger: {signal_ledger_path}\n")
        f.write(f"Run summary ledger: {run_summary_path}\n")
        f.write(f"Order sheet: {orders_path}\n")
        f.write(f"Order ledger: {orders_ledger_path}\n")
        f.write(f"Current holdings: {holdings_path}\n")

    print("")
    print("Saved original predictions:", original_path)
    print("Saved neighbor predictions:", neighbor_path)
    print("Saved branch detail:", branch_detail_path)
    print("Saved final portfolio:", final_portfolio_path)
    print("Saved regime status:", regime_path)
    print("Saved signal ledger:", signal_ledger_path)
    print("Saved run summary ledger:", run_summary_path)
    print("Saved order sheet:", orders_path)
    print("Saved order ledger:", orders_ledger_path)
    print("Saved current holdings:", holdings_path)
    print("Saved signal summary:", summary_path)
    print("")
    print("Week 22 Step 2 complete.")


if __name__ == "__main__":
    main()