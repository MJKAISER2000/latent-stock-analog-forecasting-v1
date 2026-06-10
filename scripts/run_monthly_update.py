import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "paper_trading" / "monthly_update_logs"

FINAL_PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_final_pipeline.py"
VALUE_CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_portfolio_value.py"
PERFORMANCE_TRACKER_SCRIPT = PROJECT_ROOT / "scripts" / "track_paper_performance.py"
REBALANCE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_rebalance_orders.py"

HOLDINGS_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "current_paper_holdings.csv"
RUN_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_trading_run_summary.csv"
SIGNALS_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_portfolio_signals.csv"
ORDERS_LEDGER_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_trade_orders_ledger.csv"
VALUE_LEDGER_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_portfolio_value_ledger.csv"
PERFORMANCE_LEDGER_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_performance_ledger.csv"
REBALANCE_LEDGER_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "paper_rebalance_orders_ledger.csv"
LATEST_REBALANCE_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "latest_rebalance_orders.csv"
PORTFOLIO_VS_SPY_PLOT_PATH = PROJECT_ROOT / "outputs" / "figures" / "paper_portfolio_vs_spy.png"


def run_command(command: list[str], log_file: Path, step_name: str) -> int:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write(f"RUNNING STEP: {step_name}\n")
        f.write("=" * 100 + "\n")
        f.write(" ".join(command) + "\n\n")

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        process.wait()

        f.write("\n")
        f.write(f"STEP RESULT: {step_name} | return_code={process.returncode}\n")

        return process.returncode


def check_required_paths() -> list[str]:
    required = [
        PROJECT_ROOT / "configs" / "final_model_config.yaml",
        FINAL_PIPELINE_SCRIPT,
        VALUE_CHECK_SCRIPT,
        PERFORMANCE_TRACKER_SCRIPT,
        REBALANCE_SCRIPT,
        PROJECT_ROOT / "data" / "processed" / "week20_full500_with_stock_latent_neighbors.parquet",
        PROJECT_ROOT / "data" / "processed" / "week15_500_monthly_prices.parquet",
        PROJECT_ROOT / "data" / "external" / "week15_500_stock_universe.csv",
    ]

    missing = [str(path) for path in required if not path.exists()]
    return missing


def load_current_portfolio_summary() -> dict:
    if not HOLDINGS_PATH.exists():
        return {
            "holdings_found": False,
            "total_portfolio_value": None,
            "cash_value": None,
            "cash_weight": None,
            "stock_positions": None,
            "top_holding": None,
            "top_holding_value": None,
            "top_holding_weight": None,
        }

    holdings = pd.read_csv(HOLDINGS_PATH)

    if len(holdings) == 0:
        return {
            "holdings_found": True,
            "total_portfolio_value": 0.0,
            "cash_value": 0.0,
            "cash_weight": 0.0,
            "stock_positions": 0,
            "top_holding": None,
            "top_holding_value": None,
            "top_holding_weight": None,
        }

    holdings["ticker"] = holdings["ticker"].astype(str).str.strip().str.upper()
    holdings["market_value"] = pd.to_numeric(
        holdings["market_value"],
        errors="coerce",
    ).fillna(0.0)

    total_value = float(holdings["market_value"].sum())

    cash_value = float(
        holdings.loc[holdings["ticker"] == "CASH", "market_value"].sum()
    )

    cash_weight = cash_value / total_value if total_value > 0 else 0.0

    stock_holdings = holdings[holdings["ticker"] != "CASH"].copy()
    stock_positions = len(stock_holdings)

    if len(stock_holdings) > 0:
        stock_holdings = stock_holdings.sort_values("market_value", ascending=False)
        top_row = stock_holdings.iloc[0]
        top_holding = str(top_row["ticker"])
        top_holding_value = float(top_row["market_value"])
        top_holding_weight = top_holding_value / total_value if total_value > 0 else 0.0
    else:
        top_holding = None
        top_holding_value = None
        top_holding_weight = None

    return {
        "holdings_found": True,
        "total_portfolio_value": total_value,
        "cash_value": cash_value,
        "cash_weight": cash_weight,
        "stock_positions": stock_positions,
        "top_holding": top_holding,
        "top_holding_value": top_holding_value,
        "top_holding_weight": top_holding_weight,
    }


def load_latest_row(path: Path) -> dict:
    if not path.exists():
        return {
            "found": False,
        }

    df = pd.read_csv(path)

    if len(df) == 0:
        return {
            "found": True,
            "latest": None,
        }

    latest = df.tail(1).iloc[0].to_dict()
    latest["found"] = True

    return latest


def load_latest_rebalance_summary() -> dict:
    if not LATEST_REBALANCE_PATH.exists():
        return {
            "rebalance_found": False,
        }

    rebalance = pd.read_csv(LATEST_REBALANCE_PATH)

    if len(rebalance) == 0:
        return {
            "rebalance_found": True,
            "buy_order_count": 0,
            "sell_order_count": 0,
            "total_buy_value": 0.0,
            "total_sell_value": 0.0,
            "estimated_cash_after_trades": None,
        }

    buy_value = float(
        rebalance.loc[
            (rebalance["action"] == "BUY") & (rebalance["trade_value"] > 0),
            "trade_value",
        ].sum()
    )

    sell_value = float(
        -rebalance.loc[
            (rebalance["action"] == "SELL") & (rebalance["trade_value"] < 0),
            "trade_value",
        ].sum()
    )

    estimated_cash = rebalance["estimated_cash_after_trades"].iloc[0]

    return {
        "rebalance_found": True,
        "buy_order_count": int((rebalance["action"] == "BUY").sum()),
        "sell_order_count": int((rebalance["action"] == "SELL").sum()),
        "total_buy_value": buy_value,
        "total_sell_value": sell_value,
        "estimated_cash_after_trades": estimated_cash,
    }


def write_monthly_value_summary(log_file: Path) -> None:
    portfolio_summary = load_current_portfolio_summary()
    latest_run = load_latest_row(RUN_SUMMARY_PATH)
    latest_value = load_latest_row(VALUE_LEDGER_PATH)
    latest_performance = load_latest_row(PERFORMANCE_LEDGER_PATH)
    latest_rebalance = load_latest_rebalance_summary()

    print("")
    print("=" * 100)
    print("MONTHLY PORTFOLIO VALUE SUMMARY")
    print("=" * 100)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("MONTHLY PORTFOLIO VALUE SUMMARY\n")
        f.write("=" * 100 + "\n")

        if not portfolio_summary["holdings_found"]:
            message = "Current holdings file not found yet."
            print(message)
            f.write(message + "\n")
            return

        total_value = portfolio_summary["total_portfolio_value"]
        cash_value = portfolio_summary["cash_value"]
        cash_weight = portfolio_summary["cash_weight"]
        stock_positions = portfolio_summary["stock_positions"]
        top_holding = portfolio_summary["top_holding"]
        top_holding_value = portfolio_summary["top_holding_value"]
        top_holding_weight = portfolio_summary["top_holding_weight"]

        lines = [
            f"Total portfolio value: ${total_value:,.2f}",
            f"Cash value: ${cash_value:,.2f}",
            f"Cash weight: {cash_weight:.2%}",
            f"Stock positions: {stock_positions}",
        ]

        if top_holding is not None:
            lines.extend(
                [
                    f"Top holding: {top_holding}",
                    f"Top holding value: ${top_holding_value:,.2f}",
                    f"Top holding weight: {top_holding_weight:.2%}",
                ]
            )

        if latest_run.get("found", False) and latest_run.get("latest", "exists") is not None:
            signal_date = latest_run.get("signal_date", "unknown")
            regime_risk_on = latest_run.get("regime_risk_on", "unknown")
            tech_drawdown = latest_run.get("tech_drawdown", "unknown")
            top_ticker = latest_run.get("top_ticker", "unknown")

            lines.extend(
                [
                    "",
                    f"Latest signal date: {signal_date}",
                    f"Regime risk-on: {regime_risk_on}",
                    f"Tech drawdown: {tech_drawdown}",
                    f"Top ticker from run summary: {top_ticker}",
                ]
            )

        if latest_value.get("found", False) and latest_value.get("latest", "exists") is not None:
            portfolio_total_return = latest_value.get("portfolio_total_return", None)
            spy_total_return = latest_value.get("spy_total_return", None)
            excess_return_vs_spy = latest_value.get("excess_return_vs_spy", None)

            lines.extend(["", "Latest value vs SPY:"])

            if portfolio_total_return is not None:
                lines.append(f"Portfolio total return: {float(portfolio_total_return):.2%}")

            if spy_total_return is not None:
                lines.append(f"SPY total return: {float(spy_total_return):.2%}")

            if excess_return_vs_spy is not None:
                lines.append(f"Excess return vs SPY: {float(excess_return_vs_spy):.2%}")

        if latest_performance.get("found", False) and latest_performance.get("latest", "exists") is not None:
            portfolio_return = latest_performance.get("portfolio_return", None)
            spy_return = latest_performance.get("spy_return", None)
            excess_return = latest_performance.get("excess_return", None)
            beat_spy = latest_performance.get("beat_spy", None)

            lines.extend(["", "Latest completed holding-period performance:"])

            if portfolio_return is not None:
                lines.append(f"Portfolio period return: {float(portfolio_return):.2%}")

            if spy_return is not None:
                lines.append(f"SPY period return: {float(spy_return):.2%}")

            if excess_return is not None:
                lines.append(f"Excess period return: {float(excess_return):.2%}")

            if beat_spy is not None:
                lines.append(f"Beat SPY: {beat_spy}")

        if latest_rebalance.get("rebalance_found", False):
            lines.extend(
                [
                    "",
                    "Latest rebalance summary:",
                    f"Buy orders: {latest_rebalance['buy_order_count']}",
                    f"Sell orders: {latest_rebalance['sell_order_count']}",
                    f"Total buy value: ${latest_rebalance['total_buy_value']:,.2f}",
                    f"Total sell value: ${latest_rebalance['total_sell_value']:,.2f}",
                    f"Estimated cash after trades: ${float(latest_rebalance['estimated_cash_after_trades']):,.2f}",
                ]
            )

        for line in lines:
            print(line)
            f.write(line + "\n")


def write_artifact_summary(log_file: Path) -> None:
    artifact_paths = [
        ("Current holdings", HOLDINGS_PATH),
        ("Run summary ledger", RUN_SUMMARY_PATH),
        ("Signal ledger", SIGNALS_PATH),
        ("Order ledger", ORDERS_LEDGER_PATH),
        ("Value ledger", VALUE_LEDGER_PATH),
        ("Performance ledger", PERFORMANCE_LEDGER_PATH),
        ("Latest rebalance orders", LATEST_REBALANCE_PATH),
        ("Rebalance ledger", REBALANCE_LEDGER_PATH),
        ("Portfolio vs SPY plot", PORTFOLIO_VS_SPY_PLOT_PATH),
    ]

    print("")
    print("=" * 100)
    print("MONTHLY UPDATE ARTIFACTS")
    print("=" * 100)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("MONTHLY UPDATE ARTIFACTS\n")
        f.write("=" * 100 + "\n")

        for label, path in artifact_paths:
            exists = path.exists()
            line = f"{label}: {path} | exists={exists}"
            print(line)
            f.write(line + "\n")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"monthly_update_{timestamp}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Latent Market Twin Monthly Update Log\n")
        f.write("====================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"Final pipeline script: {FINAL_PIPELINE_SCRIPT}\n")
        f.write(f"Value check script: {VALUE_CHECK_SCRIPT}\n")
        f.write(f"Performance tracker script: {PERFORMANCE_TRACKER_SCRIPT}\n")
        f.write(f"Rebalance script: {REBALANCE_SCRIPT}\n")

    print("")
    print("=" * 100)
    print("LATENT MARKET TWIN MONTHLY UPDATE")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Log file:", log_file)

    missing = check_required_paths()

    if missing:
        print("")
        print("ERROR: Missing required files:")
        for path in missing:
            print("-", path)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\nERROR: Missing required files:\n")
            for path in missing:
                f.write(f"- {path}\n")

        sys.exit(1)

    print("")
    print("Required files found.")

    steps = [
        ("Final pipeline", [sys.executable, str(FINAL_PIPELINE_SCRIPT)]),
        ("Portfolio value check", [sys.executable, str(VALUE_CHECK_SCRIPT)]),
        ("Paper performance tracker", [sys.executable, str(PERFORMANCE_TRACKER_SCRIPT)]),
        ("Rebalance order generator", [sys.executable, str(REBALANCE_SCRIPT)]),
    ]

    step_results = []

    for step_name, command in steps:
        print("")
        print("=" * 100)
        print(f"STARTING STEP: {step_name}")
        print("=" * 100)

        return_code = run_command(
            command=command,
            log_file=log_file,
            step_name=step_name,
        )

        step_results.append(
            {
                "step_name": step_name,
                "return_code": return_code,
            }
        )

        if return_code != 0:
            print("")
            print(f"ERROR: Step failed: {step_name}")
            break

    overall_success = all(row["return_code"] == 0 for row in step_results)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("MONTHLY UPDATE RESULT\n")
        f.write("=" * 100 + "\n")

        for row in step_results:
            f.write(f"{row['step_name']}: return_code={row['return_code']}\n")

        if overall_success:
            f.write("Status: SUCCESS\n")
        else:
            f.write("Status: FAILED\n")

    print("")
    print("=" * 100)
    print("MONTHLY UPDATE COMPLETE")
    print("=" * 100)

    if overall_success:
        print("Status: SUCCESS")
        write_monthly_value_summary(log_file)
        write_artifact_summary(log_file)
    else:
        print("Status: FAILED")

    print("")
    print("Log file:", log_file)

    if overall_success:
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()