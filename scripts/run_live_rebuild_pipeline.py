import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_rebuild_logs"

STEPS = [
    {
        "name": "Make live config",
        "script": PROJECT_ROOT / "scripts" / "make_live_config.py",
        "required": True,
    },
    {
        "name": "Refresh live monthly prices",
        "script": PROJECT_ROOT / "scripts" / "refresh_live_monthly_prices.py",
        "required": True,
    },
    {
        "name": "Compare live vs research prices",
        "script": PROJECT_ROOT / "scripts" / "compare_live_vs_research_prices.py",
        "required": False,
    },
    {
        "name": "Build live base features",
        "script": PROJECT_ROOT / "scripts" / "build_live_base_features.py",
        "required": True,
    },
    {
        "name": "Build live stock-state PCA",
        "script": PROJECT_ROOT / "scripts" / "build_live_stock_state_pca.py",
        "required": True,
    },
    {
        "name": "Build live latent neighbors",
        "script": PROJECT_ROOT / "scripts" / "build_live_latent_neighbors.py",
        "required": True,
    },
    {
        "name": "Compare live vs research dataset",
        "script": PROJECT_ROOT / "scripts" / "compare_live_vs_research_dataset.py",
        "required": False,
    },
    {
        "name": "Run live final pipeline",
        "script": PROJECT_ROOT / "scripts" / "run_live_final_pipeline.py",
        "required": True,
    },
    {
        "name": "Generate live rebalance orders",
        "script": PROJECT_ROOT / "scripts" / "generate_live_rebalance_orders.py",
        "required": True,
    },
    {
        "name": "Track LTSAF live performance",
        "script": PROJECT_ROOT / "scripts" / "track_ltsaf_live_performance.py",
        "required": False,
    },
]


def check_required_scripts() -> list[str]:
    missing = []

    for step in STEPS:
        if not step["script"].exists():
            missing.append(str(step["script"]))

    return missing


def run_step(step: dict, log_file: Path) -> int:
    command = [sys.executable, str(step["script"])]

    print("")
    print("=" * 100)
    print(f"STARTING STEP: {step['name']}")
    print("=" * 100)
    print("Command:", " ".join(command))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write(f"STARTING STEP: {step['name']}\n")
        f.write("=" * 100 + "\n")
        f.write("Command: " + " ".join(command) + "\n\n")

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
        f.write(f"STEP RESULT: {step['name']} | return_code={process.returncode}\n")

    print("")
    print(f"STEP RESULT: {step['name']} | return_code={process.returncode}")

    return process.returncode


def collect_live_artifacts() -> list[tuple[str, Path]]:
    return [
        (
            "Live config",
            PROJECT_ROOT / "configs" / "live_model_config.yaml",
        ),
        (
            "Live daily prices",
            PROJECT_ROOT / "data" / "processed" / "live_500_daily_prices.parquet",
        ),
        (
            "Live monthly prices",
            PROJECT_ROOT / "data" / "processed" / "live_500_monthly_prices.parquet",
        ),
        (
            "Live monthly returns",
            PROJECT_ROOT / "data" / "processed" / "live_500_monthly_returns.parquet",
        ),
        (
            "Live base dataset",
            PROJECT_ROOT / "data" / "processed" / "live_full500_modeling_dataset.parquet",
        ),
        (
            "Live PCA latents",
            PROJECT_ROOT / "data" / "processed" / "live_stock_state_pca_latents_with_metadata.parquet",
        ),
        (
            "Live neighbor features",
            PROJECT_ROOT / "data" / "processed" / "live_stock_latent_neighbor_features.parquet",
        ),
        (
            "Live full neighbor dataset",
            PROJECT_ROOT / "data" / "processed" / "live_full500_with_stock_latent_neighbors.parquet",
        ),
        (
            "Current live holdings",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "current_live_holdings.csv",
        ),
        (
            "Live signal ledger",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_portfolio_signals.csv",
        ),
        (
            "Live run summary",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_run_summary.csv",
        ),
        (
            "Live order ledger",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_order_ledger.csv",
        ),
        (
            "Latest live rebalance orders",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "latest_live_rebalance_orders.csv",
        ),
        (
            "Live rebalance ledger",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_rebalance_orders_ledger.csv",
        ),
        (
            "Live performance ledger",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "live_performance_ledger.csv",
        ),
        (
            "Latest live performance summary",
            PROJECT_ROOT / "outputs" / "paper_trading_live" / "latest_live_performance_summary.txt",
        ),
        (
            "Live output directory",
            PROJECT_ROOT / "outputs" / "paper_trading_live",
        ),
        (
            "Live price refresh report",
            PROJECT_ROOT / "outputs" / "reports" / "live_monthly_price_refresh_report.txt",
        ),
        (
            "Live base feature report",
            PROJECT_ROOT / "outputs" / "reports" / "live_base_feature_build_report.txt",
        ),
        (
            "Live PCA report",
            PROJECT_ROOT / "outputs" / "reports" / "live_stock_state_pca_report.txt",
        ),
        (
            "Live neighbor report",
            PROJECT_ROOT / "outputs" / "reports" / "live_latent_neighbor_feature_report.txt",
        ),
        (
            "Live vs research dataset report",
            PROJECT_ROOT / "outputs" / "reports" / "live_vs_research_dataset_comparison_report.txt",
        ),
    ]


def write_artifact_summary(log_file: Path) -> None:
    artifacts = collect_live_artifacts()

    print("")
    print("=" * 100)
    print("LIVE REBUILD ARTIFACT SUMMARY")
    print("=" * 100)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("LIVE REBUILD ARTIFACT SUMMARY\n")
        f.write("=" * 100 + "\n")

        for label, path in artifacts:
            exists = path.exists()

            if exists and path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                line = f"{label}: {path} | exists=True | size_mb={size_mb:.4f}"
            else:
                line = f"{label}: {path} | exists={exists}"

            print(line)
            f.write(line + "\n")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"live_rebuild_{timestamp}.txt"

    print("")
    print("=" * 100)
    print("LATENT MARKET TWIN LIVE REBUILD PIPELINE")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Log file:", log_file)
    print("Python:", sys.executable)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Latent Market Twin Live Rebuild Pipeline Log\n")
        f.write("===========================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Python executable: {sys.executable}\n")

    missing = check_required_scripts()

    if missing:
        print("")
        print("ERROR: Missing required scripts:")
        for path in missing:
            print("-", path)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\nERROR: Missing required scripts:\n")
            for path in missing:
                f.write(f"- {path}\n")

        sys.exit(1)

    step_results = []

    for step in STEPS:
        return_code = run_step(step, log_file)

        step_results.append(
            {
                "step_name": step["name"],
                "required": step["required"],
                "return_code": return_code,
            }
        )

        if return_code != 0 and step["required"]:
            print("")
            print(f"ERROR: Required step failed: {step['name']}")
            break

        if return_code != 0 and not step["required"]:
            print("")
            print(f"WARNING: Optional step failed but pipeline will continue: {step['name']}")

    required_success = all(
        row["return_code"] == 0
        for row in step_results
        if row["required"]
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("LIVE REBUILD RESULT\n")
        f.write("=" * 100 + "\n")

        for row in step_results:
            f.write(
                f"{row['step_name']} | required={row['required']} | "
                f"return_code={row['return_code']}\n"
            )

        f.write(f"\nRequired steps successful: {required_success}\n")

    write_artifact_summary(log_file)

    print("")
    print("=" * 100)
    print("LIVE REBUILD PIPELINE COMPLETE")
    print("=" * 100)

    if required_success:
        print("Status: SUCCESS")
        print("Live model files, live signal outputs, rebalance orders, and performance tracking were rebuilt.")
    else:
        print("Status: FAILED")
        print("At least one required step failed.")

    print("Log file:", log_file)

    if required_success:
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()