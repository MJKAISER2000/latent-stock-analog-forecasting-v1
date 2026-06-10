import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.yaml"
LIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "live_model_config.yaml"


def main():
    if not FINAL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Final config not found: {FINAL_CONFIG_PATH}")

    with open(FINAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["project_name"] = "latent_market_twin_live"
    config["final_model_name"] = "LTSAF_live_v1"

    config["paths"]["base_dataset"] = "data/processed/live_full500_modeling_dataset.parquet"
    config["paths"]["neighbor_dataset"] = "data/processed/live_full500_with_stock_latent_neighbors.parquet"
    config["paths"]["monthly_prices"] = "data/processed/live_500_monthly_prices.parquet"
    config["paths"]["outputs_dir"] = "outputs/paper_trading_live"
    config["paths"]["tables_dir"] = "outputs/tables"
    config["paths"]["reports_dir"] = "outputs/reports"
    config["paths"]["models_dir"] = "models"

    os.makedirs(PROJECT_ROOT / "configs", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "outputs" / "paper_trading_live", exist_ok=True)

    with open(LIVE_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("")
    print("=" * 100)
    print("LIVE CONFIG CREATED")
    print("=" * 100)
    print("Saved:", LIVE_CONFIG_PATH)
    print("")
    print("Live paths:")
    for key, value in config["paths"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()