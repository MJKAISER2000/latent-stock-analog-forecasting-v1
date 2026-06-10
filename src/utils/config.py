import os
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = "configs/final_model_config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load the final model YAML config.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def require_config_keys(config: dict[str, Any], required_keys: list[str]) -> None:
    """
    Check top-level config keys.
    """

    missing = [key for key in required_keys if key not in config]

    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """
    Create standard output directories from config.
    """

    paths = config["paths"]

    dirs = [
        paths["outputs_dir"],
        paths["tables_dir"],
        paths["reports_dir"],
        paths["models_dir"],
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def check_required_files(config: dict[str, Any]) -> None:
    """
    Check that required input files exist.
    """

    paths = config["paths"]

    required_files = [
        paths["base_dataset"],
        paths["neighbor_dataset"],
        paths["monthly_prices"],
        paths["universe"],
    ]

    missing = [p for p in required_files if not os.path.exists(p)]

    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def print_config_summary(config: dict[str, Any]) -> None:
    """
    Print a compact summary of the final model setup.
    """

    print("")
    print("=" * 80)
    print("LATENT MARKET TWIN FINAL CONFIG")
    print("=" * 80)

    print("Project:", config.get("project_name"))
    print("Final model:", config.get("final_model_name"))

    print("")
    print("Original branch:")
    print(config["portfolio"]["original_branch"])

    print("")
    print("Latent neighbor branch:")
    print(config["portfolio"]["latent_neighbor_branch"])

    print("")
    print("Portfolio weighting:")
    print(config["portfolio"]["weighting"])

    print("")
    print("Regime filter:")
    print(config["regime_filter"])

    print("")
    print("Paper trading:")
    print(config["paper_trading"])