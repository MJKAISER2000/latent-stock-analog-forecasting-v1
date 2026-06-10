from typing import Any

import numpy as np
import pandas as pd


def get_inverse_vol_weights(
    selected: pd.DataFrame,
    dataset: pd.DataFrame,
    signal_date: pd.Timestamp,
    vol_col: str = "vol_12m",
) -> pd.DataFrame:
    """
    Given selected tickers, assign inverse-vol weights within the basket.
    """

    signal_date = pd.Timestamp(signal_date)

    if vol_col not in dataset.columns:
        raise ValueError(f"Missing volatility column: {vol_col}")

    vol_lookup = dataset[dataset["date"] == signal_date][["ticker", vol_col]].copy()
    vol_lookup["ticker"] = vol_lookup["ticker"].astype(str).str.strip().str.upper()
    vol_lookup[vol_col] = pd.to_numeric(vol_lookup[vol_col], errors="coerce")

    out = selected.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()

    out = out.merge(vol_lookup, on="ticker", how="left")

    out[vol_col] = out[vol_col].replace(0, np.nan)

    if out[vol_col].isna().all():
        out["branch_inner_weight"] = 1.0 / len(out)
        return out

    out["inv_vol"] = 1.0 / out[vol_col]
    out["inv_vol"] = out["inv_vol"].replace([np.inf, -np.inf], np.nan)
    out["inv_vol"] = out["inv_vol"].fillna(out["inv_vol"].median())

    if out["inv_vol"].sum() == 0 or pd.isna(out["inv_vol"].sum()):
        out["branch_inner_weight"] = 1.0 / len(out)
    else:
        out["branch_inner_weight"] = out["inv_vol"] / out["inv_vol"].sum()

    return out


def build_branch_portfolio(
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    signal_date: pd.Timestamp,
    branch_name: str,
    top_n: int,
    branch_weight: float,
    vol_col: str = "vol_12m",
) -> pd.DataFrame:
    """
    Select top-N from one prediction branch and assign branch-adjusted weights.
    """

    selected = (
        predictions.sort_values("rank_by_date")
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    selected["branch"] = branch_name
    selected["branch_top_n"] = top_n
    selected["branch_weight"] = branch_weight

    selected = get_inverse_vol_weights(
        selected=selected,
        dataset=dataset,
        signal_date=signal_date,
        vol_col=vol_col,
    )

    selected["final_weight_before_regime"] = (
        selected["branch_weight"] * selected["branch_inner_weight"]
    )

    keep_cols = [
        "date",
        "ticker",
        "branch",
        "branch_top_n",
        "branch_weight",
        "branch_inner_weight",
        "final_weight_before_regime",
        "ranker_score",
        "rank_by_date",
    ]

    optional_cols = [c for c in keep_cols if c in selected.columns]

    return selected[optional_cols].copy()


def combine_branch_portfolios(branch_portfolios: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine branch portfolios. If a ticker appears in both branches, sum weights.
    """

    combined_raw = pd.concat(branch_portfolios, ignore_index=True)

    grouped = (
        combined_raw.groupby("ticker")
        .agg(
            date=("date", "first"),
            final_weight_before_regime=("final_weight_before_regime", "sum"),
            branches=("branch", lambda x: ", ".join(sorted(set(x)))),
            avg_ranker_score=("ranker_score", "mean"),
            best_rank=("rank_by_date", "min"),
        )
        .reset_index()
    )

    total = grouped["final_weight_before_regime"].sum()

    if total == 0 or pd.isna(total):
        grouped["final_weight_before_regime"] = 1.0 / len(grouped)
    else:
        grouped["final_weight_before_regime"] = grouped["final_weight_before_regime"] / total

    grouped = grouped.sort_values(
        "final_weight_before_regime",
        ascending=False,
    ).reset_index(drop=True)

    return grouped


def build_tech_regime_status(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute whether the tech drawdown regime filter is risk-on or risk-off.
    """

    signal_date = pd.Timestamp(signal_date)

    regime_cfg = config["regime_filter"]
    sector_name = regime_cfg.get("tech_sector_name", "Information Technology")
    threshold = float(regime_cfg.get("drawdown_threshold", -0.20))

    if "SPY" not in prices.columns:
        raise ValueError("SPY missing from prices.")

    tech_tickers = universe.loc[
        universe["sector"] == sector_name,
        "ticker",
    ].astype(str).str.strip().str.upper().tolist()

    tech_tickers = [t for t in tech_tickers if t in prices.columns]

    if len(tech_tickers) == 0:
        raise ValueError(f"No tickers found for sector: {sector_name}")

    available_prices = prices[prices.index <= signal_date].copy()

    if len(available_prices) == 0:
        raise ValueError(f"No price rows up to signal date: {signal_date}")

    monthly_returns = available_prices / available_prices.shift(1) - 1
    tech_monthly_return = monthly_returns[tech_tickers].mean(axis=1)

    tech_index = (1.0 + tech_monthly_return.fillna(0.0)).cumprod()
    tech_drawdown = tech_index / tech_index.cummax() - 1.0

    latest_date = pd.Timestamp(tech_drawdown.index.max())
    latest_drawdown = float(tech_drawdown.loc[latest_date])

    risk_on = latest_drawdown > threshold

    return {
        "signal_date": signal_date,
        "regime_price_date": latest_date,
        "rule": regime_cfg.get("rule", "tech_drawdown_20"),
        "threshold": threshold,
        "tech_drawdown": latest_drawdown,
        "risk_on": bool(risk_on),
        "cash_weight_when_off": float(regime_cfg.get("cash_weight_when_off", 1.0)),
        "tech_ticker_count": len(tech_tickers),
    }


def apply_regime_to_portfolio(
    portfolio: pd.DataFrame,
    regime_status: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the regime filter. If risk-off, scale stock weights down and add cash.
    """

    out = portfolio.copy()

    risk_on = bool(regime_status["risk_on"])
    cash_weight_when_off = float(regime_status["cash_weight_when_off"])

    if risk_on:
        out["final_weight"] = out["final_weight_before_regime"]
        cash_weight = 0.0
    else:
        out["final_weight"] = out["final_weight_before_regime"] * (1.0 - cash_weight_when_off)
        cash_weight = cash_weight_when_off

    if cash_weight > 0:
        cash_row = {
            "ticker": "CASH",
            "date": out["date"].iloc[0],
            "final_weight_before_regime": 0.0,
            "branches": "regime_filter",
            "avg_ranker_score": np.nan,
            "best_rank": np.nan,
            "final_weight": cash_weight,
        }

        out = pd.concat([out, pd.DataFrame([cash_row])], ignore_index=True)

    total_weight = out["final_weight"].sum()

    if total_weight != 0 and not pd.isna(total_weight):
        out["final_weight"] = out["final_weight"] / total_weight

    out = out.sort_values("final_weight", ascending=False).reset_index(drop=True)

    return out


def build_final_portfolio(
    original_predictions: pd.DataFrame,
    neighbor_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Build the final Week 21 portfolio:
    70% original top20 + 30% latent-neighbor top10 + regime filter.
    """

    portfolio_cfg = config["portfolio"]

    vol_col = portfolio_cfg["weighting"]["vol_column"]

    original_cfg = portfolio_cfg["original_branch"]
    neighbor_cfg = portfolio_cfg["latent_neighbor_branch"]

    original_branch = build_branch_portfolio(
        predictions=original_predictions,
        dataset=dataset,
        signal_date=signal_date,
        branch_name="original",
        top_n=int(original_cfg["top_n"]),
        branch_weight=float(original_cfg["weight"]),
        vol_col=vol_col,
    )

    neighbor_branch = build_branch_portfolio(
        predictions=neighbor_predictions,
        dataset=dataset,
        signal_date=signal_date,
        branch_name="latent_neighbor",
        top_n=int(neighbor_cfg["top_n"]),
        branch_weight=float(neighbor_cfg["weight"]),
        vol_col=vol_col,
    )

    branch_detail = pd.concat([original_branch, neighbor_branch], ignore_index=True)

    combined = combine_branch_portfolios([original_branch, neighbor_branch])

    regime_status = build_tech_regime_status(
        prices=prices,
        universe=universe,
        signal_date=signal_date,
        config=config,
    )

    final_portfolio = apply_regime_to_portfolio(
        portfolio=combined,
        regime_status=regime_status,
    )

    return final_portfolio, branch_detail, regime_status