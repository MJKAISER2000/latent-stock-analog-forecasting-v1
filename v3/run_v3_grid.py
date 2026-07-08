"""v3 experiment grid — portfolio construction sweep over saved rankings.

Consumes the full monthly rankings produced by run_v3_cv.py and evaluates
every combination of:

    original feature set   v1feats | v3feats
    original top-N         10 | 20 | 35 | 50
    latent neighbor dim    4 | 8 | 16 | 32   (or no neighbor sleeve)
    neighbor top-N         5 | 10 | 20
    branch blend           70/30 | 50/50 | 0/100 | 100/0
    hedge                  none | trend(10m MA) | vix(>30) | credit(HYG dd<-5%)
                           | trend_or_credit   (all cut exposure to 40%)

Positions are inverse-vol weighted within each sleeve (same as v1/live), 10 bps
costs on two-way turnover vs the drifted book, cash earns 0.

Because every variant is scored on the same walk-forward out-of-sample months,
the *marginal* analysis (average Sharpe along each grid dimension) is the
robust output. The single best cell of ~1,000 is partly luck — the report
says so.

Outputs (outputs/v3_backtest/):
    v3_grid_results.csv, v3_marginal_analysis.csv, v3_report.txt
"""

import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtests.cv_metrics import contiguous_folds, summarize_returns

DATA = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "v3_backtest")
DATASET_PATH = os.path.join(DATA, "v3_modeling_dataset.parquet")

TRANSACTION_COST = 0.001
N_FOLDS = 5
HEDGED_EXPOSURE = 0.4

ORIGINAL_BRANCHES = ["original_v1feats", "original_v3feats"]
NEIGHBOR_DIMS = [4, 8, 16, 32]
ORIG_TOPNS = [10, 20, 35, 50]
NEIGH_TOPNS = [5, 10, 20]
BLENDS = [(0.7, 0.3), (0.5, 0.5)]
HEDGES = ["none", "trend", "vix", "credit", "trend_or_credit"]


def load_rankings(branch: str) -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f"rankings_{branch}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def build_sleeves(
    rankings: pd.DataFrame, topns: list[int], vol: pd.Series
) -> dict[int, dict[pd.Timestamp, pd.Series]]:
    """{top_n: {date: inverse-vol weight Series summing to 1}}."""
    sleeves: dict[int, dict[pd.Timestamp, pd.Series]] = {n: {} for n in topns}
    for date, month in rankings.groupby("date"):
        ranked = month.sort_values("rank_by_date")
        month_vol = vol.loc[date] if date in vol.index.get_level_values(0) else pd.Series(dtype=float)
        for n in topns:
            top = ranked.head(n)
            v = pd.to_numeric(month_vol.reindex(top["ticker"]), errors="coerce")
            inv = 1.0 / v.replace(0, np.nan)
            inv = inv.fillna(inv.median())
            if inv.isna().all() or inv.sum() == 0:
                weights = pd.Series(1.0 / len(top), index=top["ticker"])
            else:
                weights = inv / inv.sum()
            sleeves[n][date] = weights
    return sleeves


def hedge_exposures(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-date exposure multiplier for each hedge rule (1.0 = fully invested)."""
    macro = dataset.groupby("date")[["v3m_spy_to_ma10m", "v3m_vix", "v3m_hyg_drawdown"]].first()
    out = pd.DataFrame(index=macro.index)
    out["none"] = 1.0
    out["trend"] = np.where(macro["v3m_spy_to_ma10m"] < 1.0, HEDGED_EXPOSURE, 1.0)
    out["vix"] = np.where(macro["v3m_vix"] > 30.0, HEDGED_EXPOSURE, 1.0)
    out["credit"] = np.where(macro["v3m_hyg_drawdown"] < -0.05, HEDGED_EXPOSURE, 1.0)
    out["trend_or_credit"] = out[["trend", "credit"]].min(axis=1)
    return out.fillna(1.0)


def simulate(
    books: dict[pd.Timestamp, pd.Series],
    returns_by_date: dict[pd.Timestamp, pd.Series],
    exposure: pd.Series,
) -> pd.Series:
    """Sequential net monthly returns for one variant."""
    net = {}
    prev_actual = None
    prev_realized = None
    prev_gross = 0.0

    for date in sorted(books):
        target = books[date] * float(exposure.get(date, 1.0))

        if prev_actual is None:
            turnover = float(target.abs().sum())
        else:
            drifted = prev_actual * (1.0 + prev_realized) / (1.0 + prev_gross)
            tickers = target.index.union(drifted.index)
            turnover = float(
                (target.reindex(tickers, fill_value=0.0)
                 - drifted.reindex(tickers, fill_value=0.0)).abs().sum()
            )

        month_returns = returns_by_date.get(date)
        realized = month_returns.reindex(target.index).fillna(0.0) if month_returns is not None \
            else pd.Series(0.0, index=target.index)

        gross = float((target * realized).sum())
        net[date] = gross - TRANSACTION_COST * turnover

        prev_actual, prev_realized, prev_gross = target, realized, gross

    return pd.Series(net).sort_index()


def blend_books(
    orig: dict[pd.Timestamp, pd.Series] | None,
    neigh: dict[pd.Timestamp, pd.Series] | None,
    w_orig: float,
    w_neigh: float,
    dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Series]:
    books = {}
    for date in dates:
        parts = []
        if orig is not None and w_orig > 0 and date in orig:
            parts.append(orig[date] * w_orig)
        if neigh is not None and w_neigh > 0 and date in neigh:
            parts.append(neigh[date] * w_neigh)
        if not parts:
            continue
        combined = parts[0] if len(parts) == 1 else parts[0].add(parts[1], fill_value=0.0)
        total = combined.sum()
        books[date] = combined / total if total > 0 else combined
    return books


def main() -> None:
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = pd.read_parquet(
        DATASET_PATH,
        columns=["date", "ticker", "future_1m_return", "future_1m_spy_return", "vol_12m",
                 "v3m_spy_to_ma10m", "v3m_vix", "v3m_hyg_drawdown"],
    )
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["ticker"] = dataset["ticker"].astype(str).str.strip().str.upper()

    indexed = dataset.set_index(["date", "ticker"])
    vol = indexed["vol_12m"]
    returns_by_date = {
        date: month.droplevel(0)
        for date, month in indexed["future_1m_return"].groupby(level=0)
    }

    exposures = hedge_exposures(dataset)

    print("Building sleeves...", flush=True)
    orig_sleeves = {
        b: build_sleeves(load_rankings(b), ORIG_TOPNS, vol) for b in ORIGINAL_BRANCHES
    }
    neigh_sleeves = {
        d: build_sleeves(load_rankings(f"neighbor_dim{d}"), NEIGH_TOPNS, vol)
        for d in NEIGHBOR_DIMS
    }

    # Common test dates = intersection across all branches.
    date_sets = [set(s[ORIG_TOPNS[0]].keys()) for s in orig_sleeves.values()]
    date_sets += [set(s[NEIGH_TOPNS[0]].keys()) for s in neigh_sleeves.values()]
    dates = sorted(set.intersection(*date_sets))
    print(f"Common test months: {len(dates)} ({dates[0].date()} to {dates[-1].date()})", flush=True)

    spy = dataset.groupby("date")["future_1m_spy_return"].first().reindex(dates)

    variants = []
    for feats in ORIGINAL_BRANCHES:
        for otn in ORIG_TOPNS:
            variants.append((feats, otn, None, None, 1.0, 0.0))          # pure original
            for dim in NEIGHBOR_DIMS:
                for ntn in NEIGH_TOPNS:
                    for w_o, w_n in BLENDS:
                        variants.append((feats, otn, dim, ntn, w_o, w_n))
    for dim in NEIGHBOR_DIMS:                                            # pure neighbor
        for ntn in NEIGH_TOPNS:
            variants.append((None, None, dim, ntn, 0.0, 1.0))

    print(f"Variants x hedges: {len(variants)} x {len(HEDGES)} = "
          f"{len(variants) * len(HEDGES)}", flush=True)

    rows = []
    for feats, otn, dim, ntn, w_o, w_n in variants:
        orig = orig_sleeves[feats][otn] if feats is not None else None
        neigh = neigh_sleeves[dim][ntn] if dim is not None else None
        books = blend_books(orig, neigh, w_o, w_n, dates)

        for hedge in HEDGES:
            net = simulate(books, returns_by_date, exposures[hedge])
            stats = summarize_returns(net, spy)
            fold_sharpes = [
                summarize_returns(net.loc[f])["sharpe_ratio"]
                for _, f in contiguous_folds(list(net.index), N_FOLDS)
            ]
            rows.append({
                "orig_feats": feats or "-",
                "orig_topn": otn if otn is not None else "-",
                "neigh_dim": dim if dim is not None else "-",
                "neigh_topn": ntn if ntn is not None else "-",
                "blend": f"{int(w_o * 100)}/{int(w_n * 100)}",
                "hedge": hedge,
                "ann_return": stats["annualized_return"],
                "sharpe": stats["sharpe_ratio"],
                "sortino": stats["sortino_ratio"],
                "max_dd": stats["max_drawdown"],
                "ann_vol": stats["annualized_volatility"],
                "ir_vs_spy": stats["information_ratio_vs_benchmark"],
                "fold_sharpe_mean": float(np.nanmean(fold_sharpes)),
                "fold_sharpe_std": float(np.nanstd(fold_sharpes, ddof=1)),
                "fold_sharpe_min": float(np.nanmin(fold_sharpes)),
            })

    results = pd.DataFrame(rows)
    results = results.sort_values("sharpe", ascending=False).reset_index(drop=True)
    results.to_csv(os.path.join(OUT_DIR, "v3_grid_results.csv"), index=False)

    spy_stats = summarize_returns(spy)

    marginals = []
    for dimension in ["orig_feats", "orig_topn", "neigh_dim", "neigh_topn", "blend", "hedge"]:
        g = results.groupby(dimension).agg(
            n=("sharpe", "size"),
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            mean_ann_return=("ann_return", "mean"),
            mean_max_dd=("max_dd", "mean"),
            mean_fold_min=("fold_sharpe_min", "mean"),
        ).reset_index().rename(columns={dimension: "value"})
        g.insert(0, "dimension", dimension)
        marginals.append(g)
    marginal = pd.concat(marginals, ignore_index=True)
    marginal.to_csv(os.path.join(OUT_DIR, "v3_marginal_analysis.csv"), index=False)

    report_path = os.path.join(OUT_DIR, "v3_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LTSAF v3 Experiment Grid\n" + "=" * 60 + "\n\n")
        f.write(f"Months: {len(dates)} ({dates[0].date()} to {dates[-1].date()}), "
                f"net of {TRANSACTION_COST:.4f} costs, rf=0\n")
        f.write(f"Total variants: {len(results)}\n")
        f.write(f"SPY: ann={spy_stats['annualized_return']:.2%} "
                f"sharpe={spy_stats['sharpe_ratio']:.2f} "
                f"maxDD={spy_stats['max_drawdown']:.2%}\n\n")
        f.write("!!! With ~1,000 variants on one 9.5-year sample, the top cells are\n"
                "!!! partly luck. Trust the marginal analysis, not the single best row.\n\n")
        f.write("TOP 20 BY NET SHARPE\n")
        f.write(results.head(20).to_string(index=False))
        f.write("\n\nBOTTOM 5 (for calibration)\n")
        f.write(results.tail(5).to_string(index=False))
        f.write("\n\nMARGINAL ANALYSIS (mean over all other settings)\n")
        f.write(marginal.to_string(index=False))
        f.write("\n")

    with open(report_path, encoding="utf-8") as f:
        print("\n" + f.read())
    print(f"Grid runtime: {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
