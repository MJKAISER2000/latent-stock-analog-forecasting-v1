import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIVE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_trading_live"
FROZEN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_trading"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

LIVE_HOLDINGS_PATH = LIVE_OUTPUT_DIR / "current_live_holdings.csv"
LIVE_SIGNALS_PATH = LIVE_OUTPUT_DIR / "live_portfolio_signals.csv"
LIVE_RUN_SUMMARY_PATH = LIVE_OUTPUT_DIR / "live_run_summary.csv"
LIVE_VALUE_SNAPSHOTS_PATH = LIVE_OUTPUT_DIR / "live_value_snapshots.csv"
SPY_BENCHMARK_ANCHOR_PATH = LIVE_OUTPUT_DIR / "spy_benchmark_anchor.csv"
LATEST_LIVE_VALUE_DETAIL_PATH = LIVE_OUTPUT_DIR / "latest_live_value_detail.csv"
LIVE_REBALANCE_PATH = LIVE_OUTPUT_DIR / "latest_live_rebalance_orders.csv"
LIVE_PERFORMANCE_PATH = LIVE_OUTPUT_DIR / "live_performance_ledger.csv"
LIVE_ORDER_LEDGER_PATH = LIVE_OUTPUT_DIR / "live_order_ledger.csv"

LIVE_DAILY_PRICES_PATH = DATA_DIR / "live_500_daily_prices.parquet"
LIVE_MONTHLY_PRICES_PATH = DATA_DIR / "live_500_monthly_prices.parquet"
LIVE_DATASET_PATH = DATA_DIR / "live_full500_with_stock_latent_neighbors.parquet"

FROZEN_SIGNALS_PATH = FROZEN_OUTPUT_DIR / "paper_portfolio_signals.csv"

st.set_page_config(
    page_title="LTSAF Live",
    page_icon="📈",
    layout="wide",
)


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f14;
        color: #f5f5f5;
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        background-color: #111820;
        border: 1px solid #1f2a35;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 0 0 rgba(0,0,0,0);
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa4af;
    }

    div[data-testid="stMetricValue"] {
        color: #f5f7fa;
        font-size: 1.75rem;
    }

    .rh-card {
        background: #111820;
        border: 1px solid #1f2a35;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .rh-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f5f7fa;
        margin-bottom: 8px;
    }

    .rh-subtitle {
        color: #9aa4af;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }

    .green {
        color: #00c805;
        font-weight: 700;
    }

    .red {
        color: #ff5000;
        font-weight: 700;
    }

    .muted {
        color: #9aa4af;
    }

    .big-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f5f7fa;
    }

    .small-label {
        color: #9aa4af;
        font-size: 0.85rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111820;
        border-radius: 999px;
        color: #d8dee5;
        padding-left: 18px;
        padding-right: 18px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00c805;
        color: #08130a;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Utilities
# =============================================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def fmt_money(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"${float(x):,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"


def fmt_num(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.2f}"
    except Exception:
        return "N/A"


def color_delta(value) -> str:
    try:
        value = float(value)
    except Exception:
        return "muted"

    return "green" if value >= 0 else "red"


def color_pct_text(value) -> str:
    """Inline CSS for a percentage cell: green if up, red if down, muted if flat/NA."""
    try:
        value = float(value)
    except Exception:
        return "color: #9aa4b2"

    if pd.isna(value):
        return "color: #9aa4b2"
    if value > 0:
        return "color: #16c784; font-weight: 600"
    if value < 0:
        return "color: #ea3943; font-weight: 600"
    return "color: #9aa4b2"


def apply_spy_benchmark(value_snapshots: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Recompute the SPY 'buy & hold from inception' benchmark for every snapshot row.

    Buys SPY at the inception date with the portfolio's starting value
    (shares = inception_value / spy_inception_price), holds that fixed share count,
    and values it at each row's SPY price. This is independent of how the snapshot
    column was stored and stays correct even across days that were never logged.
    """
    if len(value_snapshots) == 0 or "spy_current_price" not in value_snapshots.columns:
        return value_snapshots

    out = value_snapshots.copy()

    inception_value = None
    spy_inception_price = None

    if len(anchor) > 0 and "spy_inception_price" in anchor.columns:
        try:
            spy_inception_price = float(anchor["spy_inception_price"].iloc[0])
            inception_value = float(anchor["inception_value"].iloc[0])
        except Exception:
            spy_inception_price = None

    if spy_inception_price is None or spy_inception_price <= 0:
        # Fallback: anchor to the earliest snapshot we have.
        sp = pd.to_numeric(out["spy_current_price"], errors="coerce").dropna()
        cv = pd.to_numeric(out["current_value"], errors="coerce").dropna()
        if len(sp) > 0 and len(cv) > 0:
            spy_inception_price = float(sp.iloc[0])
            inception_value = float(cv.iloc[0])

    if not spy_inception_price or spy_inception_price <= 0 or not inception_value:
        return out

    spy_price = pd.to_numeric(out["spy_current_price"], errors="coerce")
    current_value = pd.to_numeric(out["current_value"], errors="coerce")

    out["spy_equivalent_current_value"] = inception_value * (spy_price / spy_inception_price)
    out["spy_cumulative_return"] = spy_price / spy_inception_price - 1.0
    out["excess_cumulative_return_vs_spy"] = (
        current_value / inception_value - 1.0
    ) - out["spy_cumulative_return"]
    out["excess_cumulative_pnl_vs_spy"] = current_value - out["spy_equivalent_current_value"]

    return out


def cumulative_vs_spy(value_snapshots: pd.DataFrame, anchor: pd.DataFrame):
    """Return (gap_dollars, excess_return) of the portfolio vs SPY since inception."""
    if len(value_snapshots) == 0:
        return None, None

    cols = value_snapshots.columns
    if "spy_equivalent_current_value" not in cols or "current_value" not in cols:
        return None, None

    last = value_snapshots.iloc[-1]
    cv = pd.to_numeric(pd.Series([last.get("current_value")]), errors="coerce").iloc[0]
    spv = pd.to_numeric(pd.Series([last.get("spy_equivalent_current_value")]), errors="coerce").iloc[0]

    if pd.isna(cv) or pd.isna(spv):
        return None, None

    gap = float(cv - spv)

    inception_value = None
    if len(anchor) > 0 and "inception_value" in anchor.columns:
        try:
            inception_value = float(anchor["inception_value"].iloc[0])
        except Exception:
            inception_value = None

    if not inception_value:
        first_cv = pd.to_numeric(value_snapshots["current_value"], errors="coerce").dropna()
        inception_value = float(first_cv.iloc[0]) if len(first_cv) > 0 else None

    excess = gap / inception_value if inception_value else None
    return gap, excess


def build_positions(holdings: pd.DataFrame, value_detail: pd.DataFrame) -> pd.DataFrame:
    """Combine current holdings with the latest live valuation so per-stock prices,
    day moves, and since-entry returns reflect *current* prices (the holdings CSV's
    last_price/market_value are frozen at the last rebalance)."""
    if len(holdings) == 0:
        return pd.DataFrame()

    h = holdings.copy()
    h["ticker"] = h["ticker"].astype(str).str.strip().str.upper()

    for c in ["shares", "avg_entry_price", "last_price", "market_value", "current_weight"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")

    if len(value_detail) > 0 and "current_price" in value_detail.columns:
        vd = value_detail.copy()
        vd["ticker"] = vd["ticker"].astype(str).str.strip().str.upper()
        keep = ["ticker"]
        vd["current_price"] = pd.to_numeric(vd.get("current_price"), errors="coerce")
        keep.append("current_price")
        if "day_return" in vd.columns:
            vd["day_return"] = pd.to_numeric(vd["day_return"], errors="coerce")
            keep.append("day_return")
        if "current_value" in vd.columns:
            vd["live_value"] = pd.to_numeric(vd["current_value"], errors="coerce")
            keep.append("live_value")
        h = h.merge(vd[keep].drop_duplicates("ticker"), on="ticker", how="left")

    if "current_price" not in h.columns:
        h["current_price"] = h.get("last_price")
    else:
        h["current_price"] = h["current_price"].fillna(h.get("last_price"))

    if "day_return" not in h.columns:
        h["day_return"] = np.nan

    if "live_value" in h.columns:
        h["live_market_value"] = h["live_value"].fillna(h["shares"] * h["current_price"])
    else:
        h["live_market_value"] = h["shares"] * h["current_price"]

    h["since_entry_pct"] = np.where(
        h.get("avg_entry_price", pd.Series(np.nan, index=h.index)) > 0,
        h["current_price"] / h["avg_entry_price"] - 1.0,
        np.nan,
    )

    total = float(h["live_market_value"].sum())
    h["live_weight"] = h["live_market_value"] / total if total > 0 else 0.0

    return h


def render_positions_table(positions: pd.DataFrame, include_cash: bool = True) -> None:
    """Render a positions table with green/red Day % and Since Entry % columns."""
    if len(positions) == 0:
        st.info("No positions to display.")
        return

    pos = positions.copy()
    if not include_cash:
        pos = pos[pos["ticker"] != "CASH"]

    pos = pos.sort_values("live_market_value", ascending=False)

    show = pd.DataFrame(
        {
            "Ticker": pos["ticker"],
            "Shares": pos["shares"],
            "Price": pos["current_price"],
            "Day %": pos["day_return"],
            "Since Entry %": pos["since_entry_pct"],
            "Market Value": pos["live_market_value"],
            "Portfolio %": pos["live_weight"],
        }
    )

    styler = (
        show.style
        .map(color_pct_text, subset=["Day %", "Since Entry %"])
        .format(
            {
                "Shares": "{:,.0f}",
                "Price": "${:,.2f}",
                "Day %": "{:+.2%}",
                "Since Entry %": "{:+.2%}",
                "Market Value": "${:,.2f}",
                "Portfolio %": "{:.2%}",
            },
            na_rep="N/A",
        )
    )

    st.dataframe(styler, use_container_width=True, hide_index=True)


def latest_timestamped_file(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern))

    if not files:
        return None

    return Path(files[-1])


def plot_line(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    y_title: str,
    names: dict | None = None,
) -> go.Figure:
    fig = go.Figure()

    for col in y_cols:
        if col not in df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines",
                name=names.get(col, col) if names else col,
                hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="",
        yaxis_title=y_title,
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def plot_weight_bar(df: pd.DataFrame, ticker_col: str, weight_col: str, title: str) -> go.Figure:
    plot_df = df.copy()

    if len(plot_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title=title,
            height=420,
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
        )
        return fig

    plot_df[weight_col] = pd.to_numeric(plot_df[weight_col], errors="coerce").fillna(0.0)
    plot_df = plot_df.sort_values(weight_col, ascending=True)

    fig = go.Figure(
        go.Bar(
            x=plot_df[weight_col],
            y=plot_df[ticker_col],
            orientation="h",
            hovertemplate="%{y}<br>%{x:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_tickformat=".1%",
        height=max(420, 22 * len(plot_df)),
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
    )

    return fig


def plot_stock_price(prices: pd.DataFrame, ticker: str, spy: bool = True) -> go.Figure:
    fig = go.Figure()

    if ticker in prices.columns:
        series = pd.to_numeric(prices[ticker], errors="coerce").dropna()

        if len(series) > 0:
            normalized = series / series.iloc[0] * 100.0
            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized,
                    mode="lines",
                    name=ticker,
                    hovertemplate="%{x}<br>" + ticker + ": %{y:.2f}<extra></extra>",
                )
            )

    if spy and "SPY" in prices.columns:
        spy_series = pd.to_numeric(prices["SPY"], errors="coerce").dropna()

        if len(spy_series) > 0:
            spy_normalized = spy_series / spy_series.iloc[0] * 100.0
            fig.add_trace(
                go.Scatter(
                    x=spy_normalized.index,
                    y=spy_normalized,
                    mode="lines",
                    name="SPY",
                    hovertemplate="%{x}<br>SPY: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} vs SPY — indexed to 100",
        yaxis_title="Indexed price",
        height=460,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


# =============================================================================
# Data loaders
# =============================================================================

@st.cache_data(show_spinner=False)
def load_all_data() -> dict:
    holdings = safe_read_csv(LIVE_HOLDINGS_PATH)
    signals = safe_read_csv(LIVE_SIGNALS_PATH)
    run_summary = safe_read_csv(LIVE_RUN_SUMMARY_PATH)
    value_snapshots = safe_read_csv(LIVE_VALUE_SNAPSHOTS_PATH)
    spy_anchor = safe_read_csv(SPY_BENCHMARK_ANCHOR_PATH)
    value_detail = safe_read_csv(LATEST_LIVE_VALUE_DETAIL_PATH)
    rebalance = safe_read_csv(LIVE_REBALANCE_PATH)
    performance = safe_read_csv(LIVE_PERFORMANCE_PATH)
    order_ledger = safe_read_csv(LIVE_ORDER_LEDGER_PATH)
    frozen_signals = safe_read_csv(FROZEN_SIGNALS_PATH)

    daily_prices = safe_read_parquet(LIVE_DAILY_PRICES_PATH)
    monthly_prices = safe_read_parquet(LIVE_MONTHLY_PRICES_PATH)
    live_dataset = safe_read_parquet(LIVE_DATASET_PATH)

    for df in [
        holdings,
        signals,
        run_summary,
        value_snapshots,
        value_detail,
        rebalance,
        performance,
        order_ledger,
        frozen_signals,
        live_dataset,
    ]:
        if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if len(signals) > 0 and "signal_date" in signals.columns:
        signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")

    if len(run_summary) > 0 and "signal_date" in run_summary.columns:
        run_summary["signal_date"] = pd.to_datetime(run_summary["signal_date"], errors="coerce")

    if len(value_snapshots) > 0 and "valuation_time" in value_snapshots.columns:
        value_snapshots["valuation_time"] = pd.to_datetime(value_snapshots["valuation_time"], errors="coerce")

    if len(performance) > 0:
        for col in ["signal_date", "evaluation_date"]:
            if col in performance.columns:
                performance[col] = pd.to_datetime(performance[col], errors="coerce")

    if len(rebalance) > 0 and "signal_date" in rebalance.columns:
        rebalance["signal_date"] = pd.to_datetime(rebalance["signal_date"], errors="coerce")

    if len(frozen_signals) > 0 and "signal_date" in frozen_signals.columns:
        frozen_signals["signal_date"] = pd.to_datetime(frozen_signals["signal_date"], errors="coerce")

    if len(live_dataset) > 0 and "date" in live_dataset.columns:
        live_dataset["date"] = pd.to_datetime(live_dataset["date"], errors="coerce")

    for price_df in [daily_prices, monthly_prices]:
        if isinstance(price_df, pd.DataFrame) and len(price_df) > 0:
            price_df.index = pd.to_datetime(price_df.index)
            price_df.columns = [str(c).strip().upper() for c in price_df.columns]

    return {
        "holdings": holdings,
        "signals": signals,
        "run_summary": run_summary,
        "value_snapshots": value_snapshots,
        "spy_anchor": spy_anchor,
        "value_detail": value_detail,
        "rebalance": rebalance,
        "performance": performance,
        "order_ledger": order_ledger,
        "frozen_signals": frozen_signals,
        "daily_prices": daily_prices,
        "monthly_prices": monthly_prices,
        "live_dataset": live_dataset,
    }


def latest_signal(signals: pd.DataFrame) -> pd.DataFrame:
    if len(signals) == 0 or "signal_date" not in signals.columns:
        return pd.DataFrame()

    latest_date = signals["signal_date"].max()
    out = signals[signals["signal_date"] == latest_date].copy()
    out = out.sort_values("final_weight", ascending=False).reset_index(drop=True)
    return out


def latest_frozen_signal(frozen: pd.DataFrame) -> pd.DataFrame:
    if len(frozen) == 0 or "signal_date" not in frozen.columns:
        return pd.DataFrame()

    latest_date = frozen["signal_date"].max()
    out = frozen[frozen["signal_date"] == latest_date].copy()
    out = out.sort_values("final_weight", ascending=False).reset_index(drop=True)
    return out


def get_latest_branch_files() -> dict:
    return {
        "original": latest_timestamped_file(str(LIVE_OUTPUT_DIR / "live_original_branch_predictions_*.csv")),
        "neighbor": latest_timestamped_file(str(LIVE_OUTPUT_DIR / "live_neighbor_branch_predictions_*.csv")),
        "branch_detail": latest_timestamped_file(str(LIVE_OUTPUT_DIR / "live_branch_portfolio_detail_*.csv")),
        "final_portfolio": latest_timestamped_file(str(LIVE_OUTPUT_DIR / "live_final_portfolio_weights_*.csv")),
        "orders": latest_timestamped_file(str(LIVE_OUTPUT_DIR / "live_paper_trade_orders_*.csv")),
    }


@st.cache_data(show_spinner=False)
def load_branch_files(paths_dict: dict) -> dict:
    out = {}

    for key, path in paths_dict.items():
        if path is None or str(path) == "":
            out[key] = pd.DataFrame()
            continue

        try:
            df = pd.read_csv(path)
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            out[key] = df
        except Exception:
            out[key] = pd.DataFrame()

    return out


# =============================================================================
# Sidebar
# =============================================================================

data = load_all_data()

st.sidebar.markdown("## 📈 LTSAF Live")
st.sidebar.markdown("**Model:** `LTSAF_live_v1`")
st.sidebar.markdown("**Style:** Robinhood-like local research dashboard")

if st.sidebar.button("Refresh dashboard cache"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

daily_prices = data["daily_prices"]
monthly_prices = data["monthly_prices"]
holdings = data["holdings"]
signals = data["signals"]
value_snapshots = data["value_snapshots"]
spy_anchor = data["spy_anchor"]
value_detail = data["value_detail"]

# Recompute the SPY benchmark as a true buy-and-hold from inception so every chart
# that reads spy_equivalent_current_value shows the correct gap vs the portfolio.
value_snapshots = apply_spy_benchmark(value_snapshots, spy_anchor)

# Live per-stock view: current price, day move, and since-entry return for every position.
positions = build_positions(holdings, value_detail)
rebalance = data["rebalance"]
performance = data["performance"]
live_dataset = data["live_dataset"]
frozen_signals = data["frozen_signals"]

latest_live_signal = latest_signal(signals)
latest_live_frozen = latest_frozen_signal(frozen_signals)

if len(latest_live_signal) > 0:
    st.sidebar.success(f"Latest live signal: {latest_live_signal['signal_date'].iloc[0].date()}")
else:
    st.sidebar.warning("No live signal found")

if len(value_snapshots) > 0:
    latest_sidebar_value = value_snapshots.tail(1).iloc[0].to_dict()
    st.sidebar.metric("Current live value", fmt_money(latest_sidebar_value.get("current_value")))
elif len(holdings) > 0:
    total_value_sidebar = pd.to_numeric(holdings["market_value"], errors="coerce").fillna(0).sum()
    st.sidebar.metric("Current holdings value", fmt_money(total_value_sidebar))

st.sidebar.markdown("---")
st.sidebar.caption("Run manually:")
st.sidebar.code("streamlit run dashboard\\ltsaf_live_dashboard.py")


# =============================================================================
# Header
# =============================================================================

st.markdown("# LTSAF Live")
st.markdown("### Latent Twin Stock Analog Forecasting — live paper portfolio dashboard")
# =============================================================================
# Stale-data warnings
# =============================================================================

warnings = []

now = pd.Timestamp.now()

if len(value_snapshots) == 0:
    warnings.append("No live value snapshot found. Run scripts\\check_ltsaf_live_value.py before using the dashboard.")
else:
    latest_snapshot_time = pd.to_datetime(
        value_snapshots["valuation_time"],
        errors="coerce",
    ).max()

    if pd.notna(latest_snapshot_time):
        hours_old = (now - latest_snapshot_time).total_seconds() / 3600.0

        if hours_old > 24:
            warnings.append(
                f"Latest live value snapshot is {hours_old:.1f} hours old. Run scripts\\check_ltsaf_live_value.py."
            )

if len(latest_live_signal) == 0:
    warnings.append("No live signal found. Run scripts\\run_live_final_pipeline.py or scripts\\run_live_rebuild_pipeline.py.")
else:
    latest_signal_date = pd.to_datetime(
        latest_live_signal["signal_date"],
        errors="coerce",
    ).max()

    if len(monthly_prices) > 0:
        latest_price_date = pd.Timestamp(monthly_prices.index.max())

        if pd.notna(latest_signal_date) and latest_signal_date < latest_price_date:
            warnings.append(
                f"Live signal date {latest_signal_date.date()} is behind latest monthly price date {latest_price_date.date()}."
            )

if len(rebalance) == 0:
    warnings.append("No live rebalance file found. Run scripts\\generate_live_rebalance_orders.py.")

if len(holdings) == 0:
    warnings.append("No live holdings found. Run scripts\\initialize_live_holdings.py.")

if warnings:
    with st.expander("⚠️ Dashboard warnings", expanded=True):
        for warning in warnings:
            st.warning(warning)
if len(value_snapshots) > 0:
    latest_value = value_snapshots.tail(1).iloc[0].to_dict()
else:
    latest_value = {}

# Prefer latest live mark-to-market detail over initialized holdings.
# This prevents showing $100,000 while also showing negative daily P&L.
if len(value_detail) > 0:
    live_holdings = value_detail.copy()
    live_holdings["ticker"] = live_holdings["ticker"].astype(str).str.strip().str.upper()

    if "current_value" in live_holdings.columns:
        live_holdings["market_value"] = pd.to_numeric(
            live_holdings["current_value"],
            errors="coerce",
        ).fillna(0.0)

    if "current_price" in live_holdings.columns:
        live_holdings["last_price"] = pd.to_numeric(
            live_holdings["current_price"],
            errors="coerce",
        )

    live_total_value = float(live_holdings["market_value"].sum())

    if live_total_value > 0:
        live_holdings["current_weight"] = live_holdings["market_value"] / live_total_value
    else:
        live_holdings["current_weight"] = 0.0

    holdings = live_holdings.copy()

if len(holdings) > 0:
    holdings["market_value"] = pd.to_numeric(
        holdings["market_value"],
        errors="coerce",
    ).fillna(0.0)

    holdings["current_weight"] = pd.to_numeric(
        holdings.get("current_weight", 0),
        errors="coerce",
    ).fillna(0.0)

    holdings_total_value = float(holdings["market_value"].sum())

    total_value = float(
        latest_value.get("current_value", holdings_total_value)
        if latest_value.get("current_value", None) is not None
        else holdings_total_value
    )

    cash_value = float(
        holdings.loc[holdings["ticker"] == "CASH", "market_value"].sum()
    )

    cash_weight = cash_value / total_value if total_value > 0 else 0.0
    stock_positions = len(holdings[holdings["ticker"] != "CASH"])
else:
    total_value = float(latest_value.get("current_value", 0) or 0)
    cash_value = 0.0
    cash_weight = 0.0
    stock_positions = 0

day_pnl = float(latest_value.get("day_pnl", 0) or 0)
day_return = float(latest_value.get("day_return", 0) or 0)
spy_day_return = float(latest_value.get("spy_day_return", 0) or 0)
excess_day_return = float(latest_value.get("excess_day_return_vs_spy", 0) or 0)

spy_gap_dollars, spy_excess_return = cumulative_vs_spy(value_snapshots, spy_anchor)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Portfolio Value", fmt_money(total_value), f"{day_pnl:+,.2f} today")
col_b.metric("Day Return", fmt_pct(day_return), f"SPY {fmt_pct(spy_day_return)}")
if spy_excess_return is not None:
    col_c.metric(
        "Excess vs SPY (since start)",
        fmt_pct(spy_excess_return),
        f"{spy_gap_dollars:+,.0f} $",
    )
else:
    col_c.metric("Excess vs SPY", fmt_pct(excess_day_return))
col_d.metric("Cash Weight", fmt_pct(cash_weight), f"{stock_positions} stocks")

st.markdown("---")


# =============================================================================
# Tabs
# =============================================================================

tabs = st.tabs(
    [
        "Home",
        "Holdings",
        "Stock Explorer",
        "Strategy Signals",
        "Rebalance",
        "Performance vs SPY",
        "Frozen vs Live",
        "System Health",
    ]
)


# =============================================================================
# Home
# =============================================================================

with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Portfolio Value</div>', unsafe_allow_html=True)

        if len(value_snapshots) > 0:
            plot_df = value_snapshots.copy()
            plot_df["valuation_time"] = pd.to_datetime(plot_df["valuation_time"], errors="coerce")

            y_cols = ["current_value"]
            names = {"current_value": "LTSAF Live"}

            if "spy_equivalent_current_value" in plot_df.columns:
                y_cols.append("spy_equivalent_current_value")
                names["spy_equivalent_current_value"] = "SPY same starting value"

            fig = plot_line(
                plot_df,
                "valuation_time",
                y_cols,
                "Live Portfolio Value vs SPY",
                "Value ($)",
                names,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No live value snapshots yet. Run scripts\\check_ltsaf_live_value.py.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Holdings — all positions</div>', unsafe_allow_html=True)

        if len(positions) > 0:
            render_positions_table(positions, include_cash=False)
        else:
            st.info("No holdings found.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Today</div>', unsafe_allow_html=True)

        pnl_class = color_delta(day_pnl)
        ret_class = color_delta(day_return)

        st.markdown(
            f"""
            <div class="small-label">Day P&L</div>
            <div class="{pnl_class}" style="font-size: 1.8rem;">{fmt_money(day_pnl)}</div>
            <br>
            <div class="small-label">Day Return</div>
            <div class="{ret_class}" style="font-size: 1.8rem;">{fmt_pct(day_return)}</div>
            <br>
            <div class="small-label">SPY Day Return</div>
            <div>{fmt_pct(spy_day_return)}</div>
            <br>
            <div class="small-label">Excess Day Return</div>
            <div class="{color_delta(excess_day_return)}">{fmt_pct(excess_day_return)}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Latest Signal</div>', unsafe_allow_html=True)

        if len(latest_live_signal) > 0:
            latest_signal_date = latest_live_signal["signal_date"].iloc[0]
            stocks = latest_live_signal[latest_live_signal["ticker"] != "CASH"].copy()
            top = stocks.sort_values("final_weight", ascending=False).head(5)

            st.write(f"**Signal date:** {latest_signal_date.date()}")
            st.write(f"**Stocks:** {len(stocks)}")
            st.write(f"**Top:** {', '.join(top['ticker'].tolist())}")
            st.write(f"**Largest weight:** {fmt_pct(stocks['final_weight'].max())}")
        else:
            st.info("No live signal found.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Rebalance</div>', unsafe_allow_html=True)

        if len(rebalance) > 0:
            buys = rebalance[rebalance["action"] == "BUY"]
            sells = rebalance[rebalance["action"] == "SELL"]
            st.write(f"**Buy orders:** {len(buys)}")
            st.write(f"**Sell orders:** {len(sells)}")

            if "estimated_cash_after_trades" in rebalance.columns:
                st.write(f"**Cash after trades:** {fmt_money(rebalance['estimated_cash_after_trades'].iloc[0])}")
        else:
            st.info("No rebalance file found.")

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Holdings
# =============================================================================

with tabs[1]:
    st.markdown('<div class="section-header">Holdings</div>', unsafe_allow_html=True)

    if len(holdings) == 0:
        st.warning("No live holdings found.")
    else:
        view = holdings.copy()
        view["market_value"] = pd.to_numeric(view["market_value"], errors="coerce").fillna(0.0)
        view["current_weight"] = pd.to_numeric(view.get("current_weight", 0), errors="coerce").fillna(0.0)
        view = view.sort_values("market_value", ascending=False)

        c1, c2 = st.columns([1, 1])

        with c1:
            fig = plot_weight_bar(
                view[view["ticker"] != "CASH"].head(30),
                "ticker",
                "current_weight",
                "Portfolio Allocation",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if len(live_dataset) > 0 and "sector" in live_dataset.columns:
                latest_date = live_dataset["date"].max()
                sector_map = (
                    live_dataset[live_dataset["date"] == latest_date][["ticker", "sector", "industry"]]
                    .drop_duplicates("ticker")
                )

                sector_df = view.merge(sector_map, on="ticker", how="left")
                sector_df["sector"] = sector_df["sector"].fillna("Cash/Unknown")

                sector_exposure = (
                    sector_df.groupby("sector")["market_value"]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                sector_exposure["weight"] = sector_exposure["market_value"] / sector_exposure["market_value"].sum()

                fig_sector = plot_weight_bar(
                    sector_exposure,
                    "sector",
                    "weight",
                    "Sector Exposure",
                )
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.info("No sector metadata available.")

        st.markdown("### Full Holdings Table")
        st.caption("Live prices. Day % and Since Entry % are green when up, red when down.")

        if len(positions) > 0:
            render_positions_table(positions, include_cash=True)
        else:
            st.info("No holdings found.")


# =============================================================================
# Stock Explorer
# =============================================================================

with tabs[2]:
    st.markdown('<div class="section-header">Individual Stock Explorer</div>', unsafe_allow_html=True)

    all_tickers = set()

    if len(daily_prices) > 0:
        all_tickers |= set(daily_prices.columns)

    if len(holdings) > 0:
        all_tickers |= set(holdings["ticker"].tolist())

    if len(latest_live_signal) > 0:
        all_tickers |= set(latest_live_signal["ticker"].tolist())

    all_tickers = sorted([t for t in all_tickers if t != "CASH"])

    default_ticker = "SPY"

    if len(holdings) > 0:
        non_cash = holdings[holdings["ticker"] != "CASH"].copy()
        if len(non_cash) > 0:
            default_ticker = non_cash.sort_values("market_value", ascending=False)["ticker"].iloc[0]

    selected_ticker = st.selectbox(
        "Select ticker",
        all_tickers,
        index=all_tickers.index(default_ticker) if default_ticker in all_tickers else 0,
    )

    if selected_ticker in daily_prices.columns:
        series = pd.to_numeric(daily_prices[selected_ticker], errors="coerce").dropna()
        latest_price = float(series.iloc[-1]) if len(series) else np.nan
        one_day = series.pct_change(1).iloc[-1] if len(series) > 1 else np.nan
        one_month = series.pct_change(21).iloc[-1] if len(series) > 21 else np.nan
    else:
        latest_price = np.nan
        one_day = np.nan
        one_month = np.nan

    holding_row = pd.DataFrame()
    signal_row = pd.DataFrame()
    feature_row = pd.DataFrame()

    if len(holdings) > 0:
        holding_row = holdings[holdings["ticker"] == selected_ticker].copy()

    if len(latest_live_signal) > 0:
        signal_row = latest_live_signal[latest_live_signal["ticker"] == selected_ticker].copy()

    if len(live_dataset) > 0:
        temp = live_dataset[live_dataset["ticker"] == selected_ticker].copy()
        if len(temp) > 0:
            feature_row = temp.sort_values("date").tail(1).copy()

    is_held = len(holding_row) > 0
    in_signal = len(signal_row) > 0

    held_shares = float(holding_row["shares"].iloc[0]) if is_held and "shares" in holding_row.columns else 0.0
    held_value = float(holding_row["market_value"].iloc[0]) if is_held and "market_value" in holding_row.columns else 0.0
    held_weight = float(holding_row["current_weight"].iloc[0]) if is_held and "current_weight" in holding_row.columns else 0.0
    avg_entry = float(holding_row["avg_entry_price"].iloc[0]) if is_held and "avg_entry_price" in holding_row.columns else np.nan

    target_weight = float(signal_row["final_weight"].iloc[0]) if in_signal and "final_weight" in signal_row.columns else 0.0
    branches = str(signal_row["branches"].iloc[0]) if in_signal and "branches" in signal_row.columns else "Not selected"
    best_rank = signal_row["best_rank"].iloc[0] if in_signal and "best_rank" in signal_row.columns else "N/A"
    avg_score = signal_row["avg_ranker_score"].iloc[0] if in_signal and "avg_ranker_score" in signal_row.columns else "N/A"

    unrealized_pnl = (latest_price - avg_entry) * held_shares if is_held and not pd.isna(avg_entry) and not pd.isna(latest_price) else np.nan
    unrealized_return = latest_price / avg_entry - 1.0 if is_held and avg_entry > 0 and not pd.isna(latest_price) else np.nan

    st.markdown('<div class="rh-card">', unsafe_allow_html=True)

    if len(feature_row) > 0:
        company = feature_row["company"].iloc[0] if "company" in feature_row.columns else ""
        sector = feature_row["sector"].iloc[0] if "sector" in feature_row.columns else ""
        industry = feature_row["industry"].iloc[0] if "industry" in feature_row.columns else ""
        name_line = f"{selected_ticker} — {company}"
    else:
        sector = ""
        industry = ""
        name_line = selected_ticker

    st.markdown(f'<div class="rh-title">{name_line}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="rh-subtitle">{sector} {(" / " + industry) if industry else ""}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", fmt_money(latest_price))
    c2.metric("1D", fmt_pct(one_day))
    c3.metric("1M", fmt_pct(one_month))
    c4.metric("Target Weight", fmt_pct(target_weight))
    c5.metric("Holding Weight", fmt_pct(held_weight))

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Shares", fmt_num(held_shares))
    c7.metric("Market Value", fmt_money(held_value))
    c8.metric("Avg Entry", fmt_money(avg_entry))
    c9.metric("Unrealized P&L", fmt_money(unrealized_pnl))
    c10.metric("Unrealized Return", fmt_pct(unrealized_return))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rh-card">', unsafe_allow_html=True)
    st.markdown('<div class="rh-title">Price Chart</div>', unsafe_allow_html=True)

    chart_mode = st.radio(
        "Chart range",
        ["Daily full history", "Monthly full history"],
        horizontal=True,
        key="stock_chart_range",
    )

    if chart_mode == "Daily full history":
        price_source = daily_prices
    else:
        price_source = monthly_prices

    if len(price_source) > 0:
        fig = plot_stock_price(price_source, selected_ticker, spy=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data found.")

    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Model Signal</div>', unsafe_allow_html=True)

        status_text = "Selected" if in_signal else "Not selected"
        held_text = "Currently held" if is_held else "Not currently held"

        st.write(f"**Signal status:** {status_text}")
        st.write(f"**Holding status:** {held_text}")
        st.write(f"**Branches:** {branches}")
        st.write(f"**Best rank:** {best_rank}")
        st.write(f"**Average ranker score:** {avg_score}")

        if len(signal_row) > 0:
            st.dataframe(signal_row, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Latent Twin Analog Features</div>', unsafe_allow_html=True)

        if len(feature_row) > 0:
            analog_cols = [
                "neighbor_count",
                "neighbor_distance_mean",
                "neighbor_distance_median",
                "neighbor_distance_min",
                "neighbor_avg_future_1m_return",
                "neighbor_median_future_1m_return",
                "neighbor_avg_future_1m_excess_return",
                "neighbor_outperform_spy_1m_rate",
                "neighbor_positive_1m_return_rate",
            ]

            analog_cols = [c for c in analog_cols if c in feature_row.columns]

            if analog_cols:
                analog_display = feature_row[analog_cols].T.reset_index()
                analog_display.columns = ["Feature", "Value"]
                st.dataframe(analog_display, use_container_width=True, hide_index=True)
            else:
                st.info("No latent-neighbor columns found.")
        else:
            st.info("Ticker not found in live model dataset.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rh-card">', unsafe_allow_html=True)
    st.markdown('<div class="rh-title">Feature Snapshot</div>', unsafe_allow_html=True)

    if len(feature_row) > 0:
        desired_cols = [
            "date",
            "ticker",
            "company",
            "sector",
            "industry",
            "price",
            "ret_1m",
            "ret_3m",
            "ret_6m",
            "ret_12m",
            "vol_3m",
            "vol_6m",
            "vol_12m",
            "stock_drawdown",
            "stock_drawdown_6m",
            "stock_drawdown_12m",
            "spy_ret_1m",
            "spy_ret_3m",
            "spy_ret_6m",
            "spy_ret_12m",
            "spy_drawdown",
        ]

        desired_cols = [c for c in desired_cols if c in feature_row.columns]
        st.dataframe(feature_row[desired_cols], use_container_width=True, hide_index=True)

        return_cols = [
            "ret_1m",
            "ret_3m",
            "ret_6m",
            "ret_12m",
            "vol_12m",
            "stock_drawdown",
            "neighbor_avg_future_1m_return",
            "neighbor_avg_future_1m_excess_return",
            "neighbor_outperform_spy_1m_rate",
        ]

        return_cols = [c for c in return_cols if c in feature_row.columns]

        if return_cols:
            mini = feature_row[return_cols].T.reset_index()
            mini.columns = ["Metric", "Value"]

            fig_metrics = go.Figure(
                go.Bar(
                    x=mini["Metric"],
                    y=pd.to_numeric(mini["Value"], errors="coerce"),
                    hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
                )
            )

            fig_metrics.update_layout(
                template="plotly_dark",
                title=f"{selected_ticker} Model Feature Snapshot",
                height=420,
                margin=dict(l=20, r=20, t=55, b=20),
                paper_bgcolor="#0b0f14",
                plot_bgcolor="#0b0f14",
            )

            st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("No model feature row found for this ticker.")

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Strategy Signals
# =============================================================================

with tabs[3]:
    st.markdown('<div class="section-header">Strategy Signals</div>', unsafe_allow_html=True)

    branch_paths = get_latest_branch_files()
    branch_data = load_branch_files({k: str(v) if v else "" for k, v in branch_paths.items()})

    original_df = branch_data.get("original", pd.DataFrame()).copy()
    neighbor_df = branch_data.get("neighbor", pd.DataFrame()).copy()
    branch_detail_df = branch_data.get("branch_detail", pd.DataFrame()).copy()
    final_df = latest_live_signal.copy()
    frozen_df = latest_live_frozen.copy()

    # -------------------------------------------------------------------------
    # Top summary cards
    # -------------------------------------------------------------------------
    final_stocks = final_df[final_df["ticker"] != "CASH"].copy() if len(final_df) else pd.DataFrame()
    frozen_stocks = frozen_df[frozen_df["ticker"] != "CASH"].copy() if len(frozen_df) else pd.DataFrame()

    if len(original_df) > 0 and "rank_by_date" in original_df.columns:
        original_top = original_df.sort_values("rank_by_date").head(20).copy()
    else:
        original_top = original_df.head(20).copy()

    if len(neighbor_df) > 0 and "rank_by_date" in neighbor_df.columns:
        neighbor_top = neighbor_df.sort_values("rank_by_date").head(10).copy()
    else:
        neighbor_top = neighbor_df.head(10).copy()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Final Portfolio",
        f"{len(final_stocks)} names",
        f"Top: {final_stocks.iloc[0]['ticker'] if len(final_stocks) else 'N/A'}",
    )

    c2.metric(
        "Original Branch",
        f"{len(original_top)} shown",
        f"Top: {original_top.iloc[0]['ticker'] if len(original_top) and 'ticker' in original_top.columns else 'N/A'}",
    )

    c3.metric(
        "Latent Branch",
        f"{len(neighbor_top)} shown",
        f"Top: {neighbor_top.iloc[0]['ticker'] if len(neighbor_top) and 'ticker' in neighbor_top.columns else 'N/A'}",
    )

    if len(final_stocks) > 0 and len(frozen_stocks) > 0:
        live_set = set(final_stocks["ticker"])
        frozen_set = set(frozen_stocks["ticker"])
        overlap_count = len(live_set & frozen_set)
        overlap_pct = overlap_count / len(live_set) if len(live_set) else 0.0
    else:
        overlap_count = 0
        overlap_pct = 0.0

    c4.metric(
        "Frozen Overlap",
        f"{overlap_count} names",
        fmt_pct(overlap_pct),
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Selector
    # -------------------------------------------------------------------------
    strategy_choice = st.radio(
        "Select strategy view",
        [
            "Overview",
            "Final Portfolio",
            "Original Branch",
            "Latent-Neighbor Branch",
            "Branch Overlap",
            "SPY Benchmark",
            "Frozen Baseline",
        ],
        horizontal=True,
    )

    # -------------------------------------------------------------------------
    # Overview
    # -------------------------------------------------------------------------
    if strategy_choice == "Overview":
        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="rh-card">', unsafe_allow_html=True)
            st.markdown('<div class="rh-title">Final Live Portfolio</div>', unsafe_allow_html=True)

            if len(final_stocks) > 0:
                st.plotly_chart(
                    plot_weight_bar(
                        final_stocks.head(30),
                        "ticker",
                        "final_weight",
                        "Final LTSAF Live Weights",
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No final live portfolio found.")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="rh-card">', unsafe_allow_html=True)
            st.markdown('<div class="rh-title">Branch Top Picks</div>', unsafe_allow_html=True)

            branch_rows = []

            if len(original_top) > 0 and "ticker" in original_top.columns:
                for i, row in original_top.head(10).reset_index(drop=True).iterrows():
                    branch_rows.append(
                        {
                            "Branch": "Original",
                            "Rank": i + 1,
                            "Ticker": row["ticker"],
                            "Score": row.get("ranker_score", row.get("prediction", np.nan)),
                        }
                    )

            if len(neighbor_top) > 0 and "ticker" in neighbor_top.columns:
                for i, row in neighbor_top.head(10).reset_index(drop=True).iterrows():
                    branch_rows.append(
                        {
                            "Branch": "Latent-Neighbor",
                            "Rank": i + 1,
                            "Ticker": row["ticker"],
                            "Score": row.get("ranker_score", row.get("prediction", np.nan)),
                        }
                    )

            if branch_rows:
                branch_table = pd.DataFrame(branch_rows)
                st.dataframe(branch_table, use_container_width=True, hide_index=True)
            else:
                st.info("No branch prediction files found.")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Final Portfolio Table</div>', unsafe_allow_html=True)

        if len(final_df) > 0:
            display = final_df.copy()
            keep_cols = [
                "ticker",
                "final_weight",
                "final_weight_before_regime",
                "branches",
                "avg_ranker_score",
                "best_rank",
                "signal_date",
                "regime_risk_on",
                "tech_drawdown",
            ]
            keep_cols = [c for c in keep_cols if c in display.columns]
            st.dataframe(display[keep_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No final portfolio table found.")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Final Portfolio
    # -------------------------------------------------------------------------
    elif strategy_choice == "Final Portfolio":
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Final Portfolio</div>', unsafe_allow_html=True)

        if len(final_df) == 0:
            st.warning("No final portfolio signal found.")
        else:
            stocks = final_df[final_df["ticker"] != "CASH"].copy()
            cash = final_df[final_df["ticker"] == "CASH"].copy()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stock Count", len(stocks))
            c2.metric("Cash Weight", fmt_pct(cash["final_weight"].sum() if len(cash) else 0.0))
            c3.metric("Largest Weight", fmt_pct(stocks["final_weight"].max() if len(stocks) else 0.0))
            c4.metric("Signal Date", str(final_df["signal_date"].iloc[0].date()) if "signal_date" in final_df.columns else "N/A")

            st.plotly_chart(
                plot_weight_bar(stocks.head(35), "ticker", "final_weight", "Final Live Target Weights"),
                use_container_width=True,
            )

            st.dataframe(stocks, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Original Branch
    # -------------------------------------------------------------------------
    elif strategy_choice == "Original Branch":
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Original Branch</div>', unsafe_allow_html=True)
        st.markdown('<div class="rh-subtitle">Conventional ML feature branch: returns, momentum, volatility, drawdown, sector/industry structure.</div>', unsafe_allow_html=True)

        if len(original_df) == 0:
            st.warning("No original branch prediction file found.")
        else:
            df = original_df.copy()

            if "rank_by_date" in df.columns:
                df = df.sort_values("rank_by_date")

            top_n = st.slider("Original branch names to show", 5, 100, 30, key="original_branch_top_n")
            top = df.head(top_n).copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", len(df))
            c2.metric("Shown", len(top))
            c3.metric("Top Ticker", str(top.iloc[0]["ticker"]) if len(top) and "ticker" in top.columns else "N/A")

            score_col = None
            for candidate in ["ranker_score", "prediction", "score"]:
                if candidate in top.columns:
                    score_col = candidate
                    break

            if score_col is not None:
                plot_top = top.sort_values(score_col, ascending=True)
                fig = go.Figure(
                    go.Bar(
                        x=plot_top[score_col],
                        y=plot_top["ticker"],
                        orientation="h",
                        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Original Branch Scores",
                    height=max(420, 22 * len(plot_top)),
                    margin=dict(l=20, r=20, t=55, b=20),
                    paper_bgcolor="#0b0f14",
                    plot_bgcolor="#0b0f14",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Latent-Neighbor Branch
    # -------------------------------------------------------------------------
    elif strategy_choice == "Latent-Neighbor Branch":
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Latent-Neighbor Branch</div>', unsafe_allow_html=True)
        st.markdown('<div class="rh-subtitle">Analog branch: stock state → PCA latent coordinate → nearest historical stock states → future outcome summary.</div>', unsafe_allow_html=True)

        if len(neighbor_df) == 0:
            st.warning("No latent-neighbor branch prediction file found.")
        else:
            df = neighbor_df.copy()

            if "rank_by_date" in df.columns:
                df = df.sort_values("rank_by_date")

            top_n = st.slider("Latent-neighbor names to show", 5, 100, 30, key="neighbor_branch_top_n")
            top = df.head(top_n).copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", len(df))
            c2.metric("Shown", len(top))
            c3.metric("Top Ticker", str(top.iloc[0]["ticker"]) if len(top) and "ticker" in top.columns else "N/A")

            score_col = None
            for candidate in ["ranker_score", "prediction", "score"]:
                if candidate in top.columns:
                    score_col = candidate
                    break

            if score_col is not None:
                plot_top = top.sort_values(score_col, ascending=True)
                fig = go.Figure(
                    go.Bar(
                        x=plot_top[score_col],
                        y=plot_top["ticker"],
                        orientation="h",
                        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Latent-Neighbor Branch Scores",
                    height=max(420, 22 * len(plot_top)),
                    margin=dict(l=20, r=20, t=55, b=20),
                    paper_bgcolor="#0b0f14",
                    plot_bgcolor="#0b0f14",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Branch Overlap
    # -------------------------------------------------------------------------
    elif strategy_choice == "Branch Overlap":
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Branch Overlap</div>', unsafe_allow_html=True)

        if len(original_top) == 0 or len(neighbor_top) == 0:
            st.warning("Need both original and latent-neighbor branch files.")
        else:
            original_set = set(original_top["ticker"]) if "ticker" in original_top.columns else set()
            neighbor_set = set(neighbor_top["ticker"]) if "ticker" in neighbor_top.columns else set()
            final_set = set(final_stocks["ticker"]) if len(final_stocks) else set()

            overlap_original_neighbor = sorted(original_set & neighbor_set)
            original_only = sorted(original_set - neighbor_set)
            neighbor_only = sorted(neighbor_set - original_set)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Top", len(original_set))
            c2.metric("Latent Top", len(neighbor_set))
            c3.metric("Overlap", len(overlap_original_neighbor))
            c4.metric("Final Names", len(final_set))

            overlap_table = pd.DataFrame(
                {
                    "Group": ["Overlap", "Original Only", "Latent Only"],
                    "Tickers": [
                        ", ".join(overlap_original_neighbor),
                        ", ".join(original_only),
                        ", ".join(neighbor_only),
                    ],
                    "Count": [
                        len(overlap_original_neighbor),
                        len(original_only),
                        len(neighbor_only),
                    ],
                }
            )

            st.dataframe(overlap_table, use_container_width=True, hide_index=True)

            if len(branch_detail_df) > 0:
                st.markdown("### Branch Detail")
                st.dataframe(branch_detail_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SPY Benchmark
    # -------------------------------------------------------------------------
    elif strategy_choice == "SPY Benchmark":
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">SPY Benchmark</div>', unsafe_allow_html=True)

        if len(daily_prices) > 0 and "SPY" in daily_prices.columns:
            fig = plot_stock_price(daily_prices, "SPY", spy=False)
            st.plotly_chart(fig, use_container_width=True)

            spy = pd.to_numeric(daily_prices["SPY"], errors="coerce").dropna()
            if len(spy) > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("SPY Price", fmt_money(spy.iloc[-1]))
                c2.metric("1D", fmt_pct(spy.pct_change(1).iloc[-1] if len(spy) > 1 else np.nan))
                c3.metric("1M", fmt_pct(spy.pct_change(21).iloc[-1] if len(spy) > 21 else np.nan))
                c4.metric("1Y", fmt_pct(spy.pct_change(252).iloc[-1] if len(spy) > 252 else np.nan))
        else:
            st.warning("No SPY price data found.")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Frozen Baseline
    # -------------------------------------------------------------------------
    elif strategy_choice == "Frozen Baseline":
        st.markdown('<div class="rh-card">', unsafe_allow_html=True)
        st.markdown('<div class="rh-title">Frozen Research Baseline</div>', unsafe_allow_html=True)

        if len(frozen_df) == 0:
            st.warning("No frozen signal ledger found.")
        else:
            frozen_stocks = frozen_df[frozen_df["ticker"] != "CASH"].copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Frozen Signal Date", str(frozen_df["signal_date"].iloc[0].date()) if "signal_date" in frozen_df.columns else "N/A")
            c2.metric("Frozen Names", len(frozen_stocks))
            c3.metric("Top Ticker", str(frozen_stocks.iloc[0]["ticker"]) if len(frozen_stocks) else "N/A")

            st.plotly_chart(
                plot_weight_bar(frozen_stocks.head(35), "ticker", "final_weight", "Frozen Baseline Weights"),
                use_container_width=True,
            )

            st.dataframe(frozen_stocks, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)
# =============================================================================
# Rebalance
# =============================================================================

with tabs[4]:
    st.markdown('<div class="section-header">Rebalance Orders</div>', unsafe_allow_html=True)

    if len(rebalance) == 0:
        st.warning("No live rebalance orders found.")
    else:
        buys = rebalance[rebalance["action"] == "BUY"].copy()
        sells = rebalance[rebalance["action"] == "SELL"].copy()
        holds = rebalance[rebalance["action"] == "HOLD"].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Buy Orders", len(buys))
        c2.metric("Sell Orders", len(sells))
        c3.metric("Holds", len(holds))

        if "estimated_cash_after_trades" in rebalance.columns:
            c4.metric("Cash After Trades", fmt_money(rebalance["estimated_cash_after_trades"].iloc[0]))

        active = rebalance[rebalance["action"].isin(["BUY", "SELL"])].copy()

        st.markdown("### Active Orders")
        if len(active) == 0:
            st.success("No active buy/sell orders. Portfolio is close to target.")
        else:
            st.dataframe(active, use_container_width=True, hide_index=True)

        st.markdown("### Full Rebalance Table")
        st.dataframe(rebalance, use_container_width=True, hide_index=True)


# =============================================================================
# Performance vs SPY
# =============================================================================

with tabs[5]:
    st.markdown('<div class="section-header">Performance vs SPY</div>', unsafe_allow_html=True)

    if len(value_snapshots) > 0:
        st.markdown("### Daily / Intraday Value Tracking")
        plot_df = value_snapshots.copy()
        plot_df["valuation_time"] = pd.to_datetime(plot_df["valuation_time"], errors="coerce")

        y_cols = ["current_value"]
        names = {"current_value": "LTSAF Live"}

        if "spy_equivalent_current_value" in plot_df.columns:
            y_cols.append("spy_equivalent_current_value")
            names["spy_equivalent_current_value"] = "SPY same starting value"

        st.plotly_chart(
            plot_line(plot_df, "valuation_time", y_cols, "LTSAF Live vs SPY", "Value ($)", names),
            use_container_width=True,
        )

        st.dataframe(plot_df.tail(50), use_container_width=True, hide_index=True)
    else:
        st.info("No value snapshots yet.")

    st.markdown("### Completed Monthly Performance")

    if len(performance) > 0:
        perf = performance.copy()

        if "evaluation_date" in perf.columns:
            perf["evaluation_date"] = pd.to_datetime(perf["evaluation_date"], errors="coerce")

        fig = go.Figure()

        if "portfolio_cumulative_value" in perf.columns:
            fig.add_trace(
                go.Scatter(
                    x=perf["evaluation_date"],
                    y=perf["portfolio_cumulative_value"],
                    mode="lines+markers",
                    name="LTSAF Live",
                )
            )

        if "spy_cumulative_value" in perf.columns:
            fig.add_trace(
                go.Scatter(
                    x=perf["evaluation_date"],
                    y=perf["spy_cumulative_value"],
                    mode="lines+markers",
                    name="SPY",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            title="Completed Monthly Performance",
            yaxis_title="Value ($)",
            height=430,
            margin=dict(l=20, r=20, t=55, b=20),
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(perf, use_container_width=True, hide_index=True)
    else:
        st.info("No completed monthly performance yet. This is expected until the next completed month after the live signal.")


# =============================================================================
# Frozen vs Live
# =============================================================================

with tabs[6]:
    st.markdown('<div class="section-header">Frozen vs Live Signals</div>', unsafe_allow_html=True)

    if len(latest_live_signal) == 0 or len(latest_live_frozen) == 0:
        st.warning("Need both frozen and live signal ledgers to compare.")
    else:
        live = latest_live_signal[["ticker", "final_weight"]].copy()
        frozen = latest_live_frozen[["ticker", "final_weight"]].copy()

        live = live.rename(columns={"final_weight": "live_weight"})
        frozen = frozen.rename(columns={"final_weight": "frozen_weight"})

        comp = frozen.merge(live, on="ticker", how="outer")
        comp["frozen_weight"] = comp["frozen_weight"].fillna(0.0)
        comp["live_weight"] = comp["live_weight"].fillna(0.0)
        comp["in_frozen"] = comp["frozen_weight"] > 0
        comp["in_live"] = comp["live_weight"] > 0
        comp["in_both"] = comp["in_frozen"] & comp["in_live"]
        comp["weight_diff_live_minus_frozen"] = comp["live_weight"] - comp["frozen_weight"]
        comp["abs_weight_diff"] = comp["weight_diff_live_minus_frozen"].abs()
        comp = comp.sort_values(["in_both", "abs_weight_diff"], ascending=[False, False])

        overlap = int(comp["in_both"].sum())
        frozen_count = int(comp["in_frozen"].sum())
        live_count = int(comp["in_live"].sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frozen Names", frozen_count)
        c2.metric("Live Names", live_count)
        c3.metric("Overlap", overlap)
        c4.metric("Overlap % Live", fmt_pct(overlap / live_count if live_count else 0))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=comp["ticker"], y=comp["frozen_weight"], name="Frozen"))
        fig.add_trace(go.Bar(x=comp["ticker"], y=comp["live_weight"], name="Live"))
        fig.update_layout(
            template="plotly_dark",
            title="Frozen vs Live Target Weights",
            yaxis_tickformat=".1%",
            barmode="group",
            height=520,
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp, use_container_width=True, hide_index=True)


# =============================================================================
# System Health
# =============================================================================

with tabs[7]:
    st.markdown('<div class="section-header">System Health</div>', unsafe_allow_html=True)

    file_rows = []

    files = [
        ("Live holdings", LIVE_HOLDINGS_PATH),
        ("Latest live value detail", LATEST_LIVE_VALUE_DETAIL_PATH),
        ("Live signals", LIVE_SIGNALS_PATH),
        ("Live run summary", LIVE_RUN_SUMMARY_PATH),
        ("Live value snapshots", LIVE_VALUE_SNAPSHOTS_PATH),
        ("Live rebalance", LIVE_REBALANCE_PATH),
        ("Live performance", LIVE_PERFORMANCE_PATH),
        ("Live daily prices", LIVE_DAILY_PRICES_PATH),
        ("Live monthly prices", LIVE_MONTHLY_PRICES_PATH),
        ("Live dataset", LIVE_DATASET_PATH),
        ("Frozen signals", FROZEN_SIGNALS_PATH),
    ]

    for label, path in files:
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 * 1024) if exists and path.is_file() else np.nan
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else None

        file_rows.append(
            {
                "File": label,
                "Exists": exists,
                "Size MB": size_mb,
                "Modified": modified,
                "Path": str(path),
            }
        )

    health = pd.DataFrame(file_rows)

    st.dataframe(
        health,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Size MB": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.markdown("### Latest Data Dates")

    rows = []

    if len(monthly_prices) > 0:
        rows.append(
            {
                "Object": "Live monthly prices",
                "Start": monthly_prices.index.min(),
                "End": monthly_prices.index.max(),
                "Rows": len(monthly_prices),
                "Columns": len(monthly_prices.columns),
            }
        )

    if len(daily_prices) > 0:
        rows.append(
            {
                "Object": "Live daily prices",
                "Start": daily_prices.index.min(),
                "End": daily_prices.index.max(),
                "Rows": len(daily_prices),
                "Columns": len(daily_prices.columns),
            }
        )

    if len(live_dataset) > 0:
        rows.append(
            {
                "Object": "Live model dataset",
                "Start": live_dataset["date"].min(),
                "End": live_dataset["date"].max(),
                "Rows": len(live_dataset),
                "Columns": len(live_dataset.columns),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Commands")
    st.code(
        r"""
cd C:\latent_market_twin
.\.venv312\Scripts\Activate.ps1

python -m streamlit run dashboard\ltsaf_live_dashboard.py

python scripts\run_live_rebuild_pipeline.py
python scripts\check_ltsaf_live_value.py
python scripts\generate_live_rebalance_orders.py
python scripts\track_ltsaf_live_performance.py
        """,
        language="powershell",
    )