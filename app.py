import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import date, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS = 252
BENCHMARK = "^GSPC"
BENCHMARK_LABEL = "S&P 500"

# ── Cached data functions ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Download adjusted close prices for tickers + benchmark.
    Returns (prices_df, error_list).
    """
    all_tickers = list(tickers) + [BENCHMARK]
    errors = []
    frames = {}

    for tkr in all_tickers:
        try:
            raw = yf.download(tkr, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 2:
                errors.append(tkr)
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            frames[tkr] = close
        except Exception:
            errors.append(tkr)

    if not frames:
        return pd.DataFrame(), errors

    df = pd.DataFrame(frames)
    return df, errors


@st.cache_data(ttl=3600)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


@st.cache_data(ttl=3600)
def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    stats_dict = {}
    for col in returns.columns:
        r = returns[col].dropna()
        stats_dict[col] = {
            "Ann. Return": r.mean() * TRADING_DAYS,
            "Ann. Volatility": r.std() * np.sqrt(TRADING_DAYS),
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),
            "Min Daily Return": r.min(),
            "Max Daily Return": r.max(),
        }
    df = pd.DataFrame(stats_dict).T
    return df


def wealth_index(returns: pd.DataFrame, initial: float = 10_000) -> pd.DataFrame:
    cumret = (1 + returns).cumprod()
    return cumret * initial


def portfolio_equal_weight(returns: pd.DataFrame, stock_cols: list[str]) -> pd.Series:
    return returns[stock_cols].mean(axis=1)


def two_asset_portfolio(w: float, ret_a: pd.Series, ret_b: pd.Series):
    """Return (ann_return, ann_vol) for weight w on A, (1-w) on B."""
    port_ret = w * ret_a + (1 - w) * ret_b
    ann_ret = port_ret.mean() * TRADING_DAYS
    cov = np.cov(ret_a, ret_b) * TRADING_DAYS  # annualised covariance matrix
    ann_vol = np.sqrt(w**2 * cov[0, 0] + (1 - w)**2 * cov[1, 1] + 2 * w * (1 - w) * cov[0, 1])
    return ann_ret, ann_vol


# ── Sidebar – Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Analyzer")
    st.markdown("---")

    raw_input = st.text_input(
        "Ticker Symbols (2–5, comma-separated)",
        value="AAPL, MSFT, GOOGL",
        help="Enter between 2 and 5 valid stock ticker symbols.",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=3 * 365),
            max_value=date.today() - timedelta(days=366),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            max_value=date.today(),
        )

    run_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")

    st.markdown("---")
    with st.expander("ℹ️ About / Methodology", expanded=False):
        st.markdown("""
**What this app does**

Compare and analyze 2–5 stocks using historical price data. Features include:
- Price & return analysis
- Risk & distribution analysis
- Correlation & diversification analysis
- Two-asset portfolio explorer

**Key Assumptions**
- **Returns**: Simple (arithmetic) returns — `pct_change()`
- **Annualization**: 252 trading days per year
  - Ann. Return = mean daily return × 252
  - Ann. Volatility = daily std × √252
- **Wealth index**: `(1 + r).cumprod() × $10,000`
- **Benchmark**: S&P 500 (`^GSPC`)

**Data Source**

Yahoo Finance via `yfinance`. Adjusted closing prices (accounts for dividends and splits).
        """)

# ── Input Validation ──────────────────────────────────────────────────────────
tickers_raw = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
input_valid = True
error_msgs = []

if len(tickers_raw) < 2:
    error_msgs.append("⚠️ Please enter **at least 2** ticker symbols.")
    input_valid = False
if len(tickers_raw) > 5:
    error_msgs.append("⚠️ Please enter **no more than 5** ticker symbols.")
    input_valid = False
if (end_date - start_date).days < 365:
    error_msgs.append("⚠️ Date range must be **at least 1 year**.")
    input_valid = False

# ── Main Content ──────────────────────────────────────────────────────────────
st.title("📊 Stock Comparison & Analysis")

if error_msgs:
    for msg in error_msgs:
        st.error(msg)

# Only proceed when user clicks Analyze and inputs are valid
if not run_btn and "prices" not in st.session_state:
    st.info("👈 Enter your tickers and date range in the sidebar, then click **Analyze**.")
    st.stop()

if run_btn:
    if not input_valid:
        st.stop()
    with st.spinner("Downloading market data…"):
        prices_raw, dl_errors = download_prices(
            tuple(tickers_raw), str(start_date), str(end_date)
        )

    # Report download errors (excluding benchmark failures handled separately)
    user_errors = [t for t in dl_errors if t != BENCHMARK]
    if user_errors:
        st.warning(f"⚠️ Could not download data for: **{', '.join(user_errors)}**. They have been excluded.")

    benchmark_ok = BENCHMARK in prices_raw.columns if not prices_raw.empty else False

    # Filter to only successfully downloaded user tickers
    available_tickers = [t for t in tickers_raw if t in (prices_raw.columns if not prices_raw.empty else [])]

    if len(available_tickers) < 2:
        st.error("❌ Fewer than 2 tickers returned valid data. Please try different symbols.")
        st.stop()

    # Handle partial data: drop tickers with >5% missing, then align
    stock_prices = prices_raw[available_tickers].copy()
    missing_pct = stock_prices.isnull().mean()
    dropped = missing_pct[missing_pct > 0.05].index.tolist()
    if dropped:
        st.warning(f"⚠️ Dropped due to >5% missing data: **{', '.join(dropped)}**")
        stock_prices = stock_prices.drop(columns=dropped)
        available_tickers = [t for t in available_tickers if t not in dropped]

    if len(available_tickers) < 2:
        st.error("❌ Not enough valid tickers after data cleaning.")
        st.stop()

    # Truncate to overlapping range
    stock_prices = stock_prices.dropna(how="any")
    if stock_prices.empty:
        st.error("❌ No overlapping date range found among selected tickers.")
        st.stop()

    overlap_start = stock_prices.index.min().date()
    overlap_end = stock_prices.index.max().date()
    if overlap_start != start_date or overlap_end != end_date:
        st.info(
            f"ℹ️ Data truncated to overlapping range: **{overlap_start}** → **{overlap_end}**"
        )

    # Benchmark series
    bench_prices = prices_raw[[BENCHMARK]].reindex(stock_prices.index).dropna() if benchmark_ok else None
    full_prices = stock_prices.copy()
    if bench_prices is not None:
        full_prices[BENCHMARK_LABEL] = bench_prices[BENCHMARK]

    # Compute returns
    returns_stocks = compute_returns(stock_prices)
    returns_full = compute_returns(full_prices)

    # Cache in session state
    st.session_state["prices"] = stock_prices
    st.session_state["full_prices"] = full_prices
    st.session_state["returns_stocks"] = returns_stocks
    st.session_state["returns_full"] = returns_full
    st.session_state["available_tickers"] = available_tickers
    st.session_state["benchmark_ok"] = benchmark_ok

# Load from session state
prices = st.session_state["prices"]
full_prices = st.session_state["full_prices"]
returns_stocks = st.session_state["returns_stocks"]
returns_full = st.session_state["returns_full"]
available_tickers = st.session_state["available_tickers"]
benchmark_ok = st.session_state["benchmark_ok"]
stock_cols = available_tickers  # shorthand

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Price & Returns",
    "📉 Risk & Distribution",
    "🔗 Correlation & Diversification",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Price & Return Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Price & Return Analysis")

    # ── 1. Price Chart ─────────────────────────────────────────────────────────
    st.subheader("Adjusted Closing Prices")
    selected_for_price = st.multiselect(
        "Select stocks to display:",
        options=stock_cols,
        default=stock_cols,
        key="price_multiselect",
    )
    if selected_for_price:
        fig_price = go.Figure()
        for tkr in selected_for_price:
            fig_price.add_trace(go.Scatter(
                x=prices.index, y=prices[tkr],
                mode="lines", name=tkr,
            ))
        fig_price.update_layout(
            title="Adjusted Closing Prices",
            xaxis_title="Date", yaxis_title="Price (USD)",
            hovermode="x unified", legend_title="Ticker",
            template="plotly_dark",
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("Select at least one stock to display the price chart.")

    # ── 2. Summary Statistics ──────────────────────────────────────────────────
    st.subheader("Summary Statistics")
    stats_df = summary_statistics(returns_full)
    fmt = {
        "Ann. Return": "{:.2%}",
        "Ann. Volatility": "{:.2%}",
        "Skewness": "{:.4f}",
        "Kurtosis": "{:.4f}",
        "Min Daily Return": "{:.2%}",
        "Max Daily Return": "{:.2%}",
    }
    display_stats = stats_df.copy()
    for col, f in fmt.items():
        display_stats[col] = display_stats[col].map(lambda x, f=f: f.format(x))
    display_stats.index = display_stats.index.map(
        lambda x: BENCHMARK_LABEL if x == BENCHMARK else x
    )
    st.dataframe(display_stats, use_container_width=True)

    # ── 3. Cumulative Wealth Index ─────────────────────────────────────────────
    st.subheader("Cumulative Wealth Index ($10,000 Initial Investment)")
    eq_wt_ret = portfolio_equal_weight(returns_stocks, stock_cols)
    wealth_all = wealth_index(returns_full)
    wealth_eq = (1 + eq_wt_ret).cumprod() * 10_000

    fig_wealth = go.Figure()
    for col in wealth_all.columns:
        label = BENCHMARK_LABEL if col == BENCHMARK_LABEL else col
        fig_wealth.add_trace(go.Scatter(
            x=wealth_all.index, y=wealth_all[col],
            mode="lines", name=label,
        ))
    fig_wealth.add_trace(go.Scatter(
        x=wealth_eq.index, y=wealth_eq,
        mode="lines", name="Equal-Weight Portfolio",
        line=dict(dash="dash", width=2),
    ))
    fig_wealth.update_layout(
        title="Growth of $10,000 Investment",
        xaxis_title="Date", yaxis_title="Portfolio Value (USD)",
        hovermode="x unified", legend_title="Series",
        template="plotly_dark",
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Risk & Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Risk & Distribution Analysis")

    # ── 1. Rolling Volatility ──────────────────────────────────────────────────
    st.subheader("Rolling Annualized Volatility")
    roll_window = st.select_slider(
        "Rolling window (days):",
        options=[20, 30, 60, 90, 120],
        value=60,
        key="roll_vol_window",
    )
    roll_vol = returns_stocks.rolling(roll_window).std() * np.sqrt(TRADING_DAYS)
    fig_rv = go.Figure()
    for tkr in stock_cols:
        fig_rv.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[tkr],
            mode="lines", name=tkr,
        ))
    fig_rv.update_layout(
        title=f"Rolling {roll_window}-Day Annualized Volatility",
        xaxis_title="Date", yaxis_title="Annualized Volatility",
        hovermode="x unified", yaxis_tickformat=".0%",
        legend_title="Ticker", template="plotly_dark",
    )
    st.plotly_chart(fig_rv, use_container_width=True)

    # ── 2–4. Distribution Plot, Q-Q, Jarque-Bera ──────────────────────────────
    st.subheader("Return Distribution Analysis")
    dist_stock = st.selectbox("Select stock:", options=stock_cols, key="dist_stock")
    r = returns_stocks[dist_stock].dropna()

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(r)
    normality_msg = (
        f"🔴 **Rejects normality** (p < 0.05) — JB stat: {jb_stat:.2f}, p-value: {jb_p:.4f}"
        if jb_p < 0.05
        else f"🟢 **Fails to reject normality** (p ≥ 0.05) — JB stat: {jb_stat:.2f}, p-value: {jb_p:.4f}"
    )
    st.markdown(normality_msg)

    dist_view = st.radio("View:", ["Histogram + Normal Fit", "Q-Q Plot"], horizontal=True, key="dist_view")

    if dist_view == "Histogram + Normal Fit":
        mu, sigma = stats.norm.fit(r)
        x_range = np.linspace(r.min(), r.max(), 300)
        pdf_vals = stats.norm.pdf(x_range, mu, sigma)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=r, nbinsx=80, histnorm="probability density",
            name="Daily Returns", opacity=0.65,
            marker_color="#0d6efd",
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=pdf_vals,
            mode="lines", name="Normal Fit",
            line=dict(color="#ff6b6b", width=2),
        ))
        fig_hist.update_layout(
            title=f"{dist_stock} — Daily Return Distribution",
            xaxis_title="Daily Return", yaxis_title="Density",
            xaxis_tickformat=".1%", legend_title="Series",
            template="plotly_dark",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:  # Q-Q Plot
        (osm, osr), (slope, intercept, _) = stats.probplot(r)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            name="Sample Quantiles",
            marker=dict(color="#0d6efd", size=4, opacity=0.7),
        ))
        # Reference line
        x_line = np.array([min(osm), max(osm)])
        fig_qq.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", name="Normal Reference",
            line=dict(color="#ff6b6b", width=2),
        ))
        fig_qq.update_layout(
            title=f"{dist_stock} — Q-Q Plot vs Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_dark",
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption("Points lying on the red line indicate normality. Deviations at the tails indicate fat tails.")

    # ── 5. Box Plot ────────────────────────────────────────────────────────────
    st.subheader("Daily Return Distributions — Box Plot")
    fig_box = go.Figure()
    for tkr in stock_cols:
        fig_box.add_trace(go.Box(
            y=returns_stocks[tkr].dropna(),
            name=tkr, boxmean="sd",
        ))
    fig_box.update_layout(
        title="Daily Return Distribution by Stock",
        xaxis_title="Ticker", yaxis_title="Daily Return",
        yaxis_tickformat=".1%", template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Correlation & Diversification
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Correlation & Diversification")

    # ── 1. Correlation Heatmap ─────────────────────────────────────────────────
    st.subheader("Correlation Heatmap")
    corr = returns_stocks.corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Pairwise Correlation of Daily Returns",
    )
    fig_heat.update_layout(template="plotly_dark", coloraxis_colorbar_title="Correlation")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── 2. Scatter Plot ────────────────────────────────────────────────────────
    st.subheader("Return Scatter Plot")
    col_a, col_b = st.columns(2)
    with col_a:
        scatter_a = st.selectbox("Stock A:", options=stock_cols, index=0, key="scatter_a")
    with col_b:
        remaining = [t for t in stock_cols if t != scatter_a]
        scatter_b = st.selectbox("Stock B:", options=remaining, index=0, key="scatter_b")

    if scatter_a != scatter_b:
        x_vals = returns_stocks[scatter_a].dropna()
        y_vals = returns_stocks[scatter_b].reindex(x_vals.index).dropna()
        x_vals = x_vals.reindex(y_vals.index)
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(opacity=0.5, size=4),
            name="Daily Returns",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trendline",
            line=dict(color="#ff6b6b", width=2),
        ))
        fig_scatter.update_layout(
            title=f"{scatter_a} vs {scatter_b} — Daily Returns",
            xaxis_title=f"{scatter_a} Daily Return",
            yaxis_title=f"{scatter_b} Daily Return",
            xaxis_tickformat=".1%", yaxis_tickformat=".1%",
            template="plotly_dark",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── 3. Rolling Correlation ─────────────────────────────────────────────────
    st.subheader("Rolling Correlation")
    rc_col1, rc_col2 = st.columns(2)
    with rc_col1:
        rc_a = st.selectbox("Stock A:", options=stock_cols, index=0, key="rc_a")
    with rc_col2:
        remaining_rc = [t for t in stock_cols if t != rc_a]
        rc_b = st.selectbox("Stock B:", options=remaining_rc, index=0, key="rc_b")
    rc_window = st.select_slider(
        "Rolling window (days):",
        options=[20, 30, 60, 90, 120],
        value=60,
        key="rc_window",
    )
    if rc_a != rc_b:
        roll_corr = returns_stocks[rc_a].rolling(rc_window).corr(returns_stocks[rc_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            mode="lines", name=f"{rc_a}/{rc_b}",
            line=dict(color="#0d6efd"),
        ))
        fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rc.update_layout(
            title=f"Rolling {rc_window}-Day Correlation: {rc_a} vs {rc_b}",
            xaxis_title="Date", yaxis_title="Correlation",
            yaxis_range=[-1.1, 1.1], template="plotly_dark",
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    # ── 4. Two-Asset Portfolio Explorer ───────────────────────────────────────
    st.subheader("Two-Asset Portfolio Explorer")
    st.info(
        "**Diversification in action.** Combining two stocks can produce a portfolio with *lower* "
        "volatility than either stock individually. This effect is strongest when the correlation between "
        "the two stocks is low (or negative) — because their price movements partially cancel each other out, "
        "smoothing overall portfolio swings. The curve below shows how portfolio volatility changes as you "
        "shift weight between the two stocks — notice if it dips below the individual stocks' volatilities."
    )

    pe_col1, pe_col2 = st.columns(2)
    with pe_col1:
        pe_a = st.selectbox("Stock A:", options=stock_cols, index=0, key="pe_a")
    with pe_col2:
        remaining_pe = [t for t in stock_cols if t != pe_a]
        pe_b = st.selectbox("Stock B:", options=remaining_pe, index=0, key="pe_b")

    weight_a = st.slider(
        f"Weight on {pe_a} (%):", min_value=0, max_value=100, value=50, step=5, key="pe_weight"
    )
    w = weight_a / 100.0

    ret_a = returns_stocks[pe_a].dropna()
    ret_b = returns_stocks[pe_b].dropna()
    aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
    ret_a_al = aligned.iloc[:, 0]
    ret_b_al = aligned.iloc[:, 1]

    curr_ret, curr_vol = two_asset_portfolio(w, ret_a_al, ret_b_al)

    # Display current metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Weight on {pe_a}", f"{weight_a}%")
    m2.metric("Portfolio Ann. Return", f"{curr_ret:.2%}")
    m3.metric("Portfolio Ann. Volatility", f"{curr_vol:.2%}")

    # Sweep across all weights
    weights = np.linspace(0, 1, 101)
    vols = [two_asset_portfolio(ww, ret_a_al, ret_b_al)[1] for ww in weights]
    rets = [two_asset_portfolio(ww, ret_a_al, ret_b_al)[0] for ww in weights]

    vol_a_ann = ret_a_al.std() * np.sqrt(TRADING_DAYS)
    vol_b_ann = ret_b_al.std() * np.sqrt(TRADING_DAYS)

    fig_pe = go.Figure()
    fig_pe.add_trace(go.Scatter(
        x=weights * 100, y=vols,
        mode="lines", name="Portfolio Volatility",
        line=dict(color="#0d6efd", width=2),
    ))
    # Mark current slider position
    fig_pe.add_trace(go.Scatter(
        x=[weight_a], y=[curr_vol],
        mode="markers", name="Current Weight",
        marker=dict(color="#ffd700", size=12, symbol="star"),
    ))
    # Reference lines for individual stocks
    fig_pe.add_hline(y=vol_a_ann, line_dash="dot", line_color="#ff6b6b",
                     annotation_text=f"{pe_a} Vol", annotation_position="top left")
    fig_pe.add_hline(y=vol_b_ann, line_dash="dot", line_color="#00cc96",
                     annotation_text=f"{pe_b} Vol", annotation_position="top right")
    fig_pe.update_layout(
        title=f"Portfolio Volatility vs Weight on {pe_a}",
        xaxis_title=f"Weight on {pe_a} (%)",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".1%",
        legend_title="Series",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pe, use_container_width=True)

    # Efficient frontier-style return vs risk
    with st.expander("📐 Return vs. Risk Frontier"):
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=vols, y=rets,
            mode="lines", name="Portfolio",
            line=dict(color="#0d6efd", width=2),
        ))
        fig_ef.add_trace(go.Scatter(
            x=[curr_vol], y=[curr_ret],
            mode="markers", name="Current",
            marker=dict(color="#ffd700", size=12, symbol="star"),
        ))
        fig_ef.update_layout(
            title="Return vs. Risk Frontier",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            xaxis_tickformat=".1%", yaxis_tickformat=".1%",
            template="plotly_dark",
        )
        st.plotly_chart(fig_ef, use_container_width=True)
