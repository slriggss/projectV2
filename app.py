import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
STOCK_COLORS = [
    "#8b1a32",   # rich burgundy
    "#5c2d6e",   # deep purple
    "#0a5c5c",   # deep teal
    "#1a5c35",   # forest green
    "#7a4a34",   # warm brown (5th stock fallback)
]
BENCHMARK_COLOR = "#5a5a5a"   # grey — S&P 500
EQ_WEIGHT_COLOR = "#888888"   # lighter grey — equal-weight portfolio
TEAL_ACCENT     = "#0a6e6e"   # active tab / analyze button
STAR_COLOR      = "#c4a35a"   # gold star marker

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&display=swap');

  /* Global font */
  html, body, [class*="css"], .stMarkdown, .stText,
  .stDataFrame, .stMetric, label, p, li, td, th,
  .stSelectbox, .stMultiSelect, .stSlider {{
      font-family: 'EB Garamond', Georgia, serif !important;
      color: #1a1a1a;
  }}

  /* Light background */
  .stApp, [data-testid="stAppViewContainer"] {{
      background-color: #f7f5f0 !important;
  }}
  [data-testid="stHeader"] {{
      background-color: #f7f5f0 !important;
  }}

  /* Headings */
  h1 {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 2.4rem !important;
      font-weight: 700 !important;
      letter-spacing: 0.02em;
      color: #111111 !important;
  }}
  h2 {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 1.7rem !important;
      font-weight: 700 !important;
      color: #1a1a1a !important;
  }}
  h3 {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 1.3rem !important;
      font-weight: 600 !important;
      color: #333333 !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 4px;
      border-bottom: 2px solid #d0ccc4;
      background-color: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 1.1rem !important;
      font-weight: 700 !important;
      background-color: #ede9e2;
      border-radius: 4px 4px 0 0;
      padding: 10px 26px;
      color: #555555;
      border: none;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {TEAL_ACCENT} !important;
      color: #ffffff !important;
      border-bottom: 3px solid #111111 !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
      background-color: #0d8080 !important;
      color: #ffffff !important;
  }}
  /* Remove default red underline */
  .stTabs [data-baseweb="tab-highlight"] {{
      background-color: transparent !important;
      display: none !important;
  }}

  /* Analyze button */
  .stButton > button[kind="primary"] {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 1.05rem;
      font-weight: 600;
      background-color: {TEAL_ACCENT} !important;
      border: none !important;
      color: white !important;
      letter-spacing: 0.05em;
  }}
  .stButton > button[kind="primary"]:hover {{
      background-color: #0d8080 !important;
  }}

  /* Multiselect tags: dark grey bg, light grey text */
  [data-baseweb="tag"] {{
      background-color: #3a3a3a !important;
      border: none !important;
  }}
  [data-baseweb="tag"] span {{
      color: #e0e0e0 !important;
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 0.95rem !important;
  }}
  [data-baseweb="tag"] svg {{
      fill: #bbbbbb !important;
  }}

  /* Metrics */
  [data-testid="stMetricLabel"] {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 0.9rem !important;
      color: #555555 !important;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }}
  [data-testid="stMetricValue"] {{
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 1.5rem !important;
      font-weight: 700 !important;
      color: #111111 !important;
  }}

  hr {{ border-color: #d0ccc4; }}

  /* Summary statistics table: dark navy background */
  [data-testid="stDataFrame"] {{
      background-color: #1a1f2e !important;
      border-radius: 6px;
      overflow: hidden;
  }}
  [data-testid="stDataFrame"] * {{
      color: #e8e2d5 !important;
      font-family: 'EB Garamond', Georgia, serif !important;
  }}
  [data-testid="stDataFrame"] th {{
      background-color: #232840 !important;
      color: #c4a35a !important;
  }}
  [data-testid="stDataFrame"] tr:nth-child(even) td {{
      background-color: #1e2436 !important;
  }}
  [data-testid="stDataFrame"] tr:nth-child(odd) td {{
      background-color: #1a1f2e !important;
  }}

  /* Stock dropdown / selectbox / multiselect: dark navy */
  [data-testid="stSelectbox"] > div > div,
  [data-testid="stMultiSelect"] > div > div {{
      background-color: #1a1f2e !important;
      border-color: #3a3f50 !important;
      color: #e8e2d5 !important;
  }}
  [data-testid="stSelectbox"] span,
  [data-testid="stMultiSelect"] span {{
      color: #e8e2d5 !important;
  }}
  /* Dropdown option list */
  [data-baseweb="popover"] ul,
  [data-baseweb="menu"] {{
      background-color: #1a1f2e !important;
  }}
  [data-baseweb="popover"] li,
  [data-baseweb="menu"] li {{
      color: #e8e2d5 !important;
      font-family: 'EB Garamond', Georgia, serif !important;
  }}
  [data-baseweb="popover"] li:hover,
  [data-baseweb="menu"] li:hover {{
      background-color: #0a5c5c !important;
  }}

  /* Sidebar: light cream font on dark background */
  [data-testid="stSidebar"] {{
      background-color: #1a1f2e !important;
  }}
  [data-testid="stSidebar"] * {{
      color: #e8e2d5 !important;
      font-family: 'EB Garamond', Georgia, serif !important;
  }}
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label {{
      color: #e8e2d5 !important;
  }}
  [data-testid="stSidebar"] hr {{
      border-color: #3a3f50 !important;
  }}
  /* Sidebar expander */
  [data-testid="stSidebar"] [data-testid="stExpander"] {{
      border-color: #3a3f50 !important;
      background-color: #232840 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
      color: #e8e2d5 !important;
  }}
  /* Hide the raw icon string that appears before expander label text */
  [data-testid="stSidebar"] details > summary > span:first-child,
  [data-testid="stSidebar"] details > summary > div > span:first-child {{
      display: none !important;
  }}
  [data-testid="stSidebar"] details > summary {{
      list-style: none !important;
  }}
  [data-testid="stSidebar"] details > summary::-webkit-details-marker {{
      display: none !important;
  }}
  /* Hide raw material icon text bleeding into expander labels */
  [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{
      fill: #e8e2d5 !important;
  }}
  [data-testid="stSidebar"] details summary p {{
      color: #e8e2d5 !important;
  }}
  /* Hide sidebar collapse button entirely */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"],
  [data-testid="stSidebar"] button[aria-label*="collapse"],
  [data-testid="stSidebar"] button[aria-label*="Close"],
  [data-testid="stSidebar"] > div:first-child > div:first-child > button {{
      display: none !important;
  }}
  /* Hide raw material icon text on collapse button and expanders */
  [data-testid="stSidebar"] button span,
  [data-testid="stSidebar"] [data-testid="collapsedControl"] span {{
      font-size: 0 !important;
      color: transparent !important;
  }}
  /* Expander: hide the raw _arrow_right/_arrow_down text, keep label visible */
  [data-testid="stSidebar"] [data-testid="stExpander"] summary > span:first-child {{
      font-size: 0 !important;
      width: 14px !important;
      display: inline-block;
  }}
  [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{
      fill: #e8e2d5 !important;
      flex-shrink: 0;
  }}
  /* Ensure expander label text is fully visible and cream colored */
  [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
  [data-testid="stSidebar"] [data-testid="stExpander"] summary span:not(:first-child) {{
      color: #e8e2d5 !important;
      font-family: 'EB Garamond', Georgia, serif !important;
      font-size: 1rem !important;
      font-weight: 600 !important;
  }}

  /* Sliders: black track and thumb */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
      background-color: #111111 !important;
      border-color: #111111 !important;
  }}
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {{
      color: #111111 !important;
  }}
  div[data-baseweb="slider"] > div > div > div:nth-child(1) {{
      background-color: #111111 !important;
  }}
  div[data-baseweb="slider"] > div > div {{
      background: linear-gradient(to right, #111111 var(--progress, 50%), #d0ccc4 var(--progress, 50%)) !important;
  }}
  [data-testid="stSlider"] div[role="slider"] {{
      background-color: #111111 !important;
      border: 2px solid #111111 !important;
  }}

  /* Select slider (the step ones) */
  .stSelectSlider [data-baseweb="slider"] [role="slider"] {{
      background-color: #111111 !important;
      border-color: #111111 !important;
  }}

  /* Info box: light grey instead of blue */
  [data-testid="stAlert"][kind="info"],
  div[data-baseweb="notification"] {{
      background-color: #eeebe5 !important;
      border-left-color: #888888 !important;
      color: #1a1a1a !important;
  }}
  [data-testid="stAlert"] p,
  [data-testid="stAlert"] div {{
      color: #1a1a1a !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS    = 252
BENCHMARK       = "^GSPC"
BENCHMARK_LABEL = "S&P 500"

# ── Helpers ───────────────────────────────────────────────────────────────────
def stock_color(tickers: list, tkr: str) -> str:
    try:
        return STOCK_COLORS[tickers.index(tkr) % len(STOCK_COLORS)]
    except ValueError:
        return STOCK_COLORS[0]

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="EB Garamond, Georgia, serif", size=13, color="#1a1a1a"),
    title_font=dict(family="EB Garamond, Georgia, serif", size=16, color="#111111"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(247,245,240,0.6)",
    colorway=STOCK_COLORS,
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0ccc4",
        borderwidth=1,
        font=dict(family="EB Garamond, Georgia, serif", size=12, color="#1a1a1a"),
    ),
    xaxis=dict(
        gridcolor="#e8e4de", linecolor="#c0bbb4", tickcolor="#333333",
        tickfont=dict(color="#333333", family="EB Garamond, Georgia, serif", size=12),
        title_font=dict(color="#111111", family="EB Garamond, Georgia, serif", size=13),
        title_standoff=12,
    ),
    yaxis=dict(
        gridcolor="#e8e4de", linecolor="#c0bbb4", tickcolor="#333333",
        tickfont=dict(color="#333333", family="EB Garamond, Georgia, serif", size=12),
        title_font=dict(color="#111111", family="EB Garamond, Georgia, serif", size=13),
        title_standoff=12,
    ),
)

def apply_chart_layout(fig, **extra):
    """Apply CHART_LAYOUT and force all axis/annotation/title text dark."""
    layout = dict(CHART_LAYOUT)
    layout.update(extra)
    # Force title font dark regardless of template
    if "title" in layout and isinstance(layout["title"], str):
        layout["title"] = dict(
            text=layout.pop("title"),
            font=dict(color="#111111", family="EB Garamond, Georgia, serif", size=16),
        )
    layout["title_font_color"] = "#111111"
    layout["font_color"] = "#1a1a1a"
    fig.update_layout(**layout)
    fig.update_xaxes(
        tickfont=dict(color="#333333", family="EB Garamond, Georgia, serif", size=12),
        title_font=dict(color="#111111", family="EB Garamond, Georgia, serif", size=13),
        gridcolor="#e8e4de", linecolor="#c0bbb4",
    )
    fig.update_yaxes(
        tickfont=dict(color="#333333", family="EB Garamond, Georgia, serif", size=12),
        title_font=dict(color="#111111", family="EB Garamond, Georgia, serif", size=13),
        gridcolor="#e8e4de", linecolor="#c0bbb4",
    )
    fig.update_annotations(font=dict(color="#111111", family="EB Garamond, Georgia, serif"))
    fig.update_traces(hoverlabel=dict(font=dict(family="EB Garamond, Georgia, serif")))
    return fig

# ── Cached functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_prices(tickers: tuple, start: str, end: str) -> tuple:
    all_tickers = list(tickers) + [BENCHMARK]
    errors, frames = [], {}
    for tkr in all_tickers:
        try:
            raw = yf.download(tkr, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 2:
                errors.append(tkr); continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            frames[tkr] = close
        except Exception:
            errors.append(tkr)
    if not frames:
        return pd.DataFrame(), errors
    return pd.DataFrame(frames), errors


@st.cache_data(ttl=3600)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


@st.cache_data(ttl=3600)
def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    stats_dict = {}
    for col in returns.columns:
        r = returns[col].dropna()
        stats_dict[col] = {
            "Ann. Return":      r.mean() * TRADING_DAYS,
            "Ann. Volatility":  r.std()  * np.sqrt(TRADING_DAYS),
            "Skewness":         r.skew(),
            "Kurtosis":         r.kurtosis(),
            "Min Daily Return": r.min(),
            "Max Daily Return": r.max(),
        }
    return pd.DataFrame(stats_dict).T


def wealth_index(returns: pd.DataFrame, initial: float = 10_000) -> pd.DataFrame:
    return (1 + returns).cumprod() * initial


def portfolio_equal_weight(returns: pd.DataFrame, stock_cols: list) -> pd.Series:
    return returns[stock_cols].mean(axis=1)


def two_asset_portfolio(w: float, ret_a: pd.Series, ret_b: pd.Series):
    port_ret = w * ret_a + (1 - w) * ret_b
    ann_ret  = port_ret.mean() * TRADING_DAYS
    cov      = np.cov(ret_a, ret_b) * TRADING_DAYS
    ann_vol  = np.sqrt(w**2*cov[0,0] + (1-w)**2*cov[1,1] + 2*w*(1-w)*cov[0,1])
    return ann_ret, ann_vol

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Stock Analyzer")
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
            value=date.today() - timedelta(days=2*365),
            max_value=date.today() - timedelta(days=366),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            max_value=date.today(),
        )

    run_btn = st.button("Analyze", use_container_width=True, type="primary")


# ── Input validation ──────────────────────────────────────────────────────────
tickers_raw = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
input_valid = True
error_msgs  = []

if len(tickers_raw) < 2:
    error_msgs.append("Please enter at least 2 ticker symbols.")
    input_valid = False
if len(tickers_raw) > 5:
    error_msgs.append("Please enter no more than 5 ticker symbols.")
    input_valid = False
if (end_date - start_date).days < 365:
    error_msgs.append("Date range must be at least 1 year.")
    input_valid = False

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Stock Comparison & Analysis")

# Prevent page scroll-to-top on widget interaction
st.markdown("""
<script>
// Keep scroll position on rerun
if (window.scrollY > 0) {
    sessionStorage.setItem('scrollPos', window.scrollY);
}
window.addEventListener('load', function() {
    const pos = sessionStorage.getItem('scrollPos');
    if (pos) window.scrollTo(0, parseInt(pos));
});
</script>
""", unsafe_allow_html=True)

for msg in error_msgs:
    st.error(msg)

if not run_btn and "prices" not in st.session_state:
    st.info("Enter your tickers and date range in the sidebar, then click Analyze.")
    st.stop()

if run_btn:
    if not input_valid:
        st.stop()
    with st.spinner("Downloading market data…"):
        prices_raw, dl_errors = download_prices(
            tuple(tickers_raw), str(start_date), str(end_date)
        )

    user_errors = [t for t in dl_errors if t != BENCHMARK]
    if user_errors:
        st.warning(f"Could not download data for: **{', '.join(user_errors)}**. They have been excluded.")

    benchmark_ok      = BENCHMARK in prices_raw.columns if not prices_raw.empty else False
    available_tickers = [t for t in tickers_raw if t in (prices_raw.columns if not prices_raw.empty else [])]

    if len(available_tickers) < 2:
        st.error("Fewer than 2 tickers returned valid data. Please try different symbols.")
        st.stop()

    stock_prices = prices_raw[available_tickers].copy()
    missing_pct  = stock_prices.isnull().mean()
    dropped      = missing_pct[missing_pct > 0.05].index.tolist()
    if dropped:
        st.warning(f"Dropped due to >5% missing data: **{', '.join(dropped)}**")
        stock_prices      = stock_prices.drop(columns=dropped)
        available_tickers = [t for t in available_tickers if t not in dropped]

    if len(available_tickers) < 2:
        st.error("Not enough valid tickers after data cleaning.")
        st.stop()

    stock_prices = stock_prices.dropna(how="any")
    if stock_prices.empty:
        st.error("No overlapping date range found among selected tickers.")
        st.stop()

    overlap_start = stock_prices.index.min().date()
    overlap_end   = stock_prices.index.max().date()
    if overlap_start != start_date or overlap_end != end_date:
        st.info(f"Data truncated to overlapping range: **{overlap_start}** → **{overlap_end}**")

    bench_prices = (
        prices_raw[[BENCHMARK]].reindex(stock_prices.index).dropna()
        if benchmark_ok else None
    )
    full_prices = stock_prices.copy()
    if bench_prices is not None:
        full_prices[BENCHMARK_LABEL] = bench_prices[BENCHMARK]

    returns_stocks = compute_returns(stock_prices)
    returns_full   = compute_returns(full_prices)

    st.session_state.update({
        "prices":            stock_prices,
        "full_prices":       full_prices,
        "returns_stocks":    returns_stocks,
        "returns_full":      returns_full,
        "available_tickers": available_tickers,
        "benchmark_ok":      benchmark_ok,
    })

prices            = st.session_state["prices"]
full_prices       = st.session_state["full_prices"]
returns_stocks    = st.session_state["returns_stocks"]
returns_full      = st.session_state["returns_full"]
available_tickers = st.session_state["available_tickers"]
benchmark_ok      = st.session_state["benchmark_ok"]
stock_cols        = available_tickers

# ── Tabs ──────────────────────────────────────────────────────────────────────
# Preserve active tab across reruns
_tab_labels = ["Price & Returns", "Risk & Distribution", "Correlation & Diversification", "Additional Information"]
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0

tab1, tab2, tab3, tab4 = st.tabs(_tab_labels)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Returns
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Price & Return Analysis")

    st.subheader("Adjusted Closing Prices")
    selected_for_price = st.multiselect(
        "Select stocks to display:",
        options=stock_cols, default=stock_cols, key="price_multiselect",
    )
    if selected_for_price:
        fig_price = go.Figure()
        for tkr in selected_for_price:
            fig_price.add_trace(go.Scatter(
                x=prices.index, y=prices[tkr],
                mode="lines", name=tkr,
                line=dict(color=stock_color(stock_cols, tkr), width=1.8),
            ))
        apply_chart_layout(fig_price,
            title="Adjusted Closing Prices",
            xaxis_title="Date", yaxis_title="Price (USD)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("Select at least one stock to display the price chart.")

    # Daily Returns Chart
    st.subheader("Daily Returns")
    selected_for_returns = st.multiselect(
        "Select stocks to display:",
        options=stock_cols + ([BENCHMARK_LABEL] if BENCHMARK_LABEL in returns_full.columns else []),
        default=stock_cols + ([BENCHMARK_LABEL] if BENCHMARK_LABEL in returns_full.columns else []),
        key="returns_multiselect",
    )
    if selected_for_returns:
        fig_daily = go.Figure()
        for tkr in selected_for_returns:
            if tkr == BENCHMARK_LABEL:
                color = BENCHMARK_COLOR
                r_series = returns_full[BENCHMARK_LABEL]
            else:
                color = stock_color(stock_cols, tkr)
                r_series = returns_stocks[tkr]
            fig_daily.add_trace(go.Scatter(
                x=r_series.index, y=r_series,
                mode="lines", name=tkr,
                line=dict(color=color, width=1.2),
                opacity=0.85,
            ))
        apply_chart_layout(fig_daily,
            title="Daily Returns",
            xaxis_title="Date", yaxis_title="Daily Return",
            hovermode="x unified",
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.warning("Select at least one stock to display daily returns.")

    st.subheader("Summary Statistics")
    stats_df = summary_statistics(returns_full)
    fmt = {
        "Ann. Return":      "{:.2%}",
        "Ann. Volatility":  "{:.2%}",
        "Skewness":         "{:.4f}",
        "Kurtosis":         "{:.4f}",
        "Min Daily Return": "{:.2%}",
        "Max Daily Return": "{:.2%}",
    }
    display_stats = stats_df.copy()
    for col, f in fmt.items():
        display_stats[col] = display_stats[col].map(lambda x, f=f: f.format(x))
    display_stats.index = display_stats.index.map(
        lambda x: BENCHMARK_LABEL if x == BENCHMARK_LABEL else x
    )
    st.dataframe(display_stats, use_container_width=True)

    st.subheader("Cumulative Wealth Index  —  $10,000 Initial Investment")
    eq_wt_ret  = portfolio_equal_weight(returns_stocks, stock_cols)
    wealth_all = wealth_index(returns_full)
    wealth_eq  = (1 + eq_wt_ret).cumprod() * 10_000

    fig_wealth = go.Figure()
    for col in stock_cols:
        fig_wealth.add_trace(go.Scatter(
            x=wealth_all.index, y=wealth_all[col],
            mode="lines", name=col,
            line=dict(color=stock_color(stock_cols, col), width=1.8),
        ))
    if BENCHMARK_LABEL in wealth_all.columns:
        fig_wealth.add_trace(go.Scatter(
            x=wealth_all.index, y=wealth_all[BENCHMARK_LABEL],
            mode="lines", name=BENCHMARK_LABEL,
            line=dict(color=BENCHMARK_COLOR, width=1.8, dash="dot"),
        ))
    fig_wealth.add_trace(go.Scatter(
        x=wealth_eq.index, y=wealth_eq,
        mode="lines", name="Equal-Weight Portfolio",
        line=dict(color=EQ_WEIGHT_COLOR, width=1.8, dash="dash"),
    ))
    apply_chart_layout(fig_wealth,
        title="Growth of $10,000 Investment",
        xaxis_title="Date", yaxis_title="Portfolio Value (USD)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk & Distribution
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Risk & Distribution Analysis")

    st.subheader("Rolling Annualized Volatility")
    roll_window = st.select_slider(
        "Rolling window (days):",
        options=[20, 30, 60, 90, 120], value=60, key="roll_vol_window",
    )
    roll_vol = returns_stocks.rolling(roll_window).std() * np.sqrt(TRADING_DAYS)
    fig_rv = go.Figure()
    for tkr in stock_cols:
        fig_rv.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[tkr],
            mode="lines", name=tkr,
            line=dict(color=stock_color(stock_cols, tkr), width=1.8),
        ))
    apply_chart_layout(fig_rv,
        title=f"Rolling {roll_window}-Day Annualized Volatility",
        xaxis_title="Date", yaxis_title="Annualized Volatility",
        hovermode="x unified", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_rv, use_container_width=True)

    st.subheader("Return Distribution Analysis")
    dist_stock = st.selectbox("Select stock:", options=stock_cols, key="dist_stock")
    r  = returns_stocks[dist_stock].dropna()
    sc = stock_color(stock_cols, dist_stock)

    jb_stat, jb_p = stats.jarque_bera(r)
    normality_msg = (
        f"**Rejects normality** (p < 0.05)  —  JB stat: {jb_stat:.2f},  p-value: {jb_p:.4f}"
        if jb_p < 0.05
        else f"**Fails to reject normality** (p ≥ 0.05)  —  JB stat: {jb_stat:.2f},  p-value: {jb_p:.4f}"
    )
    st.markdown(normality_msg)

    dist_view = st.radio(
        "View:", ["Histogram + Normal Fit", "Q-Q Plot"],
        horizontal=True, key="dist_view",
        on_change=None,
    )

    if dist_view == "Histogram + Normal Fit":
        mu, sigma = stats.norm.fit(r)
        x_range   = np.linspace(r.min(), r.max(), 300)
        pdf_vals  = stats.norm.pdf(x_range, mu, sigma)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=r, nbinsx=80, histnorm="probability density",
            name="Daily Returns", opacity=0.7, marker_color=sc,
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=pdf_vals,
            mode="lines", name="Normal Fit",
            line=dict(color=EQ_WEIGHT_COLOR, width=2),
        ))
        apply_chart_layout(fig_hist,
            title=f"{dist_stock} — Daily Return Distribution",
            xaxis_title="Daily Return", yaxis_title="Density",
            xaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        (osm, osr), (slope, intercept, _) = stats.probplot(r)
        x_line = np.array([min(osm), max(osm)])

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            name="Sample Quantiles",
            marker=dict(color=sc, size=4, opacity=0.7),
        ))
        fig_qq.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", name="Normal Reference",
            line=dict(color=EQ_WEIGHT_COLOR, width=2),
        ))
        apply_chart_layout(fig_qq,
            title=f"{dist_stock} — Q-Q Plot vs Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption("Points lying on the reference line indicate normality. Tail deviations indicate fat tails.")

    st.subheader("Daily Return Distributions — Box Plot")
    fig_box = go.Figure()
    for tkr in stock_cols:
        fig_box.add_trace(go.Box(
            y=returns_stocks[tkr].dropna(),
            name=tkr, boxmean="sd",
            marker_color=stock_color(stock_cols, tkr),
            line_color=stock_color(stock_cols, tkr),
        ))
    apply_chart_layout(fig_box,
        title="Daily Return Distribution by Stock",
        xaxis_title="Ticker", yaxis_title="Daily Return",
        yaxis_tickformat=".1%", showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Correlation & Diversification
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Correlation & Diversification")

    st.subheader("Correlation Heatmap")
    corr = returns_stocks.corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=[
            [0.0, "#0a5c5c"],
            [0.5, "#f7f5f0"],
            [1.0, "#8b1a32"],
        ],
        zmin=-1, zmax=1, aspect="auto",
        title="Pairwise Correlation of Daily Returns",
    )
    apply_chart_layout(fig_heat, coloraxis_colorbar_title="Correlation")
    st.plotly_chart(fig_heat, use_container_width=True)

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
        m, b_coef = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(color=stock_color(stock_cols, scatter_a), opacity=0.45, size=4),
            name="Daily Returns",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b_coef,
            mode="lines", name="Trendline",
            line=dict(color=EQ_WEIGHT_COLOR, width=2),
        ))
        apply_chart_layout(fig_scatter,
            title=f"{scatter_a} vs {scatter_b} — Daily Returns",
            xaxis_title=f"{scatter_a} Daily Return",
            yaxis_title=f"{scatter_b} Daily Return",
            xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Rolling Correlation")
    rc_col1, rc_col2 = st.columns(2)
    with rc_col1:
        rc_a = st.selectbox("Stock A:", options=stock_cols, index=0, key="rc_a")
    with rc_col2:
        remaining_rc = [t for t in stock_cols if t != rc_a]
        rc_b = st.selectbox("Stock B:", options=remaining_rc, index=0, key="rc_b")
    rc_window = st.select_slider(
        "Rolling window (days):",
        options=[20, 30, 60, 90, 120], value=60, key="rc_window",
    )
    if rc_a != rc_b:
        roll_corr = returns_stocks[rc_a].rolling(rc_window).corr(returns_stocks[rc_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            mode="lines", name=f"{rc_a} / {rc_b}",
            line=dict(color=TEAL_ACCENT, width=1.8),
        ))
        fig_rc.add_hline(y=0, line_dash="dash", line_color="#3a3f50")
        apply_chart_layout(fig_rc,
            title=f"Rolling {rc_window}-Day Correlation: {rc_a} vs {rc_b}",
            xaxis_title="Date", yaxis_title="Correlation",
            yaxis_range=[-1.1, 1.1],
        )
    st.plotly_chart(fig_rc, use_container_width=True)

    st.subheader("Two-Asset Portfolio Explorer")
    st.info(
        "**Diversification in action.** Combining two stocks can produce a portfolio with lower "
        "volatility than either stock individually. This effect is strongest when correlation is "
        "low or negative — price movements partially cancel each other out, smoothing overall "
        "portfolio swings. The curve below shows how portfolio volatility shifts with weight — "
        "notice if it dips below either stock's individual volatility."
    )

    pe_col1, pe_col2 = st.columns(2)
    with pe_col1:
        pe_a = st.selectbox("Stock A:", options=stock_cols, index=0, key="pe_a")
    with pe_col2:
        remaining_pe = [t for t in stock_cols if t != pe_a]
        pe_b = st.selectbox("Stock B:", options=remaining_pe, index=0, key="pe_b")

    weight_a = st.slider(
        f"Weight on {pe_a} (%):", min_value=0, max_value=100, value=50, step=5, key="pe_weight",
    )
    w = weight_a / 100.0

    ret_a   = returns_stocks[pe_a].dropna()
    ret_b   = returns_stocks[pe_b].dropna()
    aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
    ret_a_al, ret_b_al = aligned.iloc[:, 0], aligned.iloc[:, 1]

    curr_ret, curr_vol = two_asset_portfolio(w, ret_a_al, ret_b_al)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Weight on {pe_a}",        f"{weight_a}%")
    m2.metric("Portfolio Ann. Return",     f"{curr_ret:.2%}")
    m3.metric("Portfolio Ann. Volatility", f"{curr_vol:.2%}")

    weights   = np.linspace(0, 1, 101)
    vols      = [two_asset_portfolio(ww, ret_a_al, ret_b_al)[1] for ww in weights]
    rets      = [two_asset_portfolio(ww, ret_a_al, ret_b_al)[0] for ww in weights]
    vol_a_ann = ret_a_al.std() * np.sqrt(TRADING_DAYS)
    vol_b_ann = ret_b_al.std() * np.sqrt(TRADING_DAYS)
    color_a   = stock_color(stock_cols, pe_a)
    color_b   = stock_color(stock_cols, pe_b)

    fig_pe = go.Figure()
    fig_pe.add_trace(go.Scatter(
        x=weights * 100, y=vols,
        mode="lines", name="Portfolio Volatility",
        line=dict(color=TEAL_ACCENT, width=2),
    ))
    fig_pe.add_trace(go.Scatter(
        x=[weight_a], y=[curr_vol],
        mode="markers", name="Current Weight",
        marker=dict(color=STAR_COLOR, size=12, symbol="star"),
    ))
    fig_pe.add_hline(
        y=vol_a_ann, line_dash="dot", line_color=color_a,
        annotation_text=pe_a, annotation_position="top left",
        annotation_font_color=color_a,
    )
    fig_pe.add_hline(
        y=vol_b_ann, line_dash="dot", line_color=color_b,
        annotation_text=pe_b, annotation_position="top right",
        annotation_font_color=color_b,
    )
    apply_chart_layout(fig_pe,
        title=f"Portfolio Volatility vs Weight on {pe_a}",
        xaxis_title=f"Weight on {pe_a} (%)",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".1%",
    )
    st.plotly_chart(fig_pe, use_container_width=True)

    with st.expander("Return vs. Risk Frontier"):
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=vols, y=rets, mode="lines", name="Portfolio",
            line=dict(color=TEAL_ACCENT, width=2),
        ))
        fig_ef.add_trace(go.Scatter(
            x=[curr_vol], y=[curr_ret], mode="markers", name="Current",
            marker=dict(color=STAR_COLOR, size=12, symbol="star"),
        ))
        apply_chart_layout(fig_ef,
            title="Return vs. Risk Frontier",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        )
    st.plotly_chart(fig_ef, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Additional Information
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Additional Information")

    st.subheader("About")
    st.markdown("""
Compare and analyze 2–5 stocks using historical price data. This application was designed
to bring financial data analytics concepts to life through an interactive interface.
Features include:
- **Price & Return Analysis** — adjusted closing prices, daily returns, cumulative wealth index, and summary statistics
- **Risk & Distribution Analysis** — rolling volatility, return histograms, Q-Q plots, normality tests, and box plots
- **Correlation & Diversification** — pairwise correlation heatmap, scatter plots, rolling correlation, and a two-asset portfolio explorer
    """)

    st.markdown("---")

    st.subheader("Methodology")
    st.markdown("""
**Key Assumptions**

- **Returns:** Simple (arithmetic) returns computed as the percentage change in price from one day to the next — `df.pct_change()`
- **Annualization:** 252 trading days per year
    - Annualized Return = mean daily return × 252
    - Annualized Volatility = daily standard deviation × √252
- **Cumulative wealth index:** `(1 + r).cumprod() × $10,000`
- **Equal-weight portfolio:** Average of daily simple returns across all selected stocks
- **Two-asset portfolio variance:** σ²p = w²σ²A + (1−w)²σ²B + 2w(1−w)σAB
- **Benchmark:** S&P 500 (`^GSPC`) — used for comparison only, not included in portfolio calculations
- **Normality testing:** Jarque-Bera test; p-value < 0.05 indicates significant non-normality
    """)

    st.markdown("---")

    st.subheader("Sources")
    st.markdown("""
This stock analysis application was built by **Shelby Riggs** for the Financial Data Analytics I course.

- **Data Source:** Yahoo Finance via the `yfinance` Python library
- **Prices:** Adjusted closing prices, which account for dividends and stock splits
- **Coverage:** Historical data availability varies by ticker; the app aligns all selected tickers to their common overlapping date range
    """)
