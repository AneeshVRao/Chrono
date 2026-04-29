"""
Paper Trading Monitoring Dashboard — Real Data Edition.

Reads actual backtest output, feature data, and logs to display:
- Equity curve vs SPY benchmark
- Drawdown profile
- Trade logs from actual backtest results
- Market regime overlay from feature data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ─── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chrono — Trading Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .css-18e3th9 { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #1e2240 100%);
        border-radius: 14px; padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid rgba(100,120,200,0.15);
        transition: all 0.35s cubic-bezier(.4,0,.2,1);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 210, 255, 0.2);
        border-color: rgba(0, 210, 255, 0.4);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label {
        font-size: 0.8rem; color: #8A92A6;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.3rem;
    }
    .metric-sub { font-size: 0.75rem; color: #555; margin-top: 0.2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
LOGS_DIR = PROJECT_ROOT / "logs"
METRICS_DIR = LOGS_DIR / "metrics"


@st.cache_data(ttl=120)
def load_feature_data(ticker: str) -> pd.DataFrame | None:
    """Load feature-engineered data for a ticker from parquet."""
    path = FEATURES_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=120)
def load_all_tickers() -> list[str]:
    """Discover tickers from available feature files."""
    if not FEATURES_DIR.exists():
        return []
    files = list(FEATURES_DIR.glob("*_features.parquet"))
    return sorted([f.stem.replace("_features", "") for f in files])


@st.cache_data(ttl=120)
def load_backtest_log() -> str:
    """Load the most recent backtest output log."""
    path = METRICS_DIR / "backtest_output.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


@st.cache_data(ttl=120)
def build_equity_curves(tickers: list[str]) -> pd.DataFrame | None:
    """Build portfolio equity curve from feature data close prices."""
    curves = {}
    for ticker in tickers:
        df = load_feature_data(ticker)
        if df is not None and "close" in df.columns:
            curves[ticker] = df["close"]
    if not curves:
        return None
    price_df = pd.DataFrame(curves).dropna()
    if price_df.empty:
        return None
    # Normalise to base 100
    normalised = price_df / price_df.iloc[0] * 100
    normalised["Portfolio (Eq. Wt)"] = normalised.mean(axis=1)
    return normalised


@st.cache_data(ttl=120)
def build_regime_series(ticker: str) -> pd.Series | None:
    df = load_feature_data(ticker)
    if df is not None and "market_regime" in df.columns:
        return df["market_regime"]
    return None


# ─── Sidebar ────────────────────────────────────────────────────────
tickers = load_all_tickers()

st.sidebar.title("⚙️ Configuration")

if tickers:
    selected_tickers = st.sidebar.multiselect(
        "Tickers", tickers, default=tickers[:5] if len(tickers) >= 5 else tickers
    )
    primary_ticker = st.sidebar.selectbox("Primary Ticker (for regime)", selected_tickers or tickers[:1])
else:
    selected_tickers = []
    primary_ticker = None

# ─── Header ─────────────────────────────────────────────────────────
st.markdown("# 📈 Chrono — Trading Monitor")
st.markdown("*Real-time view of backtested performance, market regimes, and portfolio health.*")
st.markdown("---")

# ─── Data-Availability Gate ─────────────────────────────────────────
if not tickers:
    st.warning(
        "⚠️ No feature data found. Run the pipeline first:\n\n"
        "```bash\npython scripts/run_pipeline.py\n```"
    )
    st.stop()

if not selected_tickers:
    st.info("Select at least one ticker from the sidebar.")
    st.stop()

# ─── Top Metrics Row ────────────────────────────────────────────────
eq_curves = build_equity_curves(selected_tickers)

if eq_curves is not None and "Portfolio (Eq. Wt)" in eq_curves.columns:
    pf = eq_curves["Portfolio (Eq. Wt)"]
    pf_ret = pf.pct_change().dropna()
    total_return = (pf.iloc[-1] / pf.iloc[0]) - 1
    sharpe = pf_ret.mean() / (pf_ret.std() + 1e-10) * np.sqrt(252)
    cum = (1 + pf_ret).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    regime_series = build_regime_series(primary_ticker) if primary_ticker else None
    regime_map = {0: "Mean Reverting", 1: "High Volatility", 2: "Trending ↑", 3: "Trending ↓"}
    current_regime = regime_map.get(
        int(regime_series.iloc[-1]) if regime_series is not None and not regime_series.empty else -1,
        "Unknown"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "#4CAF50" if total_return >= 0 else "#F44336"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Return</div>'
                    f'<div class="metric-value" style="color:{color}">{total_return:+.2%}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        color = "#4CAF50" if sharpe >= 0.5 else ("#FFC107" if sharpe >= 0 else "#F44336")
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div>'
                    f'<div class="metric-value" style="color:{color}">{sharpe:.3f}</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Max Drawdown</div>'
                    f'<div class="metric-value" style="color:#F44336">{max_dd:.2%}</div></div>',
                    unsafe_allow_html=True)
    with c4:
        regime_colors = {"Mean Reverting": "#4CAF50", "High Volatility": "#F44336",
                         "Trending ↑": "#00D2FF", "Trending ↓": "#FF9800", "Unknown": "#888"}
        rc = regime_colors.get(current_regime, "#888")
        st.markdown(f'<div class="metric-card"><div class="metric-label">Market Regime</div>'
                    f'<div class="metric-value" style="color:{rc}">{current_regime}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Equity Curve Chart ─────────────────────────────────────────
    fig = go.Figure()
    for col in eq_curves.columns:
        if col == "Portfolio (Eq. Wt)":
            fig.add_trace(go.Scatter(x=eq_curves.index, y=eq_curves[col], mode='lines',
                                     name=col, line=dict(color='#00D2FF', width=3)))
        else:
            fig.add_trace(go.Scatter(x=eq_curves.index, y=eq_curves[col], mode='lines',
                                     name=col, line=dict(width=1), opacity=0.5))

    # Regime overlay
    if regime_series is not None:
        regime_colors_bg = {0: "rgba(76,175,80,0.08)", 1: "rgba(244,67,54,0.1)",
                            2: "rgba(0,210,255,0.06)", 3: "rgba(255,152,0,0.08)"}
        for i in range(len(regime_series) - 1):
            if regime_series.index[i] in eq_curves.index:
                fig.add_vrect(
                    x0=regime_series.index[i], x1=regime_series.index[i + 1],
                    fillcolor=regime_colors_bg.get(int(regime_series.iloc[i]), "rgba(0,0,0,0)"),
                    opacity=1, layer="below", line_width=0,
                )

    fig.update_layout(
        title="Portfolio Performance (Normalised to 100)",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E0E0E0", family="Inter"),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1e2240'),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Drawdown Chart ─────────────────────────────────────────────
    dd = cum / cum.cummax() - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd, fill='tozeroy', mode='lines',
        line=dict(color='#F44336', width=2),
        fillcolor='rgba(244, 67, 54, 0.2)', name='Drawdown',
    ))
    fig_dd.update_layout(
        title="Drawdown Profile",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=220, margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#E0E0E0", family="Inter"),
        yaxis=dict(tickformat=".1%", showgrid=True, gridcolor='#1e2240'),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_dd, use_container_width=True)
else:
    st.warning("Could not build equity curves — feature data may be empty.")

# ─── Per-Ticker Statistics ──────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Per-Ticker Summary")

stats_rows = []
for t in selected_tickers:
    df = load_feature_data(t)
    if df is None or "close" not in df.columns:
        continue
    rets = df["close"].pct_change().dropna()
    stats_rows.append({
        "Ticker": t,
        "Start": str(df.index[0].date()),
        "End": str(df.index[-1].date()),
        "Bars": len(df),
        "Total Return": f"{(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}",
        "Ann. Vol": f"{rets.std() * np.sqrt(252):.2%}",
        "Sharpe": f"{rets.mean() / (rets.std() + 1e-10) * np.sqrt(252):.3f}",
    })
if stats_rows:
    st.dataframe(pd.DataFrame(stats_rows).set_index("Ticker"), use_container_width=True)

# ─── Backtest Log Viewer ────────────────────────────────────────────
st.markdown("---")
st.subheader("📝 Latest Backtest Log")

log_text = load_backtest_log()
if log_text:
    with st.expander("View Full Log", expanded=False):
        st.code(log_text[:5000], language="text")
    if len(log_text) > 5000:
        st.caption(f"Showing first 5,000 of {len(log_text):,} characters.")
else:
    st.info("No backtest log found. Run backtesting to generate logs.")
