There's the full blueprint. A few things worth highlighting before you start building:

<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--mono:'IBM Plex Mono',monospace;--display:'Syne',sans-serif}
body{font-family:var(--display)}
.wrap{padding:2rem 1.5rem 3rem;max-width:860px;margin:0 auto}
.hero{border-left:3px solid var(--color-text-info);padding:1.2rem 1.5rem;margin-bottom:2.5rem;background:var(--color-background-secondary);border-radius:0 var(--border-radius-lg) var(--border-radius-lg) 0}
.hero-tag{font-family:var(--mono);font-size:11px;letter-spacing:.12em;color:var(--color-text-info);margin-bottom:.5rem;text-transform:uppercase}
.hero-title{font-size:clamp(18px,3vw,26px);font-weight:700;color:var(--color-text-primary);line-height:1.3;margin-bottom:.4rem}
.hero-sub{font-size:13px;color:var(--color-text-secondary);line-height:1.6}
.toc{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-bottom:2.5rem}
.toc-item{background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-md);padding:.6rem .9rem;cursor:pointer;transition:border-color .15s}
.toc-item:hover{border-color:var(--color-border-info)}
.toc-num{font-family:var(--mono);font-size:10px;color:var(--color-text-info);margin-bottom:2px}
.toc-label{font-size:12px;font-weight:500;color:var(--color-text-primary)}
.sec{margin-bottom:2.5rem}
.sec-header{display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;padding-bottom:.7rem;border-bottom:0.5px solid var(--color-border-tertiary)}
.sec-num{font-family:var(--mono);font-size:11px;color:var(--color-text-info);background:var(--color-background-info);padding:3px 7px;border-radius:4px}
.sec-title{font-size:17px;font-weight:600;color:var(--color-text-primary)}
p{font-size:14px;color:var(--color-text-secondary);line-height:1.75;margin-bottom:1rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:1.2rem}
.card{background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1rem 1.1rem}
.card-icon{font-size:16px;margin-bottom:.5rem}
.card-title{font-size:13px;font-weight:600;color:var(--color-text-primary);margin-bottom:.35rem}
.card-desc{font-size:12px;color:var(--color-text-secondary);line-height:1.6}
.arch-flow{background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1.2rem;margin-bottom:1.2rem}
.flow-row{display:flex;align-items:center;gap:0;flex-wrap:wrap;justify-content:center;gap:4px}
.flow-box{background:var(--color-background-primary);border:0.5px solid var(--color-border-secondary);border-radius:var(--border-radius-md);padding:.45rem .75rem;font-size:12px;font-weight:500;color:var(--color-text-primary);white-space:nowrap}
.flow-box.hi{border-color:var(--color-border-info);color:var(--color-text-info)}
.flow-arr{font-size:12px;color:var(--color-text-tertiary);padding:0 2px}
.stack-table{width:100%;border-collapse:collapse;margin-bottom:1.2rem;font-size:13px}
.stack-table th{text-align:left;padding:.55rem .75rem;font-size:11px;font-weight:500;letter-spacing:.08em;color:var(--color-text-secondary);border-bottom:0.5px solid var(--color-border-tertiary);font-family:var(--mono)}
.stack-table td{padding:.6rem .75rem;border-bottom:0.5px solid var(--color-border-tertiary);color:var(--color-text-secondary);vertical-align:top;line-height:1.55}
.stack-table tr:last-child td{border-bottom:none}
.badge{display:inline-block;font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:4px;font-weight:500;margin-right:4px;letter-spacing:.04em}
.badge-blue{background:var(--color-background-info);color:var(--color-text-info)}
.badge-green{background:var(--color-background-success);color:var(--color-text-success)}
.badge-amber{background:var(--color-background-warning);color:var(--color-text-warning)}
.badge-red{background:var(--color-background-danger);color:var(--color-text-danger)}
.math-block{background:var(--color-background-secondary);border-left:3px solid var(--color-border-warning);border-radius:0 var(--border-radius-md) var(--border-radius-md) 0;padding:.9rem 1.1rem;margin-bottom:1rem;font-family:var(--mono);font-size:13px;color:var(--color-text-primary)}
.math-label{font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;font-family:var(--display)}
.diff-list{list-style:none;padding:0;margin-bottom:1.2rem}
.diff-list li{display:flex;gap:10px;align-items:flex-start;padding:.55rem 0;border-bottom:0.5px solid var(--color-border-tertiary);font-size:13px;color:var(--color-text-secondary);line-height:1.6}
.diff-list li:last-child{border-bottom:none}
.diff-icon{flex-shrink:0;width:20px;height:20px;border-radius:50%;background:var(--color-background-success);display:flex;align-items:center;justify-content:center;font-size:10px;color:var(--color-text-success);font-weight:700;margin-top:1px}
.resume-box{background:var(--color-background-primary);border:1.5px solid var(--color-border-info);border-radius:var(--border-radius-lg);padding:1.3rem 1.5rem;position:relative;overflow:hidden}
.resume-label{font-family:var(--mono);font-size:10px;color:var(--color-text-info);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.7rem}
.resume-bullet{font-size:14px;color:var(--color-text-primary);line-height:1.75;padding:.5rem 0;border-bottom:0.5px solid var(--color-border-tertiary)}
.resume-bullet:last-child{border-bottom:none;padding-bottom:0}
.resume-bullet::before{content:"▸ ";color:var(--color-text-info);font-weight:700}
.outcome-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-bottom:1.2rem}
.outcome-card{background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:1rem 1.1rem}
.outcome-metric{font-family:var(--mono);font-size:20px;font-weight:600;color:var(--color-text-info);margin-bottom:.25rem}
.outcome-label{font-size:12px;font-weight:500;color:var(--color-text-primary);margin-bottom:.3rem}
.outcome-desc{font-size:12px;color:var(--color-text-secondary);line-height:1.55}
.scale-item{display:flex;gap:12px;padding:.8rem 0;border-bottom:0.5px solid var(--color-border-tertiary);align-items:flex-start}
.scale-item:last-child{border-bottom:none}
.scale-dot{width:8px;height:8px;border-radius:50%;background:var(--color-text-info);flex-shrink:0;margin-top:5px}
.scale-content .stitle{font-size:13px;font-weight:600;color:var(--color-text-primary);margin-bottom:.2rem}
.scale-content .sdesc{font-size:13px;color:var(--color-text-secondary);line-height:1.6}
hr.divider{border:none;border-top:0.5px solid var(--color-border-tertiary);margin:2rem 0}
</style>
<div class="wrap">

<div class="hero">
  <div class="hero-tag">Senior Quant Developer · Project Blueprint</div>
  <div class="hero-title">AI-Powered Quant Trading &amp;<br>Portfolio Optimization Platform</div>
  <div class="hero-sub">A resume-grade, production-architected system for algorithmic strategy development, ML-driven prediction, and multi-asset portfolio construction.</div>
</div>

<div class="toc">
  <div class="toc-item"><div class="toc-num">01</div><div class="toc-label">Problem Statement</div></div>
  <div class="toc-item"><div class="toc-num">02</div><div class="toc-label">System Architecture</div></div>
  <div class="toc-item"><div class="toc-num">03</div><div class="toc-label">Core Features</div></div>
  <div class="toc-item"><div class="toc-num">04</div><div class="toc-label">Tech Stack</div></div>
  <div class="toc-item"><div class="toc-num">05</div><div class="toc-label">Math &amp; Finance</div></div>
  <div class="toc-item"><div class="toc-num">06</div><div class="toc-label">Differentiators</div></div>
  <div class="toc-item"><div class="toc-num">07</div><div class="toc-label">Expected Outcomes</div></div>
  <div class="toc-item"><div class="toc-num">08</div><div class="toc-label">Scalability</div></div>
  <div class="toc-item"><div class="toc-num">09</div><div class="toc-label">Resume Bullets</div></div>
</div>

<!-- 01 PROBLEM STATEMENT -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">01</span><span class="sec-title">Problem Statement</span></div>
  <p>Retail and semi-institutional traders face a structural disadvantage: they lack the infrastructure to systematically test hypotheses, optimize allocations, and adapt to regime changes. Most manual or rule-based approaches fail to account for the non-stationarity of financial time series — market conditions that worked in 2018 are structurally different from those of 2023.</p>
  <p>Modern markets generate terabytes of tick data daily. Without a unified pipeline that connects raw data ingestion → signal generation → risk-adjusted position sizing → backtested validation, any strategy deployed is speculative rather than evidence-driven. This platform closes that gap.</p>
  <div class="cards">
    <div class="card">
      <div class="card-title">Non-stationary Markets</div>
      <div class="card-desc">Volatility regimes shift. Static rules break. The system must learn and adapt continuously, not fire once on historical patterns.</div>
    </div>
    <div class="card">
      <div class="card-title">Fragmented Tooling</div>
      <div class="card-desc">Data vendors, backtesting frameworks, and optimization libraries are siloed. No end-to-end system exists at the student/indie quant level.</div>
    </div>
    <div class="card">
      <div class="card-title">Survivorship Bias</div>
      <div class="card-desc">Most backtests are wrong. This platform enforces point-in-time data and walk-forward validation to produce statistically honest results.</div>
    </div>
    <div class="card">
      <div class="card-title">Black-box Risk</div>
      <div class="card-desc">ML models in finance are opaque. Explainability modules (SHAP values) make model decisions auditable and defensible.</div>
    </div>
  </div>
</div>

<hr class="divider">

<!-- 02 ARCHITECTURE -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">02</span><span class="sec-title">High-Level System Architecture</span></div>
  <p>The platform is designed as a six-layer pipeline. Each layer has a single responsibility and communicates downstream via well-defined interfaces — mimicking production-grade quant infrastructure at hedge funds.</p>
  <div class="arch-flow">
    <div style="font-size:11px;font-family:var(--mono);color:var(--color-text-secondary);margin-bottom:.8rem;letter-spacing:.06em">DATA FLOW PIPELINE</div>
    <div class="flow-row">
      <div class="flow-box hi">Market Data API</div>
      <div class="flow-arr">→</div>
      <div class="flow-box">Ingestion &amp; Normalization</div>
      <div class="flow-arr">→</div>
      <div class="flow-box hi">Feature Engine</div>
      <div class="flow-arr">→</div>
      <div class="flow-box">ML Prediction Module</div>
      <div class="flow-arr">→</div>
      <div class="flow-box hi">Strategy Engine</div>
      <div class="flow-arr">→</div>
      <div class="flow-box">Portfolio Optimizer</div>
      <div class="flow-arr">→</div>
      <div class="flow-box hi">Backtesting Engine</div>
      <div class="flow-arr">→</div>
      <div class="flow-box">Dashboard / API</div>
    </div>
  </div>
  <div class="cards">
    <div class="card">
      <div class="card-title">Layer 1 — Data Pipeline</div>
      <div class="card-desc">Pulls OHLCV + fundamentals from yfinance / Alpha Vantage / Polygon.io. Stores in a time-series-optimized format (Parquet via Arrow). Handles splits, dividends, and point-in-time integrity.</div>
    </div>
    <div class="card">
      <div class="card-title">Layer 2 — Feature Engine</div>
      <div class="card-desc">Computes 50+ technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV). Adds macro features: VIX, yield curve slope, sector momentum. All lagged to avoid lookahead bias.</div>
    </div>
    <div class="card">
      <div class="card-title">Layer 3 — ML Module</div>
      <div class="card-desc">Trains directional classifiers (Random Forest, XGBoost, LSTM) and regression models for return forecasting. Uses TimeSeriesSplit cross-validation, never random shuffle.</div>
    </div>
    <div class="card">
      <div class="card-title">Layer 4 — Strategy Engine</div>
      <div class="card-desc">Converts model signals and rule-based logic into trade decisions. Manages signal combination, conflict resolution, and position entry/exit rules with configurable thresholds.</div>
    </div>
    <div class="card">
      <div class="card-title">Layer 5 — Portfolio Optimizer</div>
      <div class="card-desc">Solves mean-variance optimization (Markowitz) and maximum-Sharpe portfolios using scipy.optimize. Supports constraints: sector limits, max drawdown, position sizing rules.</div>
    </div>
    <div class="card">
      <div class="card-title">Layer 6 — Backtesting Engine</div>
      <div class="card-desc">Walk-forward simulation with transaction costs, slippage, and borrow costs. Outputs full equity curve, per-trade P&amp;L, and risk-adjusted performance metrics.</div>
    </div>
  </div>
</div>

<hr class="divider">

<!-- 03 CORE FEATURES -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">03</span><span class="sec-title">Core Features</span></div>
  <table class="stack-table">
    <thead><tr><th>Feature</th><th>Implementation Detail</th><th>Tag</th></tr></thead>
    <tbody>
      <tr>
        <td style="font-weight:500;color:var(--color-text-primary)">Market Data Collection</td>
        <td>Multi-source ingestion (yfinance, Alpha Vantage). Async fetch for 100+ tickers. OHLCV stored in Parquet. Incremental daily updates via cron scheduler.</td>
        <td><span class="badge badge-blue">DATA</span></td>
      </tr>
      <tr>
        <td style="font-weight:500;color:var(--color-text-primary)">Technical Indicators</td>
        <td>RSI (14-period overbought/oversold), MACD (12/26/9 EMA crossover signal), SMA/EMA crossovers, Bollinger Band squeeze, ATR-based volatility. Computed via pandas-ta for vectorized speed.</td>
        <td><span class="badge badge-green">SIGNAL</span></td>
      </tr>
      <tr>
        <td style="font-weight:500;color:var(--color-text-primary)">ML Prediction Module</td>
        <td>XGBoost for directional classification (+1/-1/0), LSTM for sequence modelling of 30-day return windows, Random Forest for feature importance ranking. Trained on rolling 2-year windows.</td>
        <td><span class="badge badge-amber">ML</span></td>
      </tr>
      <tr>
        <td style="font-weight:500;color:var(--color-text-primary)">Portfolio Optimization</td>
        <td>Mean-variance frontier via cvxpy (convex solver). Max-Sharpe and min-variance tangent portfolios. Risk parity weighting as alternative. Rebalancing trigger on drift &gt; threshold.</td>
        <td><span class="badge badge-blue">OPTIM</span></td>
      </tr>
      <tr>
        <td style="font-weight:500;color:var(--color-text-primary)">Backtesting Engine</td>
        <td>Event-driven simulation (not vectorized) to faithfully model slippage and partial fills. Walk-forward validation splits data into 12-month train / 3-month test rolling windows. Reports Sharpe, Calmar, max drawdown, win rate.</td>
        <td><span class="badge badge-red">EVAL</span></td>
      </tr>
      <tr>
        <td style="font-weight:500;color:var(--color-text-primary)">Dashboard</td>
        <td>Streamlit-based interactive UI. Real-time equity curve, correlation heatmap, drawdown chart, position table, and strategy parameter tuning panel. FastAPI backend serves JSON to frontend.</td>
        <td><span class="badge badge-green">VIZ</span></td>
      </tr>
    </tbody>
  </table>
</div>

<hr class="divider">

<!-- 04 TECH STACK -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">04</span><span class="sec-title">Tech Stack Justification</span></div>
  <table class="stack-table">
    <thead><tr><th>Tool</th><th>Why This, Not the Alternative</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="badge badge-blue">Python 3.11</span></td>
        <td>Dominant in quant finance. NumPy vectorization, Pandas for time-indexed data, and the entire ML/scientific stack are Python-native. Not R — Python scales better to production.</td>
      </tr>
      <tr>
        <td><span class="badge badge-green">Pandas + NumPy</span></td>
        <td>Financial time series are fundamentally indexed DataFrames. Pandas handles resampling, rolling windows, and join-on-date-index natively. NumPy for matrix operations in optimization.</td>
      </tr>
      <tr>
        <td><span class="badge badge-amber">XGBoost + PyTorch (LSTM)</span></td>
        <td>XGBoost wins on tabular financial data — handles missing values, is fast, and gives feature importance. PyTorch LSTM captures temporal dependencies missed by tree models. Keras would be simpler but less flexible.</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">cvxpy / scipy</span></td>
        <td>cvxpy provides a disciplined convex optimization DSL — the portfolio problem is a quadratic program (QP) with linear constraints. Writing gradient descent by hand would be error-prone and slower.</td>
      </tr>
      <tr>
        <td><span class="badge badge-green">Backtrader / Vectorbt</span></td>
        <td>Backtrader for event-driven simulation (realistic), Vectorbt for ultra-fast parameter sweeps. Using both is legitimate — event-driven for final validation, vectorbt for hyperparameter search.</td>
      </tr>
      <tr>
        <td><span class="badge badge-amber">FastAPI + Streamlit</span></td>
        <td>FastAPI is async-native and auto-generates OpenAPI docs — production-grade. Streamlit for rapid dashboard iteration without React overhead. Plotly for interactive charting.</td>
      </tr>
      <tr>
        <td><span class="badge badge-red">PostgreSQL + TimescaleDB</span></td>
        <td>TimescaleDB extends Postgres with hypertables optimized for time-series queries. Faster than flat files at scale. SQLite would fail at multi-ticker, multi-year tick data.</td>
      </tr>
      <tr>
        <td><span class="badge badge-blue">Docker + GitHub Actions</span></td>
        <td>Containerized deployment ensures reproducibility. CI/CD pipeline runs backtests on every push, catching regressions before they affect strategy performance.</td>
      </tr>
    </tbody>
  </table>
</div>

<hr class="divider">

<!-- 05 MATH & FINANCE -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">05</span><span class="sec-title">Mathematical &amp; Financial Concepts</span></div>
  <div class="math-block">
    <div class="math-label">Sharpe Ratio</div>
    S = (R_p − R_f) / σ_p
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:.4rem">Annualized excess return per unit of volatility. Target: S &gt; 1.5 for a viable strategy.</div>
  </div>
  <div class="math-block">
    <div class="math-label">Mean-Variance Optimization (Markowitz)</div>
    min  w&#7488;Σw  s.t.  w&#7488;μ ≥ r*,  Σw_i = 1,  w_i ≥ 0
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:.4rem">Minimize portfolio variance for a target return r*. Σ is the covariance matrix, μ the expected returns vector.</div>
  </div>
  <div class="math-block">
    <div class="math-label">Maximum Drawdown</div>
    MDD = max [ (Peak_t − Trough_t) / Peak_t ]
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:.4rem">Largest peak-to-trough decline in the equity curve. Used in Calmar ratio = CAGR / |MDD|.</div>
  </div>
  <div class="math-block">
    <div class="math-label">RSI (Relative Strength Index)</div>
    RSI = 100 − [ 100 / (1 + RS) ]   where RS = avg_gain / avg_loss (14-period)
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:.4rem">RSI &lt; 30 → oversold signal. RSI &gt; 70 → overbought signal. Used as a feature, not a standalone strategy.</div>
  </div>
  <p>The ML layer uses <strong>binary cross-entropy loss</strong> for classification (up/down/neutral), <strong>MSE</strong> for return regression, and <strong>SHAP (SHapley Additive exPlanations)</strong> to decompose feature contributions to each prediction — a direct answer to "why did the model go long?"</p>
</div>

<hr class="divider">

<!-- 06 DIFFERENTIATORS -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">06</span><span class="sec-title">Unique Differentiators</span></div>
  <ul class="diff-list">
    <li><div class="diff-icon">✓</div><div><strong style="color:var(--color-text-primary)">Walk-Forward Validation, Not Static Backtesting.</strong> Rolling train/test windows prevent overfitting to a single historical period. Most student projects use a fixed split — this is the #1 reason their results don't hold out-of-sample.</div></li>
    <li><div class="diff-icon">✓</div><div><strong style="color:var(--color-text-primary)">Explainable AI (SHAP) Integration.</strong> Every ML trade decision is explainable via SHAP waterfall plots. This makes the system auditable — critical for anyone targeting finance roles at banks or hedge funds.</div></li>
    <li><div class="diff-icon">✓</div><div><strong style="color:var(--color-text-primary)">News Sentiment Fusion.</strong> FinBERT (pre-trained financial sentiment transformer) scores daily news headlines per ticker. Sentiment score is a feature alongside technicals — blending structured + unstructured data.</div></li>
    <li><div class="diff-icon">✓</div><div><strong style="color:var(--color-text-primary)">Paper Trading Live Loop.</strong> Integration with Alpaca Markets API for paper trading — the system runs in near-real-time on market hours, generating live signals and tracking paper P&amp;L. Proves the system works beyond historical data.</div></li>
    <li><div class="diff-icon">✓</div><div><strong style="color:var(--color-text-primary)">Regime Detection.</strong> Hidden Markov Model (HMM) classifies the market into bull/bear/sideways regimes. Strategy parameters (position size, signal thresholds) adapt per regime — not a static system.</div></li>
    <li><div class="diff-icon">✓</div><div><strong style="color:var(--color-text-primary)">Transaction Cost Modelling.</strong> Slippage (market impact), brokerage commission, and short borrow costs are explicitly modelled in backtests — most student projects ignore these and produce inflated returns.</div></li>
  </ul>
</div>

<hr class="divider">

<!-- 07 OUTCOMES -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">07</span><span class="sec-title">Expected Outputs &amp; User Benefits</span></div>
  <div class="outcome-grid">
    <div class="outcome-card">
      <div class="outcome-metric">Sharpe</div>
      <div class="outcome-label">Strategy Quality Metric</div>
      <div class="outcome-desc">Target &gt;1.5 on out-of-sample test windows. Displayed with confidence interval from bootstrap sampling.</div>
    </div>
    <div class="outcome-card">
      <div class="outcome-metric">Equity Curve</div>
      <div class="outcome-label">Portfolio Growth Visualization</div>
      <div class="outcome-desc">Full equity curve vs. S&amp;P 500 benchmark. Annotated with drawdown periods and regime transitions.</div>
    </div>
    <div class="outcome-card">
      <div class="outcome-metric">SHAP Plot</div>
      <div class="outcome-label">Model Explainability</div>
      <div class="outcome-desc">Per-trade feature attribution. Shows whether RSI, momentum, or news sentiment drove each model decision.</div>
    </div>
    <div class="outcome-card">
      <div class="outcome-metric">Weights</div>
      <div class="outcome-label">Optimized Allocation</div>
      <div class="outcome-desc">Asset weight vector on the efficient frontier. Rebalancing signals when portfolio drifts beyond threshold.</div>
    </div>
  </div>
  <p>A portfolio manager using this system gets: a daily briefing of ML-generated signals, an allocation recommendation with risk attribution, and a backtested evidence base — all from a single Streamlit dashboard.</p>
</div>

<hr class="divider">

<!-- 08 SCALABILITY -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">08</span><span class="sec-title">Scalability &amp; Production Upgrades</span></div>
  <div class="scale-item"><div class="scale-dot"></div><div class="scale-content"><div class="stitle">Replace yfinance with Polygon.io WebSocket feed</div><div class="sdesc">Move from end-of-day batch to intraday streaming. Apache Kafka for message queuing, Faust for stream processing. Enables sub-minute signal generation.</div></div></div>
  <div class="scale-item"><div class="scale-dot"></div><div class="scale-content"><div class="stitle">Online Learning for ML Models</div><div class="sdesc">Switch from static retraining to incremental learning (River library or PyTorch online SGD). Models update daily on new data without full retraining cycles.</div></div></div>
  <div class="scale-item"><div class="scale-dot"></div><div class="scale-content"><div class="stitle">Kubernetes Deployment + Airflow Orchestration</div><div class="sdesc">Containerize each layer as a microservice. Apache Airflow DAGs schedule data ingestion → feature computation → model inference → signal generation as a dependency graph.</div></div></div>
  <div class="scale-item"><div class="scale-dot"></div><div class="scale-content"><div class="stitle">Multi-Asset Expansion (Crypto, Futures, FX)</div><div class="sdesc">Architecture is asset-agnostic. Adding a new asset class requires a data adapter and asset-class-specific features. Same strategy engine and optimizer handle all instruments.</div></div></div>
  <div class="scale-item"><div class="scale-dot"></div><div class="scale-content"><div class="stitle">Risk Management Layer (VaR / CVaR)</div><div class="sdesc">Add Value-at-Risk (historical simulation) and Conditional VaR as hard portfolio constraints in the optimizer. Integrates with live position limits and margin calculations.</div></div></div>
  <div class="scale-item"><div class="scale-dot"></div><div class="scale-content"><div class="stitle">A/B Strategy Testing Framework</div><div class="sdesc">Run multiple strategy variants in parallel paper trading accounts. Statistical significance testing (t-test on Sharpe difference) determines which variant to promote to live.</div></div></div>
</div>

<hr class="divider">

<!-- 09 RESUME -->
<div class="sec">
  <div class="sec-header"><span class="sec-num">09</span><span class="sec-title">Resume Description</span></div>
  <div class="resume-box">
    <div class="resume-label">Copy-Paste Ready · Bullet Points</div>
    <div class="resume-bullet">Engineered an end-to-end algorithmic trading platform in Python integrating a 6-stage ML pipeline (XGBoost + LSTM) with Markowitz portfolio optimization, achieving a backtested Sharpe ratio &gt;1.6 on walk-forward validation across 50+ equities.</div>
    <div class="resume-bullet">Built a regime-adaptive strategy engine combining 50+ technical features with FinBERT-derived news sentiment scores; implemented SHAP explainability to audit model-driven trade signals; deployed via FastAPI + Streamlit with live Alpaca paper trading integration.</div>
    <div class="resume-bullet">Designed a transaction-cost-aware backtesting engine with event-driven simulation, rolling 2-year train/3-month test splits, and TimescaleDB-backed time-series storage; containerized with Docker and CI/CD via GitHub Actions.</div>
  </div>
</div>

</div>

**The single most important architectural decision** is using `TimeSeriesSplit` for cross-validation, not `train_test_split`. Random shuffling in financial ML is the most common mistake — it leaks future data into training. This one choice separates a credible system from a flawed one.

**On the resume bullets** — the three bullets in Section 09 are designed to hit the keywords that quant finance recruiters scan for: walk-forward validation, Sharpe ratio with a number attached, named model architectures, a real deployment target (Alpaca), and production tooling (Docker, CI/CD). Tailor the Sharpe number to whatever your actual backtest produces.

**Recommended build order:**
1. Data pipeline + feature engine first (if your features are wrong, nothing downstream works)
2. Backtesting engine second (validates features before wasting time on ML)
3. ML module third (slot into the already-working pipeline)
4. Portfolio optimizer fourth
5. Dashboard last

The SHAP integration and FinBERT sentiment fusion are the two features that will get the most questions in interviews — they demonstrate you understand *why* a model does what it does, not just that it does it.