"""
run_cvar_comparison.py — CVaR vs Risk Parity head-to-head comparison.

Runs the full portfolio backtest twice (once with CVaR, once with Risk Parity)
and produces a consolidated report with:
    • Sharpe Ratio
    • Max Drawdown
    • CVaR (5 % tail loss)
    • CAGR
    • Sortino
    • Win Rate
    • Tail-loss comparison

Usage:
    python scripts/run_cvar_comparison.py
    python scripts/run_cvar_comparison.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.config_loader import Config
from src.utils.logger import setup_logging, get_logger
from src.pipeline.backtest_runner import BacktestRunner
from src.core.backtesting.metrics import MetricsCalculator, PerformanceReport

logger = get_logger("cvar_comparison")


# ══════════════════════════════════════════════════════════════════════════
#   Helper utilities
# ══════════════════════════════════════════════════════════════════════════

def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall) at the given confidence level."""
    if returns.empty or returns.std() < 1e-10:
        return 0.0
    var_threshold = np.percentile(returns.dropna(), (1 - alpha) * 100)
    tail = returns[returns <= var_threshold]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = cum / rolling_max - 1
    return float(dd.min())


def compute_sharpe(returns: pd.Series, rf_annual: float = 0.04) -> float:
    """Annualised Sharpe ratio."""
    if returns.std() < 1e-10 or len(returns) < 10:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    return float((excess.mean() / returns.std()) * np.sqrt(252))


def compute_sortino(returns: pd.Series, rf_annual: float = 0.04) -> float:
    """Annualised Sortino ratio."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() < 1e-10:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))


def compute_tail_stats(returns: pd.Series, alpha: float = 0.95) -> dict:
    """Detailed tail-risk breakdown."""
    r = returns.dropna()
    if r.empty:
        return {"VaR_5pct": 0.0, "CVaR_5pct": 0.0, "worst_day": 0.0,
                "worst_5d": 0.0, "tail_ratio": 0.0, "skewness": 0.0, "kurtosis": 0.0}

    var_5 = np.percentile(r, (1 - alpha) * 100)
    tail = r[r <= var_5]
    cvar_5 = float(tail.mean()) if len(tail) > 0 else 0.0

    # Worst 5-day rolling return
    roll5 = r.rolling(5).sum()
    worst_5d = float(roll5.min()) if not roll5.isna().all() else 0.0

    # Tail ratio: mean of upper 5% / abs(mean of lower 5%)
    upper_5 = np.percentile(r, 95)
    upper_tail = r[r >= upper_5]
    lower_tail = r[r <= var_5]
    if len(lower_tail) > 0 and abs(lower_tail.mean()) > 1e-10:
        tail_ratio = float(upper_tail.mean() / abs(lower_tail.mean()))
    else:
        tail_ratio = 0.0

    return {
        "VaR_5pct": float(var_5),
        "CVaR_5pct": cvar_5,
        "worst_day": float(r.min()),
        "worst_5d": worst_5d,
        "tail_ratio": tail_ratio,
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
    }


# ══════════════════════════════════════════════════════════════════════════
#   Run a portfolio backtest under a specific allocation method
# ══════════════════════════════════════════════════════════════════════════

def run_portfolio_backtest(
    cfg: Config,
    allocation_type: str,
    tickers: list[str],
    strategy_filter: str = "Ensemble",
) -> tuple[PerformanceReport | None, pd.Series | None]:
    """
    Run the full portfolio pipeline under a given allocation method.

    Returns
    -------
    report : PerformanceReport for the portfolio
    daily_returns : pd.Series of daily strategy returns for further analysis
    """
    # Override allocation in config (in-memory only)
    cfg._cfg.setdefault("portfolio", {})["allocation_type"] = allocation_type

    runner = BacktestRunner(cfg)

    # Keep only the target strategy to speed things up
    runner.strategies = [s for s in runner.strategies if strategy_filter in s.name]

    _, reports = runner.run_all(tickers=tickers)

    # Find the portfolio-level report
    pf_report = next(
        (r for r in reports
         if r.ticker == "PORTFOLIO" and strategy_filter in r.strategy_name
         and "Beta-Neutral" not in r.strategy_name),
        None,
    )

    # Reconstruct daily returns from the runner (we need the raw series for tail analysis)
    # Since the runner doesn't expose daily_returns directly for the portfolio,
    # we re-derive from the report's equity curve via metrics
    # However, the runner's internal aggregation does produce them.
    # For robustness, we run a lightweight replay:
    daily_returns = _reconstruct_portfolio_returns(cfg, runner, tickers, strategy_filter)

    return pf_report, daily_returns


def _reconstruct_portfolio_returns(
    cfg: Config,
    runner: BacktestRunner,
    tickers: list[str],
    strategy_filter: str,
) -> pd.Series | None:
    """
    Reconstruct the aggregated portfolio daily returns by replaying
    the same logic as BacktestRunner.run_all but capturing returns.
    """
    from src.portfolio.portfolio_manager import PortfolioManager
    from src.risk.risk_manager import RiskManager

    dfs = {}
    for t in tickers:
        try:
            dfs[t] = runner.load_features(t)
        except FileNotFoundError:
            continue

    if not dfs:
        return None

    returns_dict = {t: df["close"].pct_change().fillna(0) for t, df in dfs.items()}

    portfolio_cfg = cfg.get("portfolio", {})
    allocation_type = portfolio_cfg.get("allocation_type", "cvar")
    cvar_params = portfolio_cfg.get("cvar", {})
    pm = PortfolioManager(
        allocation_type=allocation_type,
        returns_data=returns_dict,
        cvar_params=cvar_params,
    )
    rm = RiskManager(cfg.get("risk_management", {}))

    target_strategy = None
    for s in runner.strategies:
        if strategy_filter in s.name:
            target_strategy = s
            break
    if target_strategy is None:
        return None

    raw_signals = {}
    risk_adj_signals = {}
    for ticker, df in dfs.items():
        raw = target_strategy.generate_signals(df)
        ret = returns_dict[ticker]
        risk_adj = rm.apply_rules(df, raw, ret)
        raw_signals[ticker] = raw
        risk_adj_signals[ticker] = risk_adj

    weights_df = pm.allocate(risk_adj_signals)

    combined_returns = None
    for ticker, df in dfs.items():
        alloc_signal = weights_df[ticker]
        result = runner.engine.run(df, alloc_signal, target_strategy.name, ticker)
        if combined_returns is None:
            combined_returns = result.daily_returns.copy()
        else:
            combined_returns = combined_returns.add(result.daily_returns, fill_value=0)

    return combined_returns


# ══════════════════════════════════════════════════════════════════════════
#   Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="CVaR vs Risk Parity Comparison")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker (optional)")
    args = parser.parse_args()

    cfg = Config(args.config)
    setup_logging(level=cfg.log_level, log_dir=cfg.project_root / "logs")

    tickers = [args.ticker.upper()] if args.ticker else cfg.tickers

    print(f"\n{'═' * 80}")
    print(f"  CVaR vs RISK PARITY — Portfolio Allocation Comparison")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'═' * 80}\n")

    t0 = time.time()

    # ── Run 1: CVaR allocation ─────────────────────────────────────────────
    print("▸ Running CVaR portfolio backtest...")
    rpt_cvar, ret_cvar = run_portfolio_backtest(cfg, "cvar", tickers)

    # ── Run 2: Risk Parity allocation ──────────────────────────────────────
    print("▸ Running Risk Parity portfolio backtest...")
    rpt_rp, ret_rp = run_portfolio_backtest(cfg, "risk_parity", tickers)

    elapsed = time.time() - t0

    if rpt_cvar is None or rpt_rp is None:
        print("\n✗ Could not generate both portfolio reports. Aborting comparison.")
        return

    # ── Compute tail statistics ────────────────────────────────────────────
    tail_cvar = compute_tail_stats(ret_cvar) if ret_cvar is not None else {}
    tail_rp = compute_tail_stats(ret_rp) if ret_rp is not None else {}

    # ══════════════════════════════════════════════════════════════════════
    #   Report
    # ══════════════════════════════════════════════════════════════════════
    sep = "─" * 76
    header = f"\n{'═' * 76}\n  PORTFOLIO ALLOCATION COMPARISON: CVaR vs Risk Parity\n{'═' * 76}"
    print(header)

    # — Section 1: Core Risk-Adjusted Metrics ——————————————————————————————
    print(f"\n  ┌{'─' * 40}┬{'─' * 16}┬{'─' * 16}┐")
    print(f"  │ {'Metric':<38s} │ {'CVaR':^14s} │ {'Risk Parity':^14s} │")
    print(f"  ├{'─' * 40}┼{'─' * 16}┼{'─' * 16}┤")

    metrics = [
        ("Sharpe Ratio",          f"{rpt_cvar.sharpe_ratio:>+.3f}",      f"{rpt_rp.sharpe_ratio:>+.3f}"),
        ("Sortino Ratio",         f"{rpt_cvar.sortino_ratio:>+.3f}",     f"{rpt_rp.sortino_ratio:>+.3f}"),
        ("CAGR",                  f"{rpt_cvar.cagr:>+.2%}",             f"{rpt_rp.cagr:>+.2%}"),
        ("Total Return",          f"{rpt_cvar.total_return:>+.2%}",      f"{rpt_rp.total_return:>+.2%}"),
        ("Annualised Volatility", f"{rpt_cvar.annualized_volatility:>.2%}", f"{rpt_rp.annualized_volatility:>.2%}"),
        ("Max Drawdown",          f"{rpt_cvar.max_drawdown:>+.2%}",      f"{rpt_rp.max_drawdown:>+.2%}"),
        ("Max DD Duration (days)",f"{rpt_cvar.max_drawdown_duration_days:>d}", f"{rpt_rp.max_drawdown_duration_days:>d}"),
        ("Calmar Ratio",          f"{rpt_cvar.calmar_ratio:>+.3f}",      f"{rpt_rp.calmar_ratio:>+.3f}"),
        ("Win Rate",              f"{rpt_cvar.win_rate:>.2%}",           f"{rpt_rp.win_rate:>.2%}"),
        ("Profit Factor",         f"{rpt_cvar.profit_factor:>.3f}",      f"{rpt_rp.profit_factor:>.3f}"),
    ]

    for name, v_cvar, v_rp in metrics:
        print(f"  │ {name:<38s} │ {v_cvar:>14s} │ {v_rp:>14s} │")

    print(f"  └{'─' * 40}┴{'─' * 16}┴{'─' * 16}┘")

    # — Section 2: Tail-Risk Comparison ———————————————————————————————————
    print(f"\n  ┌{'─' * 40}┬{'─' * 16}┬{'─' * 16}┐")
    print(f"  │ {'Tail-Risk Metric':<38s} │ {'CVaR':^14s} │ {'Risk Parity':^14s} │")
    print(f"  ├{'─' * 40}┼{'─' * 16}┼{'─' * 16}┤")

    tail_metrics = [
        ("VaR (5%)",      f"{tail_cvar.get('VaR_5pct', 0):>+.4%}",   f"{tail_rp.get('VaR_5pct', 0):>+.4%}"),
        ("CVaR / ES (5%)", f"{tail_cvar.get('CVaR_5pct', 0):>+.4%}",  f"{tail_rp.get('CVaR_5pct', 0):>+.4%}"),
        ("Worst 1-Day",   f"{tail_cvar.get('worst_day', 0):>+.4%}",   f"{tail_rp.get('worst_day', 0):>+.4%}"),
        ("Worst 5-Day",   f"{tail_cvar.get('worst_5d', 0):>+.4%}",    f"{tail_rp.get('worst_5d', 0):>+.4%}"),
        ("Tail Ratio",    f"{tail_cvar.get('tail_ratio', 0):>.3f}",    f"{tail_rp.get('tail_ratio', 0):>.3f}"),
        ("Skewness",      f"{tail_cvar.get('skewness', 0):>+.3f}",    f"{tail_rp.get('skewness', 0):>+.3f}"),
        ("Excess Kurtosis",f"{tail_cvar.get('kurtosis', 0):>+.3f}",   f"{tail_rp.get('kurtosis', 0):>+.3f}"),
    ]

    for name, v_cvar_t, v_rp_t in tail_metrics:
        print(f"  │ {name:<38s} │ {v_cvar_t:>14s} │ {v_rp_t:>14s} │")

    print(f"  └{'─' * 40}┴{'─' * 16}┴{'─' * 16}┘")

    # — Section 3: Delta Summary ——————————————————————————————————————————
    delta_sharpe = rpt_cvar.sharpe_ratio - rpt_rp.sharpe_ratio
    delta_dd = rpt_cvar.max_drawdown - rpt_rp.max_drawdown
    delta_cvar_5 = tail_cvar.get("CVaR_5pct", 0) - tail_rp.get("CVaR_5pct", 0)
    delta_cagr = rpt_cvar.cagr - rpt_rp.cagr

    print(f"\n  {'─' * 50}")
    print(f"  DELTA (CVaR − Risk Parity)")
    print(f"  {'─' * 50}")
    print(f"    Sharpe:       {delta_sharpe:>+.3f}  {'✓ Better' if delta_sharpe > 0 else '✗ Worse'}")
    print(f"    Max DD:       {delta_dd:>+.2%}  {'✓ Shallower' if delta_dd > 0 else '✗ Deeper'}")
    print(f"    CVaR (5%):    {delta_cvar_5:>+.4%}  {'✓ Smaller tail' if delta_cvar_5 > 0 else '✗ Larger tail'}")
    print(f"    CAGR:         {delta_cagr:>+.2%}  {'✓ Higher' if delta_cagr > 0 else '✗ Lower'}")

    # — Verdict ———————————————————————————————————————————————————————————
    score_cvar = sum([
        delta_sharpe > 0,
        delta_dd > 0,  # less negative = shallower
        delta_cvar_5 > 0,  # less negative = smaller tail
        delta_cagr > 0,
    ])

    winner = "CVaR" if score_cvar >= 3 else ("Risk Parity" if score_cvar <= 1 else "Mixed")
    print(f"\n  ► Verdict: {winner} wins on {score_cvar}/4 headline metrics")

    print(f"\n{'═' * 76}")
    print(f"  Comparison completed in {elapsed:.1f}s")
    print(f"{'═' * 76}\n")

    # — Save to file ——————————————————————————————————————————————————————
    out_dir = cfg.project_root / "logs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cvar_vs_risk_parity.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write("Core Metrics:\n")
        for name, v_c, v_r in metrics:
            f.write(f"  {name:<38s}  {v_c:>14s}  {v_r:>14s}\n")
        f.write("\nTail-Risk Metrics:\n")
        for name, v_c, v_r in tail_metrics:
            f.write(f"  {name:<38s}  {v_c:>14s}  {v_r:>14s}\n")
        f.write(f"\nDelta Sharpe: {delta_sharpe:+.3f}\n")
        f.write(f"Delta Max DD: {delta_dd:+.2%}\n")
        f.write(f"Delta CVaR(5%): {delta_cvar_5:+.4%}\n")
        f.write(f"Delta CAGR: {delta_cagr:+.2%}\n")
        f.write(f"\nVerdict: {winner}\n")

    print(f"  Report saved → {out_path}")


if __name__ == "__main__":
    main()
