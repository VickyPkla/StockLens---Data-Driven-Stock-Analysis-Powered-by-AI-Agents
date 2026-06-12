"""Quantitative risk metrics for a stock.

Called from data_prefetch.py to compute verified risk data before agents run.
All values are computed in Python — never left to LLM generation.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import yfinance as yf


# ── Minimal helpers (duplicated from data_prefetch.py to avoid circular imports) ──

def _f(val, decimals: int = 2) -> str:
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _is_indian(ticker: str) -> bool:
    return ticker.upper().endswith((".NS", ".BO"))


# ── Sub-score helpers (1-10, higher = more risk) ──

def _vol_score(vol_pct: float | None) -> int:
    if vol_pct is None:
        return 5
    if vol_pct < 15:
        return 2
    if vol_pct < 25:
        return 4
    if vol_pct < 35:
        return 6
    if vol_pct < 50:
        return 8
    return 10


def _de_score(de: float | None) -> int:
    if de is None:
        return 5
    if de < 0.3:
        return 1
    if de < 0.8:
        return 3
    if de < 1.5:
        return 5
    if de < 3.0:
        return 7
    return 9


def _beta_score(beta: float | None) -> int:
    if beta is None:
        return 5
    if beta < 0.5:
        return 2
    if beta < 0.8:
        return 3
    if beta < 1.2:
        return 5
    if beta < 1.6:
        return 7
    return 9


def _earnings_score(beat_rate: float | None) -> int:
    """Inverted: high beat rate = low earnings quality risk."""
    if beat_rate is None:
        return 5
    if beat_rate > 75:
        return 2
    if beat_rate > 50:
        return 4
    if beat_rate > 25:
        return 6
    return 8


# ── Main computation ──

def compute_risk_metrics(ticker: str) -> dict:
    """Compute quantitative risk metrics for a stock.

    Returns a dict with raw metrics and sub-scores (1-10, higher = more risk),
    a weighted composite_risk_score, and a risk_level string.
    """
    t = yf.Ticker(ticker)

    try:
        hist = t.history(period="2y")
    except Exception:
        hist = None

    try:
        info = t.info or {}
    except Exception:
        info = {}

    result: dict = {}

    # ── Price-based metrics ──
    if hist is not None and len(hist) >= 30:
        close = hist["Close"].dropna()

        # Annualized volatility
        daily_returns = close.pct_change().dropna()
        ann_vol = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) > 1 else None
        result["annualized_volatility_pct"] = ann_vol

        # Max drawdown (worst rolling 252-day drawdown; fall back to full history if shorter)
        window = min(252, len(close))
        rolling_max = close.rolling(window, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        result["max_drawdown_pct"] = float(drawdown.min() * 100)

        # Annualised return (last 252 sessions or full history)
        idx = min(252, len(close) - 1)
        if idx > 0:
            ann_ret = float((close.iloc[-1] / close.iloc[-idx] - 1))
        else:
            ann_ret = 0.0

        # Sharpe ratio
        risk_free = 0.06 if _is_indian(ticker) else 0.05
        if ann_vol and ann_vol > 0:
            result["sharpe_ratio"] = round((ann_ret - risk_free) / (ann_vol / 100), 2)
        else:
            result["sharpe_ratio"] = None
    else:
        result["annualized_volatility_pct"] = None
        result["max_drawdown_pct"] = None
        result["sharpe_ratio"] = None

    # ── Info-based metrics ──
    beta = info.get("beta")
    try:
        result["beta"] = float(beta) if beta is not None else None
    except (TypeError, ValueError):
        result["beta"] = None

    de = info.get("debtToEquity")
    try:
        result["debt_to_equity"] = float(de) if de is not None else None
    except (TypeError, ValueError):
        result["debt_to_equity"] = None

    cr = info.get("currentRatio")
    try:
        result["current_ratio"] = float(cr) if cr is not None else None
    except (TypeError, ValueError):
        result["current_ratio"] = None

    # ── Analyst target spread ──
    try:
        at = t.analyst_price_targets
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if at is not None and not at.empty and current:
            high = float(at.get("high", 0) or 0)
            low = float(at.get("low", 0) or 0)
            if high > 0 and low > 0 and float(current) > 0:
                result["analyst_target_spread_pct"] = round((high - low) / float(current) * 100, 1)
            else:
                result["analyst_target_spread_pct"] = None
        else:
            result["analyst_target_spread_pct"] = None
    except Exception:
        result["analyst_target_spread_pct"] = None

    # ── Earnings beat rate (last 4 quarters) ──
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            reported = ed.dropna(subset=["Reported EPS"])
            recent = reported.head(4)
            beats = 0
            total = 0
            for _, row in recent.iterrows():
                actual = row.get("Reported EPS")
                estimate = row.get("EPS Estimate")
                if actual is not None and estimate is not None:
                    try:
                        if float(actual) >= float(estimate):
                            beats += 1
                        total += 1
                    except (TypeError, ValueError):
                        pass
            result["earnings_beat_rate"] = round(beats / total * 100, 1) if total > 0 else None
        else:
            result["earnings_beat_rate"] = None
    except Exception:
        result["earnings_beat_rate"] = None

    # ── Sub-scores ──
    vs = _vol_score(result.get("annualized_volatility_pct"))
    bs = _de_score(result.get("debt_to_equity"))
    ms = _beta_score(result.get("beta"))
    es = _earnings_score(result.get("earnings_beat_rate"))

    result["volatility_risk_score"] = vs
    result["balance_sheet_risk_score"] = bs
    result["market_risk_score"] = ms
    result["earnings_quality_score"] = es

    composite = vs * 0.35 + bs * 0.25 + ms * 0.25 + es * 0.15
    result["composite_risk_score"] = round(composite, 1)

    if composite < 4:
        result["risk_level"] = "Low"
    elif composite <= 7:
        result["risk_level"] = "Medium"
    else:
        result["risk_level"] = "High"

    return result


def format_risk_context(ticker: str) -> str:
    """Compute risk metrics and return a formatted VERIFIED RISK DATA context block."""
    today = date.today().strftime("%B %d, %Y")
    try:
        m = compute_risk_metrics(ticker)
    except Exception as e:
        return (
            f"╔══════════════════════════════════════════════════════════════╗\n"
            f"║  VERIFIED RISK DATA  |  {today}  |  Ticker: {ticker}\n"
            f"╚══════════════════════════════════════════════════════════════╝\n"
            f"ERROR: Could not compute risk metrics: {e}\n"
        )

    vol = _f(m.get("annualized_volatility_pct")) + "%"
    mdd = (_f(m.get("max_drawdown_pct")) + "%") if m.get("max_drawdown_pct") is not None else "N/A"
    sharpe = _f(m.get("sharpe_ratio"))
    beta = _f(m.get("beta"))
    de = _f(m.get("debt_to_equity"))
    cr = _f(m.get("current_ratio"))
    spread = (_f(m.get("analyst_target_spread_pct")) + "%") if m.get("analyst_target_spread_pct") is not None else "N/A"
    beat = (_f(m.get("earnings_beat_rate")) + "%") if m.get("earnings_beat_rate") is not None else "N/A"

    vs = m.get("volatility_risk_score", "N/A")
    bs = m.get("balance_sheet_risk_score", "N/A")
    ms = m.get("market_risk_score", "N/A")
    es = m.get("earnings_quality_score", "N/A")
    composite = m.get("composite_risk_score", "N/A")
    risk_level = m.get("risk_level", "N/A")

    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        f"║  VERIFIED RISK DATA  |  {today}  |  Ticker: {ticker}",
        "║  Source: yfinance computed metrics",
        "║  USE ONLY THESE NUMBERS. DO NOT CHANGE OR INVENT ANY VALUES.",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        "── Quantitative Risk Metrics ──────────────────────────────────────────",
        f"  Annualized Volatility    : {vol}",
        f"  Max Drawdown (1-2yr)     : {mdd}",
        f"  Sharpe Ratio             : {sharpe}",
        f"  Beta                     : {beta}",
        f"  Debt-to-Equity           : {de}",
        f"  Current Ratio            : {cr}",
        f"  Analyst Target Spread    : {spread}  (high-low / current price)",
        f"  Earnings Beat Rate       : {beat}  (last 4 quarters)",
        "",
        "── Risk Sub-Scores (1-10, higher = more risk) ─────────────────────────",
        f"  Volatility Risk Score    : {vs}/10",
        f"  Balance Sheet Risk Score : {bs}/10",
        f"  Market Risk Score        : {ms}/10",
        f"  Earnings Quality Score   : {es}/10",
        "",
        "── Composite Risk Assessment ──────────────────────────────────────────",
        f"  Composite Risk Score     : {composite}/10  (vol×0.35 + balance×0.25 + market×0.25 + earnings×0.15)",
        f"  Overall Risk Level       : {risk_level}  (<4 = Low, 4-7 = Medium, >7 = High)",
        "",
    ]
    return "\n".join(lines)
