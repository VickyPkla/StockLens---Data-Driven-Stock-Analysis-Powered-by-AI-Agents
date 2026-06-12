"""Pre-fetch verified stock data and format it as a grounded context block.

This module fetches real data directly from yfinance BEFORE agents run and
injects it into task prompts. This prevents the LLM from hallucinating prices,
ratios, or indicators from stale training data.

Works for any ticker — Indian (.NS/.BO) or US/global.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _f(val, decimals: int = 2) -> str:
    """Format a float; return 'N/A' if None or NaN."""
    try:
        f = float(val)
        if pd.isna(f):
            return "N/A"
        return f"{f:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct(val) -> str:
    """Format a value already in percentage form (e.g. 5.01 → '5.01%')."""
    try:
        f = float(val)
        return f"{f:.2f}%" if not pd.isna(f) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def _pct_frac(val) -> str:
    """Format a yfinance 0.0–1.0 fractional value as a percentage (e.g. 0.28 → '28.00%')."""
    try:
        f = float(val)
        return f"{f * 100:.2f}%" if not pd.isna(f) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def _money(val, prefix: str = "₹", indian: bool = True) -> str:
    """Format a monetary value with smart scale suffixes.

    Indian stocks: uses Cr (crore = 10^7) and L (lakh = 10^5).
    US/global stocks: uses T, B, M, K.
    """
    try:
        f = float(val)
        if pd.isna(f):
            return "N/A"
        neg = f < 0
        af = abs(f)
        if indian:
            if af >= 1e12:
                s = f"{prefix}{af / 1e12:,.2f}T"
            elif af >= 1e9:
                s = f"{prefix}{af / 1e9:,.2f}B"
            elif af >= 1e7:
                s = f"{prefix}{af / 1e7:,.2f}Cr"
            elif af >= 1e5:
                s = f"{prefix}{af / 1e5:,.2f}L"
            else:
                s = f"{prefix}{af:,.2f}"
        else:
            if af >= 1e12:
                s = f"{prefix}{af / 1e12:,.2f}T"
            elif af >= 1e9:
                s = f"{prefix}{af / 1e9:,.2f}B"
            elif af >= 1e6:
                s = f"{prefix}{af / 1e6:,.2f}M"
            elif af >= 1e3:
                s = f"{prefix}{af / 1e3:,.2f}K"
            else:
                s = f"{prefix}{af:,.2f}"
        return f"-{s}" if neg else s
    except (TypeError, ValueError):
        return "N/A"


def _df_get(df: pd.DataFrame, col, *keys) -> float | None:
    """Return the first non-null value matching any of the given row keys."""
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key, col]
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Ticker metadata
# ---------------------------------------------------------------------------

def _is_indian(ticker: str) -> bool:
    return ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")


def _currency(ticker: str) -> str:
    return "₹" if _is_indian(ticker) else "$"


# ---------------------------------------------------------------------------
# Technical data fetcher
# ---------------------------------------------------------------------------

def _fetch_technical(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        return {}

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    current_price = float(close.iloc[-1])

    rsi_series = RSIIndicator(close, window=14).rsi()
    rsi = float(rsi_series.iloc[-1]) if rsi_series.notna().any() else None

    macd_obj = MACD(close, window_fast=12, window_slow=26, window_sign=9)
    macd_val = float(macd_obj.macd().iloc[-1])
    macd_sig = float(macd_obj.macd_signal().iloc[-1])
    macd_hist = float(macd_obj.macd_diff().iloc[-1])

    bb = BollingerBands(close, window=20)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_mid = float(bb.bollinger_mavg().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    bb_pct_b = float(bb.bollinger_pband().iloc[-1])

    sma50_series = SMAIndicator(close, window=50).sma_indicator()
    sma50 = float(sma50_series.iloc[-1]) if sma50_series.notna().any() else None

    sma200_series = SMAIndicator(close, window=200).sma_indicator()
    sma200 = float(sma200_series.iloc[-1]) if sma200_series.notna().any() else None

    ema20 = float(EMAIndicator(close, window=20).ema_indicator().iloc[-1])
    atr = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])

    vol_sma20 = float(volume.rolling(20).mean().iloc[-1])
    current_vol = float(volume.iloc[-1])

    obv_series = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    if len(obv_series) >= 21:
        obv_trend = "rising" if float(obv_series.iloc[-1]) > float(obv_series.iloc[-21]) else "falling"
    else:
        obv_trend = "insufficient data"

    high_52w = float(hist["High"].max())
    low_52w = float(hist["Low"].min())
    pct_from_high = ((current_price - high_52w) / high_52w) * 100
    pct_from_low = ((current_price - low_52w) / low_52w) * 100

    fib_range = high_52w - low_52w
    fibs = {
        "23.6%": high_52w - fib_range * 0.236,
        "38.2%": high_52w - fib_range * 0.382,
        "50.0%": high_52w - fib_range * 0.500,
        "61.8%": high_52w - fib_range * 0.618,
        "78.6%": high_52w - fib_range * 0.786,
    }

    support_60 = float(close.tail(60).min())
    resistance_60 = float(close.tail(60).max())

    return {
        "price_date": str(hist.index[-1].date()),
        "current_price": current_price,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low,
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "macd_histogram": macd_hist,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_pct_b": bb_pct_b,
        "sma50": sma50,
        "sma200": sma200,
        "ema20": ema20,
        "atr": atr,
        "atr_pct": (atr / current_price) * 100 if current_price else None,
        "vol_sma20": vol_sma20,
        "current_vol": current_vol,
        "vol_above_avg": current_vol > vol_sma20,
        "price_above_sma50": sma50 is not None and current_price > sma50,
        "price_above_sma200": sma200 is not None and current_price > sma200,
        "golden_cross": sma50 is not None and sma200 is not None and sma50 > sma200,
        "obv_trend": obv_trend,
        "fibs": fibs,
        "support_60d": support_60,
        "resistance_60d": resistance_60,
    }


# ---------------------------------------------------------------------------
# Fundamental data fetcher
# ---------------------------------------------------------------------------

def _fetch_fundamental(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    result: dict = {}

    # --- Info / valuation ratios ---
    try:
        info = t.info
        result["name"] = info.get("longName") or info.get("shortName") or ticker
        result["sector"] = info.get("sector") or "N/A"
        result["industry"] = info.get("industry") or "N/A"
        result["market_cap"] = info.get("marketCap")
        result["trailing_pe"] = info.get("trailingPE")
        result["forward_pe"] = info.get("forwardPE")
        result["pb"] = info.get("priceToBook")
        result["ev_ebitda"] = info.get("enterpriseToEbitda")
        result["beta"] = info.get("beta")
        result["trailing_eps"] = info.get("trailingEps")
        result["analyst_count"] = info.get("numberOfAnalystOpinions")
        # Margins and returns — yfinance returns as 0.0-1.0 fractions
        result["gross_margin"] = info.get("grossMargins")
        result["op_margin"] = info.get("operatingMargins")
        result["net_margin"] = info.get("profitMargins")
        result["roe"] = info.get("returnOnEquity")
        result["roa"] = info.get("returnOnAssets")
        result["revenue_growth_yoy"] = info.get("revenueGrowth")
        result["earnings_growth_yoy"] = info.get("earningsGrowth")
        result["debt_to_equity"] = info.get("debtToEquity")
        result["current_ratio"] = info.get("currentRatio")
        # Dividend yield: compute from dividendRate/price for accuracy;
        # yfinance's dividendYield field uses inconsistent scaling across exchanges.
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        dividend_rate = info.get("dividendRate")
        if dividend_rate and current_price and float(current_price) > 0:
            result["dividend_yield_pct"] = round(float(dividend_rate) / float(current_price) * 100, 2)
        else:
            # Fall back: yfinance stores this as the actual % value (e.g. 0.12 = 0.12%)
            result["dividend_yield_pct"] = info.get("dividendYield")
    except Exception:
        pass

    # --- Analyst targets ---
    try:
        at = t.analyst_price_targets
        if at:
            result["analyst_target_mean"] = at.get("mean")
            result["analyst_target_high"] = at.get("high")
            result["analyst_target_low"] = at.get("low")
            result["analyst_target_current"] = at.get("current")
    except Exception:
        pass

    # --- Analyst ratings ---
    try:
        rec = t.recommendations_summary
        if rec is not None and not rec.empty:
            row = rec.iloc[0]
            sb = int(row.get("strongBuy", 0) or 0)
            b = int(row.get("buy", 0) or 0)
            h = int(row.get("hold", 0) or 0)
            s = int(row.get("sell", 0) or 0)
            ss = int(row.get("strongSell", 0) or 0)
            total = sb + b + h + s + ss
            result["ratings"] = {"strongBuy": sb, "buy": b, "hold": h, "sell": s, "strongSell": ss}
            result["ratings_total"] = total
            result["pct_bullish"] = round((sb + b) / total * 100, 1) if total else None
    except Exception:
        pass

    # --- Earnings surprises (last 4 reported quarters) ---
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            actual_col = "Reported EPS" if "Reported EPS" in ed.columns else "EPS Actual"
            if actual_col in ed.columns:
                past = ed[ed[actual_col].notna()].head(4)
                surprises = []
                for idx, row in past.iterrows():
                    actual = row.get(actual_col)
                    estimate = row.get("EPS Estimate")
                    surprise_pct = row.get("Surprise(%)")
                    if pd.isna(surprise_pct) and pd.notna(actual) and pd.notna(estimate) and float(estimate) != 0:
                        surprise_pct = ((float(actual) - float(estimate)) / abs(float(estimate))) * 100
                    surprises.append({
                        "quarter": str(idx.date()),
                        "eps_actual": float(actual) if pd.notna(actual) else None,
                        "eps_estimate": float(estimate) if pd.notna(estimate) else None,
                        "surprise_pct": round(float(surprise_pct), 2) if pd.notna(surprise_pct) else None,
                        "beat": bool(float(actual) > float(estimate)) if (pd.notna(actual) and pd.notna(estimate)) else None,
                    })
                result["earnings_surprises"] = surprises
    except Exception:
        pass

    # --- Quarterly financials (last 6 quarters) ---
    try:
        qfin = t.quarterly_financials
        qcf = t.quarterly_cashflow
        quarters: dict = {}
        if qfin is not None and not qfin.empty:
            for col in qfin.columns[:6]:
                ds = str(col.date())
                quarters[ds] = {
                    "revenue": _df_get(qfin, col, "Total Revenue"),
                    "gross_profit": _df_get(qfin, col, "Gross Profit"),
                    "op_income": _df_get(qfin, col, "Operating Income", "Total Operating Income As Reported"),
                    "net_income": _df_get(qfin, col, "Net Income", "Net Income Common Stockholders"),
                }
        if qcf is not None and not qcf.empty:
            for col in qcf.columns[:6]:
                ds = str(col.date())
                if ds not in quarters:
                    quarters[ds] = {}
                quarters[ds]["op_cashflow"] = _df_get(
                    qcf, col,
                    "Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                )
                quarters[ds]["free_cashflow"] = _df_get(qcf, col, "Free Cash Flow")
        result["quarterly"] = quarters
    except Exception:
        pass

    # --- Annual financials (last 4 years) ---
    try:
        fin = t.financials
        bal = t.balance_sheet
        annual: dict = {}
        if fin is not None and not fin.empty:
            for col in fin.columns[:4]:
                ds = str(col.date())
                annual[ds] = {
                    "revenue": _df_get(fin, col, "Total Revenue"),
                    "net_income": _df_get(fin, col, "Net Income", "Net Income Common Stockholders"),
                    "op_income": _df_get(fin, col, "Operating Income", "Total Operating Income As Reported"),
                    "gross_profit": _df_get(fin, col, "Gross Profit"),
                }
        if bal is not None and not bal.empty:
            for col in bal.columns[:4]:
                ds = str(col.date())
                if ds not in annual:
                    annual[ds] = {}
                equity = _df_get(
                    bal, col,
                    "Stockholders Equity", "Common Stock Equity",
                    "Total Stockholder Equity", "Total Equity Gross Minority Interest",
                )
                total_assets = _df_get(bal, col, "Total Assets")
                curr_liab = _df_get(bal, col, "Current Liabilities", "Total Current Liabilities")
                ebit = annual.get(ds, {}).get("op_income")
                ni = annual.get(ds, {}).get("net_income") or 0
                annual[ds]["roe_pct"] = round(ni / equity * 100, 2) if equity else None
                annual[ds]["roa_pct"] = round(ni / total_assets * 100, 2) if total_assets else None
                if total_assets and curr_liab:
                    cap_employed = total_assets - curr_liab
                    annual[ds]["roce_pct"] = round(ebit / cap_employed * 100, 2) if (ebit and cap_employed) else None
        result["annual"] = annual
    except Exception:
        pass

    # --- Institutional holders & insider transactions ---
    try:
        inst = t.institutional_holders
        if inst is not None and not inst.empty:
            result["institutional_holders"] = [
                {k: v for k, v in row.items() if pd.notna(v)}
                for _, row in inst.head(5).iterrows()
            ]
    except Exception:
        pass

    try:
        insider = t.insider_transactions
        if insider is not None and not insider.empty:
            result["insider_transactions"] = [
                {k: v for k, v in row.items()
                 if pd.notna(v) and k not in ("url", "filingUrl", "link")}
                for _, row in insider.head(5).iterrows()
            ]
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Format as prompt context blocks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sentiment context fetcher
# ---------------------------------------------------------------------------

def _fetch_sentiment(ticker: str) -> str:
    """Fetch news headlines, score with VADER, return a formatted VERIFIED SENTIMENT DATA block."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    today = date.today().strftime("%B %d, %Y")

    header = (
        "╔══════════════════════════════════════════════════════════════════════╗\n"
        f"║  VERIFIED SENTIMENT DATA  |  {today}  |  Ticker: {ticker}\n"
        "║  Source: yfinance news + VADER scoring\n"
        "║  USE ONLY THESE NUMBERS. DO NOT CHANGE OR INVENT ANY VALUES.\n"
        "╚══════════════════════════════════════════════════════════════════════╝\n"
    )

    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception as e:
        return header + f"ERROR: Could not fetch news: {e}\n"

    sia = SentimentIntensityAnalyzer()
    headlines = []
    for item in news_items[:15]:
        try:
            title = (
                item.get("content", {}).get("title")
                or item.get("title")
                or ""
            )
            if not title:
                continue
            sc = sia.polarity_scores(title)
            label = "POSITIVE" if sc["compound"] >= 0.05 else ("NEGATIVE" if sc["compound"] <= -0.05 else "NEUTRAL")
            headlines.append({"title": title, "compound": sc["compound"], "label": label})
        except Exception:
            continue

    if not headlines:
        return header + "No news headlines available.\n"

    compounds = [h["compound"] for h in headlines]
    avg = sum(compounds) / len(compounds)
    score10 = round((avg + 1) / 2 * 10, 1)
    pos = sum(1 for h in headlines if h["label"] == "POSITIVE")
    neg = sum(1 for h in headlines if h["label"] == "NEGATIVE")
    neu = len(headlines) - pos - neg

    if avg >= 0.20:
        overall = "STRONGLY POSITIVE"
    elif avg >= 0.05:
        overall = "POSITIVE"
    elif avg <= -0.20:
        overall = "STRONGLY NEGATIVE"
    elif avg <= -0.05:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"

    lines = [
        header,
        f"  Headlines Analysed    : {len(headlines)}",
        f"  Positive Headlines    : {pos}",
        f"  Neutral Headlines     : {neu}",
        f"  Negative Headlines    : {neg}",
        f"  Avg VADER Compound    : {avg:.4f}  (range: -1.0 to +1.0)",
        f"  Sentiment Score       : {score10}/10  (0=most negative, 5=neutral, 10=most positive)",
        f"  Overall Sentiment     : {overall}",
        "",
        "  Top Headlines (compound score):",
    ]
    for h in headlines[:10]:
        lines.append(f"    [{h['compound']:+.4f}] {h['label']:8s}  {h['title'][:100]}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Risk context fetcher
# ---------------------------------------------------------------------------

def _fetch_risk_metrics(ticker: str) -> str:
    """Compute quantitative risk metrics and return a formatted VERIFIED RISK DATA block."""
    from tools.risk_scoring import format_risk_context
    return format_risk_context(ticker)


# ---------------------------------------------------------------------------
# Format as prompt context blocks
# ---------------------------------------------------------------------------

def build_verified_context(ticker: str) -> tuple[str, str, str, str]:
    """Fetch live market data and return (technical_ctx, fundamental_ctx, sentiment_ctx, risk_ctx).

    All four strings are formatted as clearly labelled VERIFIED DATA blocks for
    direct injection into agent task prompts, preventing hallucination.
    Works for any ticker — Indian (.NS/.BO) or US/global.
    """
    cur = _currency(ticker)
    indian = _is_indian(ticker)
    today = date.today().strftime("%B %d, %Y")

    try:
        tech = _fetch_technical(ticker)
    except Exception as e:
        tech = {}
        print(f"[data_prefetch] WARNING: technical fetch failed for {ticker}: {e}")

    try:
        fund = _fetch_fundamental(ticker)
    except Exception as e:
        fund = {}
        print(f"[data_prefetch] WARNING: fundamental fetch failed for {ticker}: {e}")

    try:
        sentiment_ctx = _fetch_sentiment(ticker)
    except Exception as e:
        sentiment_ctx = f"ERROR: Could not fetch sentiment data for {ticker}: {e}\n"
        print(f"[data_prefetch] WARNING: sentiment fetch failed for {ticker}: {e}")

    # Append institutional/insider data from the already-fetched fund dict — no extra API call
    if fund:
        si_lines = ["  Institutional Holders (top 3):"]
        for h in fund.get("institutional_holders", [])[:3]:
            name = str(h.get("Holder", "N/A"))[:45]
            pct = h.get("% Out", h.get("pctHeld", None))
            try:
                pv = float(str(pct).replace("%", ""))
                pct_fmt = f"{pv * 100:.1f}%" if pv <= 1.0 else f"{pv:.1f}%"
            except Exception:
                pct_fmt = str(pct) if pct else "N/A"
            si_lines.append(f"    {name}: {pct_fmt} held")
        if not fund.get("institutional_holders"):
            si_lines.append("    N/A")
        si_lines.append("  Recent Insider Transactions (last 3):")
        for txn in fund.get("insider_transactions", [])[:3]:
            name = str(txn.get("Insider Trading", txn.get("Name", txn.get("Insider", "N/A"))))[:30]
            trans = str(txn.get("Transaction", "N/A"))[:15]
            shares = txn.get("Shares", txn.get("# Shares", "N/A"))
            date_val = str(txn.get("Start Date", txn.get("Date", ""))).split("T")[0][:10]
            si_lines.append(f"    {date_val or 'N/A'}  {name}  {trans}  Shares: {shares}")
        if not fund.get("insider_transactions"):
            si_lines.append("    N/A")
        si_lines.append("")
        sentiment_ctx = sentiment_ctx.rstrip() + "\n" + "\n".join(si_lines)

    try:
        risk_ctx = _fetch_risk_metrics(ticker)
    except Exception as e:
        risk_ctx = f"ERROR: Could not compute risk metrics for {ticker}: {e}\n"
        print(f"[data_prefetch] WARNING: risk fetch failed for {ticker}: {e}")

    # -----------------------------------------------------------------------
    # Technical context block
    # -----------------------------------------------------------------------
    if not tech:
        tech_block = (
            f"ERROR: Could not fetch technical data for {ticker}. "
            "Do not proceed without verified price data."
        )
    else:
        cp = tech["current_price"]

        # SMA200 — format with currency prefix; note if insufficient history
        if tech.get("sma200") is not None:
            sma200_display = f"{cur}{tech['sma200']:,.2f}"
            sma200_cross = "GOLDEN CROSS (SMA50 > SMA200) ✓" if tech.get("golden_cross") else "DEATH CROSS (SMA50 < SMA200) ✗"
        else:
            sma200_display = "N/A (< 200 trading days of history)"
            sma200_cross = "N/A (insufficient history for cross signal)"

        # SMA50
        sma50_display = f"{cur}{tech['sma50']:,.2f}" if tech.get("sma50") else "N/A"

        # RSI
        if tech.get("rsi") is not None:
            rsi_val = tech["rsi"]
            rsi_zone = "OVERBOUGHT (>70)" if rsi_val > 70 else ("OVERSOLD (<30)" if rsi_val < 30 else "NEUTRAL (30–70)")
            rsi_display = f"{rsi_val:.2f}  [{rsi_zone}]"
        else:
            rsi_display = "N/A"

        macd_cross = "BULLISH crossover" if tech["macd_histogram"] > 0 else "BEARISH crossover"
        vol_str = "ABOVE 20-day avg ✓" if tech["vol_above_avg"] else "below 20-day avg"

        # 52w high/low labels
        high_label = f"{tech['pct_from_high']:.1f}% below 52w high" if tech["pct_from_high"] < 0 else f"{tech['pct_from_high']:.1f}% above 52w high (new high)"
        low_label = f"{tech['pct_from_low']:.1f}% above 52w low"

        fib_lines = "\n".join(
            f"    Fib {level}: {cur}{v:,.2f}" for level, v in tech["fibs"].items()
        )

        tech_block = f"""
╔══════════════════════════════════════════════════════════════╗
║         VERIFIED TECHNICAL DATA — {today}
║  Source: yfinance live data  |  Ticker: {ticker}
║  USE ONLY THESE NUMBERS. DO NOT CHANGE OR INVENT ANY VALUES.
╚══════════════════════════════════════════════════════════════╝

PRICE (as of {tech['price_date']}):
  Current Price       : {cur}{cp:,.2f}
  52-Week High        : {cur}{tech['high_52w']:,.2f}  ({high_label})
  52-Week Low         : {cur}{tech['low_52w']:,.2f}  ({low_label})
  60-Day Support      : {cur}{tech['support_60d']:,.2f}
  60-Day Resistance   : {cur}{tech['resistance_60d']:,.2f}

MOVING AVERAGES:
  SMA 50              : {sma50_display}  — Price is {'ABOVE ✓' if tech['price_above_sma50'] else 'BELOW ✗'} SMA50
  SMA 200             : {sma200_display}  — Price is {'ABOVE ✓' if tech['price_above_sma200'] else 'BELOW ✗'} SMA200
  EMA 20              : {cur}{tech['ema20']:,.2f}
  Cross Status        : {sma200_cross}

MOMENTUM:
  RSI (14)            : {rsi_display}
  MACD                : {tech['macd']:.4f}
  MACD Signal         : {tech['macd_signal']:.4f}
  MACD Histogram      : {tech['macd_histogram']:.4f}  [{macd_cross}]

BOLLINGER BANDS (20-day):
  Upper Band          : {cur}{tech['bb_upper']:,.2f}
  Middle Band         : {cur}{tech['bb_mid']:,.2f}
  Lower Band          : {cur}{tech['bb_lower']:,.2f}
  %B                  : {tech['bb_pct_b']:.3f}  (>1 = above upper band; <0 = below lower band)

VOLATILITY:
  ATR (14)            : {cur}{tech['atr']:,.2f}  ({_f(tech.get('atr_pct'), 2)}% of price)

VOLUME:
  Current Volume      : {tech['current_vol']:,.0f}
  20-Day Avg Volume   : {tech['vol_sma20']:,.0f}
  Volume Signal       : {vol_str}
  OBV Trend (20d)     : {tech['obv_trend'].upper()}

FIBONACCI RETRACEMENT LEVELS (from 52-week range):
{fib_lines}
""".strip()

    # -----------------------------------------------------------------------
    # Fundamental context block
    # -----------------------------------------------------------------------
    if not fund:
        fund_block = (
            f"ERROR: Could not fetch fundamental data for {ticker}. "
            "Do not proceed without verified fundamental data."
        )
    else:
        # Use analyst_target_current price if available, else live technical price
        cp_anchor = fund.get("analyst_target_current") or (tech.get("current_price") if tech else None)

        # Analyst target upside/downside
        mean_target = fund.get("analyst_target_mean")
        if mean_target and cp_anchor and float(cp_anchor) > 0:
            upside = (float(mean_target) - float(cp_anchor)) / float(cp_anchor) * 100
            upside_str = f"  ({upside:+.1f}% from current)"
        else:
            upside_str = ""

        # Dividend yield display
        dv_pct = fund.get("dividend_yield_pct")
        dv_str = _pct(dv_pct) if dv_pct is not None else "N/A"

        # Quarterly financials table
        q_lines = []
        for ds, q in sorted(fund.get("quarterly", {}).items(), reverse=True):
            rev = _money(q.get("revenue"), prefix=cur, indian=indian)
            ni = _money(q.get("net_income"), prefix=cur, indian=indian)
            ocf = _money(q.get("op_cashflow"), prefix=cur, indian=indian)
            q_lines.append(f"    {ds}  Revenue: {rev}  Net Income: {ni}  Op CF: {ocf}")
        quarterly_str = "\n".join(q_lines) if q_lines else "    N/A"

        # Annual financials table
        a_lines = []
        for ds, a in sorted(fund.get("annual", {}).items(), reverse=True):
            rev = _money(a.get("revenue"), prefix=cur, indian=indian)
            ni = _money(a.get("net_income"), prefix=cur, indian=indian)
            roe = _pct(a.get("roe_pct")) if a.get("roe_pct") is not None else "N/A"
            roce = _pct(a.get("roce_pct")) if a.get("roce_pct") is not None else "N/A"
            a_lines.append(f"    {ds}  Revenue: {rev}  Net Income: {ni}  ROE: {roe}  ROCE: {roce}")
        annual_str = "\n".join(a_lines) if a_lines else "    N/A"

        # Earnings surprises
        es_lines = []
        for s in fund.get("earnings_surprises", []):
            beat_str = "BEAT ✓" if s.get("beat") else ("MISS ✗" if s.get("beat") is False else "N/A")
            surp = f"{s['surprise_pct']:+.1f}%" if s.get("surprise_pct") is not None else "N/A"
            es_lines.append(
                f"    {s['quarter']}  Actual EPS: {_f(s.get('eps_actual'))}  "
                f"Estimate: {_f(s.get('eps_estimate'))}  Surprise: {surp}  [{beat_str}]"
            )
        es_str = "\n".join(es_lines) if es_lines else "    N/A"

        # Institutional holders
        inst_lines = []
        for h in fund.get("institutional_holders", [])[:5]:
            name = str(h.get("Holder", "N/A"))[:45]
            pct = h.get("% Out", h.get("pctHeld", None))
            try:
                pv = float(str(pct).replace("%", ""))
                pct_fmt = f"{pv * 100:.1f}%" if pv <= 1.0 else f"{pv:.1f}%"
            except Exception:
                pct_fmt = str(pct) if pct else "N/A"
            inst_lines.append(f"    {name}: {pct_fmt} held")
        inst_str = "\n".join(inst_lines) if inst_lines else "    N/A"

        # Insider transactions
        insider_lines = []
        for txn in fund.get("insider_transactions", [])[:5]:
            name = str(txn.get("Insider Trading", txn.get("Name", txn.get("Insider", "N/A"))))[:30]
            trans = str(txn.get("Transaction", txn.get("Transaction Type", "N/A")))[:20]
            shares = txn.get("Shares", txn.get("# Shares", "N/A"))
            date_val = str(txn.get("Start Date", txn.get("Date", txn.get("Transaction Date", "")))).split("T")[0][:10]
            insider_lines.append(f"    {date_val or 'N/A'}  {name}  {trans}  Shares: {shares}")
        insider_str = "\n".join(insider_lines) if insider_lines else "    N/A"

        # Analyst ratings
        r = fund.get("ratings", {})
        total_r = fund.get("ratings_total", 0)
        pct_bull = fund.get("pct_bullish")
        if total_r:
            ratings_str = (
                f"Strong Buy: {r.get('strongBuy', 0)}  Buy: {r.get('buy', 0)}  "
                f"Hold: {r.get('hold', 0)}  Sell: {r.get('sell', 0)}  "
                f"Strong Sell: {r.get('strongSell', 0)}  "
                f"(Total: {total_r}, {pct_bull}% bullish)"
            )
        else:
            ratings_str = "N/A"

        fund_block = f"""
╔══════════════════════════════════════════════════════════════╗
║       VERIFIED FUNDAMENTAL DATA — {today}
║  Source: yfinance live data  |  Ticker: {ticker}
║  USE ONLY THESE NUMBERS. DO NOT CHANGE OR INVENT ANY VALUES.
╚══════════════════════════════════════════════════════════════╝

COMPANY:
  Name                : {fund.get('name', ticker)}
  Sector              : {fund.get('sector', 'N/A')}
  Industry            : {fund.get('industry', 'N/A')}
  Market Cap          : {_money(fund.get('market_cap'), prefix=cur, indian=indian)}

VALUATION RATIOS (current):
  P/E Trailing        : {_f(fund.get('trailing_pe'))}
  P/E Forward         : {_f(fund.get('forward_pe'))}
  Price-to-Book (P/B) : {_f(fund.get('pb'))}
  EV/EBITDA           : {_f(fund.get('ev_ebitda'))}
  Dividend Yield      : {dv_str}
  Beta                : {_f(fund.get('beta'))}
  Trailing EPS        : {cur}{_f(fund.get('trailing_eps'))}

PROFITABILITY (trailing twelve months):
  Gross Margin        : {_pct_frac(fund.get('gross_margin'))}
  Operating Margin    : {_pct_frac(fund.get('op_margin'))}
  Net Profit Margin   : {_pct_frac(fund.get('net_margin'))}
  Return on Equity    : {_pct_frac(fund.get('roe'))}
  Return on Assets    : {_pct_frac(fund.get('roa'))}

GROWTH (year-over-year, trailing):
  Revenue Growth      : {_pct_frac(fund.get('revenue_growth_yoy'))}
  Earnings Growth     : {_pct_frac(fund.get('earnings_growth_yoy'))}

BALANCE SHEET:
  Debt-to-Equity      : {_f(fund.get('debt_to_equity'))}
  Current Ratio       : {_f(fund.get('current_ratio'))}

ANALYST CONSENSUS ({fund.get('analyst_count', 'N/A')} analysts):
  Ratings             : {ratings_str}
  Target — Mean       : {cur}{_f(mean_target)}{upside_str}
  Target — High       : {cur}{_f(fund.get('analyst_target_high'))}
  Target — Low        : {cur}{_f(fund.get('analyst_target_low'))}

QUARTERLY FINANCIALS (most recent quarters first):
{quarterly_str}

ANNUAL FINANCIALS (most recent years first):
{annual_str}

EARNINGS SURPRISES (most recent reported quarters):
{es_str}

INSTITUTIONAL HOLDERS (top 5):
{inst_str}

RECENT INSIDER TRANSACTIONS (last 5):
{insider_str}
""".strip()

    return tech_block, fund_block, sentiment_ctx, risk_ctx
