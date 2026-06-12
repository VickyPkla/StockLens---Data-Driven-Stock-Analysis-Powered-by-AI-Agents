"""yfinance + ta library wrappers as CrewAI tools."""

import json
import pandas as pd
import yfinance as yf
from crewai.tools import tool

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator, denominator) -> float | None:
    try:
        n, d = float(numerator), float(denominator)
        return round(n / d, 4) if d != 0 else None
    except (TypeError, ValueError):
        return None


def _df_get(df: pd.DataFrame, col, *keys) -> float | None:
    """Try multiple row-label keys in a yfinance DataFrame column; return first non-null match."""
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key, col]
                if pd.notna(val):
                    return _safe_float(val)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Price History
# ---------------------------------------------------------------------------

@tool
def get_price_history(ticker: str) -> str:
    """Fetch 1 year of daily OHLCV price history for a stock ticker using yfinance.

    Returns a JSON array of records with Date, Open, High, Low, Close, Volume.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        return json.dumps({"error": "No price history available"})
    hist = hist.reset_index()
    hist["Date"] = hist["Date"].astype(str)
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    cols = [c for c in cols if c in hist.columns]
    records = hist[cols].fillna(0).to_dict(orient="records")
    return json.dumps(records[:252], default=str)  # cap at ~1 trading year


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

@tool
def get_technical_indicators(ticker: str) -> str:
    """Calculate key technical indicators for a stock ticker using 1 year of price data.

    Computes and returns the latest values for:
    - RSI (14-period)
    - MACD (12, 26, 9): value, signal line, histogram
    - Bollinger Bands (20): upper, middle, lower, %B
    - SMA 50 and SMA 200
    - EMA 20
    - ATR (14) for volatility
    - Volume SMA (20)
    - Booleans: price_above_sma50, price_above_sma200, golden_cross (SMA50 > SMA200)
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        return json.dumps({"error": "No price data available for indicators"})

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    rsi_val = _safe_float(RSIIndicator(close, window=14).rsi().iloc[-1])

    macd_obj = MACD(close, window_fast=12, window_slow=26, window_sign=9)
    macd_val = _safe_float(macd_obj.macd().iloc[-1])
    macd_signal = _safe_float(macd_obj.macd_signal().iloc[-1])
    macd_diff = _safe_float(macd_obj.macd_diff().iloc[-1])

    bb = BollingerBands(close, window=20)
    bb_upper = _safe_float(bb.bollinger_hband().iloc[-1])
    bb_middle = _safe_float(bb.bollinger_mavg().iloc[-1])
    bb_lower = _safe_float(bb.bollinger_lband().iloc[-1])
    bb_pct_b = _safe_float(bb.bollinger_pband().iloc[-1])

    sma50 = _safe_float(SMAIndicator(close, window=50).sma_indicator().iloc[-1])
    sma200_series = SMAIndicator(close, window=200).sma_indicator()
    sma200 = _safe_float(sma200_series.iloc[-1])
    ema20 = _safe_float(EMAIndicator(close, window=20).ema_indicator().iloc[-1])
    atr = _safe_float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])

    vol_sma20 = _safe_float(volume.rolling(20).mean().iloc[-1])
    current_price = _safe_float(close.iloc[-1])
    current_volume = _safe_float(volume.iloc[-1])

    # Support/resistance: recent swing highs and lows
    recent = close.tail(60)
    support = _safe_float(recent.min())
    resistance = _safe_float(recent.max())

    # OBV
    obv_series = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_current = _safe_float(obv_series.iloc[-1])
    obv_prev20 = _safe_float(obv_series.iloc[-21]) if len(obv_series) >= 21 else None
    obv_trend = "rising" if (obv_current and obv_prev20 and obv_current > obv_prev20) else "falling"

    # 52-week context
    high_52w = _safe_float(hist["High"].max())
    low_52w = _safe_float(hist["Low"].min())
    pct_from_52w_high = round(((current_price - high_52w) / high_52w) * 100, 2) if (current_price and high_52w) else None
    pct_from_52w_low = round(((current_price - low_52w) / low_52w) * 100, 2) if (current_price and low_52w) else None

    # Fibonacci retracement levels from 52-week range
    fib_range = (high_52w - low_52w) if (high_52w and low_52w) else None
    fibonacci_levels = {
        "fib_23_6": _safe_float(high_52w - fib_range * 0.236),
        "fib_38_2": _safe_float(high_52w - fib_range * 0.382),
        "fib_50_0": _safe_float(high_52w - fib_range * 0.500),
        "fib_61_8": _safe_float(high_52w - fib_range * 0.618),
        "fib_78_6": _safe_float(high_52w - fib_range * 0.786),
    } if fib_range else None

    result = {
        "ticker": ticker,
        "price_date": str(hist.index[-1].date()),
        "current_price": current_price,
        "rsi_14": rsi_val,
        "macd": macd_val,
        "macd_signal": macd_signal,
        "macd_histogram": macd_diff,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "bb_pct_b": bb_pct_b,
        "sma_50": sma50,
        "sma_200": sma200,
        "ema_20": ema20,
        "atr_14": atr,
        "volume_sma_20": vol_sma20,
        "current_volume": current_volume,
        "support_60d": support,
        "resistance_60d": resistance,
        "price_above_sma50": bool(current_price and sma50 and current_price > sma50),
        "price_above_sma200": bool(current_price and sma200 and current_price > sma200),
        "golden_cross": bool(sma50 and sma200 and sma50 > sma200),
        "volume_above_avg": bool(current_volume and vol_sma20 and current_volume > vol_sma20),
        "obv": obv_current,
        "obv_trend_20d": obv_trend,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_52w_high": pct_from_52w_high,
        "pct_from_52w_low": pct_from_52w_low,
        "fibonacci_levels": fibonacci_levels,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Stock Info
# ---------------------------------------------------------------------------

@tool
def get_stock_info(ticker: str) -> str:
    """Fetch key fundamental info for a stock ticker using yfinance.

    Returns a filtered subset of yfinance .info including: longName, sector,
    industry, marketCap, trailingPE, forwardPE, priceToBook, dividendYield,
    beta, fiftyTwoWeekHigh, fiftyTwoWeekLow, currentPrice, recommendationMean,
    numberOfAnalystOpinions.
    """
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception as e:
        return json.dumps({"error": str(e)})

    keys = [
        "longName", "sector", "industry", "marketCap",
        "trailingPE", "forwardPE", "priceToBook", "dividendYield",
        "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "currentPrice",
        "recommendationMean", "numberOfAnalystOpinions",
        "trailingEps", "returnOnEquity", "returnOnAssets",
        "revenueGrowth", "earningsGrowth", "debtToEquity",
    ]
    filtered = {k: info.get(k) for k in keys}
    return json.dumps(filtered, indent=2, default=str)


# ---------------------------------------------------------------------------
# Recent News
# ---------------------------------------------------------------------------

@tool
def get_recent_news(ticker: str) -> str:
    """Fetch the 5 most recent news headlines for a stock ticker using yfinance.

    Returns a JSON array with title, publisher, and publish date for each article.
    """
    t = yf.Ticker(ticker)
    try:
        news = t.news
    except Exception as e:
        return json.dumps({"error": str(e)})

    if not news:
        return json.dumps({"message": "No recent news found"})

    results = []
    for item in news[:5]:
        content = item.get("content", {})
        results.append({
            "title": content.get("title", item.get("title", "N/A")),
            "publisher": content.get("provider", {}).get("displayName", item.get("publisher", "N/A")),
            "publishedAt": content.get("pubDate", str(item.get("providerPublishTime", ""))),
            "summary": content.get("summary", "")[:200] if content.get("summary") else "",
        })
    return json.dumps(results, indent=2, default=str)


# ---------------------------------------------------------------------------
# Earnings Surprises
# ---------------------------------------------------------------------------

@tool
def get_earnings_surprises(ticker: str) -> str:
    """Fetch the last 4 quarters of earnings surprises (actual EPS vs estimated EPS) for a stock.

    Returns date, EPS actual, EPS estimate, surprise percentage, and whether the company beat.
    Consistent beats indicate high earnings quality and management credibility.
    """
    t = yf.Ticker(ticker)
    try:
        dates = t.earnings_dates
        if dates is None or dates.empty:
            return json.dumps({"message": "No earnings surprise data available"})
        # yfinance uses "Reported EPS" (not "EPS Actual")
        actual_col = "Reported EPS" if "Reported EPS" in dates.columns else "EPS Actual"
        past = dates[dates[actual_col].notna()].head(4)
        results = []
        for idx, row in past.iterrows():
            actual = row.get(actual_col)
            estimate = row.get("EPS Estimate")
            surprise_pct = _safe_float(row.get("Surprise(%)"))
            if surprise_pct is None and pd.notna(actual) and pd.notna(estimate) and float(estimate) != 0:
                surprise_pct = round(((float(actual) - float(estimate)) / abs(float(estimate))) * 100, 2)
            results.append({
                "quarter": str(idx.date()),
                "eps_actual": _safe_float(actual),
                "eps_estimate": _safe_float(estimate),
                "surprise_pct": surprise_pct,
                "beat": bool(float(actual) > float(estimate)) if (pd.notna(actual) and pd.notna(estimate)) else None,
            })
        return json.dumps({"source": "yfinance", "data": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Analyst Consensus
# ---------------------------------------------------------------------------

@tool
def get_analyst_consensus(ticker: str) -> str:
    """Fetch analyst consensus: buy/hold/sell rating breakdown and price targets for a stock.

    Returns strong buy, buy, hold, sell, strong sell counts plus mean/high/low price targets.
    """
    t = yf.Ticker(ticker)
    result = {}
    try:
        rec = t.recommendations_summary
        if rec is not None and not rec.empty:
            latest = rec.iloc[0]
            ratings = {
                "strongBuy": int(latest.get("strongBuy", 0)),
                "buy": int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)),
                "strongSell": int(latest.get("strongSell", 0)),
            }
            total = sum(ratings.values())
            bullish = ratings["strongBuy"] + ratings["buy"]
            result["ratings"] = ratings
            result["total_analysts"] = total
            result["pct_bullish"] = round(bullish / total * 100, 1) if total else None
    except Exception:
        pass
    try:
        targets = t.analyst_price_targets
        if targets is not None:
            result["price_targets"] = {
                "current": _safe_float(targets.get("current")),
                "mean": _safe_float(targets.get("mean")),
                "high": _safe_float(targets.get("high")),
                "low": _safe_float(targets.get("low")),
            }
    except Exception:
        pass
    if not result:
        return json.dumps({"message": "No analyst consensus data available"})
    return json.dumps({"source": "yfinance", "data": result}, indent=2)


# ---------------------------------------------------------------------------
# Quarterly Financials
# ---------------------------------------------------------------------------

@tool
def get_quarterly_financials(ticker: str) -> str:
    """Fetch last 6 quarters of revenue, net income, operating income, and operating cash flow.

    Useful for spotting recent acceleration or deceleration in growth that annual data misses.
    """
    t = yf.Ticker(ticker)
    try:
        qfin = t.quarterly_financials
        qcf = t.quarterly_cashflow
        rows = {}
        if qfin is not None and not qfin.empty:
            for col in qfin.columns[:6]:
                date_str = str(col.date())
                rows[date_str] = {
                    "total_revenue": _df_get(qfin, col, "Total Revenue"),
                    "gross_profit": _df_get(qfin, col, "Gross Profit"),
                    "operating_income": _df_get(qfin, col, "Operating Income", "Total Operating Income As Reported"),
                    "net_income": _df_get(qfin, col, "Net Income", "Net Income Common Stockholders"),
                }
        if qcf is not None and not qcf.empty:
            for col in qcf.columns[:6]:
                date_str = str(col.date())
                if date_str not in rows:
                    rows[date_str] = {}
                rows[date_str]["operating_cashflow"] = _df_get(
                    qcf, col, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"
                )
                rows[date_str]["free_cashflow"] = _df_get(qcf, col, "Free Cash Flow")
        if not rows:
            return json.dumps({"message": "No quarterly financial data available"})
        return json.dumps({"source": "yfinance", "data": rows}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Peer Comparison
# ---------------------------------------------------------------------------

@tool
def get_peer_comparison(tickers: str) -> str:
    """Fetch key valuation and profitability metrics for a comma-separated list of peer tickers.

    Pass 2-4 ticker symbols, e.g. "INFY.NS,WIPRO.NS,HCLTECH.NS" or "MSFT,GOOGL,META".
    Returns P/E, P/B, EV/EBITDA, revenue growth, net margin, ROE, and market cap for each peer.
    Use this to benchmark the main stock's valuation against direct competitors.
    """
    ticker_list = [tk.strip() for tk in tickers.split(",") if tk.strip()]
    keys = [
        "longName", "trailingPE", "forwardPE", "priceToBook",
        "enterpriseToEbitda", "revenueGrowth", "netMargins",
        "returnOnEquity", "returnOnAssets", "grossMargins",
        "operatingMargins", "marketCap", "beta",
    ]
    results = []
    for tk in ticker_list:
        try:
            info = yf.Ticker(tk).info
            row = {"ticker": tk}
            for k in keys:
                row[k] = info.get(k)
            results.append(row)
        except Exception as e:
            results.append({"ticker": tk, "error": str(e)})
    return json.dumps({"source": "yfinance", "data": results}, indent=2, default=str)


# ---------------------------------------------------------------------------
# Institutional & Insider Data
# ---------------------------------------------------------------------------

@tool
def get_institutional_insider_data(ticker: str) -> str:
    """Fetch top institutional holders and recent insider transactions for a stock.

    Institutional ownership level and insider buying/selling are strong sentiment signals.
    Returns top 10 institutional holders (name, % held, shares) and last 10 insider transactions.
    """
    t = yf.Ticker(ticker)
    result = {}
    try:
        inst = t.institutional_holders
        if inst is not None and not inst.empty:
            result["institutional_holders"] = inst.head(10).to_dict(orient="records")
    except Exception:
        pass
    try:
        major = t.major_holders
        if major is not None and not major.empty:
            result["major_holders_summary"] = major.to_dict(orient="records")
    except Exception:
        pass
    try:
        insider = t.insider_transactions
        if insider is not None and not insider.empty:
            result["recent_insider_transactions"] = insider.head(10).to_dict(orient="records")
    except Exception:
        pass
    if not result:
        return json.dumps({"message": "No institutional or insider data available"})
    return json.dumps({"source": "yfinance", "data": result}, indent=2, default=str)


# ---------------------------------------------------------------------------
# Return Metrics Trend
# ---------------------------------------------------------------------------

@tool
def get_return_metrics_trend(ticker: str) -> str:
    """Compute ROE, ROA, and ROCE trend over the last 4 fiscal years for a stock.

    Return on Equity (ROE), Return on Assets (ROA), and Return on Capital Employed (ROCE)
    trended over time reveal whether the business is compounding quality or eroding it.
    A consistently improving trend is a strong fundamental signal.
    """
    t = yf.Ticker(ticker)
    try:
        fin = t.financials
        bal = t.balance_sheet
        if fin is None or fin.empty or bal is None or bal.empty:
            return json.dumps({"message": "Insufficient financial data for return metrics"})

        results = []
        for col in fin.columns[:4]:
            if col not in bal.columns:
                continue
            year_str = str(col.date())

            net_income = _df_get(
                fin, col, "Net Income", "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            )
            ebit = _df_get(fin, col, "EBIT", "Operating Income", "Total Operating Income As Reported")
            total_assets = _df_get(bal, col, "Total Assets")
            equity = _df_get(
                bal, col, "Stockholders Equity", "Common Stock Equity",
                "Total Stockholder Equity", "Total Equity Gross Minority Interest",
            )
            curr_liab = _df_get(
                bal, col, "Current Liabilities", "Total Current Liabilities",
            )

            roe = _safe_ratio(net_income, equity)
            roa = _safe_ratio(net_income, total_assets)
            capital_employed = (float(total_assets) - float(curr_liab)) if (total_assets and curr_liab) else None
            roce = _safe_ratio(ebit, capital_employed) if capital_employed else None

            results.append({
                "year": year_str,
                "roe_pct": round(roe * 100, 2) if roe is not None else None,
                "roa_pct": round(roa * 100, 2) if roa is not None else None,
                "roce_pct": round(roce * 100, 2) if roce is not None else None,
            })

        if not results:
            return json.dumps({"message": "Could not compute return metrics from available data"})
        return json.dumps({"source": "yfinance", "data": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
