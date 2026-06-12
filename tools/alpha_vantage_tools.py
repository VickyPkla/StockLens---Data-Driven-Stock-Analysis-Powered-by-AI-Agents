"""Alpha Vantage API wrappers as CrewAI tools.

Each tool sleeps 12 seconds before calling the API to stay within the free-tier
rate limit of 5 requests/minute. Indian tickers (.NS/.BO) are often unsupported
by Alpha Vantage; in those cases the tool silently falls back to yfinance data.
"""

import json
import requests
import yfinance as yf
from crewai.tools import tool

from config.settings import ALPHA_VANTAGE_API_KEY, av_rate_limit

_AV_BASE = "https://www.alphavantage.co/query"


def _av_get(params: dict) -> dict:
    av_rate_limit()
    params["apikey"] = ALPHA_VANTAGE_API_KEY
    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _av_empty(data: dict, *required_keys: str) -> bool:
    """Return True when Alpha Vantage responded with an error or empty payload."""
    if not data:
        return True
    if "Information" in data or "Note" in data or "Error Message" in data:
        return True
    for key in required_keys:
        val = data.get(key)
        if isinstance(val, list) and len(val) > 0:
            return False
    return True


def _safe_ratio(numerator, denominator) -> float | None:
    try:
        n, d = float(numerator), float(denominator)
        return round(n / d, 4) if d != 0 else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Income Statement
# ---------------------------------------------------------------------------

@tool
def get_income_statement(ticker: str) -> str:
    """Fetch annual income statement data for a stock ticker.

    Returns Revenue, Net Income, EPS, Gross Profit Margin, and Net Profit Margin
    for the last 4 fiscal years as a JSON string. Falls back to yfinance for
    Indian tickers (.NS / .BO) that Alpha Vantage does not cover.
    """
    data = _av_get({"function": "INCOME_STATEMENT", "symbol": ticker})

    if not _av_empty(data, "annualReports"):
        reports = data["annualReports"][:4]
        result = []
        for r in reports:
            rev = r.get("totalRevenue")
            gp = r.get("grossProfit")
            ni = r.get("netIncome")
            result.append({
                "fiscalDateEnding": r.get("fiscalDateEnding"),
                "totalRevenue": rev,
                "netIncome": ni,
                "eps": r.get("reportedEPS"),
                "grossProfitMargin": _safe_ratio(gp, rev),
                "netProfitMargin": _safe_ratio(ni, rev),
                "operatingIncome": r.get("operatingIncome"),
                "operatingMargin": _safe_ratio(r.get("operatingIncome"), rev),
            })
        return json.dumps({"source": "alpha_vantage", "data": result}, indent=2)

    # Fallback: yfinance
    t = yf.Ticker(ticker)
    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            fin_dict = fin.fillna(0).to_dict()
            return json.dumps(
                {"source": "yfinance_fallback", "data": fin_dict}, indent=2, default=str
            )
    except Exception as e:
        return json.dumps({"error": str(e), "source": "yfinance_fallback"})

    return json.dumps({"error": "No income statement data available"})


# ---------------------------------------------------------------------------
# Balance Sheet
# ---------------------------------------------------------------------------

@tool
def get_balance_sheet(ticker: str) -> str:
    """Fetch annual balance sheet data for a stock ticker.

    Returns Total Assets, Total Liabilities, Shareholder Equity, Debt-to-Equity
    ratio, and Current Ratio for the last 4 fiscal years. Falls back to yfinance
    for Indian tickers that Alpha Vantage does not cover.
    """
    data = _av_get({"function": "BALANCE_SHEET", "symbol": ticker})

    if not _av_empty(data, "annualReports"):
        reports = data["annualReports"][:4]
        result = []
        for r in reports:
            assets = r.get("totalAssets")
            liabilities = r.get("totalLiabilities")
            equity = r.get("totalShareholderEquity")
            current_assets = r.get("totalCurrentAssets")
            current_liab = r.get("totalCurrentLiabilities")
            total_debt = r.get("longTermDebt") or r.get("shortLongTermDebtTotal")
            result.append({
                "fiscalDateEnding": r.get("fiscalDateEnding"),
                "totalAssets": assets,
                "totalLiabilities": liabilities,
                "shareholderEquity": equity,
                "debtToEquity": _safe_ratio(total_debt, equity),
                "currentRatio": _safe_ratio(current_assets, current_liab),
                "longTermDebt": total_debt,
            })
        return json.dumps({"source": "alpha_vantage", "data": result}, indent=2)

    t = yf.Ticker(ticker)
    try:
        bal = t.balance_sheet
        if bal is not None and not bal.empty:
            return json.dumps(
                {"source": "yfinance_fallback", "data": bal.fillna(0).to_dict()},
                indent=2, default=str,
            )
    except Exception as e:
        return json.dumps({"error": str(e), "source": "yfinance_fallback"})

    return json.dumps({"error": "No balance sheet data available"})


# ---------------------------------------------------------------------------
# Cash Flow
# ---------------------------------------------------------------------------

@tool
def get_cash_flow(ticker: str) -> str:
    """Fetch annual cash flow statement data for a stock ticker.

    Returns Operating Cash Flow, Capital Expenditure, and Free Cash Flow
    for the last 4 fiscal years. Falls back to yfinance for Indian tickers.
    """
    data = _av_get({"function": "CASH_FLOW", "symbol": ticker})

    if not _av_empty(data, "annualReports"):
        reports = data["annualReports"][:4]
        result = []
        for r in reports:
            ocf_raw = r.get("operatingCashflow")
            capex_raw = r.get("capitalExpenditures")
            try:
                ocf = float(ocf_raw) if ocf_raw else None
                capex = float(capex_raw) if capex_raw else None
                fcf = round(ocf - abs(capex), 2) if (ocf is not None and capex is not None) else None
            except (TypeError, ValueError):
                ocf = capex = fcf = None
            result.append({
                "fiscalDateEnding": r.get("fiscalDateEnding"),
                "operatingCashflow": ocf,
                "capitalExpenditures": capex,
                "freeCashFlow": fcf,
                "dividendPayout": r.get("dividendPayout"),
            })
        return json.dumps({"source": "alpha_vantage", "data": result}, indent=2)

    t = yf.Ticker(ticker)
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            return json.dumps(
                {"source": "yfinance_fallback", "data": cf.fillna(0).to_dict()},
                indent=2, default=str,
            )
    except Exception as e:
        return json.dumps({"error": str(e), "source": "yfinance_fallback"})

    return json.dumps({"error": "No cash flow data available"})


# ---------------------------------------------------------------------------
# Company Overview
# ---------------------------------------------------------------------------

@tool
def get_company_overview(ticker: str) -> str:
    """Fetch company overview and key valuation metrics for a stock ticker.

    Returns Sector, Industry, Market Cap, P/E, P/B, EPS, 52-week high/low,
    analyst target price, and company description. Falls back to yfinance
    for Indian tickers that Alpha Vantage does not support.
    """
    data = _av_get({"function": "OVERVIEW", "symbol": ticker})

    if data and "Symbol" in data and data.get("Name"):
        overview = {
            "name": data.get("Name"),
            "symbol": data.get("Symbol"),
            "sector": data.get("Sector"),
            "industry": data.get("Industry"),
            "marketCap": data.get("MarketCapitalization"),
            "peRatio": data.get("PERatio"),
            "pbRatio": data.get("PriceToBookRatio"),
            "eps": data.get("EPS"),
            "forwardPE": data.get("ForwardPE"),
            "dividendYield": data.get("DividendYield"),
            "beta": data.get("Beta"),
            "52weekHigh": data.get("52WeekHigh"),
            "52weekLow": data.get("52WeekLow"),
            "analystTargetPrice": data.get("AnalystTargetPrice"),
            "description": data.get("Description", "")[:500],
            "source": "alpha_vantage",
        }
        return json.dumps(overview, indent=2)

    # Fallback: yfinance .info
    t = yf.Ticker(ticker)
    try:
        info = t.info
        overview = {
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "pbRatio": info.get("priceToBook"),
            "eps": info.get("trailingEps"),
            "dividendYield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52weekHigh": info.get("fiftyTwoWeekHigh"),
            "52weekLow": info.get("fiftyTwoWeekLow"),
            "analystTargetPrice": info.get("targetMeanPrice"),
            "description": info.get("longBusinessSummary", "")[:500],
            "source": "yfinance_fallback",
        }
        return json.dumps(overview, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
