"""Resolves company names or raw tickers to a canonical ticker symbol with exchange info."""

import yfinance as yf


# yfinance Search returns exchange codes: "NSI" for NSE, "BOM" for BSE
_NSE_EXCHANGES = {"NSI", "NSE"}
_BSE_EXCHANGES = {"BOM", "BSE"}


def resolve_ticker(company_name: str) -> dict:
    """Return {"ticker": str, "exchange": str, "full_name": str} for a company name or ticker.

    Tries yfinance Search first. Appends .NS for NSE stocks, .BO for BSE.
    Falls back to treating the raw input as a ticker if search fails or returns nothing.
    """
    try:
        results = yf.Search(company_name, max_results=5).quotes
        if not results:
            raise ValueError("empty results")

        best = results[0]
        symbol: str = best.get("symbol", company_name).upper()
        exchange: str = best.get("exchange", "")
        full_name: str = (
            best.get("longname")
            or best.get("shortname")
            or company_name
        )

        if exchange in _NSE_EXCHANGES and not symbol.endswith(".NS"):
            symbol = symbol + ".NS"
        elif exchange in _BSE_EXCHANGES and not symbol.endswith(".BO"):
            symbol = symbol + ".BO"

        return {"ticker": symbol, "exchange": exchange, "full_name": full_name}

    except Exception:
        # If the input looks like a ticker (all caps, short) use it directly
        raw = company_name.strip().upper()
        return {"ticker": raw, "exchange": "UNKNOWN", "full_name": company_name}
