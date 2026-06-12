"""Resolves company names or raw tickers to a canonical ticker symbol with exchange info."""

import yfinance as yf


# yfinance Search returns exchange codes: "NSI" for NSE, "BOM" for BSE
_NSE_EXCHANGES = {"NSI", "NSE"}
_BSE_EXCHANGES = {"BOM", "BSE"}
_INDIAN_EXCHANGES = _NSE_EXCHANGES | _BSE_EXCHANGES

_EXCHANGE_DISPLAY = {
    "NSI": "NSE — National Stock Exchange (India)",
    "NSE": "NSE — National Stock Exchange (India)",
    "BOM": "BSE — Bombay Stock Exchange (India)",
    "BSE": "BSE — Bombay Stock Exchange (India)",
    "NMS": "NASDAQ",
    "NYQ": "NYSE",
    "NGM": "NASDAQ Global Market",
    "NCM": "NASDAQ Capital Market",
    "ASE": "NYSE American",
    "PCX": "NYSE Arca",
}


def _normalise_symbol(symbol: str, exchange: str) -> str:
    if exchange in _NSE_EXCHANGES and not symbol.endswith(".NS"):
        return symbol + ".NS"
    if exchange in _BSE_EXCHANGES and not symbol.endswith(".BO"):
        return symbol + ".BO"
    return symbol


def resolve_listings(company_name: str) -> dict:
    """Detect whether a company has both an Indian listing and a non-Indian listing.

    Returns one of two shapes:
      {"status": "ok",     "ticker": str, "exchange": str, "full_name": str}
      {"status": "choose", "options": [{"ticker", "exchange", "exchange_name", "full_name"}, ...]}

    When both an Indian (NSE/BSE) result and a non-Indian result appear in the top
    yfinance Search results the caller should let the user pick which listing to analyse.
    """
    try:
        results = yf.Search(company_name, max_results=10).quotes
        if not results:
            raise ValueError("empty results")

        indian: list[dict] = []
        other: list[dict] = []

        for r in results:
            symbol = r.get("symbol", "").upper()
            if not symbol:
                continue
            exchange = r.get("exchange", "")
            full_name = r.get("longname") or r.get("shortname") or company_name
            symbol = _normalise_symbol(symbol, exchange)

            entry = {
                "ticker": symbol,
                "exchange": exchange,
                "exchange_name": _EXCHANGE_DISPLAY.get(exchange, exchange or "Unknown Exchange"),
                "full_name": full_name,
            }
            if exchange in _INDIAN_EXCHANGES:
                indian.append(entry)
            elif exchange:
                other.append(entry)

        # Prefer NSE over BSE when both Indian variants exist
        best_indian = (
            next((o for o in indian if o["exchange"] in _NSE_EXCHANGES), None)
            or (indian[0] if indian else None)
        )
        best_other = other[0] if other else None

        if best_indian and best_other:
            return {"status": "choose", "options": [best_indian, best_other]}

        single = best_indian or best_other
        if single:
            return {
                "status": "ok",
                "ticker": single["ticker"],
                "exchange": single["exchange"],
                "full_name": single["full_name"],
            }

        raise ValueError("no valid results")

    except Exception:
        raw = company_name.strip().upper()
        return {"status": "ok", "ticker": raw, "exchange": "UNKNOWN", "full_name": company_name}


def resolve_ticker(company_name: str) -> dict:
    """Return {"ticker": str, "exchange": str, "full_name": str}.

    Legacy single-result resolver. Prefers the first yfinance Search hit.
    Use resolve_listings() for dual-listing detection.
    """
    result = resolve_listings(company_name)
    if result["status"] == "choose":
        # Auto-pick the first option (Indian listing takes precedence since it's listed first)
        opt = result["options"][0]
        return {"ticker": opt["ticker"], "exchange": opt["exchange"], "full_name": opt["full_name"]}
    return {"ticker": result["ticker"], "exchange": result["exchange"], "full_name": result["full_name"]}
