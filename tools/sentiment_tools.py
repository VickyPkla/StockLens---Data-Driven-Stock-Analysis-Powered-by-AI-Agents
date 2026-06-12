"""VADER-based news sentiment analysis tool for the Sentiment Agent."""

from __future__ import annotations

import json

import yfinance as yf
from crewai.tools import tool


def _vader_scores(headline: str) -> dict:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(headline)


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "POSITIVE"
    if compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


@tool
def get_sentiment_analysis(ticker: str) -> str:
    """Fetch recent news headlines for a stock and score each using VADER sentiment analysis.

    Returns JSON with per-headline compound scores and an aggregate summary.
    Compound score: +1 = most positive, -1 = most negative.
    Aggregate score_out_of_10 maps [-1, +1] → [0, 10].
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception as e:
        return json.dumps({"error": f"Could not fetch news for {ticker}: {e}", "ticker": ticker})

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
            scores = _vader_scores(title)
            headlines.append({
                "title": title,
                "compound": round(scores["compound"], 4),
                "pos": round(scores["pos"], 4),
                "neu": round(scores["neu"], 4),
                "neg": round(scores["neg"], 4),
                "label": _label(scores["compound"]),
            })
        except Exception:
            continue

    if not headlines:
        return json.dumps({
            "ticker": ticker,
            "headline_count": 0,
            "headlines": [],
            "aggregate": {
                "avg_compound": 0.0,
                "positive_count": 0,
                "neutral_count": 0,
                "negative_count": 0,
                "overall_label": "NEUTRAL",
                "score_out_of_10": 5.0,
            },
        })

    compounds = [h["compound"] for h in headlines]
    avg_compound = round(sum(compounds) / len(compounds), 4)
    score_out_of_10 = round((avg_compound + 1) / 2 * 10, 1)

    pos_count = sum(1 for h in headlines if h["label"] == "POSITIVE")
    neg_count = sum(1 for h in headlines if h["label"] == "NEGATIVE")
    neu_count = len(headlines) - pos_count - neg_count

    # Overall label with stronger thresholds for aggregate
    if avg_compound >= 0.20:
        overall_label = "STRONGLY POSITIVE"
    elif avg_compound >= 0.05:
        overall_label = "POSITIVE"
    elif avg_compound <= -0.20:
        overall_label = "STRONGLY NEGATIVE"
    elif avg_compound <= -0.05:
        overall_label = "NEGATIVE"
    else:
        overall_label = "NEUTRAL"

    return json.dumps({
        "ticker": ticker,
        "headline_count": len(headlines),
        "headlines": headlines,
        "aggregate": {
            "avg_compound": avg_compound,
            "positive_count": pos_count,
            "neutral_count": neu_count,
            "negative_count": neg_count,
            "overall_label": overall_label,
            "score_out_of_10": score_out_of_10,
        },
    }, indent=2)
