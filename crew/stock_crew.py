"""Orchestrates the 4-agent stock analysis pipeline via LangGraph.

The graph is defined in crew/analysis_graph.py. This module provides the
public run_analysis() interface used by main.py.
"""

from crew.analysis_graph import get_graph, AnalysisState


def run_analysis(ticker: str, company_name: str) -> dict:
    """Run the full 4-agent analysis for a given stock via LangGraph.

    Agents (in order):
        1. Fundamental Analyst        — financial health, valuation, peers
        2. Technical Analyst          — price action, indicators, support/resistance
        3. Sentiment Analyst          — news VADER scores, insider activity, institutional flow
        4. Chief Investment Strategist — synthesises all three analyses

    Returns a dict with keys:
        "fundamental"  — raw Fundamental Agent output
        "technical"    — raw Technical Agent output
        "sentiment"    — raw Sentiment Agent output
        "synthesis"    — raw Synthesis Agent output (final recommendation)
        "risk_ctx"     — pre-computed quantitative risk context block (for display)
        "full"         — same as "synthesis" (backward-compat key)
    """
    initial_state: AnalysisState = {
        "ticker": ticker,
        "company_name": company_name,
        "tech_ctx": "",
        "fund_ctx": "",
        "sentiment_ctx": "",
        "risk_ctx": "",
        "fundamental_report": "",
        "technical_report": "",
        "sentiment_report": "",
        "synthesis_report": "",
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    synthesis = final_state.get("synthesis_report", "")

    return {
        "fundamental": final_state.get("fundamental_report", ""),
        "technical": final_state.get("technical_report", ""),
        "sentiment": final_state.get("sentiment_report", ""),
        "synthesis": synthesis,
        "risk_ctx": final_state.get("risk_ctx", ""),
        "fund_ctx": final_state.get("fund_ctx", ""),
        "tech_ctx": final_state.get("tech_ctx", ""),
        "full": synthesis,
    }
