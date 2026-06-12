"""LangGraph StateGraph orchestrating the 4-agent stock analysis pipeline.

Flow (sequential — Ollama is single-threaded):
    START → prefetch → fundamental → technical → sentiment → synthesis → END

Each agent node creates a single-agent CrewAI Crew, runs it, and writes the
result into state. Inter-agent context is passed as formatted strings injected
into task descriptions (not as CrewAI context_tasks, which don't survive across
separate Crew instances).

Note: CrewAI 1.14.x uses asyncio.run() internally per Crew via its Flow-based
agent executor. Concurrent fan-out threads each calling asyncio.run() can cause
event-loop conflicts and connection errors against a single Ollama instance.
Sequential edges avoid this entirely.
"""

from __future__ import annotations

import re
import time
from typing import TypedDict

from crewai import Crew, Process
from langgraph.graph import StateGraph, END

from tools.data_prefetch import build_verified_context
from agents.fundamental_agent import create_fundamental_agent, create_fundamental_task
from agents.technical_agent import create_technical_agent, create_technical_task
from agents.sentiment_agent import create_sentiment_agent, create_sentiment_task
from agents.synthesis_agent import create_synthesis_agent, create_synthesis_task


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AnalysisState(TypedDict):
    ticker: str
    company_name: str
    # Prefetched verified context blocks
    tech_ctx: str
    fund_ctx: str
    sentiment_ctx: str
    risk_ctx: str
    # Agent outputs
    fundamental_report: str
    technical_report: str
    sentiment_report: str
    synthesis_report: str


# ---------------------------------------------------------------------------
# Helper: run a single-agent Crew and return the raw output
# ---------------------------------------------------------------------------

def _parse_groq_retry_wait(err: str) -> float:
    """Parse Groq's suggested retry delay, handling both '30s' and '52m0.768s' formats."""
    # "Xm Y.Zs" — e.g. "52m0.768s"
    m = re.search(r"try again in (\d+)m(\d+\.?\d*)s", err, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    # plain seconds — e.g. "30.5s"
    m = re.search(r"try again in (\d+\.?\d*)s", err, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 65.0


def _run_single_agent_crew(agent, task, max_retries: int = 8) -> str:
    for attempt in range(max_retries):
        try:
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            if result.tasks_output:
                return result.tasks_output[0].raw or ""
            return result.raw or ""
        except Exception as exc:
            err = str(exc)
            is_rate_limit = "RateLimitError" in err or "rate_limit" in err.lower() or "rate limit" in err.lower()
            if not is_rate_limit or attempt >= max_retries - 1:
                raise

            # Distinguish daily-limit (TPD) from per-minute (TPM) errors.
            is_daily_limit = "per day" in err.lower() or "TPD" in err or "tokens per day" in err.lower()
            if is_daily_limit:
                wait_s = _parse_groq_retry_wait(err)
                print(
                    f"\n[DAILY LIMIT] Groq daily token quota exhausted. "
                    f"Reset in ~{wait_s/60:.0f} minutes. "
                    f"Switch to a different model or wait for the daily reset.\n"
                )
                raise

            # Per-minute TPM limit — wait for the rolling window to clear.
            groq_wait = _parse_groq_retry_wait(err)
            wait = max(groq_wait + 5, 70)
            print(
                f"\n[RATE LIMIT] Groq TPM window full — pausing {wait:.0f}s "
                f"(auto-retry {attempt + 2}/{max_retries}). "
                f"The 'Crew Execution Failed' panel above is cosmetic.\n"
            )
            time.sleep(wait)
            continue


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def prefetch_node(state: AnalysisState) -> dict:
    """Fetch all four verified context blocks before any agent runs."""
    ticker = state["ticker"]
    print(f"\n[LangGraph] prefetch_node: fetching verified data for {ticker}...")
    tech_ctx, fund_ctx, sentiment_ctx, risk_ctx = build_verified_context(ticker)
    print("[LangGraph] prefetch_node: complete.\n")
    return {
        "tech_ctx": tech_ctx,
        "fund_ctx": fund_ctx,
        "sentiment_ctx": sentiment_ctx,
        "risk_ctx": risk_ctx,
    }


def fundamental_node(state: AnalysisState) -> dict:
    """Run the Fundamental Analyst agent."""
    print("\n[LangGraph] fundamental_node: starting Fundamental Agent...")
    agent = create_fundamental_agent()
    task = create_fundamental_task(
        agent,
        state["ticker"],
        state["company_name"],
        verified_fundamental_data=state["fund_ctx"],
    )
    report = _run_single_agent_crew(agent, task)
    print("[LangGraph] fundamental_node: complete.\n")
    return {"fundamental_report": report}


def technical_node(state: AnalysisState) -> dict:
    """Run the Technical Analyst agent."""
    print("[LangGraph] technical_node: waiting 15s inter-agent gap...")
    time.sleep(15)
    print("\n[LangGraph] technical_node: starting Technical Agent...")
    agent = create_technical_agent()
    task = create_technical_task(
        agent,
        state["ticker"],
        state["company_name"],
        verified_technical_data=state["tech_ctx"],
    )
    report = _run_single_agent_crew(agent, task)
    print("[LangGraph] technical_node: complete.\n")
    return {"technical_report": report}


def sentiment_node(state: AnalysisState) -> dict:
    """Run the Sentiment Analyst agent."""
    print("[LangGraph] sentiment_node: waiting 15s inter-agent gap...")
    time.sleep(15)
    print("\n[LangGraph] sentiment_node: starting Sentiment Agent...")
    agent = create_sentiment_agent()
    task = create_sentiment_task(
        agent,
        state["ticker"],
        state["company_name"],
        verified_sentiment_data=state["sentiment_ctx"],
    )
    report = _run_single_agent_crew(agent, task)
    print("[LangGraph] sentiment_node: complete.\n")
    return {"sentiment_report": report}


def synthesis_node(state: AnalysisState) -> dict:
    """Run the Chief Investment Strategist (Synthesis) agent."""
    print("[LangGraph] synthesis_node: waiting 15s inter-agent gap...")
    time.sleep(15)
    print("\n[LangGraph] synthesis_node: starting Synthesis Agent...")
    agent = create_synthesis_agent()
    task = create_synthesis_task(
        agent,
        state["ticker"],
        state["company_name"],
        fundamental_report=state.get("fundamental_report", ""),
        technical_report=state.get("technical_report", ""),
        sentiment_report=state.get("sentiment_report", ""),
        verified_sentiment_data=state.get("sentiment_ctx", ""),
        verified_risk_data=state.get("risk_ctx", ""),
    )
    report = _run_single_agent_crew(agent, task)
    print("[LangGraph] synthesis_node: complete.\n")
    return {"synthesis_report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_analysis_graph():
    """Build and compile the LangGraph StateGraph for the 4-agent pipeline."""
    builder = StateGraph(AnalysisState)

    builder.add_node("prefetch", prefetch_node)
    builder.add_node("fundamental", fundamental_node)
    builder.add_node("technical", technical_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("synthesis", synthesis_node)

    # Fully sequential chain — avoids concurrent asyncio.run() conflicts in CrewAI 1.14.x
    builder.set_entry_point("prefetch")
    builder.add_edge("prefetch", "fundamental")
    builder.add_edge("fundamental", "technical")
    builder.add_edge("technical", "sentiment")
    builder.add_edge("sentiment", "synthesis")
    builder.add_edge("synthesis", END)

    return builder.compile()


# Module-level cached graph instance (built once per process)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_analysis_graph()
    return _graph
