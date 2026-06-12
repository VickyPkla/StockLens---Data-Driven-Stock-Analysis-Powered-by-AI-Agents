"""Synthesis Analyst agent — Agent 4 in the 5-agent pipeline. No data-fetching tools."""

from datetime import date
from crewai import Agent, Task, LLM

from config.settings import GROQ_MODEL, GROQ_API_KEY

_ROLE = "Chief Investment Strategist"

_GOAL = (
    "Synthesise the fundamental, technical, and sentiment analyses into a clear, risk-adjusted "
    "investment recommendation with a final BUY, HOLD, or SELL verdict."
)

_BACKSTORY = (
    "Seasoned portfolio manager, 20 years across Indian and global markets. "
    "Weighs fundamental value against technical timing and sentiment; expresses high conviction "
    "only when multiple signals agree. Always anchors Risk Level to pre-verified quantitative risk metrics."
)

_MAX_REPORT_CHARS = 1500


def _truncate(text: str, max_chars: int = _MAX_REPORT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... TRUNCATED FOR CONTEXT WINDOW ...]"


_TASK_TEMPLATE = """You are synthesising investment analyses for {company_name} ({ticker}).
Today: {current_date}.

{verified_sentiment_data}

{verified_risk_data}

CRITICAL: The Risk Level in your report MUST match the Composite Risk Score in VERIFIED RISK DATA above. Do not estimate it independently.

{fundamental_report_section}

{technical_report_section}

Begin your response with "Final Answer:" on the very first line, then write the Final Investment Report with these sections:

## Executive Summary
3-4 sentences: company overview, core fundamental finding, core technical finding, sentiment picture, overall stance.

## Fundamental Signal: [Bullish / Neutral / Bearish]
Recap Fundamental Score + 2 most important supporting reasons.

## Technical Signal: [Bullish / Neutral / Bearish]
Recap Technical Score + 2 most important supporting reasons.

## Sentiment Signal: [Bullish / Neutral / Bearish]
Recap Sentiment Score + key driver (news flow, insider activity, or institutional flow).

## Signal Confluence
Do all three signals agree or diverge? If diverging, which do you weight more and why?

## RECOMMENDATION: [BUY / HOLD / SELL]

## For New Investors: Should You Buy Now?
If BUY: confirm entry zone. If HOLD: buy now / wait for specific price / avoid — state reason and target price. If SELL: what conditions make it worth revisiting.

## Action Prices
- Current Market Price: [from technical report]
- Suggested Buy Zone: [lower] – [upper]
- Stop Loss: [price] ([X]% below current)
- Short-term Profit Target: [price] ([X]%, 1-3 months)
- Medium-term Profit Target: [price] ([X]%, 3-12 months)

## Price Targets: 6-Month and 1-Year Outlook
Use exactly these labels:
- 6-Month Target: [price] ([X]% upside/downside). [1-2 sentence basis]. Bull case: [price]. Bear case: [price].
- 1-Year Outlook Target: [price] ([X]% upside/downside). [1-2 sentence basis]. Bull case: [price]. Bear case: [price].

## Conviction Level: [High / Medium / Low]

## Risk Level: [Low / Medium / High]
Must match the VERIFIED RISK DATA composite score. If you disagree, explain why.

## Investment Horizon
Short-term (1-3m), Medium-term (3-12m), Long-term (1-3y).

## Key Catalysts to Watch (3-5 items)

## Key Risks to Watch (3-5 items)

## Disclaimer
AI-generated analysis for educational purposes only. Not financial advice. Consult a licensed financial advisor before investing.
"""


def create_synthesis_agent() -> Agent:
    llm = LLM(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=1800,
    )
    return Agent(
        role=_ROLE,
        goal=_GOAL,
        backstory=_BACKSTORY,
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
    )


def create_synthesis_task(
    agent: Agent,
    ticker: str,
    company_name: str,
    context_tasks: list = None,
    fundamental_report: str = "",
    technical_report: str = "",
    sentiment_report: str = "",
    verified_sentiment_data: str = "",
    verified_risk_data: str = "",
) -> Task:
    if fundamental_report:
        fund_section = (
            "════════════════════════════════════════════════════════════\n"
            "FUNDAMENTAL ANALYSIS REPORT (from your team):\n"
            "════════════════════════════════════════════════════════════\n"
            + _truncate(fundamental_report)
            + "\n════════════════════════════════════════════════════════════\n"
        )
    else:
        fund_section = "(Fundamental Analysis Report available in your context.)"

    if technical_report:
        tech_section = (
            "════════════════════════════════════════════════════════════\n"
            "TECHNICAL ANALYSIS REPORT (from your team):\n"
            "════════════════════════════════════════════════════════════\n"
            + _truncate(technical_report)
            + "\n════════════════════════════════════════════════════════════\n"
        )
    else:
        tech_section = "(Technical Analysis Report available in your context.)"

    if sentiment_report:
        sent_section = (
            "════════════════════════════════════════════════════════════\n"
            "SENTIMENT ANALYSIS REPORT (from your team):\n"
            "════════════════════════════════════════════════════════════\n"
            + _truncate(sentiment_report)
            + "\n════════════════════════════════════════════════════════════\n"
        )
        tech_section = tech_section + "\n" + sent_section

    return Task(
        description=_TASK_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            current_date=date.today().strftime("%B %d, %Y"),
            verified_sentiment_data=verified_sentiment_data or "(No pre-computed sentiment data available.)",
            verified_risk_data=verified_risk_data or "(No pre-computed risk data available.)",
            fundamental_report_section=fund_section,
            technical_report_section=tech_section,
        ),
        expected_output=(
            "Professional investment report with sections: Executive Summary, "
            "Fundamental/Technical/Sentiment Signals, Signal Confluence, "
            "RECOMMENDATION: BUY/HOLD/SELL, For New Investors, Action Prices, "
            "6-Month and 1-Year Price Targets (with bull/bear cases), "
            "Conviction Level, Risk Level, Investment Horizon, Catalysts, Risks, Disclaimer."
        ),
        agent=agent,
        context=context_tasks or None,
    )
