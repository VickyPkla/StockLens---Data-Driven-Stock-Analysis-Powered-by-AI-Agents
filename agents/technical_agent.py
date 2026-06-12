"""Technical Analyst agent — Agent 2 in the sequential crew."""

from datetime import date
from crewai import Agent, Task, LLM

from config.settings import GROQ_MODEL, GROQ_API_KEY

_ROLE = "Senior Technical Analyst"

_GOAL = (
    "Analyse price action, trend, momentum, and volatility to determine the technical "
    "outlook of a stock and produce a structured Technical Analysis Report."
)

_BACKSTORY = (
    "CMT-certified technical analyst, 12 years across NSE/BSE and US markets. "
    "Precise and disciplined; reads price structure, indicator confluence, and volume."
)

_TASK_TEMPLATE = """Perform a comprehensive technical analysis of {company_name} ({ticker}).

{verified_technical_data}

⚠️ STOP — DATA AVAILABILITY CHECK (evaluate before writing anything):
If the verified data block above starts with "ERROR:" or contains no actual data values:
  → Your ONLY allowed output is: "ANALYSIS UNAVAILABLE: [copy the exact error text]. No estimates or placeholders substituted for missing data."
  → Do NOT write any analysis sections, scores, or price levels.
Only continue if real data values are present in the block above.

ANCHOR TO VERIFIED DATA:
- Every numerical value in this report MUST come from the verified block above.
- If a value shows "N/A" in the verified block, write "N/A" — never substitute an estimate.
- If a metric is absent from the verified block, write "Not available" — do NOT use training knowledge to fill it in.
- Do NOT invent, estimate, or approximate any price, indicator value, support/resistance level, or target price.
Today: {current_date}.

Write the Technical Analysis Report directly from the verified data above.
Begin your response with "Final Answer:" on the very first line.

## Current Trend
Exact current price, SMA50, SMA200. Golden/death cross? Overall trend direction.

## Momentum Analysis
Exact RSI(14): overbought/oversold/neutral. Exact MACD, signal line, histogram. Bullish/bearish crossover? Divergence?

## Bollinger Bands
Exact upper/middle/lower band values and %B. Price position relative to bands. Squeezing or expanding?

## Support & Resistance
Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%). Which level is nearest? 60-day swing high/low.

## 52-Week Context
Exact 52w high and low. Distance from each. Near 52w high (momentum) or low (oversold/distress)?

## Volatility Assessment
Exact ATR(14) as price and % of current price. Volatility high or low?

## Volume & OBV Analysis
Current vs 20-day avg volume — confirming the trend? OBV trend direction. Diverging from price?

## Recent News Sentiment
Briefly note any market context relevant to the technical picture.

## Technically Derived Price Levels
- Technical Support Zone
- Technical Resistance Zone
- Suggested Entry Zone
- Stop Loss: price and % below current
- Short-term Target (1-3 months)
- Medium-term Target (6-12 months)

## Technical Score: X/10
Justify in 2-3 sentences based on indicator confluence.
"""


def create_technical_agent() -> Agent:
    llm = LLM(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
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


def create_technical_task(
    agent: Agent,
    ticker: str,
    company_name: str,
    verified_technical_data: str = "",
) -> Task:
    return Task(
        description=_TASK_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            current_date=date.today().strftime("%B %d, %Y"),
            verified_technical_data=verified_technical_data,
        ),
        expected_output=(
            "Structured technical analysis report with sections: Current Trend, "
            "Momentum Analysis, Bollinger Bands, Support & Resistance, 52-Week Context, "
            "Volatility Assessment, Volume & OBV Analysis, Recent News Sentiment, "
            "Technically Derived Price Levels (support/resistance/entry/stop-loss/targets), "
            "Technical Score (1-10). Plain text."
        ),
        agent=agent,
    )
