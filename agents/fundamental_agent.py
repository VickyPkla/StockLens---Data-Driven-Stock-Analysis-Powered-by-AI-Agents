"""Fundamental Analyst agent — Agent 1 in the sequential crew."""

from datetime import date
from crewai import Agent, Task, LLM

from config.settings import GROQ_MODEL, GROQ_API_KEY

_ROLE = "Senior Fundamental Analyst"

_GOAL = (
    "Analyse the financial health, valuation, and business quality of a stock using "
    "the pre-verified fundamental data provided in the task."
)

_BACKSTORY = (
    "CFA-level analyst, 15 years across Indian and US equities. "
    "Methodical and data-driven; anchor conclusions in numbers before qualitative commentary."
)

_TASK_TEMPLATE = """Perform a comprehensive fundamental analysis of {company_name} ({ticker}).

{verified_fundamental_data}

⚠️ STOP — DATA AVAILABILITY CHECK (evaluate before writing anything):
If the verified data block above starts with "ERROR:" or contains no actual data values:
  → Your ONLY allowed output is: "ANALYSIS UNAVAILABLE: [copy the exact error text]. No estimates or placeholders substituted for missing data."
  → Do NOT write any analysis sections, scores, or price levels.
Only continue if real data values are present in the block above.

ANCHOR TO VERIFIED DATA:
- Every numerical value in this report MUST come from the verified block above.
- If a value shows "N/A" in the verified block, write "N/A" — never substitute an estimate.
- If a metric is absent from the verified block, write "Not available" — do NOT use training knowledge to fill it in.
- Do NOT invent, estimate, or approximate any price, ratio, indicator, or financial figure.
Today: {current_date}.

Write a Fundamental Analysis Report directly from the verified data above.
Begin your response with "Final Answer:" on the very first line.

## Business Overview
Sector, industry, primary business model.

## Revenue & Earnings Trend
4 years annual revenue/net income + 3-year CAGR. 4-6 recent quarters. Quarterly trend accelerating, stable, or decelerating?

## Profitability & Return Quality
Gross/operating/net margins (most recent year). ROE/ROA/ROCE trend (3-4 years). Improving or eroding?

## Balance Sheet Health
Debt-to-Equity and Current Ratio. Flag concerns.

## Cash Flow Quality
Operating CF and Free CF (2-3 years). FCF positive and growing?

## Earnings Surprise Track Record
Last 4 quarters: actual vs estimate EPS with surprise %. Consistent beats, misses, or mixed?

## Valuation vs Peers
Report ONLY the P/E (trailing/forward), P/B, and EV/EBITDA values from the verified block above for {company_name}.
Name 2-3 direct competitors for qualitative context only. Do NOT state specific valuation multiples, ratios, or price figures for competitors — you do not have verified data for them.
State whether {company_name} appears premium/discount/fairly valued based solely on the verified ratios above.

## Analyst & Institutional Sentiment
Buy/hold/sell breakdown, mean price target and implied upside. Top institutional holders; notable insider activity.

## Key Risks
3-5 key risks from the financial data.

## Fundamental Score: X/10
Justify in 2-3 sentences covering growth, quality, valuation, and sentiment.
"""


def create_fundamental_agent() -> Agent:
    llm = LLM(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=2000,
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


def create_fundamental_task(
    agent: Agent,
    ticker: str,
    company_name: str,
    verified_fundamental_data: str = "",
) -> Task:
    return Task(
        description=_TASK_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            current_date=date.today().strftime("%B %d, %Y"),
            verified_fundamental_data=verified_fundamental_data,
        ),
        expected_output=(
            "Structured fundamental analysis report with sections: Business Overview, "
            "Revenue & Earnings Trend, Profitability & Return Quality, Balance Sheet Health, "
            "Cash Flow Quality, Earnings Surprise Track Record, Valuation vs Peers, "
            "Analyst & Institutional Sentiment, Key Risks, Fundamental Score (1-10). "
            "Plain text, no JSON."
        ),
        agent=agent,
    )
