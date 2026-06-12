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

ANCHOR TO VERIFIED DATA: Use only the numbers above. Never substitute training-knowledge values for any figure present in the verified block.
Today: {current_date}.

Write a Fundamental Analysis Report directly from the verified data above. For the Valuation vs Peers section, use your domain knowledge to name 2-3 direct competitors and state their typical valuation ranges — clearly noting these are approximate.
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
P/E (trailing/forward), P/B, EV/EBITDA for {company_name}. Name 2-3 direct competitors and compare typical multiples. Premium/discount/fairly valued?

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
