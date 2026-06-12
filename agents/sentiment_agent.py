"""Sentiment Analyst agent — Agent 3 in the 5-agent pipeline."""

from datetime import date
from crewai import Agent, Task, LLM

from config.settings import GROQ_MODEL, GROQ_API_KEY

_ROLE = "Market Sentiment & News Analyst"

_GOAL = (
    "Gauge market mood from news flow, VADER sentiment scores, analyst activity, "
    "and institutional/insider behaviour to produce a structured Sentiment Analysis Report."
)

_BACKSTORY = (
    "Specialist in market psychology and news-driven price action, 10 years at a quantitative hedge fund. "
    "Interprets VADER scores precisely; distinguishes noisy headlines from genuinely sentiment-shifting events."
)

_TASK_TEMPLATE = """Perform a comprehensive sentiment analysis of {company_name} ({ticker}).

{verified_sentiment_data}

ANCHOR TO VERIFIED DATA: Use only the VADER scores and institutional data above. Never substitute training-knowledge values.
Today: {current_date}.

Write the Sentiment Analysis Report directly from the verified data above.
Begin your response with "Final Answer:" on the very first line.

## News Sentiment Score
VADER aggregate (avg_compound) as score/10. Top 5 headlines with individual scores and labels.
Positive/neutral/negative counts. Improving or deteriorating trend?

## Analyst Sentiment
Based on the headlines above, note any upgrade/downgrade signals or price target revisions.
Trend moving bullish or bearish?

## Insider Activity Sentiment
From the insider transaction data above: predominantly buying or selling? Buy-to-sell ratio.

## Institutional Flow
Top 3 institutional holders and % stake from the verified data above. Ownership concentrated or diversified?

## Overall Sentiment Assessment
Synthesise all signals. State: STRONGLY BULLISH / BULLISH / NEUTRAL / BEARISH / STRONGLY BEARISH
with 2-3 sentence justification.

## Sentiment Score: X/10
Justify in 1-2 sentences.
"""


def create_sentiment_agent() -> Agent:
    llm = LLM(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=1500,
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


def create_sentiment_task(
    agent: Agent,
    ticker: str,
    company_name: str,
    verified_sentiment_data: str = "",
) -> Task:
    return Task(
        description=_TASK_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            current_date=date.today().strftime("%B %d, %Y"),
            verified_sentiment_data=verified_sentiment_data,
        ),
        expected_output=(
            "Structured sentiment report with sections: News Sentiment Score (VADER aggregate + "
            "per-headline breakdown), Analyst Sentiment, Insider Activity Sentiment, "
            "Institutional Flow, Overall Sentiment (STRONGLY BULLISH/BULLISH/NEUTRAL/BEARISH/"
            "STRONGLY BEARISH with justification), Sentiment Score (1-10). Plain text."
        ),
        agent=agent,
    )
