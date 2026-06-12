"""Devil's Advocate agent — Agent 5 (final) in the 5-agent pipeline.

Challenges the Synthesis agent's conclusions, surfaces confirmation bias,
and stress-tests the investment thesis to improve final recommendation accuracy.
"""

from datetime import date
from crewai import Agent, Task, LLM

from config.settings import GROQ_MODEL, GROQ_API_KEY

_ROLE = "Contrarian Risk Analyst / Devil's Advocate"

_GOAL = (
    "Rigorously challenge the investment thesis, identify confirmation bias in the synthesis "
    "report, surface ignored warning signals, and stress-test the key assumptions so the "
    "investor has a balanced, adversarial perspective before making a decision."
)

_BACKSTORY = (
    "Contrarian portfolio risk manager, 18 years identifying overconfident calls. "
    "Not negative for its own sake — stress-tests reasoning. "
    "Finds what could make a BUY fail and what bears miss in a SELL case."
)

_MAX_REPORT_CHARS = 1500


def _truncate(text: str, max_chars: int = _MAX_REPORT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... TRUNCATED FOR CONTEXT WINDOW ...]"


_TASK_TEMPLATE = """You are reviewing an investment analysis for {company_name} ({ticker}).
Today: {current_date}.

════════════════════════════════════════════
SYNTHESIS REPORT (PRIMARY RECOMMENDATION):
════════════════════════════════════════════
{synthesis_report}

════════════════════════════════════════════
FUNDAMENTAL ANALYSIS REPORT:
════════════════════════════════════════════
{fundamental_report}

════════════════════════════════════════════
TECHNICAL ANALYSIS REPORT:
════════════════════════════════════════════
{technical_report}
════════════════════════════════════════════

⚠️ STOP — DATA AVAILABILITY CHECK (evaluate before writing anything):
If ANY of the reports above contain "ANALYSIS UNAVAILABLE" or "RECOMMENDATION UNAVAILABLE":
  → Your ONLY allowed output is: "CHALLENGE UNAVAILABLE: One or more upstream analyses failed. No challenge can be produced without complete verified data."
  → Do NOT write any challenge sections or invent figures to fill the gap.
Only continue if all three reports contain real analysis.

ANCHOR TO REPORTS ONLY:
- Every data point, price, ratio, and metric you cite MUST appear in the reports above.
- Do NOT introduce any financial figure, price level, ratio, or valuation not present in those reports.
- Do NOT use your training knowledge to supply missing numbers or to estimate what the data "probably" shows.
- If a figure is "N/A" or "Not available" in the source reports, note that gap — do not substitute a value.

Begin your response with "Final Answer:" on the very first line, then challenge the synthesis recommendation rigorously. Do NOT simply agree with it.

## Main Thesis Challenges
3-5 specific, evidence-based objections. For each: state the claim from the synthesis → explain why it may be flawed → cite contrary data points from the reports with specific numbers.

## Ignored Warning Signs
2-4 bearish signals (if BUY/HOLD) or bullish signals (if SELL) the synthesis downplayed or missed entirely. State the potential downside scenario for each.

## Assumption Stress Test
2-3 critical assumptions the recommendation depends on. For each: "What if this is wrong?" Describe the realistic downside.

## Bull/Bear Balance Check
Was the synthesis appropriately balanced? Were risks adequately quantified? Did it over-anchor on analyst consensus? Was peer valuation comparison fair?

## CHALLENGE: [STRONG / MODERATE / WEAK]
Exactly one verdict. 2-3 sentence justification.
STRONG = material flaws that could change the recommendation.
MODERATE = notable weaknesses worth flagging.
WEAK = solid thesis with only minor concerns.

## Revised Confidence Level
STRONG: suggest replacement conviction level. MODERATE: lower by one level? WEAK: confirm original stands.
Also state whether the Risk Level should be revised up or down.
"""


def create_devil_advocate_agent() -> Agent:
    llm = LLM(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.4,
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
        max_retry_limit=2,
    )


def create_devil_advocate_task(
    agent: Agent,
    ticker: str,
    company_name: str,
    synthesis_report: str,
    fundamental_report: str = "",
    technical_report: str = "",
) -> Task:
    return Task(
        description=_TASK_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            current_date=date.today().strftime("%B %d, %Y"),
            synthesis_report=_truncate(synthesis_report),
            fundamental_report=_truncate(fundamental_report),
            technical_report=_truncate(technical_report),
        ),
        expected_output=(
            "Devil's advocate counter-report with sections: Main Thesis Challenges (3-5 objections), "
            "Ignored Warning Signs (2-4 signals), Assumption Stress Test (2-3 assumptions), "
            "Bull/Bear Balance Check, CHALLENGE: STRONG/MODERATE/WEAK with justification, "
            "Revised Confidence Level. Direct, specific prose."
        ),
        agent=agent,
    )
