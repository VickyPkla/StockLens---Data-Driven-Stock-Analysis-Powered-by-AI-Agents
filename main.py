"""Stock Analyser — CLI entrypoint.

Usage:
    python main.py "Reliance Industries"
    python main.py AAPL
    python main.py          # interactive prompt
"""

import sys
from pathlib import Path

# Load .env before any project imports so settings are populated
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from tools.ticker_resolver import resolve_ticker
from tools.data_prefetch import build_verified_context
from crew.stock_crew import run_analysis
from display.rich_output import (
    print_banner,
    print_ticker_resolved,
    print_agent_start,
    print_fundamental_table,
    print_technical_table,
    print_final_recommendation,
    print_sentiment_panel,
    print_risk_table,
    print_devil_advocate_panel,
    print_error,
    save_report,
    console,
)


def _get_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return Prompt.ask("[bold cyan]Enter stock name or ticker symbol[/bold cyan]").strip()


def main() -> None:
    print_banner()

    query = _get_query()
    if not query:
        print_error("No stock name or ticker provided.")
        sys.exit(1)

    # Resolve ticker
    console.print("\n[dim]Resolving ticker symbol...[/dim]")
    try:
        ticker_info = resolve_ticker(query)
    except Exception as exc:
        print_error(f"Ticker resolution failed: {exc}")
        sys.exit(1)

    print_ticker_resolved(ticker_info)
    ticker = ticker_info["ticker"]
    company_name = ticker_info["full_name"]

    if not Confirm.ask(
        f"\nProceed with full analysis of [bold]{company_name}[/bold] ([cyan]{ticker}[/cyan])?",
        default=False,
    ):
        console.print("[yellow]Analysis cancelled.[/yellow]")
        sys.exit(0)

    # Pre-fetch and display verified data so the user can confirm accuracy BEFORE analysis
    console.print("\n[dim]Fetching live market data to verify before analysis...[/dim]")
    try:
        tech_ctx, fund_ctx, sentiment_ctx, risk_ctx = build_verified_context(ticker)
        console.print("\n[bold green]✓ Live data verified[/bold green]")
        # Show just the price block so user can spot-check
        for line in tech_ctx.split("\n"):
            if any(k in line for k in ["Current Price", "52-Week High", "52-Week Low", "SMA 50", "RSI"]):
                console.print(f"  [dim]{line.strip()}[/dim]")
        console.print()
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not pre-fetch verified data: {exc}[/yellow]")

    console.print(
        "[dim]Starting 5-agent analysis. This typically takes 10–25 minutes "
        "depending on Ollama response time and Alpha Vantage rate limits.[/dim]\n"
    )

    # Run the pipeline via LangGraph
    try:
        print_agent_start("Starting Analysis Pipeline (LangGraph + CrewAI)")
        result = run_analysis(ticker, company_name)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        print_error(f"Analysis failed: {exc}")
        raise  # re-raise so the user sees the full traceback for debugging

    # ── Display results ────────────────────────────────────────────────────

    print_agent_start("Fundamental Analysis Results")
    if result.get("fundamental"):
        print_fundamental_table(result["fundamental"])

    print_agent_start("Technical Analysis Results")
    if result.get("technical"):
        print_technical_table(result["technical"])

    print_agent_start("Sentiment Analysis Results")
    if result.get("sentiment"):
        print_sentiment_panel(result["sentiment"])

    print_agent_start("Quantitative Risk Analysis")
    if result.get("risk_ctx"):
        print_risk_table(result["risk_ctx"])

    print_agent_start("Final Investment Recommendation")
    synthesis = result.get("synthesis") or result.get("full", "")
    if synthesis:
        print_final_recommendation(synthesis)

    print_agent_start("Devil's Advocate Counter-Report")
    if result.get("devil_advocate"):
        print_devil_advocate_panel(result["devil_advocate"])

    # ── Build full markdown report ─────────────────────────────────────────

    full_md = (
        f"# Stock Analysis Report: {company_name} ({ticker})\n\n"
        f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        f"---\n\n"
        f"## Fundamental Analysis\n\n{result.get('fundamental', 'N/A')}\n\n"
        f"---\n\n"
        f"## Technical Analysis\n\n{result.get('technical', 'N/A')}\n\n"
        f"---\n\n"
        f"## Sentiment Analysis\n\n{result.get('sentiment', 'N/A')}\n\n"
        f"---\n\n"
        f"## Quantitative Risk Metrics\n\n```\n{result.get('risk_ctx', 'N/A')}\n```\n\n"
        f"---\n\n"
        f"## Investment Synthesis & Recommendation\n\n{synthesis}\n\n"
        f"---\n\n"
        f"## Devil's Advocate Counter-Report\n\n{result.get('devil_advocate', 'N/A')}\n\n"
        f"---\n\n"
        f"*Disclaimer: This report is AI-generated for educational purposes only. "
        f"Not financial advice.*\n"
    )

    try:
        saved_path = save_report(ticker, full_md)
        console.print(f"\n[dim]Full report saved to: [bold]{saved_path}[/bold][/dim]")
    except Exception as exc:
        console.print(f"\n[yellow]Could not save report: {exc}[/yellow]")


if __name__ == "__main__":
    main()
