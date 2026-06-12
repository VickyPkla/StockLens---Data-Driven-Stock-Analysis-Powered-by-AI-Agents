"""All Rich CLI rendering: banners, tables, panels, report saving.

This module has NO internal project imports so it can be safely imported
without triggering agent or crew initialisation.
"""

import re
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

# Strip deepseek-r1 <think>...</think> reasoning blocks before displaying
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Print the application banner."""
    title = Text("STOCK ANALYSER", style="bold cyan", justify="center")
    subtitle = Text(
        "Multi-Agent AI Analysis  •  CrewAI + LangGraph + Ollama (deepseek-r1)",
        style="dim cyan",
        justify="center",
    )
    content = Text.assemble(title, "\n", subtitle)
    console.print(Panel(content, box=box.DOUBLE_EDGE, style="cyan", padding=(1, 4)))


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

def print_ticker_resolved(info: dict) -> None:
    """Print a green panel showing the resolved ticker details."""
    content = (
        f"[bold]Ticker :[/bold]  [green]{info.get('ticker', 'N/A')}[/green]\n"
        f"[bold]Exchange:[/bold]  {info.get('exchange', 'N/A')}\n"
        f"[bold]Company :[/bold]  {info.get('full_name', 'N/A')}"
    )
    console.print(Panel(content, title="[bold green]Ticker Resolved[/bold green]", style="green"))


# ---------------------------------------------------------------------------
# Agent section dividers
# ---------------------------------------------------------------------------

def print_agent_start(agent_name: str) -> None:
    """Print a yellow divider rule when an agent section begins."""
    console.print()
    console.rule(f"[bold yellow]{agent_name}[/bold yellow]", style="yellow")
    console.print()


# ---------------------------------------------------------------------------
# Fundamental table
# ---------------------------------------------------------------------------

def _extract_score(text: str, label: str) -> str:
    """Pull 'X/10' score from a labelled line in a report."""
    pattern = re.compile(rf"{re.escape(label)}[^\d]*(\d+)[/\\]10", re.IGNORECASE)
    m = pattern.search(text)
    return f"{m.group(1)}/10" if m else "N/A"


def _extract_value(text: str, *patterns: str) -> str:
    """Extract first numeric value following any of the given keyword patterns."""
    for pat in patterns:
        m = re.search(rf"{pat}[^\d\-]*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    return "N/A"


def _score_color(score_str: str) -> str:
    try:
        s = float(score_str.split("/")[0])
        if s >= 7:
            return "green"
        if s >= 4:
            return "yellow"
        return "red"
    except (ValueError, IndexError):
        return "white"


def print_fundamental_table(report_text: str) -> None:
    """Parse the fundamental report and render key metrics in a Rich table."""
    text = _strip_think(report_text)

    table = Table(
        title="Fundamental Analysis — Key Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Metric", style="bold", min_width=28)
    table.add_column("Value", min_width=16)
    table.add_column("Signal", min_width=10)

    def _add(metric: str, value: str, signal_fn=None) -> None:
        signal = ""
        style = "white"
        if signal_fn and value != "N/A":
            try:
                v = float(value.replace(",", ""))
                sig, style = signal_fn(v)
                signal = sig
            except (ValueError, TypeError):
                pass
        table.add_row(metric, value, f"[{style}]{signal}[/{style}]")

    score = _extract_score(text, "Fundamental Score")
    sc = _score_color(score)
    table.add_row("Fundamental Score", f"[bold {sc}]{score}[/bold {sc}]", "")

    pe = _extract_value(text, r"P/E[^:]*:", r"Price.to.Earnings[^:]*:", r"PE ratio[^:]*:")
    _add("P/E Ratio", pe, lambda v: ("Expensive", "red") if v > 30 else (("Cheap", "green") if v < 15 else ("Fair", "yellow")))

    pb = _extract_value(text, r"P/B[^:]*:", r"Price.to.Book[^:]*:", r"PB ratio[^:]*:")
    _add("P/B Ratio", pb, lambda v: ("High", "yellow") if v > 3 else ("OK", "green"))

    de = _extract_value(text, r"Debt.to.Equity[^:]*:", r"D/E[^:]*:")
    _add("Debt/Equity", de, lambda v: ("High Leverage", "red") if v > 1.5 else ("Healthy", "green"))

    cr = _extract_value(text, r"Current Ratio[^:]*:")
    _add("Current Ratio", cr, lambda v: ("Low", "red") if v < 1 else ("OK", "green"))

    gm = _extract_value(text, r"Gross Margin[^:]*:", r"Gross Profit Margin[^:]*:")
    _add("Gross Margin %", gm, lambda v: ("Strong", "green") if v > 40 else (("OK", "yellow") if v > 20 else ("Weak", "red")))

    nm = _extract_value(text, r"Net (?:Profit )?Margin[^:]*:", r"Net Margin[^:]*:")
    _add("Net Margin %", nm, lambda v: ("Strong", "green") if v > 15 else (("OK", "yellow") if v > 5 else ("Weak", "red")))

    console.print(table)
    console.print()
    console.print(Panel(text, title="[magenta]Fundamental Analysis Report[/magenta]", style="magenta", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Technical table
# ---------------------------------------------------------------------------

def print_technical_table(report_text: str) -> None:
    """Parse the technical report and render indicators in a Rich table."""
    text = _strip_think(report_text)

    table = Table(
        title="Technical Analysis — Indicator Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
        expand=False,
    )
    table.add_column("Indicator", style="bold", min_width=24)
    table.add_column("Reading", min_width=14)
    table.add_column("Signal", min_width=18)

    score = _extract_score(text, "Technical Score")
    sc = _score_color(score)
    table.add_row("Technical Score", f"[bold {sc}]{score}[/bold {sc}]", "")

    # Trend
    trend_signal = "Bullish" if re.search(r"above.*SMA.*200|golden cross|uptrend", text, re.I) else (
        "Bearish" if re.search(r"below.*SMA.*200|death cross|downtrend", text, re.I) else "Neutral"
    )
    trend_style = {"Bullish": "green", "Bearish": "red", "Neutral": "yellow"}[trend_signal]
    table.add_row("Overall Trend", trend_signal, f"[{trend_style}]{trend_signal}[/{trend_style}]")

    rsi_val = _extract_value(text, r"RSI[^:]*:", r"RSI\(?14\)?[^:]*:")
    rsi_sig, rsi_style = ("Overbought", "red") if _gt(rsi_val, 70) else (
        ("Oversold", "green") if _lt(rsi_val, 30) else ("Neutral", "yellow")
    )
    table.add_row("RSI (14)", rsi_val, f"[{rsi_style}]{rsi_sig}[/{rsi_style}]")

    macd_signal_word = "Bullish" if re.search(r"bullish.*crossover|MACD.*above.*signal", text, re.I) else (
        "Bearish" if re.search(r"bearish.*crossover|MACD.*below.*signal", text, re.I) else "Neutral"
    )
    ms = {"Bullish": "green", "Bearish": "red", "Neutral": "yellow"}[macd_signal_word]
    table.add_row("MACD", macd_signal_word, f"[{ms}]{macd_signal_word}[/{ms}]")

    bb_pos = "Upper" if re.search(r"near.*upper|above.*upper band|upper band", text, re.I) else (
        "Lower" if re.search(r"near.*lower|below.*lower band|lower band", text, re.I) else "Middle"
    )
    bb_s = "yellow" if bb_pos == "Upper" else ("green" if bb_pos == "Lower" else "white")
    table.add_row("Bollinger Band Position", bb_pos, f"[{bb_s}]{bb_pos}[/{bb_s}]")

    vol_signal = "Above Avg" if re.search(r"above.*average.*volume|volume.*above", text, re.I) else (
        "Below Avg" if re.search(r"below.*average.*volume|volume.*below", text, re.I) else "N/A"
    )
    table.add_row("Volume vs Avg", vol_signal, "")

    console.print(table)
    console.print()
    console.print(Panel(text, title="[blue]Technical Analysis Report[/blue]", style="blue", padding=(1, 2)))


def _gt(val: str, threshold: float) -> bool:
    try:
        return float(val) > threshold
    except (ValueError, TypeError):
        return False


def _lt(val: str, threshold: float) -> bool:
    try:
        return float(val) < threshold
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Final recommendation
# ---------------------------------------------------------------------------

def _extract_price(text: str, *patterns: str) -> str:
    """Extract first price value (with optional ₹ prefix) after any of the given patterns."""
    for pat in patterns:
        m = re.search(rf"{pat}[^\d₹]*₹?\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            return "₹" + m.group(1).replace(",", "")
    return "N/A"


def print_final_recommendation(report_text: str) -> None:
    """Display the synthesis report with a prominent BUY/HOLD/SELL panel."""
    text = _strip_think(report_text)

    # Detect verdict
    verdict = "HOLD"
    if re.search(r"RECOMMENDATION:\s*BUY", text, re.IGNORECASE):
        verdict = "BUY"
    elif re.search(r"RECOMMENDATION:\s*SELL", text, re.IGNORECASE):
        verdict = "SELL"

    verdict_styles = {
        "BUY": ("bold green", "green", "green"),
        "HOLD": ("bold yellow", "yellow", "yellow"),
        "SELL": ("bold red", "red", "red"),
    }
    text_style, border_style, _ = verdict_styles[verdict]

    # Extract conviction and risk if present
    conviction_m = re.search(r"Conviction Level[:\s]+([^\n]+)", text, re.IGNORECASE)
    risk_m = re.search(r"Risk Level[:\s]+([^\n]+)", text, re.IGNORECASE)
    conviction = conviction_m.group(1).strip() if conviction_m else "N/A"
    risk = risk_m.group(1).strip() if risk_m else "N/A"

    # Prominent verdict panel
    verdict_content = Text.assemble(
        Text(f"  {verdict}  ", style=f"bold white on {border_style}", justify="center"),
        "\n\n",
        Text(f"Conviction: {conviction}", style=text_style),
        "   |   ",
        Text(f"Risk: {risk}", style=text_style),
    )
    console.print()
    console.print(
        Panel(
            verdict_content,
            title=f"[{text_style}]FINAL RECOMMENDATION[/{text_style}]",
            style=border_style,
            box=box.DOUBLE_EDGE,
            padding=(1, 6),
        )
    )

    # Action Prices table
    price_table = Table(
        title="Action Prices & Price Targets",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    price_table.add_column("Level", style="bold", min_width=30)
    price_table.add_column("Price", min_width=16)
    price_table.add_column("Note", min_width=28)

    cmp = _extract_price(text, r"Current Market Price")
    price_table.add_row("Current Market Price", f"[bold]{cmp}[/bold]", "Live market price")

    buy_zone_m = re.search(
        r"Suggested Buy Zone[^₹\d]*₹?\s*([\d,]+\.?\d*)\s*[–\-]+\s*₹?\s*([\d,]+\.?\d*)",
        text, re.IGNORECASE,
    )
    if buy_zone_m:
        buy_zone = f"₹{buy_zone_m.group(1)} – ₹{buy_zone_m.group(2)}"
    else:
        buy_zone = _extract_price(text, r"Suggested Buy Zone", r"Buy Zone")
    price_table.add_row("Suggested Buy Zone", f"[green]{buy_zone}[/green]", "Entry range")

    stop_loss = _extract_price(text, r"Stop Loss[^:]*:", r"Stop.Loss")
    stop_pct_m = re.search(r"Stop Loss[^%\d]*(\d+\.?\d*)%\s*below", text, re.IGNORECASE)
    stop_note = f"{stop_pct_m.group(1)}% below entry" if stop_pct_m else "Cut loss here"
    price_table.add_row("Stop Loss", f"[red]{stop_loss}[/red]", stop_note)

    st_target = _extract_price(text, r"Short.term (?:Profit )?Target", r"Short.term Target")
    price_table.add_row("Short-term Target (1-3 mo)", f"[yellow]{st_target}[/yellow]", "First resistance")

    mt_target = _extract_price(text, r"Medium.term (?:Profit )?Target", r"Medium.term Target")
    price_table.add_row("Medium-term Target (3-12 mo)", f"[yellow]{mt_target}[/yellow]", "Second resistance")

    t6m = _extract_price(text, r"6.Month Price Target", r"6.Month Target")
    price_table.add_row("6-Month Price Target", f"[cyan]{t6m}[/cyan]", "Technical + fundamental")

    t1y = _extract_price(text, r"1.Year Price Target", r"1.Year Target")
    price_table.add_row("1-Year Price Target", f"[cyan]{t1y}[/cyan]", "Valuation + analyst target")

    console.print()
    console.print(price_table)
    console.print()
    console.print(Panel(text, title="[bold]Investment Synthesis Report[/bold]", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Sentiment panel
# ---------------------------------------------------------------------------

def print_sentiment_panel(report_text: str) -> None:
    """Display the sentiment report with an OVERALL SENTIMENT badge."""
    text = _strip_think(report_text)

    # Parse overall sentiment
    sent_m = re.search(
        r"Overall Sentiment[:\s]+([A-Z ]+(?:BULLISH|BEARISH|NEUTRAL))",
        text, re.IGNORECASE,
    )
    overall = sent_m.group(1).strip().upper() if sent_m else "NEUTRAL"

    # Score
    score_m = re.search(r"Sentiment Score[:\s]*(\d+(?:\.\d+)?)/10", text, re.IGNORECASE)
    score_str = f"{score_m.group(1)}/10" if score_m else "N/A"

    # Color mapping
    if "STRONGLY BULLISH" in overall:
        badge_style, border_style = "bold white on green", "green"
    elif "BULLISH" in overall:
        badge_style, border_style = "bold green", "green"
    elif "STRONGLY BEARISH" in overall:
        badge_style, border_style = "bold white on red", "red"
    elif "BEARISH" in overall:
        badge_style, border_style = "bold red", "red"
    else:
        badge_style, border_style = "bold yellow", "yellow"

    badge_content = Text.assemble(
        Text(f"  {overall}  ", style=badge_style, justify="center"),
        "\n\n",
        Text(f"Sentiment Score: {score_str}", style=border_style),
    )
    console.print()
    console.print(
        Panel(
            badge_content,
            title="[bold]MARKET SENTIMENT[/bold]",
            style=border_style,
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )
    )
    console.print()
    console.print(Panel(text, title="[bold]Sentiment Analysis Report[/bold]", style="cyan", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Risk table
# ---------------------------------------------------------------------------

def print_risk_table(risk_ctx: str) -> None:
    """Parse the VERIFIED RISK DATA block and render as a structured Rich table."""
    table = Table(
        title="Quantitative Risk Analysis",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold red",
        expand=False,
    )
    table.add_column("Metric", style="bold", min_width=28)
    table.add_column("Value", min_width=18)
    table.add_column("Risk Signal", min_width=16)

    def _parse(label: str) -> str:
        # Use a flexible regex that allows extra text between label and colon
        m = re.search(rf"{re.escape(label)}[^:\n]*:\s*([^\n]+)", risk_ctx, re.IGNORECASE)
        if not m:
            return "N/A"
        # Take only the value portion before any explanatory '(' comment
        parts = m.group(1).strip().split("  ")
        return parts[0].strip()

    def _add_row(metric: str, value: str, signal: str = "", sig_style: str = "white") -> None:
        table.add_row(metric, value, f"[{sig_style}]{signal}[/{sig_style}]")

    vol = _parse("Annualized Volatility")
    mdd = _parse("Max Drawdown")
    sharpe = _parse("Sharpe Ratio")
    beta = _parse("Beta")
    de = _parse("Debt-to-Equity")
    cr = _parse("Current Ratio")
    composite_m = re.search(r"Composite Risk Score\s*:\s*([\d.]+)/10", risk_ctx, re.IGNORECASE)
    composite_str = composite_m.group(1) if composite_m else "N/A"
    risk_level_m = re.search(r"Overall Risk Level\s*:\s*(\w+)", risk_ctx, re.IGNORECASE)
    risk_level = risk_level_m.group(1) if risk_level_m else "N/A"

    _add_row("Annualized Volatility", vol)
    _add_row("Max Drawdown (1-2yr)", mdd)
    _add_row("Sharpe Ratio", sharpe)
    _add_row("Beta (Market Risk)", beta)
    _add_row("Debt-to-Equity", de)
    _add_row("Current Ratio", cr)

    # Composite score with color
    try:
        cs = float(composite_str)
        cs_style = "green" if cs < 4 else ("yellow" if cs <= 7 else "red")
        cs_signal = "Low Risk" if cs < 4 else ("Medium Risk" if cs <= 7 else "High Risk")
    except (ValueError, TypeError):
        cs_style, cs_signal = "white", ""

    table.add_row(
        "Composite Risk Score",
        f"[bold {cs_style}]{composite_str}/10[/bold {cs_style}]",
        f"[bold {cs_style}]{cs_signal}[/bold {cs_style}]",
    )
    rl_style = "green" if risk_level == "Low" else ("yellow" if risk_level == "Medium" else "red")
    table.add_row(
        "Overall Risk Level",
        f"[bold {rl_style}]{risk_level}[/bold {rl_style}]",
        "",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Devil's Advocate panel
# ---------------------------------------------------------------------------

def print_devil_advocate_panel(report_text: str) -> None:
    """Display the Devil's Advocate counter-report with a CHALLENGE badge."""
    text = _strip_think(report_text)

    # Detect challenge level
    challenge_m = re.search(r"CHALLENGE[:\s]+(STRONG|MODERATE|WEAK)", text, re.IGNORECASE)
    challenge = challenge_m.group(1).upper() if challenge_m else "MODERATE"

    challenge_styles = {
        "STRONG": ("bold white on red", "red"),
        "MODERATE": ("bold white on yellow", "yellow"),
        "WEAK": ("bold white on green", "green"),
    }
    badge_style, border_style = challenge_styles.get(challenge, ("bold yellow", "yellow"))

    challenge_labels = {
        "STRONG": "STRONG CHALLENGE — Primary thesis has material flaws",
        "MODERATE": "MODERATE CHALLENGE — Notable weaknesses worth flagging",
        "WEAK": "WEAK CHALLENGE — Thesis is solid; only minor concerns",
    }
    label = challenge_labels.get(challenge, f"CHALLENGE: {challenge}")

    badge_content = Text.assemble(
        Text(f"  CHALLENGE: {challenge}  ", style=badge_style, justify="center"),
        "\n\n",
        Text(label, style=border_style),
    )
    console.print()
    console.print(
        Panel(
            badge_content,
            title="[bold]DEVIL'S ADVOCATE COUNTER-REPORT[/bold]",
            style=border_style,
            box=box.DOUBLE_EDGE,
            padding=(1, 4),
        )
    )
    console.print()
    console.print(
        Panel(
            text,
            title="[bold]Devil's Advocate Analysis[/bold]",
            style=border_style,
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

def print_error(msg: str) -> None:
    """Print an error message in a red panel."""
    console.print(Panel(f"[bold red]{msg}[/bold red]", title="[red]Error[/red]", style="red"))


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def save_report(ticker: str, full_report: str) -> Path:
    """Save the full analysis report as a markdown file in reports/."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = reports_dir / f"{ticker}_{timestamp}.md"
    filepath.write_text(full_report, encoding="utf-8")
    return filepath
