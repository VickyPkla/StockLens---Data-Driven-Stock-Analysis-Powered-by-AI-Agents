# Stock Analyser

A multi-agent AI stock analysis CLI. Three sequential CrewAI agents — a Fundamental Analyst, a Technical Analyst, and a Synthesis Strategist — collaborate to produce a structured **BUY / HOLD / SELL** recommendation for any US or Indian stock.

**Powered by:** CrewAI · Ollama (`deepseek-r1:7b`) · Alpha Vantage · yfinance · Rich

---

## Architecture

```
User Input
    │
    ▼
Ticker Resolver (yfinance Search)
    │
    ▼
Agent 1: Fundamental Analyst
    │   Tools: Alpha Vantage (income, balance sheet, cash flow, overview)
    │          yfinance (stock info)
    ▼
Agent 2: Technical Analyst
    │   Tools: yfinance (price history, indicators, news)
    ▼
Agent 3: Synthesis Strategist
        Context: outputs from Agents 1 & 2
        Output: BUY / HOLD / SELL + conviction + risk
```

---

## Prerequisites

### 1. Python 3.10+

crewai requires Python ≥ 3.10. Check with `python3 --version`.

If you need to upgrade, use [pyenv](https://github.com/pyenv/pyenv):
```bash
pyenv install 3.12.7
pyenv local 3.12.7
```

### 2. Ollama

Install Ollama from [https://ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull deepseek-r1:7b
```

Verify Ollama is running:
```bash
ollama list   # should show deepseek-r1:7b
```

### 3. Alpha Vantage API Key (free)

Get a free key at [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key).

Free tier: 25 calls/day, 5 calls/minute. The app enforces the rate limit automatically.

> **Note for Indian stocks (.NS / .BO):** Alpha Vantage coverage for NSE/BSE tickers is limited. The app automatically falls back to yfinance for any missing data.

---

## Setup

```bash
# 1. Clone / navigate to project
cd "Stock Analyser/stock_analyser"

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ALPHA_VANTAGE_API_KEY

# 5. Ensure Ollama is running
ollama serve &   # or start the Ollama desktop app
```

---

## Usage

```bash
# Analyse by company name
python main.py "Reliance Industries"
python main.py "Tata Consultancy Services"
python main.py "Apple"
python main.py "NVIDIA"

# Analyse by ticker symbol directly
python main.py TCS.NS
python main.py AAPL
python main.py TSLA

# Interactive mode (no argument)
python main.py
```

Reports are saved to `reports/` as markdown files with timestamps.

---

## Expected Runtime

| Phase | Typical Duration |
|-------|-----------------|
| Ticker resolution | < 5 seconds |
| Alpha Vantage data fetch (4 calls × 12s) | ~50 seconds |
| Fundamental agent analysis | 2–5 minutes |
| Technical agent analysis | 1–3 minutes |
| Synthesis agent | 1–2 minutes |
| **Total** | **~5–15 minutes** |

---

## Project Structure

```
stock_analyser/
├── main.py                    # CLI entrypoint
├── requirements.txt
├── .env.example
├── README.md
├── config/
│   └── settings.py            # Config constants + AV rate limiter
├── tools/
│   ├── ticker_resolver.py     # Company name → ticker symbol
│   ├── alpha_vantage_tools.py # AV API tools (with yfinance fallback)
│   └── yfinance_tools.py      # Price, indicators, news tools
├── agents/
│   ├── fundamental_agent.py   # Agent 1: Fundamental Analyst
│   ├── technical_agent.py     # Agent 2: Technical Analyst
│   └── synthesis_agent.py     # Agent 3: Synthesis Strategist
├── crew/
│   └── stock_crew.py          # Sequential crew wiring
├── display/
│   └── rich_output.py         # Rich terminal rendering
└── reports/                   # Auto-created; stores .md reports
```

---

## Troubleshooting

**`ollama: connection refused`** — Start Ollama: `ollama serve`

**`model not found`** — Pull the model: `ollama pull deepseek-r1:7b`

**Alpha Vantage returns empty data for Indian stocks** — This is expected. The app silently falls back to yfinance for .NS / .BO tickers.

**`ModuleNotFoundError`** — Ensure your virtual environment is activated and `pip install -r requirements.txt` completed successfully.

**Slow responses** — `deepseek-r1:7b` requires ~8 GB RAM. Close other applications to free memory. Consider `deepseek-r1:1.5b` for faster (but lower quality) results by changing `OLLAMA_MODEL` in `.env`.

---

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always conduct your own due diligence and consult a licensed financial advisor before making investment decisions.
