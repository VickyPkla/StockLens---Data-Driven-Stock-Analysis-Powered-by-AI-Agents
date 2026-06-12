import os
import uuid
import threading
import io
import sys
import re
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load env before imports
from dotenv import load_dotenv
load_dotenv()

# crewAI 1.14.x adds cache_breakpoint to messages for prompt caching but only strips
# it for Anthropic providers. Groq (via litellm) rejects this key on system messages.
# Patch _format_messages_for_provider to strip it for all providers.
from crewai.llm import LLM as _CrewLLM
_orig_fmt = _CrewLLM._format_messages_for_provider

def _fmt_strip_cache_breakpoint(self, messages):
    result = _orig_fmt(self, messages)
    for msg in result:
        msg.pop("cache_breakpoint", None)
    return result

_CrewLLM._format_messages_for_provider = _fmt_strip_cache_breakpoint

# Verify that GROQ_API_KEY is available
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in the environment or .env file.")

# Project imports
from tools.ticker_resolver import resolve_listings
from tools.data_prefetch import build_verified_context
from crew.stock_crew import run_analysis

app = FastAPI(title="Stock Analyser Web App Backend")

# In-memory status store
analyses = {}
analyses_lock = threading.Lock()

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
_BOX_CHARS = set('╭╮╰╯│─═╔╗╚╝║═')
_CREWAI_NOISE = re.compile(
    r'^(╭|╰|│\s*$|Name:|ID:|Final Output:|Info: Tracing|To enable tracing|Set tracing|Run: crewai)',
    re.IGNORECASE,
)

def _is_useful_log_line(line: str) -> bool:
    """Return True only for lines worth showing in the status log."""
    if not line:
        return False
    # Skip lines that are mostly box-drawing characters
    non_box = sum(1 for c in line if c not in _BOX_CHARS and not c.isspace())
    if non_box < 4:
        return False
    # Skip crewAI verbose panel noise
    if _CREWAI_NOISE.match(line):
        return False
    # Skip lines that are just dividers or crewAI panel borders starting with │
    if line.startswith('│') and len(line.replace('│', '').strip()) == 0:
        return False
    return True


class LogRedirector(io.TextIOBase):
    """Redirects stdout/stderr to collect meaningful logs for a specific analysis."""
    def __init__(self, analysis_id, original_stdout):
        self.analysis_id = analysis_id
        self.original_stdout = original_stdout

    def write(self, s):
        self.original_stdout.write(s)
        if s.strip():
            clean = _ANSI_RE.sub('', s)
            # Strip non-printable control chars (keep \n for splitting)
            clean = ''.join(c if c >= ' ' or c == '\n' else ' ' for c in clean)
            lines = [l.strip() for l in clean.split('\n') if _is_useful_log_line(l.strip())]
            if lines:
                with analyses_lock:
                    if self.analysis_id in analyses:
                        log = analyses[self.analysis_id]["logs"]
                        log.extend(lines)
                        # Keep only the last 200 lines to bound memory and response size
                        if len(log) > 200:
                            analyses[self.analysis_id]["logs"] = log[-200:]
        return len(s)

    def flush(self):
        self.original_stdout.flush()

def run_analysis_worker(analysis_id: str, ticker: str, company_name: str):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_redirector = LogRedirector(analysis_id, original_stdout)
    
    # Redirect stdout and stderr for the execution duration
    sys.stdout = log_redirector
    sys.stderr = log_redirector
    
    try:
        with analyses_lock:
            analyses[analysis_id]["status"] = "running"
            analyses[analysis_id]["logs"].append(f"Starting sequential multi-agent stock analysis for {company_name}...")

        # 1. Running the data pre-fetch context step
        print(f"[Backend] Pre-fetching verified market data for {ticker}...")
        tech_ctx, fund_ctx, sentiment_ctx, risk_ctx = build_verified_context(ticker)
        print("[Backend] Verified live market data pre-fetch completed.")
        
        # 2. Running the main sequential analysis graph
        result = run_analysis(ticker, company_name)
        
        with analyses_lock:
            analyses[analysis_id]["status"] = "completed"
            analyses[analysis_id]["results"] = result
            analyses[analysis_id]["logs"].append("Analysis finished! Generating interactive reports...")
            
    except Exception as exc:
        print(f"Error in background worker: {exc}")
        with analyses_lock:
            analyses[analysis_id]["status"] = "failed"
            analyses[analysis_id]["error"] = str(exc)
            analyses[analysis_id]["logs"].append(f"Analysis failed: {exc}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

@app.get("/api/analyze")
def start_analysis(stock: str, background_tasks: BackgroundTasks):
    if not stock:
        raise HTTPException(status_code=400, detail="Stock query parameter is required")

    try:
        listing_info = resolve_listings(stock)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ticker resolution failed: {exc}")

    # When the company has both an Indian and a US listing, ask the user to choose.
    if listing_info["status"] == "choose":
        return {"status": "choose", "options": listing_info["options"]}

    ticker = listing_info["ticker"]
    company_name = listing_info["full_name"]

    analysis_id = uuid.uuid4().hex

    with analyses_lock:
        analyses[analysis_id] = {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "company_name": company_name,
            "status": "pending",
            "logs": [],
            "results": None,
            "error": None
        }

    background_tasks.add_task(run_analysis_worker, analysis_id, ticker, company_name)

    return {
        "status": "started",
        "analysis_id": analysis_id,
        "ticker": ticker,
        "company_name": company_name,
    }

@app.get("/api/status/{analysis_id}")
def get_analysis_status(analysis_id: str):
    with analyses_lock:
        if analysis_id not in analyses:
            raise HTTPException(status_code=404, detail="Analysis session not found")
        return analyses[analysis_id]

# Set up static files and frontend
web_dir = Path("web")
web_dir.mkdir(exist_ok=True)
(web_dir / "assets").mkdir(exist_ok=True)

@app.get("/")
def serve_index():
    index_file = web_dir / "index.html"
    if not index_file.exists():
        return {"message": "Frontend files are still being generated. Please wait..."}
    return FileResponse(index_file)

# Mount the static files directory to serve CSS, JS, and graphics
app.mount("/", StaticFiles(directory="web"), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

