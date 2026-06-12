import os
import time
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")

# Alpha Vantage free tier allows 5 calls/minute (25/day).
# Sleeping 12 seconds between every call ensures we never exceed the rate limit.
def av_rate_limit() -> None:
    time.sleep(12)

