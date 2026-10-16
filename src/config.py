"""Project paths and environment settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("OLIST_DB_PATH", ROOT / "db" / "olist.duckdb"))
SCHEMA_PATH = ROOT / "schema.md"
SCHEMA_COMPACT_PATH = ROOT / "schema_compact.md"
# Which schema goes into LLM prompts: "compact" (fewer tokens) or "full" (schema.md).
PROMPT_SCHEMA = os.getenv("PROMPT_SCHEMA", "compact").lower()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
# Groq gpt-oss reasoning effort: "low" (frozen eval default) | "medium" | "high"; "none" = don't send it.
_effort = os.getenv("REASONING_EFFORT", "").lower() or "low"
REASONING_EFFORT = None if _effort == "none" else _effort
# Per-agent Groq models (Gemini always uses GEMINI_MODEL).
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "openai/gpt-oss-120b")
DATA_MODEL = os.getenv("DATA_MODEL", "openai/gpt-oss-120b")
REPORTER_MODEL = os.getenv("REPORTER_MODEL", "openai/gpt-oss-20b")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "openai/gpt-oss-120b")
MAX_CRITIC_ROUNDS = 3
