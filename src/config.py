"""Project paths and environment settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("OLIST_DB_PATH", ROOT / "db" / "olist.duckdb"))
SCHEMA_PATH = ROOT / "schema.md"
# run_sql safety limits: rows returned per query, and seconds before a query is cancelled.
MAX_ROWS = int(os.getenv("MAX_ROWS", "1000"))
QUERY_TIMEOUT_S = float(os.getenv("QUERY_TIMEOUT_S", "10"))
SCHEMA_COMPACT_PATH = ROOT / "schema_compact.md"
# Schema explorer: hand-written keys/traps, and precomputed column stats (scripts/export_schema_stats.py).
SCHEMA_META_PATH = ROOT / "schema_meta.json"
SCHEMA_STATS_PATH = ROOT / "schema_stats.json"
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

# Local caches (gitignored): daily token ledger and live-run traces.
CACHE_DIR = Path(os.getenv("ANALYST_CACHE_DIR", ROOT / ".cache"))
# Committed zero-token example traces shown in the app.
EXAMPLES_DIR = ROOT / "examples"
# Daily token budgets per model: local/eval use, and a smaller one for the deployed app.
DAILY_TOKEN_BUDGET = int(os.getenv("DAILY_TOKEN_BUDGET", "200000"))
APP_DAILY_TOKEN_BUDGET = int(os.getenv("APP_DAILY_TOKEN_BUDGET", "100000"))
BUDGET_STOP_FRACTION = 0.9
# Planner question gate (JSON {answerable, reason, plan}); off by default so evals are unchanged.
PLANNER_GATE = os.getenv("PLANNER_GATE", "1").lower() in ("1", "true", "yes")
