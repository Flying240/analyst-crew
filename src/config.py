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
# Groq gpt-oss reasoning effort: "low" | "medium" | "high". Empty = don't send it (provider default).
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "").lower() or None
