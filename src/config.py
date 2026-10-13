"""Project paths and environment settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("OLIST_DB_PATH", ROOT / "db" / "olist.duckdb"))
SCHEMA_PATH = ROOT / "schema.md"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
