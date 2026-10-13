"""Read-only SQL execution against the Olist DuckDB database."""
import re
from dataclasses import dataclass, field
from typing import Any

import duckdb

from src.config import DB_PATH

FORBIDDEN = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "PRAGMA", "INSTALL", "LOAD",
    "SET", "CALL", "CHECKPOINT", "VACUUM", "GRANT", "MERGE",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN) + r")\b", re.IGNORECASE)


@dataclass
class SqlResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _strip(sql: str) -> str:
    """Remove comments and quoted literals so keywords inside them are ignored."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    return sql.strip()


def is_read_only(sql: str) -> bool:
    """True only for a single SELECT/WITH statement with no write keywords."""
    clean = _strip(sql).rstrip(";").strip()
    if not clean or ";" in clean:
        return False
    if not re.match(r"^(SELECT|WITH)\b", clean, re.IGNORECASE):
        return False
    return _FORBIDDEN_RE.search(clean) is None


def run_sql(sql: str, limit: int = 1000) -> SqlResult:
    """Validate and run a read-only query, returning at most `limit` rows."""
    if not is_read_only(sql):
        return SqlResult(error="Rejected: only a single read-only SELECT/WITH query is allowed.")
    try:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            cur = con.execute(sql.strip().rstrip(";"))
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(limit + 1)
    except Exception as exc:  # surface DB errors to the caller/agent
        return SqlResult(error=f"{type(exc).__name__}: {exc}")
    truncated = len(rows) > limit
    rows = rows[:limit]
    return SqlResult(columns=columns, rows=rows, row_count=len(rows), truncated=truncated)
