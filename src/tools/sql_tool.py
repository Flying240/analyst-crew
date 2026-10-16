"""Read-only SQL execution against the Olist DuckDB database."""
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import duckdb

from src.config import DB_PATH, MAX_ROWS, QUERY_TIMEOUT_S

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

    @property
    def notice(self) -> str | None:
        """Plain-language note when the row limit cut the result."""
        return f"Result truncated to the first {self.row_count} rows (row limit)." if self.truncated else None


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


def run_sql(sql: str, limit: int | None = None, timeout_s: float | None = None) -> SqlResult:
    """Validate and run a read-only query: at most `limit` rows, cancelled after `timeout_s` seconds."""
    limit = MAX_ROWS if limit is None else limit
    timeout_s = QUERY_TIMEOUT_S if timeout_s is None else timeout_s
    if not is_read_only(sql):
        return SqlResult(error="Rejected: only a single read-only SELECT/WITH query is allowed.")
    timed_out = threading.Event()
    try:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            timer = threading.Timer(timeout_s, lambda: (timed_out.set(), con.interrupt()))
            timer.start()
            try:
                cur = con.execute(sql.strip().rstrip(";"))
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(limit + 1)
            finally:
                timer.cancel()
    except Exception as exc:  # surface DB errors to the caller/agent
        if timed_out.is_set():
            return SqlResult(error=f"Timeout: query cancelled after {timeout_s:g}s; simplify it or aggregate more.")
        return SqlResult(error=f"{type(exc).__name__}: {exc}")
    truncated = len(rows) > limit
    rows = rows[:limit]
    return SqlResult(columns=columns, rows=rows, row_count=len(rows), truncated=truncated)
