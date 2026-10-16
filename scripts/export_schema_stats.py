"""Precompute per-table and per-column stats for the Schema explorer into schema_stats.json (committed).

Run after build_db.py: python scripts/export_schema_stats.py
"""
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH, SCHEMA_STATS_PATH  # noqa: E402

EXAMPLES = 3
MAX_CHARS = 60


def short(value: object) -> str:
    text = str(value)
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def column_stats(con: duckdb.DuckDBPyConnection, table: str, column: str, dtype: str, rows: int) -> dict:
    nulls, distinct = con.execute(f'SELECT COUNT(*) - COUNT("{column}"), COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()
    examples = con.execute(
        f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL '
        f'GROUP BY 1 ORDER BY COUNT(*) DESC, 1 LIMIT {EXAMPLES}'
    ).fetchall()
    return {
        "column": column, "type": dtype, "null_pct": round(100 * nulls / rows, 2) if rows else 0.0,
        "distinct": distinct, "examples": [short(e[0]) for e in examples],
    }


def table_stats(con: duckdb.DuckDBPyConnection, table: str) -> dict:
    rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    cols = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position", [table],
    ).fetchall()
    return {"rows": rows, "columns": [column_stats(con, table, c, t, rows) for c, t in cols]}


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY 1").fetchall()]
        stats = {"tables": {t: table_stats(con, t) for t in tables}}
    SCHEMA_STATS_PATH.write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SCHEMA_STATS_PATH} ({len(tables)} tables)")


if __name__ == "__main__":
    main()
