"""Schema explorer data: committed keys/traps (schema_meta.json) and precomputed stats (schema_stats.json)."""
import json
from functools import lru_cache
from typing import Any

from src.config import SCHEMA_META_PATH, SCHEMA_STATS_PATH


@lru_cache(maxsize=1)
def meta() -> dict[str, Any]:
    return json.loads(SCHEMA_META_PATH.read_text(encoding="utf-8"))["tables"]


@lru_cache(maxsize=1)
def stats() -> dict[str, Any]:
    return json.loads(SCHEMA_STATS_PATH.read_text(encoding="utf-8"))["tables"]


def overview() -> list[dict[str, Any]]:
    """One row per table: name, rows, columns, primary key, description."""
    return [
        {
            "table": t, "rows": stats()[t]["rows"], "columns": len(stats()[t]["columns"]),
            "primary key": ", ".join(m["primary_key"]) or "none", "description": m["description"],
        }
        for t, m in sorted(meta().items())
    ]


def links(table: str | None = None) -> list[tuple[str, str, bool]]:
    """(from "table.column", to "table.column", soft) for every relationship touching `table` (all if None)."""
    out = []
    for t, m in sorted(meta().items()):
        for fk in m["foreign_keys"]:
            out.append((f"{t}.{fk['column']}", fk["references"], False))
        for fk in m.get("soft_links", []):
            out.append((f"{t}.{fk['column']}", fk["references"], True))
    return [l for l in out if table is None or table in (l[0].split(".")[0], l[1].split(".")[0])]


def er_dot() -> str:
    """Graphviz ER diagram: one box per table with its key columns; edges from foreign key to referenced key."""
    lines = ['digraph ER {', '  rankdir=LR; bgcolor="transparent";',
             '  node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];',
             '  edge [fontname="Helvetica", fontsize=9, arrowhead=crow, arrowtail=none];']
    for t, m in sorted(meta().items()):
        keys = [f"PK {', '.join(m['primary_key'])}" if m["primary_key"] else "no key"]
        keys += [f"FK {fk['column']}" for fk in m["foreign_keys"]]
        label = r"\n".join([t, f"{stats()[t]['rows']:,} rows", *keys])
        lines.append(f'  {t} [label="{label}"];')
    for child, parent, soft in links():
        style = ', style=dashed, label="zip prefix"' if soft else f', label="{child.split(".")[1]}"'
        # parent -> child with a crow's foot at the "many" end
        lines.append(f'  {parent.split(".")[0]} -> {child.split(".")[0]} [{style[2:]}];')
    lines.append("}")
    return "\n".join(lines)


def column_table(table: str) -> list[dict[str, Any]]:
    return [
        {"column": c["column"], "type": c["type"], "% NULL": c["null_pct"], "distinct": c["distinct"],
         "examples": ", ".join(c["examples"])}
        for c in stats()[table]["columns"]
    ]
