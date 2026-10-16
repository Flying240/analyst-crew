import json
import subprocess
import sys

from src.config import ROOT, SCHEMA_STATS_PATH
from src.tools.sql_tool import table_names
from src.ui import schema_info


def test_meta_and_stats_cover_every_table():
    assert set(schema_info.meta()) == set(schema_info.stats()) == set(table_names())


def test_keys_and_references_are_real_columns():
    cols = {t: {c["column"] for c in s["columns"]} for t, s in schema_info.stats().items()}
    for t, m in schema_info.meta().items():
        assert set(m["primary_key"]) <= cols[t]
    for child, parent, _ in schema_info.links():
        for ref in (child, parent):
            table, column = ref.split(".")
            assert column in cols[table], ref


def test_join_keys_match_schema_compact():
    text = (ROOT / "schema_compact.md").read_text(encoding="utf-8")
    for child, parent, soft in schema_info.links():
        if not soft:
            assert child.split(".")[1] in text and parent in text.replace("order_items/payments/reviews.", "orders.")


def test_primary_keys_are_unique_in_the_data():
    from src.tools.sql_tool import run_sql

    for t, m in schema_info.meta().items():
        if m["primary_key"]:
            keys = ", ".join(m["primary_key"])
            n, distinct = run_sql(f"SELECT COUNT(*), COUNT(DISTINCT ({keys})) FROM {t}").rows[0]
            assert n == distinct, t


def test_er_dot_has_every_table_and_edge():
    dot = schema_info.er_dot()
    assert all(f"  {t} [label=" in dot for t in schema_info.meta())
    assert "orders -> order_items" in dot and "customers -> orders" in dot and "style=dashed" in dot


def test_traps_cover_known_pitfalls():
    traps = " ".join(t for m in schema_info.meta().values() for t in m["traps"])
    for phrase in ("customer_unique_id", "NULL category", "more than one review", "no product names"):
        assert phrase in traps


def test_column_table_has_stats():
    row = schema_info.column_table("products")[1]
    assert row["column"] == "product_category_name" and row["% NULL"] > 0 and len(row["examples"].split(", ")) == 3


def test_export_script_matches_committed_stats(tmp_path, monkeypatch):
    committed = json.loads(SCHEMA_STATS_PATH.read_text(encoding="utf-8"))
    out = tmp_path / "stats.json"
    code = ("import scripts.export_schema_stats as e, pathlib; "
            f"e.SCHEMA_STATS_PATH = pathlib.Path(r'{out}'); e.main()")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)
    assert json.loads(out.read_text(encoding="utf-8")) == committed
