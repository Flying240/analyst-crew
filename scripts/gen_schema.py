"""Generate schema.md from db/olist.duckdb for agents to read."""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH, SCHEMA_PATH  # noqa: E402

RULES = """## Business rules
- Revenue = `order_items.price + order_items.freight_value`, delivered orders only (`orders.order_status = 'delivered'`).
- Count customers with `customers.customer_unique_id`, not `customer_id` (customer_id is per order).
- Trend analysis: exclude 2016 and Sept-Oct 2018 (incomplete months); use 2017-01 to 2018-08.
- Category names are Portuguese; join `category_translation` for English.
- Late = `orders.order_delivered_customer_date > orders.order_estimated_delivery_date` (timestamp comparison, not calendar date).

## Joins
- orders.customer_id -> customers.customer_id
- order_items.order_id / payments.order_id / reviews.order_id -> orders.order_id
- order_items.product_id -> products.product_id
- order_items.seller_id -> sellers.seller_id
- products.product_category_name -> category_translation.product_category_name
"""


def table_section(con: duckdb.DuckDBPyConnection, table: str) -> str:
    rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    cols = con.execute(f"DESCRIBE {table}").fetchall()
    lines = [f"### {table} ({rows:,} rows)", "| column | type |", "|---|---|"]
    lines += [f"| {c[0]} | {c[1]} |" for c in cols]
    return "\n".join(lines)


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        sections = [table_section(con, t) for t in tables]
    body = "# Olist schema (DuckDB)\n\n" + RULES + "\n## Tables\n\n" + "\n\n".join(sections) + "\n"
    SCHEMA_PATH.write_text(body, encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
