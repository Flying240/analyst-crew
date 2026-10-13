"""Load all Olist CSVs from data/ into db/olist.duckdb."""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, DB_PATH  # noqa: E402

TABLES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_table(con: duckdb.DuckDBPyConnection, table: str, csv: Path) -> int:
    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
        [str(csv)],
    )
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def main() -> None:
    missing = [f for f in TABLES.values() if not (DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"Missing CSVs in {DATA_DIR}: {missing}. Run scripts/download_data.py first.")
    DB_PATH.parent.mkdir(exist_ok=True)
    with duckdb.connect(str(DB_PATH)) as con:
        for table, file in TABLES.items():
            print(f"{table:<22} {load_table(con, table, DATA_DIR / file):>10,} rows")
    print(f"Built {DB_PATH}")


if __name__ == "__main__":
    main()
