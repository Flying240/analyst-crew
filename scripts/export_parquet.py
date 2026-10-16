"""Export every table in db/olist.duckdb to data/parquet/<table>.parquet (committed, so a host can rebuild the DB).

Usage:
    python scripts/export_parquet.py
Fails if any file is over 50 MB (GitHub warns at 50 MB and blocks 100 MB).
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH  # noqa: E402
from src.db_bootstrap import PARQUET_DIR  # noqa: E402

MAX_FILE_MB = 50


def main() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1").fetchall()]
        for t in tables:
            out = PARQUET_DIR / f"{t}.parquet"
            con.execute(f"COPY \"{t}\" TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
            size = out.stat().st_size
            total += size
            print(f"{t:<22} {size / 1e6:8.2f} MB")
            if size > MAX_FILE_MB * 1e6:
                sys.exit(f"{out.name} is over {MAX_FILE_MB} MB")
    print(f"Total {total / 1e6:.2f} MB in {PARQUET_DIR}")


if __name__ == "__main__":
    main()
