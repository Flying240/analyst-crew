"""Build db/olist.duckdb from the committed Parquet export when the database file is missing (e.g. on a fresh host)."""
from pathlib import Path

import duckdb

from src.config import DB_PATH, ROOT

PARQUET_DIR = ROOT / "data" / "parquet"


def build_from_parquet(db_path: Path = DB_PATH, parquet_dir: Path = PARQUET_DIR) -> list[str]:
    """Create one table per data/parquet/<table>.parquet; return the table names."""
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {parquet_dir}; run scripts/export_parquet.py")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    with duckdb.connect(str(tmp)) as con:
        for f in files:
            con.execute(f'CREATE TABLE "{f.stem}" AS SELECT * FROM read_parquet(?)', [str(f)])
    tmp.replace(db_path)  # a half-built file never sits at DB_PATH
    return [f.stem for f in files]


def ensure_db(db_path: Path = DB_PATH, parquet_dir: Path = PARQUET_DIR) -> bool:
    """Return True if the database had to be built."""
    if db_path.exists():
        return False
    build_from_parquet(db_path, parquet_dir)
    return True
