import duckdb
import pytest

from src.config import DB_PATH
from src.db_bootstrap import PARQUET_DIR, build_from_parquet, ensure_db


def schema(path):
    with duckdb.connect(str(path), read_only=True) as con:
        cols = con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema='main' ORDER BY 1, ordinal_position"
        ).fetchall()
        counts = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in {c[0] for c in cols}}
    return cols, counts


@pytest.mark.skipif(not DB_PATH.exists(), reason="needs the original db/olist.duckdb")
def test_parquet_rebuild_matches_original(tmp_path):
    built = tmp_path / "olist.duckdb"
    assert ensure_db(built) is True and ensure_db(built) is False
    assert schema(built) == schema(DB_PATH)


def test_missing_parquet_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_from_parquet(tmp_path / "x.duckdb", tmp_path)
    assert not (tmp_path / "x.duckdb").exists()


def test_parquet_is_committed_size():
    total = sum(f.stat().st_size for f in PARQUET_DIR.glob("*.parquet"))
    assert len(list(PARQUET_DIR.glob("*.parquet"))) == 9 and total < 100e6
