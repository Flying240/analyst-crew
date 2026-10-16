import pytest

from src.tools.sql_tool import is_read_only, run_sql

ALLOWED = [
    "SELECT 1",
    "select * from orders limit 5;",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT 'drop table orders' AS note",
    "SELECT 1 -- delete everything",
    "SELECT order_status FROM orders WHERE order_status = 'created'",
]

BLOCKED = [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "INSERT INTO orders VALUES (1)",
    "UPDATE orders SET order_status = 'x'",
    "ALTER TABLE orders ADD COLUMN x INT",
    "SELECT 1; DROP TABLE orders",
    "SELECT 1; SELECT 2",
    "/* hi */ DELETE FROM orders",
    "CREATE TABLE x AS SELECT 1",
    "WITH t AS (SELECT 1) DELETE FROM orders",
    "COPY orders TO 'out.csv'",
    "ATTACH 'other.db'",
    "PRAGMA database_list",
    "",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql: str) -> None:
    assert is_read_only(sql)


@pytest.mark.parametrize("sql", BLOCKED)
def test_blocked(sql: str) -> None:
    assert not is_read_only(sql)


def test_run_sql_rejects_write() -> None:
    result = run_sql("DROP TABLE orders")
    assert not result.ok and "Rejected" in result.error


def test_run_sql_truncates_at_limit() -> None:
    result = run_sql("SELECT * FROM range(50)", limit=10)
    assert result.ok and result.row_count == 10 and result.truncated
    assert "truncated" in result.notice


def test_run_sql_under_limit_not_truncated() -> None:
    result = run_sql("SELECT * FROM range(10)", limit=10)
    assert result.row_count == 10 and not result.truncated and result.notice is None


def test_run_sql_default_limit_is_max_rows() -> None:
    from src.config import MAX_ROWS

    result = run_sql(f"SELECT * FROM range({MAX_ROWS + 5})")
    assert result.row_count == MAX_ROWS and result.truncated


def test_run_sql_times_out_slow_query() -> None:
    import time

    slow = "SELECT COUNT(*) FROM range(100000000) a, range(100000000) b WHERE a.range * b.range = 7"
    start = time.monotonic()
    result = run_sql(slow, timeout_s=0.5)
    assert not result.ok and result.error.startswith("Timeout")
    assert time.monotonic() - start < 10


def test_connection_is_read_only_even_without_sql_check() -> None:
    import duckdb

    from src.config import DB_PATH

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        with pytest.raises(duckdb.Error, match="read-only"):
            con.execute("CREATE TABLE should_fail (x INT)")
