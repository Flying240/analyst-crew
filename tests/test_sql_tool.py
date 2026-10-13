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
