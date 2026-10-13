from evals.run_eval import score
from src.tools.sql_tool import SqlResult

GOLD_ROWS = SqlResult(columns=["category", "revenue"], rows=[("a", 10.0), ("b", 5.0)])
ROWS_CHECK = {"type": "rows", "tolerance": 0.01, "ordered": True}


def res(rows: list[tuple]) -> SqlResult:
    return SqlResult(rows=rows)


def test_swapped_columns_pass() -> None:
    assert score(res([(10.0, "a"), (5.0, "b")]), GOLD_ROWS, ROWS_CHECK)


def test_extra_column_passes() -> None:
    assert score(res([("a", 3, 10.0), ("b", 2, 5.0)]), GOLD_ROWS, ROWS_CHECK)


def test_wrong_values_fail() -> None:
    assert not score(res([("a", 10.0), ("b", 6.0)]), GOLD_ROWS, ROWS_CHECK)


def test_wrong_order_fails_when_ordered() -> None:
    assert not score(res([("b", 5.0), ("a", 10.0)]), GOLD_ROWS, ROWS_CHECK)


def test_wrong_order_passes_when_unordered() -> None:
    check = {**ROWS_CHECK, "ordered": False}
    assert score(res([("b", 5.0), ("a", 10.0)]), GOLD_ROWS, check)


def test_equal_valued_columns_backtrack() -> None:
    # Gold col 0 (1.0) matches got cols 0 and 1; gold col 1 (1.0015) matches only got col 0.
    # Greedy would give got col 0 to gold col 0 and then fail on gold col 1.
    check = {"type": "rows", "tolerance": 0.001, "ordered": True}
    assert score(res([(1.0008, 1.0)]), res([(1.0, 1.0015)]), check)


def test_unordered_rows_must_stay_intact() -> None:
    check = {**ROWS_CHECK, "ordered": False}
    assert not score(res([("a", 5.0), ("b", 10.0)]), GOLD_ROWS, check)


def test_scalar_not_first_column() -> None:
    gold = res([(42.0,)])
    assert score(res([("label", 42.001)]), gold, {"type": "scalar", "tolerance": 0.01})
    assert not score(res([("label", 50.0)]), gold, {"type": "scalar", "tolerance": 0.01})


def test_scalar_rejects_ambiguous_wide_row() -> None:
    gold = res([(42.0,)])
    check = {"type": "scalar", "tolerance": 0.01}
    assert not score(res([(7, 42.0)]), gold, check)  # two numbers: ambiguous
    assert not score(res([(42.0,), (1.0,)]), gold, check)  # more than one row
