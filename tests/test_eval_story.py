import json

from src.agents import planner
from src.ui import eval_story


def summary(answerer, score, official=None, failed=(), total=50, complete=True, date="2026-10-01"):
    ids = [f"q{i:02d}" for i in range(total)]
    return {
        "answerer": answerer, "official": official, "score": score, "total": total, "complete": complete, "date": date,
        "tokens": {"mean_prompt": 900, "mean_completion": 100},
        "questions": [{"id": q, "passed": q not in failed} for q in ids],
    }


def write(tmp_path, runs):
    for i, r in enumerate(runs):
        (tmp_path / f"{i:02d}.json").write_text(json.dumps(r), encoding="utf-8")
    return eval_story.load_summaries(tmp_path)


def test_ladder_fills_rows_that_exist(tmp_path):
    runs = write(tmp_path, [
        summary("src.agents.baseline:answer", 42, "baseline_50q_runs"),
        summary("src.agents.baseline:answer", 44, "baseline_50q_runs"),
        summary("src.agents.baseline:answer", 47, "retry_50q_runs", date="2026-10-05"),
        summary("src.graph:answer", 5, total=5),  # spot check: not a full run
        summary("oracle", 50),
    ])
    rows = {r["step"]: r for r in eval_story.ladder(runs)}
    assert rows["Baseline"]["mean score %"] == 86.0 and rows["Baseline"]["range"] == "42–44 / 50"
    assert rows["+ retry"]["runs"] == 1 and rows["+ retry"]["tokens / question"] == 1000
    assert rows["Multi-agent graph"]["mean score %"] is None and "not measured" in rows["Multi-agent graph"]["range"]
    assert eval_story.last_updated(runs) == "2026-10-05"


def test_always_failed_is_the_intersection(tmp_path):
    runs = write(tmp_path, [
        summary("src.agents.baseline:answer", 48, "baseline_50q_runs", failed=("q01", "q02")),
        summary("src.agents.baseline:answer", 49, "retry_50q_runs", failed=("q02",)),
    ])
    assert eval_story.always_failed(runs) == ["q02"]


def test_committed_summaries_match_known_failure():
    runs = eval_story.load_summaries()
    assert "m19_avg_freight_per_order" in eval_story.always_failed(runs)
    assert all(q in eval_story.question_text() for q in eval_story.FAILURE_NOTES)


def test_meta_answer_puts_not_in_data_in_its_own_paragraph():
    text = planner.meta_answer()
    assert "zip code prefix\n\nIt has no product names" in text
