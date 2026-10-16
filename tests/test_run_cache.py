from src import run_cache


def test_normalize_ignores_case_space_and_trailing_punctuation():
    assert run_cache.normalize("  What is  Revenue? ") == run_cache.normalize("what is revenue")


def test_key_depends_on_settings():
    a = run_cache.settings(critic=False, retries=2, gate=True)
    b = run_cache.settings(critic=True, retries=2, gate=True)
    assert run_cache.key("q", a) != run_cache.key("q", b)
    assert run_cache.key("Q?", a) == run_cache.key("q", a)


def test_save_and_load_live_round_trip(tmp_path):
    s = run_cache.settings(critic=True, retries=2, gate=True)
    state = {"sql": "SELECT 1", "plan": "- p", "rows": [(1,)], "chart": object(), "critic_rounds": []}
    nodes = [{"node": "planner", "seconds": 1.0, "calls": [{"model": "m", "prompt": 10, "completion": 5}]}]
    rec = run_cache.make_record("What is X?", s, state, nodes)
    run_cache.save(rec, tmp_path)
    got = run_cache.load_live("what is x", s, tmp_path)
    assert got["sql"] == "SELECT 1" and "rows" not in got and "chart" not in got
    assert run_cache.total_tokens(got) == 15
    assert run_cache.load_live("what is x", run_cache.settings(False, 2, True), tmp_path) is None


def test_find_example_matches_question_only(tmp_path):
    run_cache.write_json(tmp_path / "e01.json", {"question": "Total revenue?", "sql": "SELECT 1"})
    assert run_cache.find_example("total revenue", tmp_path)["sql"] == "SELECT 1"
    assert run_cache.find_example("other", tmp_path) is None


def test_committed_examples_cover_all_questions():
    import json
    from src.config import ROOT

    questions = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
    assert {e["id"] for e in run_cache.load_examples()} <= {q["id"] for q in questions}
    assert all(e["source"] == "eval" and e["sql"] for e in run_cache.load_examples())


def test_replay_rebuilds_rounds_from_eval_trace():
    a1, a2, a3 = ({"sql": f"S{i}", "error": None} for i in range(3))
    rec = {"plan": "- p", "sql": "S2", "attempts": [a1, a2, a3],
           "critic_rounds": [{"verdict": "revise", "attempts": [a1]}, {"verdict": "ok", "attempts": [a2]}]}
    steps = run_cache.replay_steps(rec)
    assert [s["node"] for s in steps] == ["planner", "data", "critic", "data", "critic", "data"]
    assert steps[-1]["update"]["round_attempts"] == [a3]


def test_replay_sql_only_example():
    steps = run_cache.replay_steps({"sql": "SELECT 1"})
    assert [s["node"] for s in steps] == ["data"] and steps[0]["update"]["round_attempts"][0]["sql"] == "SELECT 1"


def test_replay_uses_stored_live_steps():
    nodes = [{"node": "planner", "update": {"plan": "x"}, "seconds": 1, "calls": []}]
    assert run_cache.replay_steps({"nodes": nodes}) == nodes
