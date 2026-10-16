"""Reporter: writes a short plain-language answer from the query result."""
from src import llm
from src.config import REPORTER_MODEL
from src.state import CrewState

MAX_ROWS = 20


def build_prompt(state: CrewState) -> str:
    if state.get("error"):
        result = f"The query failed: {state['error']}"
    else:
        rows = state.get("rows", [])
        lines = [" | ".join(state.get("columns", []))] + [" | ".join(map(str, r)) for r in rows[:MAX_ROWS]]
        more = f"\n({len(rows) - MAX_ROWS} more rows not shown)" if len(rows) > MAX_ROWS else ""
        result = "\n".join(lines) + more
    return (
        "Answer the question in 2-4 sentences using only the query result. "
        "Give exact numbers; do not speculate beyond the data.\n\n"
        f"# Question\n{state['question']}\n\n# Query result\n{result}\n\nAnswer:"
    )


def report(state: CrewState) -> CrewState:
    return {"report": llm.complete(build_prompt(state), REPORTER_MODEL).strip()}
