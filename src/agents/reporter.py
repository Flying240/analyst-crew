"""Reporter: writes a short plain-language answer from the query result."""
from src import llm
from src.config import REPORTER_MODEL
from src.state import CrewState

MAX_ROWS = 20

INSTRUCTIONS = (
    "Answer the question in 2-4 sentences.\n"
    "- The query result below is the computed answer from the database. Never judge whether it is sufficient; "
    "state the answer directly using the values in the result, with exact numbers.\n"
    "- Use the plan and the final SQL only to understand what each column means; do not describe the SQL.\n"
    "- Only say data is missing if the result is empty or the query failed.\n"
    "- All money values are Brazilian reais: write them as R$ with thousands separators (e.g. R$ 45,585.09), "
    "never $.\n"
    "- Do not speculate beyond the result."
)


def result_text(state: CrewState) -> str:
    if state.get("error"):
        return f"The query failed: {state['error']}"
    rows = state.get("rows", [])
    if not rows:
        return "The query returned no rows."
    lines = [" | ".join(state.get("columns", []))] + [" | ".join(map(str, r)) for r in rows[:MAX_ROWS]]
    more = f"\n({len(rows) - MAX_ROWS} more rows not shown)" if len(rows) > MAX_ROWS else ""
    if state.get("truncated"):
        more += f"\n(The query hit the row limit; only the first {len(rows)} rows were returned. Say so.)"
    return "\n".join(lines) + more


def build_prompt(state: CrewState) -> str:
    caveat = ""
    if state.get("answerable") == "partly":
        caveat = (
            f"# Caveat\nThe data only partly answers the question: {state.get('gate_reason', '')}\n"
            "The app shows this caveat next to the answer, so do not repeat it; just answer what the data covers.\n\n"
        )
    return (
        f"{INSTRUCTIONS}\n\n# Question\n{state['question']}\n\n{caveat}"
        f"# Plan\n{state.get('plan') or '(none)'}\n\n# Final SQL\n{state.get('sql') or '(none)'}\n\n"
        f"# Query result\n{result_text(state)}\n\nAnswer:"
    )


def report(state: CrewState) -> CrewState:
    return {"report": llm.complete(build_prompt(state), REPORTER_MODEL).strip()}
