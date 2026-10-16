"""Streamlit rendering of the agent trace (one card per node), shared by live runs and cached/eval replays."""
import os
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from src.config import CRITIC_MODEL, DATA_MODEL, LLM_PROVIDER, PLANNER_MODEL, REPORTER_MODEL

GEMINI = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
AGENTS = {
    "planner": ("Planner", ":material/checklist:", PLANNER_MODEL),
    "data": ("Data (SQL)", ":material/database:", DATA_MODEL),
    "critic": ("Critic", ":material/rate_review:", CRITIC_MODEL),
    "viz": ("Viz", ":material/bar_chart:", None),
    "reporter": ("Reporter", ":material/edit_note:", REPORTER_MODEL),
}
VERDICT = {"ok": ":green-badge[ok]", "revise": ":orange-badge[revise]"}
ANSWERABLE = {"yes": ":green-badge[answerable]", "partly": ":orange-badge[partly answerable]", "no": ":red-badge[not answerable]",
              "meta": ":blue-badge[about the dataset]"}
RETURNED = {
    "no_critic": "Critic off", "ok": "Critic accepted the SQL", "first_ok_fallback": "fell back to the first SQL that ran",
    "data_disagreed": "Data agent kept its SQL after Critic feedback", "data_error": "SQL failed", "unanswerable": "stopped: not answerable",
    "meta": "answered from the dataset description (no SQL)",
}


def model_for(node: str) -> str:
    """Provider and model shown on a card; Viz uses rules, not an LLM."""
    model = AGENTS[node][2]
    if model is None:
        return "rules, no LLM"
    return f"gemini · {GEMINI}" if LLM_PROVIDER == "gemini" else f"groq · {model}"


def tokens(calls: list[dict[str, Any]]) -> int:
    return sum(c["prompt"] + c["completion"] for c in calls)


def header(node: str, step: dict[str, Any]) -> str:
    label, _, _ = AGENTS[node]
    bits = [f"**{label}**", model_for(node)]
    if step.get("calls"):
        bits.append(f"{tokens(step['calls']):,} tokens")
    if step.get("seconds") is not None:
        bits.append(f"{step['seconds']:.1f}s")
    return " · ".join(bits)


def render_attempts(attempts: list[dict[str, Any]]) -> None:
    for i, a in enumerate(attempts, 1):
        status = ":red-badge[error]" if a.get("error") else ":green-badge[ran]"
        st.markdown(f"Attempt {i} {status}")
        st.code(a["sql"], language="sql")
        if a.get("error"):
            st.error(a["error"], icon=":material/error:")


def render_critic(update: dict[str, Any]) -> None:
    rnd = update["critic_rounds"][-1]
    st.markdown(f"Round {rnd.get('round', len(update['critic_rounds']))} {VERDICT.get(rnd['verdict'], rnd['verdict'])}")
    if rnd.get("checklist"):
        rows = [{"requirement": c.get("requirement", ""), "SQL": c.get("sql", ""),
                 "status": "✅ match" if c.get("status") == "match" else "❌ mismatch"} for c in rnd["checklist"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if rnd.get("parse_error"):
        st.caption("The Critic's reply was not valid JSON, so it counts as ok.")
    if rnd.get("feedback"):
        st.markdown(f"**Feedback:** {rnd['feedback']}")


def render_body(node: str, update: dict[str, Any]) -> None:
    if node == "planner":
        if update.get("answerable"):
            st.markdown(f"{ANSWERABLE[update['answerable']]} {update.get('gate_reason', '')}")
        if update.get("plan"):
            st.markdown(update["plan"])
    elif node == "data":
        render_attempts(update.get("round_attempts", []))
        if update.get("returned") in ("data_disagreed", "first_ok_fallback"):
            st.caption(RETURNED[update["returned"]])
    elif node == "critic":
        render_critic(update)
        if update.get("returned") == "first_ok_fallback":
            st.caption(RETURNED["first_ok_fallback"])
    elif node == "viz":
        st.markdown("Picked a chart from the result shape (months: line, categories: bar), shown below.")
    elif node == "reporter":
        st.markdown("Wrote the answer, shown below.")


def render_step(step: dict[str, Any], expanded: bool = True) -> None:
    node = step["node"]
    if node not in AGENTS:
        return
    with st.expander(header(node, step), icon=AGENTS[node][1], expanded=expanded):
        render_body(node, step["update"])


def render_totals(steps: list[dict[str, Any]], seconds: float | None, returned: str | None) -> None:
    per_model: dict[str, int] = defaultdict(int)
    for s in steps:
        for c in s.get("calls", []):
            per_model[c["model"]] += c["prompt"] + c["completion"]
    with st.container(horizontal=True):
        st.metric("Tokens", f"{sum(per_model.values()):,}", border=True)
        st.metric("Time", f"{seconds:.1f}s" if seconds is not None else "n/a", border=True)
        st.metric("LLM calls", sum(len(s.get("calls", [])) for s in steps), border=True)
    if per_model:
        st.caption(" · ".join(f"{m}: {t:,}" for m, t in per_model.items()))
    if returned:
        st.caption(f"Outcome: {RETURNED.get(returned, returned)}")
