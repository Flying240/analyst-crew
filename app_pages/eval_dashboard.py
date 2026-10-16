"""Eval dashboard: committed run summaries (evals/summaries/) as charts, plus per-question pass/fail."""
import json

import pandas as pd
import streamlit as st

from src.config import ROOT

SUMMARIES_DIR = ROOT / "evals" / "summaries"


@st.cache_data(ttl=600)
def load_runs() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SUMMARIES_DIR.glob("*.json"))]


runs = load_runs()
if not runs:
    st.info("No eval summaries yet. Run `python scripts/export_eval_summaries.py` after an eval.")
    st.stop()

full = st.toggle("Only full 50-question runs", value=True)
shown = [r for r in runs if r["answerer"] != "oracle" and (not full or r["total"] == 50)]
table = pd.DataFrame([{
    "run": f"{r['date']} {r['time']} {r['run']}", "answerer": r["run"], "date": r["date"],
    "score %": round(100 * r["score"] / r["total"], 1) if r["total"] else 0.0,
    "score": f"{r['score']}/{r['total']}", "official": r["official"] or "",
    "tokens / question": round((r["tokens"] or {}).get("mean_prompt", 0) + (r["tokens"] or {}).get("mean_completion", 0)),
    "complete": r["complete"],
} for r in shown])

st.subheader("Score per run")
st.bar_chart(table, x="run", y="score %", horizontal=True, sort=False, height=max(240, 28 * len(table)))
st.dataframe(table.drop(columns=["answerer", "date"]), hide_index=True)

st.subheader("Tokens per question")
st.bar_chart(table, x="run", y="tokens / question", horizontal=True, sort=False, height=max(240, 28 * len(table)))

st.subheader("One run in detail")
labels = [f"{r['date']} {r['time']} {r['run']} ({r['score']}/{r['total']})" for r in runs]
pick = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i], index=len(runs) - 1)
run = runs[pick]
breakdown = [
    {"group": f"{kind}: {name}", "score %": round(100 * b["score"] / b["total"], 1) if b["total"] else 0.0, "score": f"{b['score']}/{b['total']}"}
    for kind, groups in (("difficulty", run.get("by_difficulty") or {}), ("split", run.get("by_split") or {}))
    for name, b in groups.items()
]
if breakdown:
    st.bar_chart(pd.DataFrame(breakdown), x="group", y="score %", horizontal=True, sort=False)
status = st.segmented_control("Show", ["All", "Failed", "Passed"], default="All")
questions = pd.DataFrame([{
    "id": q["id"], "difficulty": q["difficulty"], "split": q["split"],
    "result": "⚠️ infra error" if q.get("infra_error") else "✅ pass" if q["passed"] else "❌ fail",
    "seconds": q.get("seconds"), "tokens": sum((q.get("tokens") or {}).values()) or None,
    "retries": q.get("retries_used"),
} for q in run["questions"]])
if status == "Failed":
    questions = questions[questions["result"] != "✅ pass"]
elif status == "Passed":
    questions = questions[questions["result"] == "✅ pass"]
st.dataframe(questions, hide_index=True)
