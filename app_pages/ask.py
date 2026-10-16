"""Ask page: question -> live agent cards -> result table, chart and plain-English answer."""
import time
from typing import Any

import pandas as pd
import streamlit as st

from src import graph, quota, run_cache, usage_ledger
from src.agents import reporter, viz
from src.agents.gate import DATASET_SUMMARY, precheck
from src.config import LLM_PROVIDER
from src.llm_retry import DailyLimitError, InfraError
from src.tools.sql_tool import run_sql
from src.ui.trace_view import AGENTS, GEMINI, render_step, render_totals

RETRIES = 2  # error-feedback retries, as in the official eval rows
FEATURED = [
    "e01_total_revenue", "m04_health_beauty_revenue", "h04_drop_category", "m19_avg_freight_per_order",
    "e10_avg_delivery_days", "m11_null_category_products", "h18_five_star_revenue",
]

if "live_starts" not in st.session_state:
    st.session_state.live_starts = []
    st.session_state.last = None
    st.session_state.pending = False
    st.session_state.force_live = False
    st.session_state.question = ""


@st.cache_data(ttl=600)
def examples() -> list[dict[str, Any]]:
    return run_cache.load_examples()


@st.cache_data(ttl=600, max_entries=200)
def result_of(sql: str) -> tuple[list[str], list[tuple], str | None]:
    r = run_sql(sql)
    return r.columns, r.rows, r.error


def ask(question: str | None = None, force: bool = False) -> None:
    """Button callback: queue a question (None keeps the text box) for this rerun."""
    if question is not None:
        st.session_state.question = question
    st.session_state.pending = True
    st.session_state.force_live = force


def run_models(critic: bool) -> list[str]:
    if LLM_PROVIDER == "gemini":
        return [GEMINI]
    nodes = ["planner", "data", "reporter"] + (["critic"] if critic else [])
    return [AGENTS[n][2] for n in nodes]


def usage_help(model: str, b: quota.Budget) -> str:
    m = usage_ledger.load()["models"].get(model, {})
    lines = [
        f"Tracked locally today (UTC): {b.used:,} tokens ({m.get('prompt', 0):,} prompt + {m.get('completion', 0):,} completion), "
        f"{m.get('requests', 0)} requests. Budget {b.budget:,}; live runs stop at 90%.",
    ]
    h = usage_ledger.load()["headers"].get(model)
    if h:
        lines.append(
            f"From Groq at {h.get('seen_at', '?')}: {h.get('x-ratelimit-remaining-requests', '?')}/"
            f"{h.get('x-ratelimit-limit-requests', '?')} requests left today (resets in {h.get('x-ratelimit-reset-requests', '?')}); "
            f"{h.get('x-ratelimit-remaining-tokens', '?')}/{h.get('x-ratelimit-limit-tokens', '?')} tokens left this minute."
        )
    return "\n\n".join(lines)


def sidebar() -> bool:
    with st.sidebar:
        critic = st.toggle("Critic loop", value=False, help="The Critic reviews the SQL and can send it back up to 3 times.")
        st.caption(f"Live questions left this hour: {quota.live_left(st.session_state.live_starts)}/{quota.LIVE_PER_HOUR}")
        st.subheader("Examples")
        st.caption("Cached from eval runs: no tokens used.")
        by_id = {e["id"]: e for e in examples()}
        for qid in FEATURED:
            if qid in by_id:
                st.button(by_id[qid]["question"], key=f"ex_{qid}", on_click=ask, args=(by_id[qid]["question"],), width="stretch")
        pick = st.selectbox("All 50 eval questions", [e["question"] for e in examples()], index=None, placeholder="Choose a question")
        if pick:
            st.button("Show this example", on_click=ask, args=(pick,), icon=":material/play_arrow:")
        st.subheader("Token usage today")
        for model in dict.fromkeys(run_models(True)):
            b = quota.budget(model)
            st.markdown(f"`{model}`", help=usage_help(model, b))
            st.progress(b.fraction, text=f"~{b.used:,} / {b.budget:,} (estimate)")
        st.caption("Counts calls made since tracking started; estimate")
    return critic


def not_answerable(reason: str) -> None:
    st.warning(reason, title="This question can't be answered from the dataset", icon=":material/help:")
    st.markdown(DATASET_SUMMARY)
    st.markdown("Try one of these:\n" + "\n".join(f"- {e['question']}" for e in examples()[:5]))


def limit_reason(critic: bool) -> str | None:
    wait = quota.session_wait(st.session_state.live_starts)
    if wait:
        return f"You've asked {quota.LIVE_PER_HOUR} live questions this hour. Try again in {int(wait // 60) + 1} min; cached examples still work."
    blocked = quota.blocked_models(run_models(critic))
    if blocked:
        hours = quota.seconds_to_utc_midnight() / 3600
        names = ", ".join(b.model for b in blocked)
        return f"Daily token limit reached for {names}. It resets at 00:00 UTC (in {hours:.1f} h). Cached examples still work."
    return None


def run_live(question: str, critic: bool, run_settings: dict[str, Any]) -> dict[str, Any] | None:
    """Stream the graph, drawing each agent's card as it finishes; return the saved record, or None on failure."""
    st.session_state.live_starts.append(time.time())
    steps: list[dict[str, Any]] = []
    with st.status("Planner is reading the question…", expanded=True) as status:
        try:
            for node, update, meta in graph.stream(question, retries=RETRIES, critic=critic, gate=True):
                if node == "done":
                    record = run_cache.make_record(question, run_settings, update, steps)
                    record["seconds"] = meta["seconds"]
                    run_cache.save(record)
                    status.update(label=f"Done in {meta['seconds']:.1f}s", state="complete", expanded=False)
                    return record
                step = {"node": node, "update": run_cache.safe_update(update), **meta}
                steps.append(step)
                render_step(step)
                status.update(label=f"{AGENTS[node][0]} finished…")
        except DailyLimitError:
            status.update(label="Daily limit reached", state="error")
            st.error("The model's daily token limit is used up. It resets at 00:00 UTC; cached examples still work.")
        except InfraError as exc:
            status.update(label="Rate limited", state="error")
            st.error(f"Groq is rate-limiting right now. Try again in about {int(exc.retry_after or 30)} s.")
        except Exception as exc:  # never crash the page
            status.update(label="Something went wrong", state="error")
            st.error("The agents hit an unexpected error.")
            st.exception(exc)
    return None


def partly_note(record: dict[str, Any]) -> None:
    """The Planner's caveat, shown once, plus its suggested fully answerable question if it gave one."""
    st.info(record.get("gate_reason") or "The data only partly answers this question.", title="Partly answerable", icon=":material/info:")
    if record.get("suggestion"):
        st.button(f"Try instead: {record['suggestion']}", icon=":material/lightbulb:", on_click=ask, args=(record["suggestion"],))


def show_answer(record: dict[str, Any]) -> None:
    if record.get("returned") == "unanswerable":
        not_answerable(record.get("gate_reason") or "The dataset doesn't hold this data.")
        return
    if record.get("returned") == "meta":
        st.subheader("Answer")
        st.markdown(record.get("report", ""))
        return
    if not record.get("sql"):
        st.error("No SQL was produced.")
        return
    columns, rows, error = result_of(record["sql"])
    if error:
        st.error(error, title="The final SQL failed")
        return
    st.subheader("Result")
    st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True)
    chart = viz.make_chart({"columns": columns, "rows": rows})["chart"]
    if chart is not None:
        st.plotly_chart(chart)
    st.subheader("Answer")
    if record.get("answerable") == "partly":
        partly_note(record)
    if record.get("report"):
        st.markdown(record["report"])
    elif st.button("Generate summary (1 small Reporter call)", icon=":material/edit_note:"):
        if quota.blocked_models([AGENTS["reporter"][2]]):
            st.error("Daily token limit reached for the Reporter model; it resets at 00:00 UTC.")
        else:
            try:
                state = {"question": record["question"], "columns": columns, "rows": rows, **record}
                record["report"] = reporter.report(state)["report"]
                st.rerun()
            except InfraError:
                st.error("Groq is rate-limiting right now. Try again in a minute.")


def show_record(record: dict[str, Any], label: str) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(label)
        st.button("Re-run live", icon=":material/refresh:", on_click=ask, args=(record["question"], True))
    steps = run_cache.replay_steps(record)
    with st.expander("Agent trace", icon=":material/account_tree:", expanded=True):
        for step in steps:
            render_step(step, expanded=False)
        render_totals(steps, record.get("seconds"), record.get("returned"))
        if record.get("eval_tokens"):
            t = record["eval_tokens"]
            st.caption(f"Tokens when this ran in the eval: {t['prompt'] + t['completion']:,} (none used now).")
    show_answer(record)


def label_for(record: dict[str, Any], fresh: bool) -> str:
    if fresh:
        return ":blue-badge[live run]"
    date = record.get("created_at", "")[:10]
    if record.get("source") == "eval":
        return f":gray-badge[cached from {date}] eval run `{record.get('eval_run')}`, Critic {'on' if record['settings'].get('critic') else 'off'}"
    return f":gray-badge[cached from {date}]"


critic = sidebar()
st.text_area("Ask a question about Olist orders, revenue, deliveries, reviews, sellers or customers", key="question", height=80)
st.button("Ask", type="primary", icon=":material/send:", on_click=ask)

if st.session_state.pending:
    st.session_state.pending = False
    question = st.session_state.question.strip()
    run_settings = run_cache.settings(critic, RETRIES, gate=True)
    cached = None if st.session_state.force_live else run_cache.lookup(question, run_settings)
    reason = precheck(question)
    if reason:
        st.session_state.last = None
        not_answerable(reason)
    elif cached:
        st.session_state.last = (cached, False)
    elif blocked := limit_reason(critic):
        st.warning(blocked, icon=":material/hourglass:")
    else:
        record = run_live(question, critic, run_settings)
        if record:
            st.session_state.last = (record, True)

if st.session_state.last:
    record, fresh = st.session_state.last
    show_record(record, label_for(record, fresh))
