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
from src.ui import formatting
from src.ui.trace_view import AGENTS, GEMINI, render_step, render_totals, summary_line

RETRIES = 2  # error-feedback retries, as in the official eval rows
FEATURED = [
    "e01_total_revenue", "m04_health_beauty_revenue", "h04_drop_category", "m19_avg_freight_per_order",
    "e10_avg_delivery_days", "m11_null_category_products", "h18_five_star_revenue",
]
STARTERS = ["e01_total_revenue", "m04_health_beauty_revenue", "h04_drop_category"]  # empty-state suggestions
DIFFICULTY = {"e": "Easy", "m": "Medium", "h": "Hard"}

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


def difficulty(example: dict[str, Any]) -> str:
    return DIFFICULTY.get(example["id"][:1], "Other")


def token_usage() -> None:
    st.subheader("Token usage today")
    for model in dict.fromkeys(run_models(True)):
        b = quota.budget(model)
        st.markdown(f"`{model}`", help=usage_help(model, b))
        st.progress(b.fraction, text=f"~{b.used:,} / {b.budget:,} (estimate)")
    st.caption("Counts calls made since tracking started; estimate")


def example_picker() -> None:
    """Featured examples as buttons and all eval questions, both grouped by difficulty."""
    st.subheader("Examples")
    st.caption("Cached from eval runs: no tokens used.")
    by_id = {e["id"]: e for e in examples()}
    featured = [by_id[q] for q in FEATURED if q in by_id]
    for level in DIFFICULTY.values():
        group = [e for e in featured if difficulty(e) == level]
        if group:
            st.markdown(f"**{level}**")
            for e in group:
                st.button(e["question"], key=f"ex_{e['id']}", on_click=ask, args=(e["question"],), width="stretch")
    level = st.segmented_control("All eval questions", list(DIFFICULTY.values()), default="Easy", key="ex_level")
    pool = [e["question"] for e in examples() if difficulty(e) == level]
    pick = st.selectbox(f"{len(pool)} {(level or '').lower()} questions", pool, index=None, placeholder="Choose a question")
    if pick:
        st.button("Show this example", on_click=ask, args=(pick,), icon=":material/play_arrow:")


def sidebar() -> bool:
    with st.sidebar:
        token_usage()
        st.divider()
        critic = st.toggle("Critic loop", value=False, help="The Critic reviews the SQL and can send it back up to 3 times.")
        st.caption(f"Live questions left this hour: {quota.live_left(st.session_state.live_starts)}/{quota.LIVE_PER_HOUR}")
        example_picker()
    return critic


def empty_state() -> None:
    """First visit: what the app does, and three free (cached) questions to try."""
    st.markdown(
        "Ask a question about the Olist Brazilian e-commerce data, and a crew of AI agents plans it, writes and checks "
        "the SQL, runs it on DuckDB, and explains the answer."
    )
    by_id = {e["id"]: e for e in examples()}
    with st.container(horizontal=True):
        for e in (by_id[q] for q in STARTERS if q in by_id):
            st.button(e["question"], key=f"start_{e['id']}", on_click=ask, args=(e["question"],), icon=":material/arrow_forward:")


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
    live = st.empty()  # agent cards while running; cleared once the answer is ready
    with live.status("Planner is reading the question…", expanded=True) as status:
        try:
            for node, update, meta in graph.stream(question, retries=RETRIES, critic=critic, gate=True):
                if node == "done":
                    record = run_cache.make_record(question, run_settings, update, steps)
                    record["seconds"] = meta["seconds"]
                    run_cache.save(record)
                    live.empty()
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


def show_report(record: dict[str, Any], columns: list[str], rows: list[tuple]) -> None:
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


def show_result(columns: list[str], rows: list[tuple]) -> None:
    """Chart (if the result's shape suits one), then the result table with a CSV download."""
    df = pd.DataFrame(rows, columns=columns)
    chart = viz.make_chart({"columns": columns, "rows": rows})["chart"]
    if chart is not None:
        st.plotly_chart(chart)
    st.subheader("Result")
    st.dataframe(df, hide_index=True, column_config=formatting.column_config(df))
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="result.csv",
                       mime="text/csv", icon=":material/download:")


def show_trace(record: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    with st.expander("How the agents got here", icon=":material/account_tree:"):
        for step in steps:
            render_step(step, expanded=False)
        render_totals(steps, record.get("seconds"), record.get("returned"))
        if record.get("eval_tokens"):
            t = record["eval_tokens"]
            st.caption(f"Tokens when this ran in the eval: {t['prompt'] + t['completion']:,} (none used now).")


def show_record(record: dict[str, Any], label: str) -> None:
    """Answer (with caveat), summary line, final SQL, chart, result table, then the collapsed agent trace."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(label)
        st.button("Re-run live", icon=":material/refresh:", on_click=ask, args=(record["question"], True))
    steps = run_cache.replay_steps(record)
    returned = record.get("returned")
    has_sql = bool(record.get("sql")) and returned not in ("unanswerable", "meta")
    columns, rows, error = result_of(record["sql"]) if has_sql else ([], [], None)

    st.subheader("Answer")
    if returned == "unanswerable":
        not_answerable(record.get("gate_reason") or "The dataset doesn't hold this data.")
    elif returned == "meta":
        st.markdown(record.get("report", ""))
    elif not has_sql:
        st.error("No SQL was produced.")
    elif error:
        st.error(error, title="The final SQL failed")
    else:
        if record.get("answerable") == "partly":
            partly_note(record)
        show_report(record, columns, rows)
    st.caption(summary_line(steps, record.get("seconds"), returned, record.get("eval_tokens")))
    if has_sql:
        with st.expander("Final SQL", icon=":material/code:"):
            st.code(record["sql"], language="sql")
        if not error:
            show_result(columns, rows)
    show_trace(record, steps)


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
elif not st.session_state.question.strip():
    empty_state()
