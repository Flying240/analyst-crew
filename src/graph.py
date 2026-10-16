"""LangGraph crew: Planner -> Data -> [Critic loop] -> Viz -> Reporter."""
import time
from functools import lru_cache
from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph

from src import llm_usage, retry_info, trace_info
from src.agents import critic, data, planner, reporter, viz
from src.agents.data import use_sql
from src.config import MAX_CRITIC_ROUNDS
from src.state import CrewState


def finish(state: CrewState) -> str:
    if state.get("skip_report"):
        return END
    return "viz" if viz.chart_kind(state.get("columns", []), state.get("rows", [])) else "reporter"


def after_data(state: CrewState) -> str:
    if state.get("use_critic") and not state.get("returned") and state.get("error") is None:
        return "critic"
    return finish(state)


def after_critic(state: CrewState) -> str:
    return "data" if state.get("critic_feedback") else finish(state)


def critic_step(state: CrewState) -> CrewState:
    """Review; stop on ok, or fall back to the first SQL that ran after MAX_CRITIC_ROUNDS flags."""
    update = critic.review(state)
    if update["critic_feedback"] is None:
        return {**update, "returned": "ok"}
    if len(update["critic_rounds"]) >= MAX_CRITIC_ROUNDS:
        return {**update, **use_sql(state["first_ok_sql"]), "critic_feedback": None, "returned": "first_ok_fallback"}
    return update


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(CrewState)
    g.add_node("planner", planner.plan)
    g.add_node("data", data.write_sql)
    g.add_node("critic", critic_step)
    g.add_node("viz", viz.make_chart)
    g.add_node("reporter", reporter.report)
    g.add_edge(START, "planner")
    g.add_edge("planner", "data")
    g.add_conditional_edges("data", after_data, ["critic", "viz", "reporter", END])
    g.add_conditional_edges("critic", after_critic, ["data", "viz", "reporter", END])
    g.add_edge("viz", "reporter")
    g.add_edge("reporter", END)
    return g.compile()


def initial_state(question: str, retries: int, skip_report: bool, critic: bool) -> CrewState:
    return {
        "question": question, "max_retries": retries, "skip_report": skip_report,
        "use_critic": critic, "critic_feedback": None, "critic_rounds": [],
    }


def finalize(state: CrewState) -> CrewState:
    critic = state.get("use_critic")
    state.setdefault("returned", "no_critic" if not critic else "data_error" if state.get("error") else "ok")
    return state


def run(question: str, retries: int = 0, skip_report: bool = False, critic: bool = False) -> CrewState:
    return finalize(build_graph().invoke(initial_state(question, retries, skip_report, critic)))


def stream(question: str, retries: int = 0, critic: bool = False) -> Iterator[tuple[str, CrewState, dict[str, Any]]]:
    """Yield (node, update, meta) as each node finishes, then ("done", final state, totals).

    meta has the node's wall seconds and its LLM calls [{model, prompt, completion}].
    """
    state = initial_state(question, retries, False, critic)
    llm_usage.take_calls()
    start = last = time.perf_counter()
    calls: list[dict[str, Any]] = []
    for chunk in build_graph().stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            now = time.perf_counter()
            meta = {"seconds": round(now - last, 2), "calls": llm_usage.take_calls()}
            last = now
            calls += meta["calls"]
            state = {**state, **(update or {})}
            yield node, update or {}, meta
    yield "done", finalize(state), {"seconds": round(time.perf_counter() - start, 2), "calls": calls}


def answer(question: str, retries: int = 0, critic: bool = False) -> str:
    """Eval entry point: return the final SQL (Viz and Reporter are skipped); the trace goes to the eval result."""
    state = run(question, retries, skip_report=True, critic=critic)
    retry_info.record(state.get("retries_used", 0))  # error retries summed over all Data runs
    trace_info.record({
        "plan": state.get("plan"),
        "attempts": state.get("attempts", []),
        "critic_rounds": state.get("critic_rounds", []),
        "critic_feedback": state.get("critic_feedback"),
        "returned": state["returned"],
    })
    return state["sql"]
