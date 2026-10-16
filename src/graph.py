"""LangGraph crew: Planner -> Data -> Viz -> Reporter."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src import trace_info
from src.agents import data, planner, reporter, viz
from src.state import CrewState


def after_data(state: CrewState) -> str:
    if state.get("skip_report"):
        return END
    return "viz" if viz.chart_kind(state.get("columns", []), state.get("rows", [])) else "reporter"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(CrewState)
    g.add_node("planner", planner.plan)
    g.add_node("data", data.write_sql)
    g.add_node("viz", viz.make_chart)
    g.add_node("reporter", reporter.report)
    g.add_edge(START, "planner")
    g.add_edge("planner", "data")
    g.add_conditional_edges("data", after_data, ["viz", "reporter", END])
    g.add_edge("viz", "reporter")
    g.add_edge("reporter", END)
    return g.compile()


def run(question: str, retries: int = 0, skip_report: bool = False) -> CrewState:
    state: CrewState = {"question": question, "max_retries": retries, "skip_report": skip_report, "critic_feedback": None}
    return build_graph().invoke(state)


def answer(question: str, retries: int = 0) -> str:
    """Eval entry point: return the final SQL (Viz and Reporter are skipped); plan and attempts go to the eval result."""
    state = run(question, retries, skip_report=True)
    trace_info.record({"plan": state.get("plan"), "attempts": state.get("attempts", [])})
    return state["sql"]
