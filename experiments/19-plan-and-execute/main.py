"""Plan-and-Execute — a planner writes a step-by-step plan up front, then an
executor carries out each step in order. Separates high-level planning from
execution (and makes re-planning possible).
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


class State(TypedDict):
    goal: str
    plan: list
    step: int
    scratch: str
    answer: str


def plan(state: State) -> State:
    raw = model.invoke(f"Make a numbered plan (max 4 steps) to achieve:\n{state['goal']}").content
    steps = [ln.strip() for ln in raw.splitlines() if ln.strip()[:1].isdigit()][:4]
    return {"plan": steps, "step": 0, "scratch": ""}


def execute(state: State) -> State:
    step = state["plan"][state["step"]]
    result = model.invoke(
        f"Goal: {state['goal']}\nWork so far:\n{state['scratch']}\nDo this step: {step}"
    ).content
    return {"scratch": state["scratch"] + f"\n[{step}] {result}", "step": state["step"] + 1}


def more(state: State) -> str:
    return "execute" if state["step"] < len(state["plan"]) else "finish"


def finish(state: State) -> State:
    return {"answer": model.invoke(
        f"Given this work, give the final result for '{state['goal']}':\n{state['scratch']}"
    ).content}


g = StateGraph(State)
g.add_node("plan", plan)
g.add_node("execute", execute)
g.add_node("finish", finish)
g.add_edge(START, "plan")
g.add_edge("plan", "execute")
g.add_conditional_edges("execute", more, {"execute": "execute", "finish": "finish"})
g.add_edge("finish", END)
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"goal": "Draft a launch checklist for a new public API.",
                        "plan": [], "step": 0, "scratch": "", "answer": ""})
    print(out["answer"])
