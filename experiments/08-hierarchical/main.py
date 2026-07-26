"""Hierarchical Task Decomposition — a parent agent breaks an open-ended task
into sub-tasks, delegates each to a worker, then synthesizes the results.

Google Cloud: agents organized in a multi-level hierarchy; a top-level parent
decomposes the task into smaller manageable sub-tasks. Best for ambiguous,
open-ended problems (research, planning, synthesis). Powerful but adds
architectural complexity and multiplies model calls.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


class State(TypedDict):
    goal: str
    subtasks: list
    results: list
    answer: str


def orchestrate(state: State) -> State:
    # Parent (top of the hierarchy) decomposes the goal into sub-tasks.
    raw = model.invoke(f"Break this goal into 3 short sub-tasks, one per line:\n{state['goal']}").content
    subs = [ln.strip("-* ").strip() for ln in raw.splitlines() if ln.strip()][:3]
    return {"subtasks": subs}


def workers(state: State) -> State:
    # Each worker (lower level) handles one sub-task independently.
    results = [model.invoke(f"Complete this sub-task concisely:\n{s}").content for s in state["subtasks"]]
    return {"results": results}


def synthesize(state: State) -> State:
    joined = "\n\n".join(f"- {s}: {r}" for s, r in zip(state["subtasks"], state["results"]))
    return {"answer": model.invoke(f"Combine into a final answer for '{state['goal']}':\n{joined}").content}


g = StateGraph(State)
g.add_node("orchestrate", orchestrate)
g.add_node("workers", workers)
g.add_node("synthesize", synthesize)
g.add_edge(START, "orchestrate")
g.add_edge("orchestrate", "workers")
g.add_edge("workers", "synthesize")
g.add_edge("synthesize", END)
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"goal": "Plan a launch for a new open-source RAG library.",
                        "subtasks": [], "results": [], "answer": ""})
    print(out["answer"])
