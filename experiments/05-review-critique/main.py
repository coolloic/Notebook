"""Review and Critique pattern — a generator produces output, a critic evaluates
it against criteria, and the generator revises until it passes.

Google Cloud: two agents (generator + critic); best for high-accuracy or
compliance-sensitive tasks (e.g. code + security audit). Higher quality, more
latency/cost per iteration.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
MAX_REVISIONS = 3


class State(TypedDict):
    task: str
    draft: str
    verdict: str
    revisions: int


def generate(state: State) -> State:
    prompt = (f"Task: {state['task']}\nPrevious critique: {state.get('verdict', '')}\n"
              f"Previous draft: {state.get('draft', '')}\nProduce an improved answer.")
    return {"draft": model.invoke(prompt).content, "revisions": state.get("revisions", 0) + 1}


def critique(state: State) -> State:
    prompt = (f"Critique this answer for the task '{state['task']}'. "
              f"Reply 'PASS' if good, else list concrete fixes.\n{state['draft']}")
    return {"verdict": model.invoke(prompt).content}


def route(state: State) -> str:
    passed = state["verdict"].strip().upper().startswith("PASS")
    return "done" if passed or state["revisions"] >= MAX_REVISIONS else "revise"


g = StateGraph(State)
g.add_node("generate", generate)
g.add_node("critique", critique)
g.add_edge(START, "generate")
g.add_edge("generate", "critique")
g.add_conditional_edges("critique", route, {"revise": "generate", "done": END})
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"task": "Write a haiku about retrieval-augmented generation.",
                        "draft": "", "verdict": "", "revisions": 0})
    print(out["draft"])
