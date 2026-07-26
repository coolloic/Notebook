"""Iterative Refinement pattern — one agent repeatedly refines a single artifact
until it clears a quality threshold.

Google Cloud: "one or more agents work within a loop to modify a result...during
each iteration." Good for complex generation hard to nail in one shot (code
debugging, long-form drafting). Polished output at proportional cost.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
TARGET_SCORE = 8
MAX_ITERS = 4


class State(TypedDict):
    goal: str
    artifact: str
    score: int
    iters: int


def refine(state: State) -> State:
    prompt = (f"Goal: {state['goal']}\nCurrent draft: {state.get('artifact', '')}\n"
              f"Return an improved version only.")
    return {"artifact": model.invoke(prompt).content, "iters": state.get("iters", 0) + 1}


def score(state: State) -> State:
    prompt = f"Rate 1-10 for the goal '{state['goal']}'. Reply with just the number.\n{state['artifact']}"
    raw = model.invoke(prompt).content.strip()
    digits = "".join(c for c in raw if c.isdigit())
    return {"score": int(digits[:2]) if digits else 0}


def gate(state: State) -> str:
    # Quality gate: stop when good enough or out of budget.
    return "done" if state["score"] >= TARGET_SCORE or state["iters"] >= MAX_ITERS else "refine"


g = StateGraph(State)
g.add_node("refine", refine)
g.add_node("score", score)
g.add_edge(START, "refine")
g.add_edge("refine", "score")
g.add_conditional_edges("score", gate, {"refine": "refine", "done": END})
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"goal": "A clear one-sentence definition of an AI agent.",
                        "artifact": "", "score": 0, "iters": 0})
    print(f"[score {out['score']} in {out['iters']} iters] {out['artifact']}")
