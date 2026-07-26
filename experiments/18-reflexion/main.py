"""Reflexion — attempt a task, reflect on why it fell short, store that
reflection in memory, and retry using the self-feedback. Improves across
attempts with no weight updates.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
MAX_ATTEMPTS = 3


class State(TypedDict):
    task: str
    answer: str
    reflections: list
    attempts: int
    done: bool


def attempt(state: State) -> State:
    memo = "\n".join(state.get("reflections", []))
    ans = model.invoke(f"Task: {state['task']}\nPast reflections:\n{memo}\nProduce your best answer.").content
    return {"answer": ans, "attempts": state.get("attempts", 0) + 1}


def evaluate(state: State) -> State:
    verdict = model.invoke(
        f"Does this fully solve '{state['task']}'? Reply PASS, or one line on why it fails.\n{state['answer']}"
    ).content
    if verdict.strip().upper().startswith("PASS"):
        return {"done": True}
    # store the self-reflection so the next attempt can use it
    return {"done": False, "reflections": state.get("reflections", []) + [verdict.strip()]}


def route(state: State) -> str:
    return "done" if state["done"] or state["attempts"] >= MAX_ATTEMPTS else "retry"


g = StateGraph(State)
g.add_node("attempt", attempt)
g.add_node("evaluate", evaluate)
g.add_edge(START, "attempt")
g.add_edge("attempt", "evaluate")
g.add_conditional_edges("evaluate", route, {"retry": "attempt", "done": END})
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"task": "Write a regex matching valid IPv4 addresses.",
                        "answer": "", "reflections": [], "attempts": 0, "done": False})
    print(out["answer"])
