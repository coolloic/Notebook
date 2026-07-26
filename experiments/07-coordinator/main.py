"""Coordinator pattern — a central agent analyzes the request and routes it (or
sub-tasks) to the right specialized agent.

Google Cloud: a central agent "decomposes a user's request...dispatches each
sub-task to a specialized agent." Adaptive routing for varied inputs (e.g.
customer service). More flexible than fixed workflows; more model calls/cost.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
SPECIALISTS = ("billing", "technical", "sales")


class State(TypedDict):
    request: str
    route: str
    answer: str


def coordinator(state: State) -> State:
    # AI-driven (dynamic) routing: the model chooses the specialist.
    choice = model.invoke(
        f"Classify into one word ({', '.join(SPECIALISTS)}):\n{state['request']}"
    ).content.strip().lower()
    route = next((r for r in SPECIALISTS if r in choice), "technical")
    return {"route": route}


def specialist(name: str):
    def _node(state: State) -> State:
        ans = model.invoke(f"You are the {name} specialist. Help with:\n{state['request']}").content
        return {"answer": ans}
    return _node


g = StateGraph(State)
g.add_node("coordinator", coordinator)
for r in SPECIALISTS:
    g.add_node(r, specialist(r))
g.add_edge(START, "coordinator")
g.add_conditional_edges("coordinator", lambda s: s["route"], {r: r for r in SPECIALISTS})
for r in SPECIALISTS:
    g.add_edge(r, END)
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"request": "I was charged twice for my subscription.",
                        "route": "", "answer": ""})
    print(f"[routed to {out['route']}]\n{out['answer']}")
