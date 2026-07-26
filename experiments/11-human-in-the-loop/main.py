"""Human-in-the-Loop — the agent pauses at a checkpoint and waits for a person to
approve before taking a high-stakes action.

Google Cloud: integrates points for human intervention; the agent pauses and
waits for a person to review. Best for high-stakes decisions, safety-critical
operations, and compliance validation.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


class State(TypedDict):
    request: str
    draft: str
    approved: bool
    sent: str


def draft_reply(state: State) -> State:
    return {"draft": model.invoke(f"Draft a refund reply to:\n{state['request']}").content}


def human_gate(state: State) -> State:
    # Pause the graph and surface the draft; the run resumes when a human
    # supplies a decision via Command(resume=...).
    decision = interrupt({"draft": state["draft"], "question": "Approve sending?"})
    return {"approved": bool(decision.get("approve"))}


def send(state: State) -> State:
    return {"sent": state["draft"] if state["approved"] else "(cancelled by reviewer)"}


g = StateGraph(State)
g.add_node("draft_reply", draft_reply)
g.add_node("human_gate", human_gate)
g.add_node("send", send)
g.add_edge(START, "draft_reply")
g.add_edge("draft_reply", "human_gate")
g.add_edge("human_gate", "send")
g.add_edge("send", END)
graph = g.compile(checkpointer=MemorySaver())  # checkpointer enables pause/resume

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "demo"}}
    graph.invoke({"request": "I want a refund for order #42.",
                  "draft": "", "approved": False, "sent": ""}, cfg)
    # Execution pauses at human_gate. A human reviews, then resumes with a decision:
    final = graph.invoke(Command(resume={"approve": True}), cfg)
    print(final["sent"])
