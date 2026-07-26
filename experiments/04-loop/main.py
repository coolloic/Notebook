"""Loop pattern — repeat a step until a termination condition is met.

Google Cloud: "repeatedly executes a sequence...until a specific termination
condition." Good for iterative refinement / self-correction. A clear exit
condition is required to avoid infinite loops.
"""
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
MAX_ROUNDS = 3


class State(TypedDict):
    draft: str
    rounds: int


def improve(state: State) -> State:
    prompt = f"Make this tagline punchier. Return only the tagline.\n{state['draft']}"
    return {"draft": model.invoke(prompt).content, "rounds": state["rounds"] + 1}


def should_continue(state: State) -> str:
    # Termination condition: a bounded loop that stops after MAX_ROUNDS.
    return "loop" if state["rounds"] < MAX_ROUNDS else "done"


g = StateGraph(State)
g.add_node("improve", improve)
g.add_edge(START, "improve")
g.add_conditional_edges("improve", should_continue, {"loop": "improve", "done": END})
graph = g.compile()

if __name__ == "__main__":
    out = graph.invoke({"draft": "A database for vectors", "rounds": 0})
    print(out["draft"])
