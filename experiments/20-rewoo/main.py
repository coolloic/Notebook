"""ReWOO (Reasoning WithOut Observation) — plan ALL tool calls up front with
placeholders, execute them in one pass, then solve using the collected evidence.
Uses far fewer LLM calls than ReAct because the model doesn't re-plan after every
observation.
"""
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

load_dotenv()


@tool
def wiki(query: str) -> str:
    """Look up a short fact (stub knowledge base)."""
    facts = {"eiffel tower height": "330 meters", "great wall length": "21196 km"}
    return facts.get(query.lower().strip(), "unknown")


TOOLS = {"wiki": wiki}
planner = ChatAnthropic(model="claude-opus-4-8", temperature=0)
solver = ChatAnthropic(model="claude-opus-4-8", temperature=0)


def solve(task: str) -> str:
    # 1) PLAN: model emits steps like  #E1 = wiki[eiffel tower height]
    plan = planner.invoke(
        f"Task: {task}\nWrite a plan. For each lookup use exactly: #E<n> = wiki[<query>]. "
        f"Only the 'wiki' tool is available."
    ).content
    # 2) EXECUTE every planned tool call (no LLM in this loop)
    evidence = {name: TOOLS["wiki"].invoke(arg.strip())
                for name, arg in re.findall(r"#(E\d+)\s*=\s*wiki\[([^\]]+)\]", plan)}
    # 3) SOLVE using the gathered evidence
    ev = "\n".join(f"{k}: {v}" for k, v in evidence.items())
    return solver.invoke(f"Task: {task}\nPlan:\n{plan}\nEvidence:\n{ev}\nGive the final answer.").content


if __name__ == "__main__":
    print(solve("How tall is the Eiffel Tower in meters?"))
