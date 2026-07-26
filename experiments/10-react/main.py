"""ReAct (Reason + Act) — the agent loops thought -> action -> observation until
it can answer, interleaving reasoning with tool calls.

Google Cloud: "the agent operates in an iterative loop of thought, action, and
observation." Good for complex, dynamic tasks needing continuous adaptation;
simpler than multi-agent systems and easy to debug via the reasoning transcript,
but susceptible to error propagation across observations.

Unlike 01 (which uses the prebuilt agent), this spells out the loop by hand.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 2)'."""
    return str(eval(expression, {"__builtins__": {}}))


TOOLS = {"calculator": calculator}
model = ChatAnthropic(model="claude-opus-4-8", temperature=0).bind_tools(list(TOOLS.values()))


def react(question: str, max_steps: int = 5) -> str:
    messages = [SystemMessage("Reason step by step; call tools when useful."),
                HumanMessage(question)]
    for _ in range(max_steps):
        ai = model.invoke(messages)          # THOUGHT (+ optional ACTION)
        messages.append(ai)
        if not ai.tool_calls:                # no action -> final answer
            return ai.content
        for call in ai.tool_calls:           # ACT, then OBSERVE
            result = TOOLS[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "stopped: max steps reached"


if __name__ == "__main__":
    print(react("What is 15% of 240, then add 12?"))
