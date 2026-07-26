"""Single-Agent pattern — one model + tools + system prompt handles the task.

Google Cloud: "Uses an AI model, a defined set of tools, and a comprehensive
system prompt to autonomously handle a user request." Best for early-stage
agents and moderate-complexity, multi-step tasks that don't need delegation.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()  # ANTHROPIC_API_KEY, LANGSMITH_* from .env


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return {"paris": "18C, cloudy", "tokyo": "26C, sunny"}.get(city.lower(), "unknown")


# One agent = one model + a tool belt + a system prompt. The prebuilt ReAct
# agent internally loops reason -> call tool -> observe until it can answer.
agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8", temperature=0),
    tools=[get_weather],
    prompt="You are a concise travel assistant. Use tools when needed.",
)

if __name__ == "__main__":
    out = agent.invoke({"messages": [("user", "What's the weather in Paris?")]})
    print(out["messages"][-1].content)
