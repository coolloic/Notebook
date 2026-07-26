"""Memory Agent — a conversational agent with SHORT-term memory: the running
message thread is persisted by a checkpointer, so follow-up turns keep context.
(Long-term memory would add a cross-session store keyed by user id.)
"""
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

load_dotenv()

# The checkpointer persists state per thread_id; reusing the id recalls history.
agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8", temperature=0),
    tools=[],
    checkpointer=MemorySaver(),
)

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "user-1"}}
    agent.invoke({"messages": [("user", "My name is Sam and I love hiking.")]}, cfg)
    out = agent.invoke({"messages": [("user", "What hobby did I mention?")]}, cfg)
    print(out["messages"][-1].content)  # remembers 'hiking' from the earlier turn
