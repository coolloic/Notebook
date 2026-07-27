"""Customer-support agent.

Ties together the whole stack:
  - LangGraph  — the stateful agent loop (agent ⇄ tools) with conversation memory
  - OpenAI     — gpt-4o for reasoning, text-embedding-3-small for retrieval
  - Chroma     — the persistent vector database holding the knowledge base
  - RAG        — the `search_kb` tool retrieves grounding context on demand
  - LangSmith  — automatic tracing when LANGSMITH_TRACING=true

Run `python ingest.py` first to build the vector store, then `python agent.py`.
"""
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from config import CHROMA_DIR, COLLECTION, CHAT_MODEL, EMBED_MODEL

load_dotenv()  # OPENAI_API_KEY, LANGSMITH_TRACING, LANGSMITH_API_KEY

# Connect to the vector store built by ingest.py.
_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
    persist_directory=CHROMA_DIR,
)


@tool
def search_kb(query: str) -> str:
    """Search the customer-support knowledge base for relevant policy/help text."""
    docs = _store.similarity_search(query, k=4)  # RAG retrieval over Chroma
    return "\n\n".join(d.page_content for d in docs) or "No relevant policy found."


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to a human agent when the KB can't answer or the issue is sensitive."""
    return f"Escalated to a human specialist. Reason: {reason}"


TOOLS = [search_kb, escalate_to_human]
model = ChatOpenAI(model=CHAT_MODEL, temperature=0).bind_tools(TOOLS)

SYSTEM = SystemMessage(
    "You are Acme's support assistant. Ground every policy answer by calling search_kb first. "
    "If the knowledge base lacks the answer, or the request is sensitive (refunds over $500, "
    "legal, account security), call escalate_to_human instead of guessing. "
    "Cite the policy you relied on and keep replies concise."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state: State) -> State:
    # The model reasons over the conversation and decides: answer, or call a tool.
    return {"messages": [model.invoke([SYSTEM] + state["messages"])]}


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # agent -> tools, or -> END
builder.add_edge("tools", "agent")                        # loop back after a tool runs
# The checkpointer gives the agent short-term memory across turns of a thread.
graph = builder.compile(checkpointer=MemorySaver())


def ask(question: str, thread_id: str = "demo") -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    out = graph.invoke({"messages": [HumanMessage(question)]}, cfg)
    return out["messages"][-1].content


if __name__ == "__main__":
    # A grounded policy question (agent should call search_kb):
    print("Q1:", ask("How many days do I have to return an item, and who pays return shipping?"))
    print("-" * 60)
    # A sensitive request on the same thread (agent should escalate):
    print("Q2:", ask("I want a $2000 refund for a laptop that arrived damaged."))
