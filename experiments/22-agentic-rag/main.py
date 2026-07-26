"""Agentic RAG — instead of always retrieving, the agent treats retrieval as a
TOOL and decides when (and what) to look up, via a ReAct loop.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

load_dotenv()

KB = ["Refunds are allowed within 30 days.",
      "Shipping fees are non-refundable.",
      "Support is available 24/7."]
store = FAISS.from_texts(KB, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))


@tool
def search_policy(query: str) -> str:
    """Search the company policy knowledge base."""
    return "\n".join(d.page_content for d in store.similarity_search(query, k=2))


# The agent decides whether to call search_policy — retrieval is a tool, not a
# fixed first step (contrast with the straight-line RAG in experiment 13).
agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8", temperature=0),
    tools=[search_policy],
    prompt="Use the policy tool when the question is about company policy; otherwise answer directly.",
)

if __name__ == "__main__":
    out = agent.invoke({"messages": [("user", "Can I get a refund after 2 weeks?")]})
    print(out["messages"][-1].content)
