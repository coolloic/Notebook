# 22 · Agentic RAG

**What:** Retrieval becomes a **tool** the agent calls on demand (via a ReAct loop) rather than a fixed pipeline step — the agent decides *whether* and *what* to retrieve.

**When to use:** mixed workloads where some questions need the knowledge base and others don't, or that need multiple/targeted lookups.

**This experiment:** a `create_react_agent` with a `search_policy` retriever tool over a FAISS store. Contrast with 13 (RAG), which always retrieves.

**Stack:** LangChain + LangGraph (ReAct agent) + FAISS · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
