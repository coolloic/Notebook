# 25 · Memory Agent

**What:** A conversational agent with **short-term memory** — the message thread is persisted by a checkpointer, so later turns remember earlier ones. Long-term memory would add a store keyed by user/session.

**When to use:** multi-turn assistants where context from earlier in the conversation must carry forward.

**This experiment:** a `create_react_agent` with a `MemorySaver` checkpointer; two turns on the same `thread_id` show the second recalling the first.

**Stack:** LangChain + LangGraph (checkpointer) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
