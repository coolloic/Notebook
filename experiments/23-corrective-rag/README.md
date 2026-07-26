# 23 · Corrective RAG (CRAG)

**What:** Retrieve, then **grade** the chunks for relevance; if they're weak, take a corrective action (broaden the query, web search) before answering.

**When to use:** open-domain questions where the local index may not contain the answer and you must avoid answering from irrelevant context.

**This experiment:** retrieve from FAISS, grade relevance with the model, and fall back to a stub web search when the grade fails.

**Stack:** LangChain + FAISS · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
