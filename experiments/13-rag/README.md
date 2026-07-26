# 13 · RAG (Retrieval-Augmented Generation)

**What:** Retrieve the chunks of your own data relevant to a question and have the model answer grounded in them.

**When to use:** the agent must answer from a private/large/changing knowledge base rather than its training data — the retrieval backbone behind most of the patterns above.

**This experiment:** `split → embed → FAISS → retrieve top-k → grounded answer`, composed as one LCEL chain over a tiny policy doc.

**Stack:** LangChain + FAISS + sentence-transformers · LangSmith tracing. See the repo's [Vector Databases guide](../../vector-databases.md) for swapping the store.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
