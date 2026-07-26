# 14 · RAG + CAG

**What:** Combine **RAG** (retrieve the big, changing corpus) with **CAG** — Cache-Augmented Generation, where small, always-relevant context is preloaded into the prompt and kept warm via prompt caching.

**When to use:** you have a stable core of facts needed on nearly every call (policies, product/company facts) plus a large volatile knowledge base. CAG removes a retrieval round-trip for the stable part; RAG covers the rest.

**Tradeoff:** cached context costs tokens on every call and can go stale — cache only what's stable and bounded. This experiment preloads company facts (CAG) and retrieves per-order facts (RAG).

**Stack:** LangChain + FAISS + sentence-transformers · LangSmith tracing · Anthropic prompt caching (in production).

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
