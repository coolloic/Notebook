# 24 · Self-RAG

**What:** The model emits "reflection" decisions — *should I retrieve?* and *is my answer grounded in the context?* — retrieving more and retrying when the answer isn't supported.

**When to use:** high-stakes answers where you want the model to gate its own retrieval and self-verify grounding, reducing unsupported claims.

**This experiment:** decide-to-retrieve → answer → grounded-check → retrieve-more/retry, bounded by `MAX_TRIES`.

**Stack:** LangChain + FAISS · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
