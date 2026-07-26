# 11 · Human-in-the-Loop

**What:** Human review points are built into the workflow — the agent pauses and waits for a person before proceeding.

**When to use (per the doc):** high-stakes decisions; safety-critical operations; compliance validation (e.g. sensitive-document or anonymization review). Adds an external review dependency and architectural complexity.

**This experiment:** a LangGraph flow that drafts a refund reply, then **interrupts** at a `human_gate` checkpoint; the run resumes only when a human supplies an approve/reject decision (`Command(resume=...)`), enabled by a checkpointer.

**Stack:** LangChain + LangGraph (`interrupt` + checkpointer) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
