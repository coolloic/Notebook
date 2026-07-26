# 05 · Review and Critique

**What:** Two specialized agents — a generator creates output, a critic evaluates it against criteria, with a revision loop.

**When to use (per the doc):** tasks requiring high accuracy or strict compliance; code generation with security auditing; content validation. Improves quality at the cost of latency/expense per iteration.

**This experiment:** a `generate ⇄ critique` LangGraph loop; the critic emits `PASS` or concrete fixes, and generation repeats until it passes or hits `MAX_REVISIONS`.

**Stack:** LangChain + LangGraph · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
