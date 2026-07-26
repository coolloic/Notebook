# 04 · Loop

**What:** Repeatedly execute a step until a termination condition is met.

**When to use (per the doc):** iterative refinement; self-correction; content generation with quality-validation cycles. Powerful, but needs a well-defined exit to avoid infinite loops.

**This experiment:** a LangGraph cycle that keeps improving a tagline, with a bounded exit (`MAX_ROUNDS`) enforced by a conditional edge.

**Stack:** LangChain + LangGraph (cyclic graph) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
