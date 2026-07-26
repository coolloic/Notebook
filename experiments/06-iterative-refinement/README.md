# 06 · Iterative Refinement

**What:** One (or more) agents work in a loop, modifying a single result each pass until it reaches a quality threshold.

**When to use (per the doc):** complex generation hard to achieve in one step — code debugging, detailed planning, long-form drafting. Produces polished output; cost grows with cycle count.

**This experiment:** a `refine → score` LangGraph loop with a numeric quality gate (`TARGET_SCORE`) and a hard cap (`MAX_ITERS`). Contrast with 05: here the same agent self-scores rather than a separate critic.

**Stack:** LangChain + LangGraph · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
