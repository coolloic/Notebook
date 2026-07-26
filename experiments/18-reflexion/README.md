# 18 · Reflexion

**What:** The agent attempts the task, reflects on failures, stores the reflection in memory, and retries with that self-feedback in context.

**When to use:** tasks with a checkable outcome where a first attempt often misses (code, structured output) and verbal self-feedback can guide the next try.

**This experiment:** an `attempt → evaluate` LangGraph loop; failures append a reflection to state that the next attempt reads. Bounded by `MAX_ATTEMPTS`. Contrast with 05 (a separate critic) — here the agent reflects on *itself* and accumulates memory.

**Stack:** LangChain + LangGraph (reflection memory) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
