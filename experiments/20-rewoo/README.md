# 20 · ReWOO (Reasoning WithOut Observation)

**What:** Plan the entire chain of tool calls up front (with `#E1`, `#E2` placeholders), execute them in one batch, then solve from the collected evidence.

**When to use:** tool-heavy tasks where you want to cut LLM calls/latency versus ReAct — the model plans once instead of re-reasoning after every observation.

**This experiment:** a planner emits `#En = wiki[query]` steps, a plain loop runs them, and a solver produces the final answer. Contrast with 10 (ReAct), which interleaves reasoning and observation.

**Stack:** LangChain (planner + solver, stub tool) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
