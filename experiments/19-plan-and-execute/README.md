# 19 · Plan-and-Execute

**What:** A planner produces a step-by-step plan first; an executor then runs each step in turn. Planning is decoupled from execution.

**When to use:** multi-step tasks that benefit from committing to a plan up front (and optionally re-planning) rather than deciding each move reactively. Fewer planning calls than ReAct.

**This experiment:** a `plan → execute (loop over steps) → finish` LangGraph. Contrast with 08 (hierarchical decomposition + synthesis) — here a single explicit plan is executed sequentially.

**Stack:** LangChain + LangGraph · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
