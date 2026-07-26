# 08 · Hierarchical Task Decomposition

**What:** Agents in a multi-level hierarchy — a top-level parent decomposes the task into sub-tasks handed to worker agents, then synthesizes.

**When to use (per the doc):** ambiguous, open-ended problems requiring multi-step reasoning (research, planning, synthesis). Solves complex problems but adds architectural complexity and multiplies model calls.

**This experiment:** an `orchestrate → workers → synthesize` LangGraph. The parent splits a goal into sub-tasks; workers solve each; a synthesis step merges them.

**Stack:** LangChain + LangGraph (orchestrator–worker) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
