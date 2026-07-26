# 09 · Swarm

**What:** Multiple specialized agents collaborate via all-to-all communication, iteratively refining a shared solution (debate-style).

**When to use (per the doc):** ambiguous or highly complex problems that benefit from debate and collective refinement; collaborative design decisions. Highest quality — and the most complex/costly pattern, with a risk of uncontrolled conversation.

**This experiment:** three personas (Optimist / Skeptic / Pragmatist) take turns improving a single shared proposal over bounded rounds. `ROUNDS` caps the conversation.

**Stack:** LangChain (multi-persona loop) · LangSmith tracing. *(For production, see `langgraph-swarm` for handoff-based swarms.)*

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
