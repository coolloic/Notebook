# 03 · Parallel

**What:** Multiple specialized subagents work independently at the same time; a synthesizer combines their outputs.

**When to use (per the doc):** sub-tasks that can run concurrently; gathering diverse perspectives or data from disparate sources at once. Reduces latency, increases resource/token use and synthesis complexity.

**This experiment:** three "lenses" (pros / cons / risks) evaluate the same topic via `RunnableParallel`, then a synthesis step merges them.

**Stack:** LangChain (LCEL `RunnableParallel`) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
