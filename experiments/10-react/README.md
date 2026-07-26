# 10 · ReAct (Reason + Act)

**What:** A single agent loops **thought → action → observation**, interleaving reasoning with tool calls until it can answer.

**When to use (per the doc):** complex, dynamic tasks needing continuous planning adaptation and real-time constraint handling. Simpler than multi-agent systems and debuggable via the reasoning transcript; watch for error propagation across observations.

**This experiment:** a hand-written ReAct loop with a `calculator` tool — you can see each thought/action/observation step explicitly (contrast with 01's prebuilt agent).

**Stack:** LangChain (manual ReAct loop) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
