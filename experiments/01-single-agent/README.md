# 01 · Single-Agent

**What:** One model, a defined tool set, and a system prompt autonomously handle a request.

**When to use (per the doc):** early-stage agent development; moderate-complexity, multi-step tasks needing external data — where no specialized delegation is required.

**This experiment:** a `create_react_agent` (LangGraph prebuilt) with a single `get_weather` tool. The agent decides on its own when to call the tool before answering.

**Stack:** LangChain + LangGraph (prebuilt ReAct agent) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env   # add your key
python main.py
```
