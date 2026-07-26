# 12 · Custom Logic

**What:** Maximum flexibility — you implement the orchestration in plain code, mixing steps and branching however the task needs.

**When to use (per the doc):** fine-grained process control; complex mixed-pattern workflows; unique business logic that doesn't fit a standard template. Maximum control, but more development and debugging effort.

**This experiment:** a code-driven handler that classifies an inbound message and branches — complaints get summarize→apologize; questions get answer→suggest-follow-up.

**Stack:** LangChain (plain-Python orchestration) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
