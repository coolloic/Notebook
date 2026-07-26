# 07 · Coordinator

**What:** A central agent analyzes/decomposes the request and dispatches to a specialized agent — AI-driven (dynamic) routing.

**When to use (per the doc):** adaptive routing for structured business processes; customer-service workflows with varied input types. More flexible than rigid workflows; increases model calls and cost.

**This experiment:** a `coordinator` node classifies a support request and routes it to a `billing` / `technical` / `sales` specialist via a conditional edge.

**Stack:** LangChain + LangGraph (router) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
