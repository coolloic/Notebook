# 02 · Sequential

**What:** A series of specialized agents in a predefined linear order — each one's output is the next one's input.

**When to use (per the doc):** highly structured, repeatable processes with an unchanging sequence; data-processing pipelines. Reduces latency/cost, sacrifices flexibility.

**This experiment:** an `outline → draft → polish` writing pipeline composed with the LCEL pipe. The order is hard-coded — the model never decides the route.

**Stack:** LangChain (LCEL) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
