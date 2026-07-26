# 15 · Chain-of-Thought (CoT)

**What:** Prompt the model to reason step by step before answering. No tools, no loop — one call whose prompt elicits explicit intermediate reasoning.

**When to use:** multi-step arithmetic/logic/planning where a direct answer is error-prone. The cheapest reasoning upgrade — start here before reaching for agents.

**This experiment:** a single LCEL chain that asks for step-by-step working ending in `Answer: <result>`.

**Stack:** LangChain (LCEL) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
