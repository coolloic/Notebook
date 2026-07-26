# 16 · Self-Consistency

**What:** Sample multiple CoT solutions at nonzero temperature, then take a majority vote over their final answers.

**When to use:** problems with a single verifiable answer (math, logic) where one CoT sometimes slips. Trades extra tokens for higher reliability.

**This experiment:** runs the CoT chain `N=5` times, extracts each `Answer:`, and returns the most common one.

**Stack:** LangChain (LCEL, temperature sampling) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
