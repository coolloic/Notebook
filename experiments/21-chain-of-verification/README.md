# 21 · Chain-of-Verification (CoVe)

**What:** Draft → generate verification questions → answer them independently → revise the draft to resolve inconsistencies.

**When to use:** factual answers prone to hallucination (lists, entities, dates) where a self-check pass meaningfully improves accuracy.

**This experiment:** four sequential model calls implementing the CoVe loop over a factual question.

**Stack:** LangChain · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
