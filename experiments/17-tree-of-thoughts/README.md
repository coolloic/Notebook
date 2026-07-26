# 17 · Tree of Thoughts (ToT)

**What:** Branch into several candidate reasoning steps, score them, and expand the best — a search over a tree of thoughts rather than one linear chain.

**When to use:** problems that need exploration/backtracking (puzzles, planning, constraint satisfaction) where a single chain gets stuck.

**This experiment:** a simplified beam search — at each depth propose `K` next-steps, score each, keep the best, repeat for `DEPTH` levels, then answer.

**Stack:** LangChain (proposer + scorer models) · LangSmith tracing.

```bash
pip install -r requirements.txt && cp .env.example .env
python main.py
```
