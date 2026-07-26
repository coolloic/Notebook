# Agentic AI Pattern Experiments

[![Experiments CI](https://github.com/coolloic/Notebook/actions/workflows/experiments-ci.yml/badge.svg)](https://github.com/coolloic/Notebook/actions/workflows/experiments-ci.yml)

Runnable scaffolds — one per agentic AI design pattern from Google Cloud's
[**Choose a design pattern for your agentic AI system**](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system),
plus two retrieval experiments (RAG and RAG + CAG).

Each folder is self-contained: a `README.md` (what the pattern is, when to use it per the doc, how the demo works), a runnable `main.py`, a `requirements.txt`, and a `.env.example`. Every `main.py` loads env via **python-dotenv**, defaults to the Anthropic model `claude-opus-4-8`, and is wired for **LangSmith** tracing.

```bash
cd experiments/01-single-agent
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY (and LangSmith keys)
python main.py
```

> All `main.py` files are verified to compile with `python -m py_compile` — they are scaffolds meant to run once you add API keys; the CI-style check does not call any LLM.

## Index

| # | Pattern | Category | Stack | When to use |
|---|---------|----------|-------|-------------|
| [01](01-single-agent/) | Single-Agent | Single-agent | LangChain + LangGraph (prebuilt ReAct) | Early-stage agents; moderate multi-step tasks, no delegation |
| [02](02-sequential/) | Sequential | Structural (fixed route) | LangChain (LCEL) | Structured, repeatable pipelines; low latency/cost |
| [03](03-parallel/) | Parallel | Structural (fixed route) | LangChain (`RunnableParallel`) | Independent sub-tasks; diverse perspectives at once |
| [04](04-loop/) | Loop | Structural (fixed route) | LangGraph (cycle) | Iterative refinement / self-correction with an exit condition |
| [05](05-review-critique/) | Review and Critique | Structural (fixed route) | LangGraph (generator + critic) | High accuracy / compliance; validation loops |
| [06](06-iterative-refinement/) | Iterative Refinement | Structural (fixed route) | LangGraph (self-score loop) | Complex output hard to nail in one shot; quality gate |
| [07](07-coordinator/) | Coordinator | Dynamic (AI routing) | LangGraph (router) | Adaptive routing of varied inputs (e.g. support) |
| [08](08-hierarchical/) | Hierarchical Task Decomposition | Dynamic (AI routing) | LangGraph (orchestrator–worker) | Ambiguous, open-ended problems; research/planning |
| [09](09-swarm/) | Swarm | Dynamic (AI routing) | LangChain (multi-persona debate) | Complex problems benefiting from debate; highest quality/cost |
| [10](10-react/) | ReAct (Reason + Act) | Reasoning loop | LangChain (manual ReAct loop) | Dynamic tasks needing continuous adaptation + tools |
| [11](11-human-in-the-loop/) | Human-in-the-Loop | Special requirement | LangGraph (`interrupt` + checkpointer) | High-stakes / safety / compliance approvals |
| [12](12-custom-logic/) | Custom Logic | Special requirement | LangChain (plain-Python orchestration) | Mixed-pattern, branching, bespoke business logic |
| [13](13-rag/) | RAG | Retrieval | LangChain + FAISS | Ground answers in your own data |
| [14](14-rag-cag/) | RAG + CAG | Retrieval | LangChain + FAISS + prompt caching | Stable core facts (cached) + volatile corpus (retrieved) |

## Taxonomy (from the doc)

- **Agent count:** single-agent (01) vs. multi-agent (the rest).
- **Orchestration routing:** *predetermined* (Sequential, Parallel, Loop, Review/Critique, Iterative Refinement) vs. *AI-driven* (Coordinator, Hierarchical, Swarm).
- **Iteration mechanism:** ReAct (thought–action–observation), Loop, Iterative Refinement, Review & Critique.
- **Human integration:** Human-in-the-Loop checkpoints.
- **Escape hatch:** Custom Logic for anything that doesn't fit a template.

## Related guides in this repo

- [LangChain / LangGraph / LangSmith guide](../README.md)
- [Vector Databases](../vector-databases.md)
- [Interactive tutorial](https://coolloic.github.io/Notebook/)
