# Agentic AI Pattern Experiments

[![Experiments CI](https://github.com/coolloic/Notebook/actions/workflows/experiments-ci.yml/badge.svg)](https://github.com/coolloic/Notebook/actions/workflows/experiments-ci.yml)

Runnable scaffolds for building agentic AI systems, in two groups:

1. **Design patterns** — one per pattern from Google Cloud's [**Choose a design pattern for your agentic AI system**](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system), plus RAG and RAG + CAG.
2. **Reasoning & agentic techniques** — prompting/reasoning/retrieval methods (CoT, self-consistency, ToT, Reflexion, plan-and-execute, ReWOO, CoVe, agentic/corrective/self-RAG, memory) that make individual agents smarter.

Each folder is self-contained: a `README.md` (what the pattern is, when to use it per the doc, how the demo works), a runnable `main.py`, a `requirements.txt`, and a `.env.example`. Every `main.py` loads env via **python-dotenv**, defaults to the Anthropic model `claude-opus-4-8`, and is wired for **LangSmith** tracing.

```bash
cd experiments/01-single-agent
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY (and LangSmith keys)
python main.py
```

> All `main.py` files are verified to compile with `python -m py_compile` — they are scaffolds meant to run once you add API keys; the CI-style check does not call any LLM.

## Index — design patterns (Google Cloud)

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

## Index — reasoning & agentic techniques

Prompting, reasoning, and retrieval techniques that make individual agents smarter — complementary to the orchestration patterns above.

| # | Technique | Category | Stack | When to use |
|---|-----------|----------|-------|-------------|
| [15](15-chain-of-thought/) | Chain-of-Thought (CoT) | Reasoning | LangChain (LCEL) | Multi-step problems; the cheapest reasoning upgrade |
| [16](16-self-consistency/) | Self-Consistency | Reasoning | LangChain (sampling + vote) | Verifiable answers where one CoT sometimes slips |
| [17](17-tree-of-thoughts/) | Tree of Thoughts (ToT) | Reasoning (search) | LangChain (propose + score) | Puzzles/planning needing exploration & backtracking |
| [18](18-reflexion/) | Reflexion | Self-improvement | LangGraph (reflection memory) | Retry with verbal self-feedback across attempts |
| [19](19-plan-and-execute/) | Plan-and-Execute | Planning | LangGraph (planner + executor) | Commit to a plan up front; fewer planning calls than ReAct |
| [20](20-rewoo/) | ReWOO | Planning (tools) | LangChain (plan-then-execute) | Tool-heavy tasks; cut LLM calls vs ReAct |
| [21](21-chain-of-verification/) | Chain-of-Verification (CoVe) | Fact-checking | LangChain | Factual answers prone to hallucination |
| [22](22-agentic-rag/) | Agentic RAG | Retrieval (agent) | LangGraph (ReAct) + FAISS | Retrieve on demand as a tool, not a fixed step |
| [23](23-corrective-rag/) | Corrective RAG (CRAG) | Retrieval (self-check) | LangChain + FAISS | Grade retrieval; fall back when it's irrelevant |
| [24](24-self-rag/) | Self-RAG | Retrieval (self-check) | LangChain + FAISS | Model gates its own retrieval + grounding |
| [25](25-memory-agent/) | Memory Agent | Memory | LangGraph (checkpointer) | Multi-turn assistants that recall earlier turns |

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
