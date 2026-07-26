"""Tree of Thoughts (ToT) — explore multiple reasoning branches, evaluate them,
and expand the most promising one, instead of committing to a single chain.

Simplified beam search: at each depth, propose K next-thoughts, score each, keep
the best, then continue.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

proposer = ChatAnthropic(model="claude-opus-4-8", temperature=0.7)  # diverse branches
scorer = ChatAnthropic(model="claude-opus-4-8", temperature=0)      # deterministic judging
K, DEPTH = 3, 2


def propose(problem: str, partial: str) -> str:
    return proposer.invoke(
        f"Problem: {problem}\nReasoning so far:\n{partial}\nPropose ONE next reasoning step."
    ).content.strip()


def score(problem: str, partial: str) -> int:
    raw = scorer.invoke(
        f"Rate 1-10 how promising this reasoning is for '{problem}':\n{partial}\nReply with just a number."
    ).content
    digits = "".join(c for c in raw if c.isdigit())
    return int(digits[:2]) if digits else 0


def solve(problem: str) -> str:
    best = ""
    for _ in range(DEPTH):
        branches = [best + "\n" + propose(problem, best) for _ in range(K)]  # branch out
        best = max(branches, key=lambda c: score(problem, c))                # evaluate, keep best
    return scorer.invoke(f"Given this reasoning, answer '{problem}':\n{best}").content


if __name__ == "__main__":
    print(solve("Use 4, 9, 10, 13 with + - * / to make 24."))
