"""Swarm pattern — multiple specialized agents collaborate with all-to-all
communication to iteratively refine a shared solution.

Google Cloud: collaborative, all-to-all communication; agents iteratively refine
a solution together. Best for ambiguous/complex problems that benefit from
debate. Highest quality but the most complex and costly pattern, with a risk of
uncontrolled conversation.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
AGENTS = ["Optimist", "Skeptic", "Pragmatist"]
ROUNDS = 2


def swarm(problem: str) -> str:
    shared = "(no proposal yet)"
    # All-to-all: every agent reads the full shared state and revises it, over
    # several rounds. Bounded by ROUNDS to control cost / runaway conversation.
    for _ in range(ROUNDS):
        for name in AGENTS:
            shared = model.invoke(
                f"You are the {name}. Current shared proposal:\n{shared}\n\n"
                f"Problem: {problem}\nImprove the proposal in 2-3 sentences."
            ).content
    return shared


if __name__ == "__main__":
    print(swarm("Design a caching strategy for an LLM chatbot."))
