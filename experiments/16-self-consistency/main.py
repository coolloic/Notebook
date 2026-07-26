"""Self-Consistency — sample several independent Chain-of-Thought solutions with
temperature, then majority-vote the final answers. More robust than a single
greedy CoT because diverse reasoning paths tend to agree on the correct answer.
"""
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# temperature > 0 so the N samples explore *different* reasoning paths
model = ChatAnthropic(model="claude-opus-4-8", temperature=0.8)
chain = ChatPromptTemplate.from_template(
    "Solve step by step, then end with 'Answer: <number>'.\n\n{problem}"
) | model | StrOutputParser()
N = 5


def final_answer(text: str) -> str:
    hits = re.findall(r"Answer:\s*(.+)", text)
    return hits[-1].strip() if hits else text.strip().splitlines()[-1]


def solve(problem: str) -> str:
    votes = [final_answer(chain.invoke({"problem": problem})) for _ in range(N)]
    return Counter(votes).most_common(1)[0][0]  # majority vote


if __name__ == "__main__":
    print(solve("If 3 pens cost $6.90, how much do 7 pens cost?"))
