"""Sequential pattern — specialized steps run in a fixed linear order, each
step's output feeding the next.

Google Cloud: predefined, linear order; best for "highly structured, repeatable
processes" and data pipelines. Lower latency/cost, less flexibility.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
parser = StrOutputParser()

# Three specialized stages. The route is fixed in code (no model routing):
# outline -> draft -> polish.
outline = ChatPromptTemplate.from_template("Write a 3-bullet outline about: {topic}") | model | parser
draft   = ChatPromptTemplate.from_template("Expand this outline into a paragraph:\n{outline}") | model | parser
polish  = ChatPromptTemplate.from_template("Tighten and fix grammar:\n{draft}") | model | parser


def run(topic: str) -> str:
    o = outline.invoke({"topic": topic})
    d = draft.invoke({"outline": o})
    return polish.invoke({"draft": d})


if __name__ == "__main__":
    print(run("why vector databases matter for RAG"))
