"""Parallel pattern — independent subtasks run concurrently, then a synthesizer
merges their outputs.

Google Cloud: sub-tasks executed "at the same time"; good for gathering diverse
perspectives from disparate sources. Lower latency, higher token/resource use.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
parser = StrOutputParser()


def lens(instruction: str):
    return ChatPromptTemplate.from_template(instruction + "\n\nTopic: {topic}") | model | parser


# RunnableParallel fans the same input out to three branches that run
# concurrently; their results are gathered into one dict.
branches = RunnableParallel(
    pros=lens("List the pros"),
    cons=lens("List the cons"),
    risks=lens("List the key risks"),
)

synth = ChatPromptTemplate.from_template(
    "Synthesize into a balanced summary:\nPROS:{pros}\nCONS:{cons}\nRISKS:{risks}"
) | model | parser

chain = branches | synth

if __name__ == "__main__":
    print(chain.invoke({"topic": "adopting a multi-agent architecture"}))
