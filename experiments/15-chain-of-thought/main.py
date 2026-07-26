"""Chain-of-Thought (CoT) — ask the model to reason step by step before giving a
final answer. A single call (no agent loop); the accuracy gain comes purely from
eliciting explicit intermediate reasoning.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)

# The whole technique is in the instruction: think step by step, then answer.
prompt = ChatPromptTemplate.from_template(
    "Solve step by step, then end with a line 'Answer: <result>'.\n\nProblem: {problem}"
)
chain = prompt | model | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({"problem": "A shirt costs $40 after a 20% discount. What was the original price?"}))
