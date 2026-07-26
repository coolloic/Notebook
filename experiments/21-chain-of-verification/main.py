"""Chain-of-Verification (CoVe) — draft an answer, generate verification
questions about it, answer those independently, then revise the draft to fix any
inconsistencies. Reduces hallucinations on factual tasks.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


def cove(question: str) -> str:
    draft = model.invoke(f"Answer concisely: {question}").content
    checks = model.invoke(f"List 3 verification questions that would catch errors in this answer:\n{draft}").content
    verified = model.invoke(f"Answer each verification question factually:\n{checks}").content
    return model.invoke(
        f"Original question: {question}\nDraft: {draft}\nVerification Q&A:\n{verified}\n"
        f"Revise the draft to fix any inconsistencies, then give the final answer."
    ).content


if __name__ == "__main__":
    print(cove("Name three countries that border Switzerland."))
