"""Custom Logic — orchestrate with plain code for complex, branching workflows
that don't fit a standard template.

Google Cloud: maximum flexibility; implement specific orchestration logic in
code for complex workflows with multiple branching paths. Maximum control at the
cost of more development and debugging effort.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


def classify(text: str) -> str:
    return model.invoke(f"Reply 'question' or 'complaint' only:\n{text}").content.strip().lower()


def handle(text: str) -> str:
    # Arbitrary branching and mixing of steps — the control flow lives in code,
    # not in a predefined pattern.
    if "complaint" in classify(text):
        summary = model.invoke(f"Summarize the complaint in one line:\n{text}").content
        return model.invoke(f"Write an empathetic apology addressing: {summary}").content
    # otherwise treat as a question: answer, then propose a follow-up
    answer = model.invoke(f"Answer concisely:\n{text}").content
    follow = model.invoke(f"Suggest one follow-up question for:\n{text}").content
    return f"{answer}\n\nYou might also ask: {follow}"


if __name__ == "__main__":
    print(handle("Your app crashed and I lost my work!"))
