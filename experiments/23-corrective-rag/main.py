"""Corrective RAG (CRAG) — retrieve, GRADE whether the chunks are relevant, and
if they're weak, fall back (broaden the query / web search) before answering.
Guards against answering confidently from irrelevant context.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
KB = ["Refunds within 30 days.", "Shipping fees are non-refundable."]
store = FAISS.from_texts(KB, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))


def web_fallback(query: str) -> str:  # stub for an external search
    return "(web) General info retrieved from the internet about: " + query


def relevant(question: str, context: str) -> bool:
    v = model.invoke(f"Is this context relevant to '{question}'? Reply yes/no.\n{context}").content
    return v.strip().lower().startswith("y")


def answer(question: str) -> str:
    ctx = "\n".join(d.page_content for d in store.similarity_search(question, k=2))
    if not relevant(question, ctx):          # corrective step: retrieved junk -> fall back
        ctx = web_fallback(question)
    return model.invoke(f"Answer using ONLY this context:\n{ctx}\n\nQuestion: {question}").content


if __name__ == "__main__":
    # KB is about refunds, so this triggers the corrective fallback:
    print(answer("What is the capital of France?"))
