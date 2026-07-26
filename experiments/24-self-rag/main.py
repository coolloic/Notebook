"""Self-RAG — the model decides whether retrieval is needed, and after answering
critiques whether the answer is GROUNDED in the retrieved context, retrieving
more and retrying if not.
"""
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-8", temperature=0)
KB = ["Refunds within 30 days.", "Shipping fees are non-refundable.",
      "The customer pays return shipping."]
store = FAISS.from_texts(KB, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
MAX_TRIES = 2


def needs_retrieval(q: str) -> bool:
    v = model.invoke(f"Does answering '{q}' need the company policy docs? yes/no.").content
    return v.strip().lower().startswith("y")


def grounded(ans: str, ctx: str) -> bool:
    v = model.invoke(f"Is this answer fully supported by the context? yes/no.\nContext:\n{ctx}\nAnswer:\n{ans}").content
    return v.strip().lower().startswith("y")


def solve(question: str) -> str:
    ctx = ""
    if needs_retrieval(question):                       # reflection token: retrieve?
        ctx = "\n".join(d.page_content for d in store.similarity_search(question, k=3))
    ans = ""
    for _ in range(MAX_TRIES):
        ans = model.invoke(f"Context:\n{ctx}\n\nAnswer: {question}").content
        if not ctx or grounded(ans, ctx):              # reflection token: grounded?
            return ans
        ctx += "\n" + "\n".join(d.page_content for d in store.similarity_search(question, k=5))
    return ans


if __name__ == "__main__":
    print(solve("How long do I have to return an item?"))
