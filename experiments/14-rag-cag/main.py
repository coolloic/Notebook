"""RAG + CAG — combine Retrieval-Augmented Generation with Cache-Augmented
Generation.

Stable, always-relevant context is PRELOADED into the prompt (Cache-Augmented
Generation, kept warm via prompt caching); only the large/volatile knowledge is
RETRIEVED per query.

Tradeoff: CAG skips a retrieval round-trip for context that's small and always
needed (policies, product facts), but it consumes tokens on every call and can
go stale — so cache only what is stable and bounded, and let RAG handle the big,
changing corpus.
"""
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- CAG: stable context, preloaded on every call (small, always relevant) ---
CACHED_CONTEXT = """Company: Acme. Support hours: 24/7. Currency: USD.
Refunds: within 30 days. Shipping fees: non-refundable."""

# --- RAG: large / volatile corpus, retrieved on demand ---
CORPUS = """Order #42 shipped on 2026-07-01 via DHL, tracking 12345.
Order #43 is delayed due to customs. Promo SAVE10 gives 10% off until Aug 1."""
chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=15).split_text(CORPUS)
retriever = FAISS.from_texts(
    chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
).as_retriever(search_kwargs={"k": 2})

# The cached system block holds the stable facts (mark it for Anthropic prompt
# caching in production); the human turn carries the freshly retrieved facts.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Cached facts (stable):\n{cached}"),
    ("human", "Retrieved facts (volatile):\n{retrieved}\n\nQuestion: {question}"),
])
model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


def fmt(docs):
    return "\n".join(d.page_content for d in docs)


def answer(question: str) -> str:
    retrieved = fmt(retriever.invoke(question))          # RAG for volatile info
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"cached": CACHED_CONTEXT, "retrieved": retrieved, "question": question})


if __name__ == "__main__":
    print(answer("When did order #42 ship, and what is the refund window?"))
