"""RAG — Retrieval-Augmented Generation. Split docs, embed, store in FAISS,
retrieve top-k relevant chunks, and answer grounded in them.

Ground the model in your own data instead of its parametric memory. The pipeline
is a straight line (no loops), which makes it a natural LangChain chain.
"""
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

KB = """Our refund policy allows returns within 30 days of purchase.
Shipping fees are non-refundable. The customer pays for return shipping."""

# split -> embed -> store
chunks = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20).split_text(KB)
store = FAISS.from_texts(
    chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY this context:\n{context}\n\nQuestion: {question}")
model = ChatAnthropic(model="claude-opus-4-8", temperature=0)


def fmt(docs):
    return "\n".join(d.page_content for d in docs)


chain = ({"context": retriever | fmt, "question": RunnablePassthrough()}
         | prompt | model | StrOutputParser())

if __name__ == "__main__":
    print(chain.invoke("How long do I have to return an item?"))
