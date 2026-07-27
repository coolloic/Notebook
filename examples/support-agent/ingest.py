"""Ingest step — load the knowledge base, chunk it, embed with OpenAI, and store
the vectors in a persistent Chroma database.

Run this ONCE before the agent (re-run to rebuild the index):

    python ingest.py
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, KB_PATH

load_dotenv()  # OPENAI_API_KEY from .env


def build_index() -> int:
    text = Path(KB_PATH).read_text(encoding="utf-8")

    # Split into overlapping chunks so retrieval returns focused, self-contained
    # passages while preserving context across boundaries.
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.create_documents([text], metadatas=[{"source": KB_PATH}])

    # Embed with OpenAI and persist to Chroma on disk.
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )
    store.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Ingested {n} chunks into Chroma collection '{COLLECTION}' at {CHROMA_DIR}")
