"""Shared settings for the support agent (overridable via environment variables)."""
import os

# Where the Chroma vector store persists on disk.
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "support_kb")

# OpenAI models.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Source knowledge base (relative to this folder).
KB_PATH = os.getenv("KB_PATH", "data/knowledge_base.md")
