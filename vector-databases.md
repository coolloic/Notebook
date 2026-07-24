# Vector Databases: Concepts, Popular Options & Storing Chunk Data

A practical guide to **vector databases** — what problem they solve, the core concepts, the popular options and how they differ, and **working examples of storing chunked document data** for retrieval (RAG).

> Companion to [the LangChain / LangGraph / LangSmith guide](README.md). That doc's RAG example glosses over *where the chunks live* — this one goes deep on that layer.

> **TL;DR**
> - A vector database stores **embeddings** (lists of numbers that capture *meaning*) and finds the ones **most similar** to a query embedding — fast, even over billions of vectors.
> - You use one whenever you need **semantic search**: RAG, recommendations, deduplication, image/audio search, anomaly detection.
> - The storage flow is always the same: **chunk → embed → upsert (vector + metadata) → query by similarity → filter by metadata**.

---

## Table of Contents

1. [Why a vector database?](#why-a-vector-database)
2. [Core concepts](#core-concepts)
3. [The chunk-storage flow](#the-chunk-storage-flow)
4. [Popular vector databases](#popular-vector-databases)
5. [Comparison table](#comparison-table)
6. [Choosing one](#choosing-one)
7. [Examples — storing chunk data](#examples--storing-chunk-data)
   - [Chroma (local, zero-setup)](#chroma-local-zero-setup)
   - [FAISS (in-process library)](#faiss-in-process-library)
   - [Qdrant (open-source server)](#qdrant-open-source-server)
   - [Pinecone (managed cloud)](#pinecone-managed-cloud)
   - [pgvector (Postgres extension)](#pgvector-postgres-extension)
8. [Practical tips](#practical-tips)

---

## Why a vector database?

Traditional databases match on **exact values** or keywords: `WHERE title = 'refund policy'`. But "how do I get my money back?" and "what's the refund policy?" share **no keywords** — yet mean the same thing.

An **embedding model** turns text (or an image, audio clip, etc.) into a **vector** — a list of hundreds/thousands of numbers positioned so that *semantically similar things sit close together* in that space. A vector database:

1. **Stores** those vectors alongside the original data and metadata.
2. Given a **query vector**, returns the **nearest** stored vectors — by meaning, not keywords.
3. Does this in **milliseconds over millions/billions of vectors**, using approximate-nearest-neighbor (ANN) indexes.

That "find the most similar items fast" capability is the backbone of **RAG** (retrieval-augmented generation), semantic search, recommendations, and more.

---

## Core concepts

| Concept | What it means |
|---|---|
| **Embedding** | A fixed-length vector (e.g. 384, 768, 1536 dims) representing meaning. Produced by a model (OpenAI `text-embedding-3-small`, Cohere, or local `sentence-transformers`). |
| **Dimension** | The length of the vector. Must be **consistent** within a collection and match the embedding model that produced it. |
| **Distance / similarity metric** | How "closeness" is measured: **cosine** (angle — most common for text), **dot product**, or **Euclidean (L2)**. Must match how the model was trained. |
| **ANN index** | The data structure that makes search fast by trading a little accuracy for huge speed. Common types: **HNSW** (graph-based, great recall/latency), **IVF** (clustering), **PQ** (compression). |
| **Collection / Index / Namespace** | The container holding a set of vectors of the same dimension (terminology varies by product). |
| **Metadata / payload** | Extra fields stored with each vector (source doc, page, author, timestamp). Enables **filtered search**: "nearest chunks *from this document, published after 2024*". |
| **Upsert** | Insert-or-update a vector by ID. The primary write operation. |
| **Top-k** | How many nearest neighbors to return per query. |

---

## The chunk-storage flow

For RAG you rarely embed whole documents — they're too big and dilute relevance. You **split** them into overlapping **chunks**, embed each, and store them. Retrieval then pulls only the chunks relevant to a question.

```mermaid
flowchart LR
    D["📄 Documents"] --> S["✂️ Split into chunks<br/>(e.g. 500 chars, 50 overlap)"]
    S --> E["🧠 Embed each chunk<br/>(embedding model)"]
    E --> U["⬆️ Upsert<br/>vector + text + metadata"]
    U --> VDB[("🗄️ Vector database")]

    Q(["❓ User question"]) --> QE["🧠 Embed query"]
    QE --> SR["🔍 Similarity search<br/>top-k + metadata filter"]
    VDB -.-> SR
    SR --> C["📚 Relevant chunks"]
    C --> LLM["🤖 LLM answers<br/>grounded in chunks"]

    classDef store fill:#bfdbfe,stroke:#1e40af,color:#000;
    classDef query fill:#bbf7d0,stroke:#15803d,color:#000;
    class D,S,E,U,VDB store
    class Q,QE,SR,C,LLM query
```

**Every example below implements the left half of this diagram** (chunk → embed → store), plus a similarity query to prove it works.

---

## Popular vector databases

- **Chroma** — Open-source, developer-first. Runs embedded in your Python process or as a lightweight server. The fastest way to prototype RAG locally. Best for small/medium projects and notebooks.
- **FAISS** (Meta) — Not a database — a **library** for similarity search. Blazing fast, runs in-process, no server. You manage persistence and metadata yourself. Great for read-heavy, static indexes and research.
- **Qdrant** — Open-source vector database written in Rust. Excellent filtering, HNSW indexing, easy Docker deploy, plus a managed cloud. Strong all-rounder.
- **Weaviate** — Open-source, feature-rich (hybrid search, built-in vectorizer modules, GraphQL API). Good when you want batteries-included semantic + keyword search.
- **Milvus** (Zilliz) — Open-source, built for **billion-scale** workloads with a distributed architecture. Heavier to operate; shines at very large scale.
- **Pinecone** — Fully **managed cloud** service. No infra to run; serverless, scalable, low-latency. Popular for production teams that don't want to operate a database.
- **pgvector** — An **extension for PostgreSQL**. Store vectors right next to your relational data and query with SQL. Ideal when you already run Postgres and want one system.
- **Redis** (Redis Stack) — Adds vector search to the in-memory store you may already use for caching. Very low latency.
- **MongoDB Atlas Vector Search** / **Elasticsearch / OpenSearch** — Vector search bolted onto document/search engines you may already operate.

---

## Comparison table

| Database | Type | Hosting | Best for | Metadata filtering | Notes |
|---|---|---|---|---|---|
| **Chroma** | OSS DB | Embedded or self-host | Prototyping, local RAG | ✅ Good | Simplest start; Python-native |
| **FAISS** | OSS library | In-process | Static, read-heavy, research | ⚠️ DIY | Fastest, but you build persistence/metadata |
| **Qdrant** | OSS DB | Self-host / cloud | Balanced prod use, rich filters | ✅ Excellent | Rust; easy Docker; good defaults |
| **Weaviate** | OSS DB | Self-host / cloud | Hybrid search, all-in-one | ✅ Excellent | Built-in vectorizers, GraphQL |
| **Milvus** | OSS DB | Self-host / cloud (Zilliz) | Billion-scale | ✅ Good | Distributed; heavier ops |
| **Pinecone** | Managed | Cloud only | Hands-off production | ✅ Good | Serverless; no infra to run |
| **pgvector** | PG extension | Anywhere Postgres runs | Already on Postgres | ✅ Via SQL | One system for relational + vectors |
| **Redis** | OSS (module) | Self-host / cloud | Ultra-low latency, caching combo | ✅ Good | In-memory; fast |

---

## Choosing one

- **Just prototyping / a notebook?** → **Chroma** or **FAISS**. Zero infra.
- **Already run Postgres and want simplicity?** → **pgvector**. One database to operate.
- **Want managed, hands-off production?** → **Pinecone** (or Qdrant Cloud / Zilliz Cloud).
- **Want open-source you self-host with great filtering?** → **Qdrant** or **Weaviate**.
- **Billion-scale, dedicated team?** → **Milvus**.
- **Need vectors next to caching, sub-ms latency?** → **Redis**.

**Rule of thumb:** Start local with Chroma/FAISS to build the pipeline, then swap the storage layer for a server (Qdrant/Pinecone/pgvector) when you need scale, concurrency, or persistence. Because the flow (chunk → embed → upsert → query) is identical, swapping is mostly a client change — especially if you go through a framework like LangChain.

---

## Examples — storing chunk data

All examples do the same thing: **split a document into chunks, embed them, store them with metadata, then run a similarity query.** They use a local embedding model (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims) so they run offline with no API key. Swap in OpenAI/Cohere embeddings for production quality.

Shared setup used by every example:

```python
# pip install langchain-text-splitters sentence-transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter

raw_text = """Our refund policy allows returns within 30 days of purchase.
Shipping fees are non-refundable. The customer pays for return shipping.
Digital goods are non-refundable once downloaded."""

# 1. CHUNK — overlap preserves context across boundaries
splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_text(raw_text)

# Metadata to attach to each chunk (source tracking, filtering later)
metadatas = [{"source": "policy.txt", "chunk_id": i} for i in range(len(chunks))]
```

### Chroma (local, zero-setup)

```python
# pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")   # persists to disk
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection("policies", embedding_function=embed_fn)

# 2. STORE — Chroma embeds the documents for you via embedding_function
collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=metadatas,
)

# 3. QUERY — Chroma embeds the query string and returns nearest chunks
results = collection.query(query_texts=["Can I get money back?"], n_results=2)
print(results["documents"][0])
# Filtered query: only chunks from a given source
# collection.query(query_texts=["..."], n_results=2, where={"source": "policy.txt"})
```

### FAISS (in-process library)

FAISS is a **library**, not a server — you compute embeddings yourself and keep the text/metadata in a parallel list. Great when the index is static and you want maximum speed.

```python
# pip install faiss-cpu sentence-transformers numpy
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(chunks, normalize_embeddings=True)   # normalize -> cosine via dot product
dim = vectors.shape[1]                                       # 384

# 2. STORE — build an index; keep chunks[]/metadatas[] aligned by position (the "ID")
index = faiss.IndexFlatIP(dim)          # inner-product = cosine on normalized vectors
index.add(np.asarray(vectors, dtype="float32"))
faiss.write_index(index, "policy.faiss")  # persistence is your responsibility

# 3. QUERY
q = model.encode(["Can I get money back?"], normalize_embeddings=True)
scores, ids = index.search(np.asarray(q, dtype="float32"), k=2)
for score, idx in zip(scores[0], ids[0]):
    print(round(float(score), 3), metadatas[idx], "->", chunks[idx])
```

### Qdrant (open-source server)

Run it locally in one command: `docker run -p 6333:6333 qdrant/qdrant`

```python
# pip install qdrant-client sentence-transformers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")   # or QdrantClient(":memory:") for tests

client.recreate_collection(
    collection_name="policies",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 2. STORE — upsert points: id + vector + payload (metadata, incl. the text itself)
vectors = model.encode(chunks, normalize_embeddings=True)
points = [
    PointStruct(id=i, vector=v.tolist(), payload={"text": chunks[i], **metadatas[i]})
    for i, v in enumerate(vectors)
]
client.upsert(collection_name="policies", points=points)

# 3. QUERY — with an optional metadata filter
q = model.encode(["Can I get money back?"], normalize_embeddings=True)[0].tolist()
hits = client.search(collection_name="policies", query_vector=q, limit=2)
for h in hits:
    print(round(h.score, 3), h.payload["text"])
```

### Pinecone (managed cloud)

No server to run — a fully hosted index. Needs a `PINECONE_API_KEY`.

```python
# pip install pinecone sentence-transformers
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if "policies" not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name="policies", dimension=384, metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index("policies")

# 2. STORE — upsert (id, vector, metadata). Store the text in metadata to retrieve it later.
vectors = model.encode(chunks, normalize_embeddings=True)
index.upsert(vectors=[
    (f"chunk-{i}", v.tolist(), {"text": chunks[i], **metadatas[i]})
    for i, v in enumerate(vectors)
])

# 3. QUERY
q = model.encode(["Can I get money back?"], normalize_embeddings=True)[0].tolist()
res = index.query(vector=q, top_k=2, include_metadata=True)
for m in res["matches"]:
    print(round(m["score"], 3), m["metadata"]["text"])
```

### pgvector (Postgres extension)

Store vectors beside your relational data. Enable once: `CREATE EXTENSION vector;`

```python
# pip install psycopg[binary] pgvector sentence-transformers
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg.connect("postgresql://user:pass@localhost/mydb", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id        bigserial PRIMARY KEY,
        text      text,
        source    text,
        embedding vector(384)
    )
""")

# 2. STORE — insert each chunk with its embedding
vectors = model.encode(chunks, normalize_embeddings=True)
for i, v in enumerate(vectors):
    conn.execute(
        "INSERT INTO chunks (text, source, embedding) VALUES (%s, %s, %s)",
        (chunks[i], metadatas[i]["source"], v),
    )
# For scale, add an ANN index: CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

# 3. QUERY — <=> is cosine distance; ORDER BY ... LIMIT gives nearest neighbors
q = model.encode(["Can I get money back?"], normalize_embeddings=True)[0]
rows = conn.execute(
    "SELECT text, embedding <=> %s AS distance FROM chunks ORDER BY distance LIMIT 2",
    (q,),
).fetchall()
for text, distance in rows:
    print(round(distance, 3), text)
```

> **Shortcut:** frameworks like **LangChain** wrap all of these behind one `VectorStore` interface (`FAISS.from_documents(...)`, `Chroma.from_documents(...)`, `QdrantVectorStore`, `PineconeVectorStore`, `PGVector`), so you can switch backends without rewriting your pipeline. The examples above show the **native** clients so you understand what the wrapper does underneath.

---

## Practical tips

- **Match the metric to the model.** Most text embedding models want **cosine** similarity. Normalize vectors and cosine ≈ dot product (faster).
- **Keep dimensions consistent.** Every vector in a collection must share the embedding model and dimension. Changing models means re-embedding everything.
- **Store the text in metadata/payload.** The database returns IDs/vectors + metadata; keep the chunk text there (or a pointer to it) so you can feed it to the LLM.
- **Tune chunk size.** Too big → diluted relevance and wasted tokens; too small → lost context. 300–800 chars with 10–20% overlap is a common starting point; measure and adjust.
- **Add an ANN index at scale.** Flat/brute-force search is exact but O(n). Switch to **HNSW** (or IVF/PQ) once you have tens of thousands+ of vectors.
- **Use metadata filters** to scope search (per-tenant, per-document, by date) — cheaper and more precise than relying on similarity alone.
- **Re-embed on model upgrades.** A better embedding model usually beats a fancier database. The store is swappable; the embeddings are the substance.

---

## Further reading

- Chroma — https://docs.trychroma.com
- FAISS — https://faiss.ai
- Qdrant — https://qdrant.tech/documentation/
- Pinecone — https://docs.pinecone.io
- pgvector — https://github.com/pgvector/pgvector
- Weaviate — https://weaviate.io/developers/weaviate
- Milvus — https://milvus.io/docs

---

*The vector-store layer is swappable; the embeddings are the substance. Prototype locally, then move the storage to a server when scale demands — the chunk → embed → upsert → query flow never changes.*
