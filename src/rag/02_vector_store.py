"""
============================================================
MODULE 03 — RAG Pipeline: Step 2 — Vector Store
============================================================

WHAT YOU WILL LEARN:
  - What an embedding is and why it enables semantic search
  - How to store embeddings in Chroma (local) or Pinecone (cloud)
  - How cosine similarity retrieves relevant chunks

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is a vector embedding?
  A: A list of numbers (e.g. 1536 floats for text-embedding-3-small)
     that captures the semantic meaning of text. Similar meanings
     are close together in vector space.

  Q: What is cosine similarity?
  A: A measure of angle between two vectors. Score of 1.0 means
     identical direction (very similar meaning); 0 means unrelated.

  Q: Pinecone vs Chroma vs FAISS?
  A: Chroma = local/dev, great for prototyping.
     Pinecone = managed cloud, production-grade, scales to billions.
     FAISS = Facebook's library, runs in-memory, fast but no persistence.

  Q: What is RBAC in a RAG system?
  A: Role-Based Access Control — different users can only retrieve
     chunks they are authorised to see (enforced via metadata filters).
============================================================
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ─────────────────────────────────────────────────────────────
# EMBEDDINGS — conceptual wrapper (no API key needed for demo)
# ─────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps an embedding API.

    In production this calls OpenAI text-embedding-3-small:
      client.embeddings.create(input=texts, model="text-embedding-3-small")

    For this tutorial we simulate it so you can run without an API key.
    When you have a key, swap simulate=False.
    """

    def __init__(self, model: str = "text-embedding-3-small", simulate: bool = True):
        self.model = model
        self.simulate = simulate
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension

        if not simulate:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of text strings into embedding vectors.

        INTERVIEW TIP: Always batch your embedding calls.
        OpenAI allows up to 2048 texts per request — much cheaper
        than calling one-by-one.
        """
        if self.simulate:
            import random
            logger.info(f"[SIMULATED] Embedding {len(texts)} texts → dim={self.dimension}")
            return [[random.uniform(-1, 1) for _ in range(self.dimension)] for _ in texts]

        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


# ─────────────────────────────────────────────────────────────
# LOCAL VECTOR STORE (Chroma — for development)
# ─────────────────────────────────────────────────────────────

class LocalVectorStore:
    """
    A simple in-memory vector store using Chroma.

    WHY CHROMA FOR DEVELOPMENT?
    - No API key needed
    - Runs locally
    - Supports metadata filtering (simulates RBAC)
    - Easy to swap for Pinecone in production
    """

    def __init__(self, collection_name: str = "tutorial_docs"):
        try:
            import chromadb
            self.client = chromadb.Client()  # In-memory mode
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
            self.backend = "chroma"
            logger.info(f"ChromaDB collection '{collection_name}' ready")
        except ImportError:
            logger.warning("chromadb not installed — using pure Python fallback store")
            self.collection = None
            self._fallback_store: List[dict] = []
            self.backend = "fallback"

    def add(self, texts: List[str], embeddings: List[List[float]], metadatas: List[dict], ids: List[str]):
        """Store chunks with their embeddings and metadata."""
        if self.backend == "chroma":
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Added {len(texts)} chunks to ChromaDB")
        else:
            for text, emb, meta, id_ in zip(texts, embeddings, metadatas, ids):
                self._fallback_store.append({"id": id_, "text": text, "embedding": emb, "metadata": meta})
            logger.info(f"Added {len(texts)} chunks to fallback store")

    def query(self, query_embedding: List[float], top_k: int = 3,
              filter_metadata: Optional[dict] = None) -> List[dict]:
        """
        Retrieve the top-k most similar chunks.

        RBAC FILTER EXAMPLE:
            filter_metadata={"department": "finance"}
        This ensures only documents tagged for finance are returned.
        """
        if self.backend == "chroma":
            kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = self.collection.query(**kwargs)
            output = []
            for i in range(len(results["documents"][0])):
                output.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
            return output
        else:
            # Simple cosine similarity fallback
            import math

            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x**2 for x in a))
                norm_b = math.sqrt(sum(x**2 for x in b))
                return dot / (norm_a * norm_b + 1e-9)

            scored = [
                {**item, "score": cosine(query_embedding, item["embedding"])}
                for item in self._fallback_store
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

    def count(self) -> int:
        if self.backend == "chroma":
            return self.collection.count()
        return len(self._fallback_store)


# ─────────────────────────────────────────────────────────────
# PRODUCTION VECTOR STORE (Pinecone — conceptual)
# ─────────────────────────────────────────────────────────────

class PineconeVectorStore:
    """
    Production vector store using Pinecone.

    INTERVIEW TIP — WHY PINECONE?
    - Managed service: no infra to maintain
    - Scales to billions of vectors
    - Sub-100ms query latency at scale
    - Supports metadata filtering for RBAC
    - Namespace isolation (multi-tenancy)

    ARCHITECTURE USED IN CV:
    "RAG system processing 10,000+ queries/day with sub-200ms latency"
    → Pinecone + LangChain + OpenAI API
    """

    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "llm-tutorial")
        self.index = None

    def connect(self, dimension: int = 1536):
        """Connect to Pinecone and get/create index."""
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=self.api_key)

        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Created Pinecone index: {self.index_name}")

        self.index = pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def upsert(self, vectors: List[dict]):
        """
        vectors = [{"id": "...", "values": [...], "metadata": {...}}, ...]
        'upsert' = insert or update (idempotent — safe to re-run)
        """
        self.index.upsert(vectors=vectors)

    def query(self, vector: List[float], top_k: int = 5, filter: Optional[dict] = None):
        """
        Retrieve similar vectors with optional metadata filter.

        RBAC Example:
          filter={"user_role": {"$in": ["admin", "analyst"]}}
        """
        return self.index.query(vector=vector, top_k=top_k, include_metadata=True, filter=filter)


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    from src.rag.01_document_ingestion import DocumentLoader, TextChunker  # type: ignore

    # Simulate — in real use these come from 01_document_ingestion.py
    documents_text = [
        "MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.",
        "Pinecone is a managed vector database that enables fast semantic search at scale with sub-100ms latency.",
        "LoRA (Low-Rank Adaptation) allows fine-tuning large language models by training only small adapter matrices while keeping the base model frozen.",
        "RAGAS is a framework for evaluating RAG pipelines using metrics like faithfulness, answer relevancy, and context recall.",
        "PSI (Population Stability Index) measures how much a distribution has shifted between two time periods. PSI > 0.2 indicates significant drift.",
    ]

    # Step 1: Embed
    embedder = EmbeddingModel(simulate=True)
    embeddings = embedder.embed(documents_text)

    # Step 2: Store
    store = LocalVectorStore(collection_name="demo")
    metadatas = [{"source": f"doc_{i}", "topic": "mlops"} for i in range(len(documents_text))]
    ids = [f"chunk_{i}" for i in range(len(documents_text))]
    store.add(documents_text, embeddings, metadatas, ids)

    print(f"\nStored {store.count()} chunks in vector store")

    # Step 3: Query
    query_text = "How does LoRA reduce training cost?"
    query_embedding = embedder.embed([query_text])[0]
    results = store.query(query_embedding, top_k=2)

    print("\n" + "=" * 60)
    print(f"Query: '{query_text}'")
    print(f"Top {len(results)} results:")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['text'][:120]}...")
        print(f"     Metadata: {r.get('metadata', {})}")

    print("\nKEY CONCEPTS:")
    print("  - Embeddings convert text to numbers capturing meaning")
    print("  - Cosine similarity finds the closest meaning in vector space")
    print("  - Metadata filters enable RBAC (role-based access control)")
    print("  - Pinecone scales this to millions of documents in production")


if __name__ == "__main__":
    main()
