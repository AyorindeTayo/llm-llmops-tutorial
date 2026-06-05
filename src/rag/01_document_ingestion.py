"""
============================================================
MODULE 03 — RAG Pipeline: Step 1 — Document Ingestion
============================================================

WHAT YOU WILL LEARN:
  - How to load documents from different sources (text, PDF, web)
  - How to chunk documents for vector storage
  - Why chunking strategy matters for retrieval quality

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is chunking in a RAG pipeline?
  A: Breaking documents into smaller pieces so that the most
     relevant section can be retrieved, not an entire document.
     Too large → dilutes relevance; too small → loses context.

  Q: What chunking strategies exist?
  A: Fixed-size, sentence-based, recursive character splitting,
     semantic chunking (split on meaning, not just characters).

  Q: What metadata do you store with chunks?
  A: Source, page number, section heading, timestamp — so the
     LLM can cite where the answer came from.
============================================================
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single chunk of text ready for embedding."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


# ─────────────────────────────────────────────────────────────
# DOCUMENT LOADER
# ─────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Loads documents from multiple sources.

    In production you would use LangChain loaders:
      - PyPDFLoader, WebBaseLoader, S3FileLoader, etc.
    Here we demonstrate the core logic clearly.
    """

    def load_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a plain text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        logger.info(f"Loaded {len(text)} characters from {file_path}")

        return [{"text": text, "source": str(path), "page": 1}]

    def load_from_string(self, text: str, source: str = "manual") -> List[Dict[str, Any]]:
        """Load from a raw string — useful for demos and tests."""
        return [{"text": text, "source": source, "page": 1}]


# ─────────────────────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits documents into overlapping chunks.

    WHY OVERLAP?
    If an answer spans a chunk boundary, overlap ensures neither
    chunk is missing critical context. Typical overlap is 10-20%
    of chunk size.

    INTERVIEW TIP:
    Chunk size of 512 tokens is common. Smaller (128-256) gives
    precise retrieval; larger (1024+) gives more context per chunk.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Split documents into overlapping chunks."""
        chunks = []

        for doc in documents:
            text = doc["text"]
            source = doc.get("source", "unknown")
            page = doc.get("page", 1)

            # Sliding window chunking
            start = 0
            chunk_index = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        metadata={
                            "source": source,
                            "page": page,
                            "chunk_index": chunk_index,
                            "char_start": start,
                            "char_end": end,
                        },
                        chunk_id=f"{source}_{page}_{chunk_index}",
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Move forward by chunk_size - overlap (sliding window)
                start += self.chunk_size - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    # Sample document about MLOps (simulates your knowledge base)
    sample_text = """
    MLOps is the practice of applying DevOps principles to machine learning systems.
    It covers the full ML lifecycle: data ingestion, model training, evaluation,
    deployment, and monitoring.

    A key component of MLOps is experiment tracking. MLflow is the most popular tool
    for tracking experiments. You log parameters, metrics, and artifacts so you can
    reproduce any run.

    Model versioning is also critical. DVC (Data Version Control) allows you to
    version datasets alongside code, so training is always reproducible. MLflow's
    model registry allows promotion of models from staging to production with full
    audit trails.

    CI/CD for ML means automating the process from code commit to deployed model.
    GitHub Actions is commonly used to trigger training, evaluation, and deployment
    pipelines on every pull request.

    Monitoring in production involves tracking data drift (input distribution changes)
    and concept drift (relationship between input and output changes). PSI and the
    KS-test are standard statistical tools for detecting these shifts.
    """

    # Step 1: Load
    loader = DocumentLoader()
    docs = loader.load_from_string(sample_text, source="mlops_guide")
    logger.info(f"Loaded {len(docs)} document(s)")

    # Step 2: Chunk
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.split(docs)

    # Step 3: Inspect
    print("\n" + "=" * 60)
    print(f"Total chunks created: {len(chunks)}")
    print("=" * 60)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} (ID: {chunk.chunk_id}) ---")
        print(f"Text: {chunk.text[:150]}...")
        print(f"Metadata: {chunk.metadata}")

    # KEY INSIGHT: Show overlap
    if len(chunks) >= 2:
        print("\n" + "=" * 60)
        print("OVERLAP DEMONSTRATION:")
        print(f"End of chunk 0:   ...{chunks[0].text[-80:]}")
        print(f"Start of chunk 1: {chunks[1].text[:80]}...")
        print("Notice the repeated text — that is the overlap at work.")
        print("=" * 60)


if __name__ == "__main__":
    main()
