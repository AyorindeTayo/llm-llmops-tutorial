"""
============================================================
MODULE 03 — RAG Pipeline: Step 4 — Production RAG System
============================================================

WHAT THIS FILE DEMONSTRATES:
  The complete end-to-end RAG pipeline as described in the CV:
  "Production-grade RAG system processing 10,000+ queries/day
   with sub-200ms latency, reducing manual document retrieval
   time by 70%; leveraged Pinecone, LangChain, and OpenAI API."

ARCHITECTURE:
  Document → Chunk → Embed → Store (Pinecone/Chroma)
                                         ↓
  Query → Embed → Retrieve → Inject into Prompt → LLM → Answer

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is the difference between RAG and fine-tuning?
  A: RAG retrieves external knowledge at inference time — good for
     frequently changing facts. Fine-tuning bakes knowledge into
     weights at training time — good for style/format/domain tasks.

  Q: What is "hallucination" in LLMs and how does RAG help?
  A: LLMs sometimes confidently generate false information.
     RAG grounds the answer in retrieved real documents, so the
     model cites actual content rather than inventing facts.

  Q: What is the system prompt role in RAG?
  A: It instructs the LLM to answer ONLY from the provided context
     and to say "I don't know" if the answer isn't there —
     reducing hallucinations.

  Q: What is "faithfulness" in RAG evaluation?
  A: How much of the generated answer is actually supported by
     the retrieved context (measured by RAGAS).
============================================================
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float
    model_used: str
    tokens_used: int = 0


# ─────────────────────────────────────────────────────────────
# SIMULATED LLM CLIENT (swap for real OpenAI/Anthropic call)
# ─────────────────────────────────────────────────────────────

class LLMClient:
    """
    Wraps the LLM API call.

    In production:
      from openai import OpenAI
      client = OpenAI()
      response = client.chat.completions.create(
          model="gpt-4o",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user",   "content": user_prompt},
          ]
      )

    INTERVIEW TIP — Model selection strategy:
      - Use gpt-4o for complex reasoning queries
      - Use gpt-4o-mini for simple factual lookups (10x cheaper)
      - This routing alone can cut costs 30-50%
    """

    def __init__(self, model: str = "gpt-4o", simulate: bool = True):
        self.model = model
        self.simulate = simulate

        if not simulate:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if self.simulate:
            # Simulate a plausible answer based on the question keywords
            return {
                "content": (
                    "Based on the provided context: "
                    "The answer involves the key concepts mentioned in the retrieved documents. "
                    "In a real deployment this would be an LLM-generated answer grounded in the context. "
                    "[SIMULATED — set simulate=False and add your OPENAI_API_KEY to get real answers]"
                ),
                "tokens": 150,
                "model": self.model,
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature → more factual, less creative
            max_tokens=512,
        )

        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "model": response.model,
        }


# ─────────────────────────────────────────────────────────────
# RAG SYSTEM
# ─────────────────────────────────────────────────────────────

class ProductionRAGSystem:
    """
    End-to-end RAG system.

    Components:
      1. Knowledge base (list of documents for this demo)
      2. Retriever (finds relevant chunks)
      3. Generator (LLM that answers using retrieved context)
    """

    # This is our "knowledge base" — in production this comes from
    # Pinecone/ChromaDB after ingestion via 01_document_ingestion.py
    KNOWLEDGE_BASE = [
        {
            "id": "k1",
            "text": "MLflow is an open-source platform for managing the ML lifecycle. It provides experiment tracking (log params/metrics/artifacts), model registry (versioning and stage transitions), and model serving.",
            "source": "mlops_guide.txt",
        },
        {
            "id": "k2",
            "text": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all model weights, it freezes the base model and trains two small matrices A and B (where A×B approximates the weight update). This reduces trainable parameters by 10,000x.",
            "source": "finetuning_guide.txt",
        },
        {
            "id": "k3",
            "text": "RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG systems. Key metrics: Faithfulness (is the answer supported by the context?), Answer Relevancy (does the answer address the question?), Context Precision (are retrieved chunks relevant?), Context Recall (are all needed chunks retrieved?).",
            "source": "evaluation_guide.txt",
        },
        {
            "id": "k4",
            "text": "PSI (Population Stability Index) measures feature distribution drift between two datasets. PSI < 0.1: stable, no action. PSI 0.1-0.2: minor drift, investigate. PSI > 0.2: significant drift, retrain model.",
            "source": "monitoring_guide.txt",
        },
        {
            "id": "k5",
            "text": "Canary deployment routes a small percentage (e.g. 5-10%) of traffic to the new model while the old model handles the rest. If metrics (latency, error rate, accuracy) remain healthy, traffic is gradually shifted to 100%. This minimises rollout risk.",
            "source": "deployment_guide.txt",
        },
        {
            "id": "k6",
            "text": "QLoRA combines quantization with LoRA. The base model is loaded in 4-bit precision (using NF4 quantization from bitsandbytes), dramatically reducing GPU memory. Then LoRA adapters are trained in 16-bit precision. This allows fine-tuning 70B models on a single A100 GPU.",
            "source": "finetuning_guide.txt",
        },
        {
            "id": "k7",
            "text": "MCP (Model Context Protocol) is an open standard developed by Anthropic for connecting LLMs to external tools, data sources, and services. It provides a standard interface so any LLM can use any tool (file systems, databases, APIs) without custom integration code.",
            "source": "agents_guide.txt",
        },
    ]

    def __init__(self, llm_simulate: bool = True):
        self.llm = LLMClient(simulate=llm_simulate)
        logger.info("RAG system initialised with in-memory knowledge base")

    def _simple_retrieve(self, question: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        Simple keyword-based retrieval for demo (no embedding API needed).
        In production this is replaced by vector similarity search.
        """
        question_lower = question.lower()

        scored = []
        for doc in self.KNOWLEDGE_BASE:
            # Count keyword overlaps (simplified TF-IDF simulation)
            words = set(question_lower.split())
            doc_words = set(doc["text"].lower().split())
            overlap = len(words & doc_words)
            score = overlap / (len(words) + 1)

            scored.append(RetrievedChunk(
                text=doc["text"],
                source=doc["source"],
                score=score,
                metadata={"id": doc["id"]},
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _build_system_prompt(self) -> str:
        """
        The system prompt is critical in RAG.
        It tells the LLM to ONLY use the provided context.

        INTERVIEW TIP:
        Without this instruction, the LLM may ignore the context
        and answer from its training data, defeating the purpose of RAG.
        """
        return """You are a helpful AI assistant for an MLOps knowledge base.

RULES:
1. Answer ONLY based on the provided context below.
2. If the answer is not in the context, say "I don't have enough information to answer that."
3. Always cite which document your answer comes from.
4. Be concise and accurate.
5. Never make up information not present in the context."""

    def _build_user_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Build the user prompt by injecting retrieved context."""
        context_str = "\n\n".join([
            f"[Source: {c.source}]\n{c.text}"
            for c in chunks
        ])

        return f"""CONTEXT:
{context_str}

QUESTION: {question}

Answer based on the context above:"""

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks
          2. Build prompt with context
          3. Call LLM
          4. Return structured response with latency
        """
        start_time = time.time()

        # Step 1: Retrieve
        logger.info(f"Retrieving top-{top_k} chunks for: '{question}'")
        chunks = self._simple_retrieve(question, top_k=top_k)

        # Step 2: Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, chunks)

        # Step 3: Generate
        result = self.llm.complete(system_prompt, user_prompt)

        latency_ms = (time.time() - start_time) * 1000

        return RAGResponse(
            question=question,
            answer=result["content"],
            retrieved_chunks=chunks,
            latency_ms=round(latency_ms, 2),
            model_used=result["model"],
            tokens_used=result.get("tokens", 0),
        )


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    rag = ProductionRAGSystem(llm_simulate=True)

    questions = [
        "What is LoRA and how does it reduce training cost?",
        "How do I detect data drift in a deployed model?",
        "What is RAGAS and which metrics does it measure?",
        "What is MCP and who created it?",
        "What is canary deployment?",
    ]

    print("\n" + "=" * 70)
    print("PRODUCTION RAG SYSTEM — DEMO")
    print("=" * 70)

    for question in questions:
        response = rag.query(question)

        print(f"\n🔍 QUESTION: {question}")
        print(f"\n📚 RETRIEVED CHUNKS:")
        for i, chunk in enumerate(response.retrieved_chunks):
            print(f"   [{i+1}] (score={chunk.score:.3f}) [{chunk.source}] {chunk.text[:80]}...")
        print(f"\n🤖 ANSWER: {response.answer}")
        print(f"\n⏱  Latency: {response.latency_ms}ms | Model: {response.model_used}")
        print("-" * 70)

    print("\n\nKEY PRODUCTION CONSIDERATIONS:")
    print("  1. Replace _simple_retrieve() with Pinecone/Chroma vector search")
    print("  2. Add caching (semantic cache) to reuse answers for similar queries")
    print("  3. Add RBAC: filter chunks by user department/role")
    print("  4. Log every query to MLflow for monitoring and evaluation")
    print("  5. Track latency in Prometheus — alert if P95 > 500ms")
    print("  6. Run RAGAS evaluation weekly to track answer quality over time")


if __name__ == "__main__":
    main()
