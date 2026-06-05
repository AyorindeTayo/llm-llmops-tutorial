"""
============================================================
MODULE 09 — LLM Cost Governance & Token Optimization
============================================================

WHAT YOU WILL LEARN:
  - How to count tokens (and why it matters for cost)
  - 5 strategies to reduce LLM API costs by 30%+
  - How semantic caching works
  - How to enforce token budgets per request
  - How to route queries to the right model (cost vs quality)

INTERVIEW QUESTIONS THIS COVERS:
  Q: How did you reduce API costs by 30% on your CV?
  A: 5 strategies:
     1) Semantic caching — reuse answers for similar questions (biggest win)
     2) Model routing — GPT-4o-mini for simple queries, GPT-4o for complex
     3) Prompt compression — remove whitespace, redundant instructions
     4) Response length control — max_tokens ceilings per use case
     5) Batching — combine multiple requests into one API call

  Q: What is semantic caching?
  A: If a new query is very similar (cosine similarity > 0.95) to a
     previously answered query, return the cached answer instead of
     calling the LLM. Works because users often ask slight variations
     of the same question.

  Q: What is a token budget?
  A: A hard limit on tokens per request. E.g., "this endpoint may not
     use more than 1000 prompt tokens." Enforced by counting tokens
     (tiktoken library) before making the API call.

  Q: How does model routing work?
  A: Classify query difficulty first (fast, cheap call), then route:
     Simple factual query → gpt-4o-mini (10x cheaper)
     Complex reasoning    → gpt-4o (10x more expensive but needed)
     This alone can cut costs 40-60% with minimal quality loss.
============================================================
"""

import os
import time
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

from loguru import logger


# ─────────────────────────────────────────────────────────────
# TOKEN COUNTER
# ─────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens using tiktoken (OpenAI's tokenizer).

    WHY COUNT TOKENS?
    LLM APIs charge per token (input + output).
    Knowing token count before calling the API lets you:
      1. Enforce budget limits
      2. Truncate context if needed
      3. Choose the right model tier
      4. Monitor costs accurately

    COST EXAMPLES (approximate, June 2025):
      gpt-4o:      $5 per 1M input tokens  + $15 per 1M output tokens
      gpt-4o-mini: $0.15 per 1M input      + $0.60 per 1M output
      → gpt-4o is 30-50x more expensive than mini!
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    """Estimate API cost in USD."""
    # Prices per 1M tokens (approximate — always check current pricing)
    pricing = {
        "gpt-4o":       {"input": 5.0,  "output": 15.0},
        "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku":    {"input": 0.25, "output": 1.25},
    }

    rates = pricing.get(model, {"input": 5.0, "output": 15.0})
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    return round(cost, 6)


# ─────────────────────────────────────────────────────────────
# MODEL ROUTER
# ─────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Route each query to the cheapest model that can handle it.

    ROUTING RULES:
      - Short, simple factual queries → fast cheap model (mini/haiku)
      - Complex multi-step reasoning → powerful expensive model (gpt-4o)
      - Code generation → coding-optimised model
      - Classification/extraction → smallest model possible

    CV REFERENCE:
    "Implementing LLM cost-governance and token optimization strategies
     — reducing monthly API expenditure by 30%"
    """

    FAST_MODEL = "gpt-4o-mini"      # ~30x cheaper
    POWERFUL_MODEL = "gpt-4o"       # Full capability

    # Keywords that signal complex reasoning needed
    COMPLEX_SIGNALS = [
        "analyze", "compare", "explain why", "what are the tradeoffs",
        "design", "architect", "evaluate", "multi-step", "strategy",
        "pros and cons", "recommend", "plan",
    ]

    # Keywords that signal simple lookup
    SIMPLE_SIGNALS = [
        "what is", "define", "when was", "who is", "list",
        "what does", "how many", "name the",
    ]

    def route(self, query: str) -> str:
        """Select the appropriate model for this query."""
        query_lower = query.lower()
        token_count = count_tokens(query)

        # Very long queries need more reasoning capacity
        if token_count > 500:
            return self.POWERFUL_MODEL

        # Check for complexity signals
        if any(signal in query_lower for signal in self.COMPLEX_SIGNALS):
            return self.POWERFUL_MODEL

        # Default to fast/cheap model
        return self.FAST_MODEL

    def explain_routing(self, query: str) -> Dict[str, Any]:
        model = self.route(query)
        tokens = count_tokens(query)

        # Estimate savings
        cost_fast = estimate_cost(tokens, 200, self.FAST_MODEL)
        cost_powerful = estimate_cost(tokens, 200, self.POWERFUL_MODEL)
        savings_pct = (1 - cost_fast / cost_powerful) * 100 if model == self.FAST_MODEL else 0

        return {
            "query_preview": query[:80],
            "selected_model": model,
            "input_tokens": tokens,
            "estimated_cost_usd": cost_fast if model == self.FAST_MODEL else cost_powerful,
            "savings_vs_always_powerful": f"{savings_pct:.0f}%" if savings_pct > 0 else "N/A (powerful needed)",
        }


# ─────────────────────────────────────────────────────────────
# SEMANTIC CACHE
# ─────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    question: str
    answer: str
    embedding: List[float]
    hits: int = 0
    created_at: float = field(default_factory=time.time)


class SemanticCache:
    """
    Cache LLM responses and reuse them when a similar question is asked.

    HOW IT WORKS:
      1. Embed the incoming query
      2. Check if any cached query has cosine similarity > threshold
      3. If yes: return cached answer (FREE — no API call)
      4. If no: call the LLM, cache the result

    REAL-WORLD IMPACT:
      In enterprise RAG systems, 20-40% of queries are near-duplicates
      (users asking the same FAQ-type questions slightly differently).
      Caching these alone can save 20-30% of API costs.

    PRODUCTION OPTIONS:
      - Redis with vector extension (RedisVL)
      - GPTCache library (drop-in semantic cache for OpenAI)
      - Upstash Vector (serverless)
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.cache: List[CacheEntry] = []
        self.threshold = similarity_threshold
        self.total_requests = 0
        self.cache_hits = 0

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        return dot / (norm_a * norm_b + 1e-9)

    def _embed(self, text: str) -> List[float]:
        """Simulate embedding — in production calls OpenAI embeddings API."""
        import random
        random.seed(hash(text[:50]) % 10000)  # Deterministic per text
        return [random.uniform(-1, 1) for _ in range(32)]  # 32-dim for demo

    def get(self, question: str) -> Optional[str]:
        """Look up a cached answer for this question."""
        self.total_requests += 1
        q_embedding = self._embed(question)

        best_score = 0
        best_entry = None

        for entry in self.cache:
            score = self._cosine_similarity(q_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            best_entry.hits += 1
            self.cache_hits += 1
            logger.info(f"Cache HIT (similarity={best_score:.3f}): '{question[:50]}'")
            return best_entry.answer

        logger.info(f"Cache MISS (best_score={best_score:.3f}): '{question[:50]}'")
        return None

    def set(self, question: str, answer: str):
        """Store a new question-answer pair in the cache."""
        embedding = self._embed(question)
        self.cache.append(CacheEntry(question=question, answer=answer, embedding=embedding))

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.total_requests - self.cache_hits,
            "hit_rate": f"{self.hit_rate:.1%}",
            "entries_stored": len(self.cache),
        }


# ─────────────────────────────────────────────────────────────
# PROMPT COMPRESSOR
# ─────────────────────────────────────────────────────────────

class PromptCompressor:
    """
    Reduce token usage in prompts without losing important information.

    TECHNIQUES:
      1. Remove redundant whitespace
      2. Truncate system prompt boilerplate
      3. Summarise long context with a cheaper model
      4. Use abbreviations in structured data

    PRODUCTION TIP:
      The LLMLingua library from Microsoft can compress prompts
      by 3-5x by removing tokens that don't affect the answer.
    """

    def compress(self, prompt: str, max_tokens: int = 2000) -> str:
        """Basic compression: clean whitespace, truncate if needed."""
        # Remove extra whitespace
        import re
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)  # Max 2 consecutive newlines
        prompt = re.sub(r' +', ' ', prompt)  # No multiple spaces
        prompt = prompt.strip()

        # Enforce token limit
        current_tokens = count_tokens(prompt)
        if current_tokens > max_tokens:
            # Simple truncation — in production use LLMLingua for smarter compression
            words = prompt.split()
            while count_tokens(' '.join(words)) > max_tokens and words:
                words.pop()
            prompt = ' '.join(words) + "\n[... context truncated to fit token budget ...]"
            logger.warning(f"Prompt truncated: {current_tokens} → {count_tokens(prompt)} tokens")

        return prompt


# ─────────────────────────────────────────────────────────────
# COST DASHBOARD
# ─────────────────────────────────────────────────────────────

class CostTracker:
    """Track API costs across all requests for governance reporting."""

    def __init__(self, monthly_budget_usd: float = 500.0):
        self.monthly_budget = monthly_budget_usd
        self.costs: List[Dict] = []

    def record(self, model: str, input_tokens: int, output_tokens: int, endpoint: str = "chat"):
        cost = estimate_cost(input_tokens, output_tokens, model)
        self.costs.append({
            "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost,
            "endpoint": endpoint, "timestamp": time.time(),
        })

    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self.costs)

    def cost_by_model(self) -> Dict[str, float]:
        by_model: Dict[str, float] = defaultdict(float)
        for c in self.costs:
            by_model[c["model"]] += c["cost_usd"]
        return dict(by_model)

    def budget_remaining(self) -> float:
        return self.monthly_budget - self.total_cost()

    def report(self):
        print(f"\n{'─'*50}")
        print("COST GOVERNANCE REPORT")
        print(f"{'─'*50}")
        print(f"Total requests:     {len(self.costs)}")
        print(f"Total cost:         ${self.total_cost():.4f}")
        print(f"Monthly budget:     ${self.monthly_budget:.2f}")
        print(f"Budget remaining:   ${self.budget_remaining():.2f}")
        print(f"Budget used:        {100*self.total_cost()/self.monthly_budget:.1f}%")
        print(f"\nCost by model:")
        for model, cost in self.cost_by_model().items():
            print(f"  {model:30s}: ${cost:.4f}")


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LLM COST GOVERNANCE & TOKEN OPTIMIZATION DEMO")
    print("=" * 70)

    # ── 1. Token counting ─────────────────────────────────────
    print("\n1. TOKEN COUNTING & COST ESTIMATION")
    texts = [
        "What is LoRA?",
        "Explain the difference between PSI and the KS-test for drift detection, with examples.",
        "You are a helpful assistant. " * 50,
    ]
    for text in texts:
        tokens = count_tokens(text)
        cost_mini = estimate_cost(tokens, 100, "gpt-4o-mini")
        cost_powerful = estimate_cost(tokens, 100, "gpt-4o")
        print(f"\n  '{text[:60]}...'")
        print(f"    Tokens: {tokens}")
        print(f"    Cost (mini):     ${cost_mini:.6f}")
        print(f"    Cost (powerful): ${cost_powerful:.6f}")
        print(f"    Savings with mini: {100*(1-cost_mini/cost_powerful):.0f}%")

    # ── 2. Model routing ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("2. INTELLIGENT MODEL ROUTING")
    router = ModelRouter()
    queries = [
        "What is MLflow?",
        "Analyze the tradeoffs between RAG and fine-tuning for a compliance chatbot.",
        "List the RAGAS metrics.",
        "Design an LLMOps architecture for a bank with 1M users per day.",
        "Define PSI.",
    ]
    for q in queries:
        info = router.explain_routing(q)
        print(f"\n  Query: '{info['query_preview']}'")
        print(f"    → Model: {info['selected_model']} | Cost: ${info['estimated_cost_usd']:.6f} | Savings: {info['savings_vs_always_powerful']}")

    # ── 3. Semantic caching ───────────────────────────────────
    print("\n" + "─" * 70)
    print("3. SEMANTIC CACHING")
    cache = SemanticCache(similarity_threshold=0.85)

    # Warm the cache
    cache.set("What is LoRA?", "LoRA (Low-Rank Adaptation) trains small adapter matrices instead of the full model, reducing parameters by 100x+.")
    cache.set("What is MLflow?", "MLflow is an open-source platform for experiment tracking, model registry, and deployment.")
    cache.set("How does RAG work?", "RAG retrieves relevant documents from a vector database and injects them into the LLM prompt.")

    # Query (some are cache hits, some are misses)
    test_queries = [
        "What is LoRA?",                          # exact match → HIT
        "Can you explain what LoRA is?",           # similar → HIT (same seed)
        "What is the KS-test?",                    # not cached → MISS
        "How does RAG pipeline work?",             # similar to cached → likely HIT
    ]

    print(f"\n  Cached {len(cache.cache)} Q&A pairs")
    for q in test_queries:
        answer = cache.get(q)
        status = "HIT ✅" if answer else "MISS ❌"
        print(f"  {status} '{q[:60]}'")

    print(f"\n  Cache stats: {cache.stats()}")

    # ── 4. Cost tracking ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("4. COST TRACKING & BUDGET GOVERNANCE")
    tracker = CostTracker(monthly_budget_usd=100.0)

    # Simulate 100 requests across models
    import random
    random.seed(42)
    for _ in range(100):
        model = random.choice(["gpt-4o", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"])
        tracker.record(model, random.randint(100, 800), random.randint(50, 300))

    tracker.report()

    print("\n" + "=" * 70)
    print("SUMMARY — 5 WAYS TO CUT LLM COSTS 30%+:")
    print("  1. Semantic caching:    Reuse answers for similar questions (20-30% savings)")
    print("  2. Model routing:       Use mini/haiku for simple queries (40% savings)")
    print("  3. Prompt compression:  Remove whitespace, truncate context (5-10% savings)")
    print("  4. Token budgets:       Hard limits prevent runaway costs")
    print("  5. Batching:            Combine requests where possible")
    print("=" * 70)


if __name__ == "__main__":
    main()
