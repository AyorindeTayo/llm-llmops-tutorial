"""
============================================================
MODULE 06 — Multi-Agent Orchestration & MCP
============================================================

WHAT YOU WILL LEARN:
  - What an LLM agent is (LLM + tools + memory + planning)
  - The ReAct (Reason + Act) loop
  - How to build a simple agent with tool use
  - What MCP (Model Context Protocol) is and why it matters
  - Multi-agent patterns (sequential, parallel, supervisor)

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is an LLM agent?
  A: An LLM that can take actions — it has:
     - Tools (web search, code execution, database queries, APIs)
     - Memory (conversation history, vector memory)
     - Planning (breaks tasks into steps, reasons about which tool to use)
     - An execution loop that runs until the task is complete

  Q: What is the ReAct pattern?
  A: Reason → Act → Observe → Reason → Act → ...
     The model first THINKS (Thought: "I need to search for this"),
     then ACTS (Action: search["query"]),
     then OBSERVES the result (Observation: "result found"),
     then reasons again until task complete.

  Q: What is MCP (Model Context Protocol)?
  A: Open standard by Anthropic (Nov 2024). Defines a common
     interface for LLMs to connect to tools and data sources.
     Like USB-C but for AI tools — one standard, any LLM + any tool.
     Supported by: Claude, GPT-4, Llama, Gemini, Cursor, Zed, etc.

  Q: What is the difference between tool use and function calling?
  A: Same concept, different names:
     OpenAI: "function calling" or "tool use"
     Anthropic: "tool use"
     MCP: standardises this across ALL providers

  Q: What are multi-agent patterns?
  A: Sequential: Agent A → Agent B → Agent C (pipeline)
     Parallel:   Agent A, B, C run simultaneously, results merged
     Supervisor: Boss agent delegates to specialist agents
     Debate:     Agents argue, best answer wins
============================================================
"""

import json
import os
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field

from loguru import logger


# ─────────────────────────────────────────────────────────────
# TOOL DEFINITION
# ─────────────────────────────────────────────────────────────

@dataclass
class Tool:
    """
    A tool the agent can use.

    In production with LangChain:
      from langchain.tools import tool
      @tool
      def search(query: str) -> str:
          "Search the web for information"
          ...

    In production with Anthropic (tool use):
      tools = [{
          "name": "search",
          "description": "Search the web",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
      }]
    """
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str] = field(default_factory=dict)

    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        return str(self.function(**kwargs))


# ─────────────────────────────────────────────────────────────
# BUILT-IN TOOLS
# ─────────────────────────────────────────────────────────────

def search_tool(query: str) -> str:
    """Simulated web search."""
    knowledge = {
        "psi": "PSI (Population Stability Index) measures distribution drift. PSI>0.2 indicates significant shift requiring model retraining.",
        "lora": "LoRA (Low-Rank Adaptation) freezes the base model and trains small adapter matrices A and B, reducing trainable parameters by 100x.",
        "ragas": "RAGAS evaluates RAG systems on: Faithfulness, Answer Relevancy, Context Precision, Context Recall.",
        "mlflow": "MLflow is an open-source ML lifecycle platform with experiment tracking, model registry, and serving.",
        "rag": "RAG (Retrieval-Augmented Generation) retrieves relevant documents from a vector DB and injects them into the LLM prompt.",
        "mcp": "MCP (Model Context Protocol) is Anthropic's open standard for connecting LLMs to tools and data sources.",
    }
    for key, answer in knowledge.items():
        if key in query.lower():
            return answer
    return f"Search results for '{query}': No specific results found in demo knowledge base."


def calculator_tool(expression: str) -> str:
    """Safe calculator."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math operations allowed"
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def get_current_metrics_tool(model_name: str) -> str:
    """Simulate fetching live model metrics from monitoring system."""
    import random
    random.seed(hash(model_name) % 100)
    return json.dumps({
        "model": model_name,
        "psi_score": round(random.uniform(0.05, 0.25), 3),
        "latency_p95_ms": random.randint(100, 400),
        "error_rate": round(random.uniform(0.001, 0.02), 4),
        "requests_per_day": random.randint(5000, 15000),
    })


# ─────────────────────────────────────────────────────────────
# SIMPLE REACT AGENT
# ─────────────────────────────────────────────────────────────

class ReActAgent:
    """
    A simple ReAct (Reason + Act) agent.

    REACT LOOP:
      1. THOUGHT: "What do I need to find out?"
      2. ACTION: Use a tool
      3. OBSERVATION: See the result
      4. Repeat until FINAL ANSWER

    In production this is driven by the LLM itself.
    Here we simulate the reasoning to show the pattern clearly.
    """

    def __init__(self, tools: List[Tool], max_steps: int = 5):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        """Execute the ReAct loop."""
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")

        steps = []

        # Simulated ReAct reasoning (in production: LLM decides each step)
        # We hardcode a few demo steps to illustrate the pattern

        # Step 1: Think → Act → Observe
        if "psi" in task.lower() or "drift" in task.lower():
            steps = self._drift_analysis_steps()
        elif "cost" in task.lower() or "token" in task.lower():
            steps = self._cost_analysis_steps()
        else:
            steps = self._generic_steps(task)

        final_answer = ""
        for i, step in enumerate(steps[:self.max_steps], 1):
            print(f"\nStep {i}:")
            print(f"  💭 THOUGHT:      {step['thought']}")
            print(f"  ⚡ ACTION:       {step['action']}({step['input']})")

            # Actually execute the tool
            if step['action'] in self.tools:
                tool = self.tools[step['action']]
                result = tool.run(**step['params'])
            else:
                result = "[Tool not found]"

            print(f"  👁 OBSERVATION:  {result[:120]}")
            final_answer = step.get('answer', result)

        print(f"\n✅ FINAL ANSWER: {final_answer}")
        print(f"{'='*60}")
        return final_answer

    def _drift_analysis_steps(self):
        return [
            {
                "thought": "I need to check the current model metrics first.",
                "action": "get_metrics",
                "input": "model_name='rag-v2'",
                "params": {"model_name": "rag-v2"},
            },
            {
                "thought": "Now I'll look up what the PSI score means for this value.",
                "action": "search",
                "input": "query='psi drift threshold'",
                "params": {"query": "psi drift threshold"},
                "answer": "Based on the metrics, the model PSI score and the PSI threshold definition, the model may need investigation if PSI > 0.1.",
            },
        ]

    def _cost_analysis_steps(self):
        return [
            {
                "thought": "Calculate the monthly cost if we make 10000 GPT-4o calls/day at ~500 tokens each.",
                "action": "calculator",
                "input": "expression='10000 * 30 * 500 * 5 / 1000000'",
                "params": {"expression": "10000 * 30 * 500 * 5 / 1000000"},
                "answer": "Monthly GPT-4o cost for 10k calls/day = $7,500. Switch 70% to mini → $1,125 for mini + $2,250 for GPT-4o = $3,375 total. That's a 55% saving.",
            }
        ]

    def _generic_steps(self, task):
        query = task.split()[-1] if task.split() else task
        return [
            {
                "thought": f"I should search for information about this topic.",
                "action": "search",
                "input": f"query='{query}'",
                "params": {"query": query},
                "answer": f"Based on search results for '{query}': this topic relates to ML/LLM concepts covered in the knowledge base.",
            }
        ]


# ─────────────────────────────────────────────────────────────
# MULTI-AGENT PATTERNS
# ─────────────────────────────────────────────────────────────

def explain_multi_agent_patterns():
    """Explain the 4 main multi-agent patterns with code examples."""
    print("""
MULTI-AGENT ORCHESTRATION PATTERNS:
═══════════════════════════════════════════════════════════════

1. SEQUENTIAL (Pipeline) — Most common in LLMOps
   ─────────────────────────────────────────────
   Agent 1 (Retriever) → Agent 2 (Summariser) → Agent 3 (Responder)

   Use case: RAG pipeline where each step is a specialist agent.
   Code (LangChain):
     chain = retriever_chain | summariser_chain | responder_chain

2. PARALLEL — For speed
   ─────────────────────
   Query → Agent A (search web)  ─┐
          → Agent B (search DB)   ├─→ Aggregator → Final Answer
          → Agent C (run calc)   ─┘

   Use case: Gather information from multiple sources simultaneously.
   Code:
     import asyncio
     results = await asyncio.gather(agent_a(query), agent_b(query), agent_c(query))

3. SUPERVISOR — For complex multi-step tasks
   ────────────────────────────────────────────
   Boss Agent (plans & delegates)
     → Worker 1 (web search specialist)
     → Worker 2 (code execution specialist)
     → Worker 3 (writing specialist)

   Use case: AutoGPT-style agents, complex report generation.
   Code (LangGraph):
     graph = StateGraph(AgentState)
     graph.add_node("supervisor", supervisor_agent)
     graph.add_node("worker_1", web_search_agent)
     graph.add_conditional_edges("supervisor", route_to_worker)

4. DEBATE / REFLECTION — For quality improvement
   ─────────────────────────────────────────────
   Agent 1 writes answer
   Agent 2 critiques it
   Agent 1 revises
   Judge Agent picks the best

   Use case: Code review, medical diagnosis, legal analysis.

MCP (MODEL CONTEXT PROTOCOL):
═══════════════════════════════════════════════════════════════

MCP is like a USB-C port for AI tools.
Instead of writing custom integration code for every LLM + every tool,
MCP provides ONE standard interface.

Architecture:
  MCP Host (Claude Desktop, Cursor IDE)
    → MCP Client (handles communication)
      → MCP Server (wraps your tool: file system, GitHub, database)

Benefits:
  - Any MCP-compatible LLM can use any MCP server
  - Security: servers can have different trust levels
  - Discoverability: LLM can list available tools automatically

Example — connecting to a file system via MCP:
  {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]
      }
    }
  }

From CV: "Designed MCP-based multi-agent orchestration workflows
integrating external tools and APIs into enterprise conversational AI systems"
═══════════════════════════════════════════════════════════════
""")


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    # Set up tools
    tools = [
        Tool("search", "Search the knowledge base", search_tool, {"query": "str"}),
        Tool("calculator", "Perform mathematical calculations", calculator_tool, {"expression": "str"}),
        Tool("get_metrics", "Get live model monitoring metrics", get_current_metrics_tool, {"model_name": "str"}),
    ]

    agent = ReActAgent(tools)

    # Demo tasks
    tasks = [
        "Check the current drift metrics for rag-v2 model and explain if action is needed",
        "Calculate the monthly API cost if we process 10000 queries per day",
        "What is RAGAS and how does it work?",
    ]

    for task in tasks:
        agent.run(task)

    # Multi-agent patterns explanation
    explain_multi_agent_patterns()


if __name__ == "__main__":
    main()
