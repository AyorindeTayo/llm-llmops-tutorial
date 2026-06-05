# 🚀 LLM & LLMOps End-to-End Tutorial
topics.

---

## 📋 What You'll Learn

| Module | Topics Covered |
|--------|---------------|
| 01 | LLM Fundamentals — Transformers, Attention, Tokenization |
| 02 | Prompt Engineering — Zero/Few-shot, Chain-of-Thought, Structured Output |
| 03 | RAG Pipelines — Vector DBs, Semantic Search, Retrieval |
| 04 | LLM Fine-Tuning — LoRA, QLoRA, RLHF |
| 05 | LLM Evaluation — RAGAS, LLM-as-Judge, Metrics |
| 06 | Multi-Agent Orchestration — LangChain Agents, MCP |
| 07 | LLMOps CI/CD — MLflow, GitHub Actions, Docker |
| 08 | Model Monitoring & Drift Detection — PSI, KS-test |
| 09 | LLM Cost Governance — Token Optimization, Caching |
| 10 | AIOps & Observability — Prometheus, Grafana, Alerting |

---

## 🗂 Repository Structure

```
llm-llmops-tutorial/
├── README.md                        ← You are here
├── requirements.txt                 ← All Python dependencies
├── .env.example                     ← Environment variables template
├── docker-compose.yml               ← Local MLflow + monitoring stack
│
├── src/
│   ├── rag/                         ← Module 03: RAG Pipeline
│   │   ├── 01_document_ingestion.py
│   │   ├── 02_vector_store.py
│   │   ├── 03_retrieval_chain.py
│   │   └── 04_production_rag.py
│   │
│   ├── llm_finetuning/              ← Module 04: Fine-Tuning
│   │   ├── 01_lora_finetuning.py
│   │   ├── 02_qlora_finetuning.py
│   │   └── 03_rlhf_basics.py
│   │
│   ├── evaluation/                  ← Module 05: Evaluation
│   │   ├── 01_ragas_evaluation.py
│   │   ├── 02_llm_as_judge.py
│   │   └── 03_custom_metrics.py
│   │
│   ├── agents/                      ← Module 06: Agents & MCP
│   │   ├── 01_langchain_agents.py
│   │   ├── 02_multi_agent.py
│   │   └── 03_mcp_integration.py
│   │
│   ├── mlops/                       ← Module 07: LLMOps CI/CD
│   │   ├── 01_mlflow_tracking.py
│   │   ├── 02_model_registry.py
│   │   └── 03_pipeline_automation.py
│   │
│   ├── monitoring/                  ← Module 08: Monitoring
│   │   ├── 01_drift_detection.py
│   │   ├── 02_prometheus_metrics.py
│   │   └── 03_alerting.py
│   │
│   └── cost_governance/             ← Module 09: Cost Governance
│       ├── 01_token_optimization.py
│       ├── 02_caching_strategies.py
│       └── 03_cost_dashboard.py
│
├── notebooks/                       ← Jupyter notebooks (interactive)
│   ├── 01_llm_fundamentals.ipynb
│   └── 02_prompt_engineering.ipynb
│
├── configs/                         ← Config files
│   ├── model_config.yaml
│   └── monitoring_config.yaml
│
├── tests/                           ← Unit tests
│   ├── test_rag.py
│   └── test_evaluation.py
│
├── scripts/                         ← Utility scripts
│   ├── setup_env.sh
│   └── run_pipeline.sh
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml                ← GitHub Actions CI/CD
│
└── docker-compose.yml
```

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/AyorindeTayo/llm-llmops-tutorial.git
cd llm-llmops-tutorial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Set Your API Keys

```bash
# In your .env file:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...
```

### 3. Start Local Stack (MLflow + Monitoring)

```bash
docker-compose up -d
# MLflow UI → http://localhost:5000
# Grafana  → http://localhost:3000
# Prometheus → http://localhost:9090
```

### 4. Run the Modules

```bash
# Module 03: RAG Pipeline
python src/rag/04_production_rag.py

# Module 05: Evaluation
python src/evaluation/01_ragas_evaluation.py

# Module 07: MLflow Tracking
python src/mlops/01_mlflow_tracking.py

# Module 08: Drift Detection
python src/monitoring/01_drift_detection.py

# Module 09: Cost Governance
python src/cost_governance/01_token_optimization.py
```

---

## 📚 Module-by-Module Learning Guide

### Module 01 — LLM Fundamentals
**Key interview concepts:**
- What is a Transformer? (Attention is All You Need — encoder/decoder)
- What is tokenization? BPE vs WordPiece
- What is temperature, top-k, top-p sampling?
- What is the context window? Why does it matter?

**See:** `notebooks/01_llm_fundamentals.ipynb`

---

### Module 02 — Prompt Engineering
**Key interview concepts:**
- Zero-shot vs few-shot prompting
- Chain-of-thought (CoT) prompting
- System prompts vs user prompts
- Structured output prompting (JSON mode)
- Prompt injection and how to defend against it

**See:** `notebooks/02_prompt_engineering.ipynb`

---

### Module 03 — RAG Pipelines
**Key interview concepts:**
- What is RAG and why is it better than fine-tuning for factual Q&A?
- What is a vector database? (Pinecone, Weaviate, Chroma, FAISS)
- What is semantic search vs keyword search?
- What is chunking strategy and why it matters?
- What is RBAC in a RAG system?

**See:** `src/rag/`

---

### Module 04 — LLM Fine-Tuning
**Key interview concepts:**
- What is LoRA? (Low-Rank Adaptation — freeze base model, train adapter layers)
- What is QLoRA? (Quantized LoRA — 4-bit quantization + LoRA)
- What is RLHF? (Reward model + PPO to align with human preferences)
- When to fine-tune vs use RAG?
- What is catastrophic forgetting?

**See:** `src/llm_finetuning/`

---

### Module 05 — LLM Evaluation
**Key interview concepts:**
- What is RAGAS? (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- What is LLM-as-Judge?
- What metrics measure hallucination?
- What is BLEU/ROUGE and their limitations?

**See:** `src/evaluation/`

---

### Module 06 — Multi-Agent Orchestration
**Key interview concepts:**
- What is an LLM agent? (LLM + tools + memory + planning)
- What is ReAct? (Reason + Act loop)
- What is MCP? (Model Context Protocol — standard for tool/context integration)
- What is the difference between sequential and parallel agent chains?

**See:** `src/agents/`

---

### Module 07 — LLMOps CI/CD
**Key interview concepts:**
- What is MLflow? (Experiment tracking, model registry, serving)
- How do you do CI/CD for ML models? (GitHub Actions + Docker + MLflow)
- What is DVC? (Data Version Control)
- What is a model registry and why use it?
- What is blue/green deployment vs canary deployment?

**See:** `src/mlops/` and `.github/workflows/ci_cd.yml`

---

### Module 08 — Monitoring & Drift Detection
**Key interview concepts:**
- What is data drift vs concept drift?
- What is PSI (Population Stability Index)?
- What is the KS-test?
- What is model degradation and how do you detect it?
- What triggers a model retraining?

**See:** `src/monitoring/`

---

### Module 09 — LLM Cost Governance
**Key interview concepts:**
- How do you reduce LLM API costs? (caching, shorter prompts, smaller models)
- What is semantic caching?
- How do you enforce token budgets?
- How do you balance cost vs quality?

**See:** `src/cost_governance/`

---

### Module 10 — AIOps & Observability
**Key interview concepts:**
- What is Prometheus? (metrics scraping and storage)
- What is Grafana? (metrics dashboards)
- What is the ELK Stack? (Elasticsearch + Logstash + Kibana — log analysis)
- What is SLA governance?
- How do you set up alerts for model incidents?

**See:** `src/monitoring/02_prometheus_metrics.py`

---

## 🎯 Top Interview Questions & Quick Answers

| Question | Quick Answer |
|----------|-------------|
| What is RAG? | Retrieval-Augmented Generation — retrieve relevant docs from a vector DB and inject into LLM context so it answers with up-to-date, grounded facts |
| LoRA vs full fine-tuning? | LoRA freezes the base model and trains small low-rank adapter matrices — 10-100x fewer parameters, same quality |
| What is RAGAS? | Framework to evaluate RAG pipelines on faithfulness, relevancy, context precision, and recall |
| PSI vs KS-test? | PSI measures distributional shift in categorical/continuous features (>0.2 = significant); KS-test is a statistical test for continuous distribution change |
| What is an LLM agent? | LLM + tools (web search, code exec, APIs) + memory + a planning loop (ReAct) to solve multi-step tasks |
| How to reduce hallucination? | RAG + structured prompting + temperature tuning + RLHF + output validation |
| What is MCP? | Model Context Protocol — Anthropic's open standard for connecting LLMs to external tools and data sources in a consistent way |
| What is canary deployment? | Route small % of traffic to new model, monitor metrics, gradually increase if healthy — reduces rollout risk |
| How to cut LLM costs 30%? | Semantic caching, prompt compression, model routing (GPT-4 only for hard queries), batching, token budget enforcement |

---

## 🏗 Tech Stack

| Category | Tools |
|----------|-------|
| LLM APIs | OpenAI (GPT-4), Anthropic (Claude), Hugging Face |
| LLM Frameworks | LangChain, LlamaIndex |
| Vector DBs | Pinecone, Chroma, FAISS |
| Fine-Tuning | PEFT (LoRA/QLoRA), TRL (RLHF), BitsAndBytes |
| Evaluation | RAGAS, DeepEval |
| MLOps | MLflow, DVC, Weights & Biases |
| Monitoring | Prometheus, Grafana, PSI, KS-test |
| Infrastructure | Docker, Kubernetes (concepts), GitHub Actions |
| Languages | Python 3.10+, YAML, Bash |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📄 License

MIT License — free to use, fork, and adapt.

---

## 👤 Author

**Ayorinde Olanipekun**  
Machine Learning Engineer | MLOps / LLMOps / AIOps Specialist  
📧 olanipekunayo2012@gmail.com  
🔗 [github.com/AyorindeTayo](https://github.com/AyorindeTayo)  
🌐 [codedatawithayo.com](https://codedatawithayo.com)
