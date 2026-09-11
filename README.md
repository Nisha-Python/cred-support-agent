# Cred Banking & FinTech Support Agent

## Domain

Banking & FinTech — Cred

## Project Overview

The Cred Banking & FinTech Support Agent is a support automation system designed to answer banking and financial-service queries using a grounded Retrieval-Augmented Generation (RAG) pipeline.

The system combines deterministic dataset generation, knowledge-base retrieval, embeddings, ChromaDB, LangGraph orchestration, persistent memory, structured output validation, guardrails, FastAPI, JSONL logging, FastMCP, SQLite checkpointing, retries, and timeout handling.

The project is implemented as a capstone demonstrating software engineering and AI-agent concepts.

---

# Technology Stack

- Python
- LangGraph
- ChromaDB
- SentenceTransformers
- FastAPI
- Pydantic
- FastMCP
- SQLite
- JSON / JSONL
- MOCK_LLM

---

# MOCK_LLM / Zero API Keys

This capstone is designed and demonstrated using `MOCK_LLM`.

The completed demonstrations do not require an external LLM API key.

No API keys are required to execute the validation workflows included in this repository.

Private environment files such as `.env` are intentionally excluded from the public repository.

---

# Project Structure

```text
cred-support-agent/
│
├── agent.py
├── api.py
├── dataset.py
├── mcp_server.py
├── mcp_client.py
├── checkpoint_demo.py
├── task16_resilience.py
├── README.md
│
├── knowledge_base/
│   ├── loan_eligibility.md
│   ├── emi_rules.md
│   ├── credit_card_fees.md
│   ├── kyc_requirements.md
│   ├── fraud_dispute.md
│   ├── account_closure.md
│   ├── interest_rates.md
│   ├── prepayment_rules.md
│   ├── minimum_balance.md
│   ├── credit_score.md
│   ├── joint_account.md
│   └── nri_account.md
│
├── rag/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   └── evaluation.py
│
├── tools/
│   └── loan_status.py
│
└── transcripts/