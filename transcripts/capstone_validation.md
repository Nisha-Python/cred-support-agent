# Cred Banking & FinTech Support Agent
## Capstone Validation Transcript

PROJECT DOMAIN: Banking & FinTech — Cred
LLM MODE: MOCK_LLM
EXTERNAL API KEYS: NOT REQUIRED


============================================================
TASK 1 — DETERMINISTIC LOAN DATASET
============================================================

Objective:
Create a deterministic loan application dataset containing at least
40 records and satisfying the required category, status and fraud
review distribution requirements.

Implementation:
The dataset is generated using dataset.py.

Required fields implemented:
- record_id
- category
- status
- loan_amount_inr
- days_since_created
- flagged_for_fraud_review

The dataset contains 50 loan application records.

The dataset generation is deterministic using a seeded approach,
so repeated executions produce consistent results.

Validation:
- Required record count verified
- Required fields verified
- Category coverage verified
- Status coverage verified
- Fraud-review distribution verified
- Deterministic generation verified

RESULT:
TASK 1 VALIDATION: PASSED


============================================================
TASK 2 — KNOWLEDGE BASE
============================================================

Objective:
Create a banking knowledge base containing at least 12 documents
covering all required banking topics.

Knowledge-base documents:

1. loan_eligibility.md
2. emi_rules.md
3. credit_card_fees.md
4. kyc_requirements.md
5. fraud_dispute.md
6. account_closure.md
7. interest_rates.md
8. prepayment_rules.md
9. minimum_balance.md
10. credit_score.md
11. joint_account.md
12. nri_account.md

Topics covered:
- Loan eligibility by type
- EMI rules
- Credit-card fees
- KYC requirements
- Fraud dispute
- Account closure
- Interest slabs
- Prepayment penalties
- Minimum balance
- Credit score
- Joint account
- NRI account

Validation:
- 12 required knowledge-base documents present
- Required topics covered
- Documents contain banking support information
- Documents are usable by the RAG pipeline

RESULT:
TASK 2 VALIDATION: PASSED


============================================================
TASK 3 — CHUNKING, EMBEDDINGS AND CHROMADB
============================================================

Objective:
Implement two chunking strategies, generate embeddings and index
the results into separate ChromaDB collections.

Chunking strategies implemented:

1. Fixed-size chunking with overlap
2. Sentence-based chunking

Embedding model:

all-MiniLM-L6-v2

Embedding framework:
SentenceTransformers

Separate ChromaDB collections:

- loan_kb_fixed
- loan_kb_sentence

Both collections were populated and tested with sample queries.

Validation:
- Fixed-size chunking implemented
- Sentence-based chunking implemented
- Embeddings generated
- Separate ChromaDB collections created
- Retrieval tested

RESULT:
TASK 3 VALIDATION: PASSED


============================================================
TASK 4 — GROUNDED GENERATION AND CALIBRATION
============================================================

Objective:
Implement grounded generation and empirically determine a similarity
threshold that separates in-scope and out-of-scope queries.

Observed in-scope top-1 similarity scores:

0.7397
0.8024
0.7381

Observed out-of-scope similarity scores:

0.0495
0.1757

Selected similarity threshold:

0.4569

The threshold was selected between the observed in-scope and
out-of-scope similarity clusters.

Grounded generation was implemented using retrieved knowledge-base
context.

Out-of-scope queries correctly trigger the fallback response:

"I don't know."

Validation:
- In-scope queries tested
- Out-of-scope queries tested
- Similarity calibration completed
- Threshold selected
- Grounded fallback demonstrated
- At least 5 in-scope and 1 out-of-scope cases supported by the
  implementation

RESULT:
TASK 4 VALIDATION: PASSED


============================================================
TASK 5 — PRECISION@3 AND RECALL@3
============================================================

Objective:
Evaluate both retrieval strategies using Precision@3 and Recall@3.

Fixed-size strategy:

Average Precision@3 = 0.3333
Average Recall@3    = 1.0000

Sentence-based strategy:

Average Precision@3 = 0.3333
Average Recall@3    = 1.0000

Document-level deduplication was implemented.

Per-query arithmetic was used to calculate the evaluation metrics.

Recommendation:
Both strategies achieved full average Recall@3.
The sentence-based strategy provides focused sentence-level
retrieval and is suitable for the subsequent support-agent workflow.

RESULT:
TASK 5 VALIDATION: PASSED


============================================================
TASK 6 — LOAN APPLICATION STATUS TOOL
============================================================

Objective:
Implement a loan application status tool that returns application
information and computes an escalation score.

Tool:

check_loan_application_status(record_id: str) -> dict

Implementation file:

tools/loan_status.py

Escalation score combines:

- Fraud-review signal
- Normalized recency signal

Weights:

FRAUD_WEIGHT = 0.60
RECENCY_WEIGHT = 0.40

Maximum application age used for normalization:

30 days

Escalation threshold:

0.68

Example validated record:

record_id = CRD-0001

Example result:

status = Approved
loan_amount_inr = 3870614
days_since_created = 18
flagged_for_fraud_review = True
recency_signal = 0.4
escalation_score = 0.76
escalation_threshold = 0.68
recommend_escalation = True

Invalid record IDs are handled with a controlled error.

RESULT:
TASK 6 VALIDATION: PASSED


============================================================
TASK 7 — LANGGRAPH AGENT AND CONDITIONAL ROUTING
============================================================

Objective:
Build a LangGraph agent containing at least four nodes and a
conditional edge for routing different types of queries.

The graph contains multiple processing nodes for:

- Router
- Loan-status tool
- RAG processing
- Fallback handling
- Output validation / response processing

Conditional routing is used to decide whether a query should be
handled by the RAG workflow or the loan-status tool.

Banking knowledge queries are routed to RAG.

Loan application status queries are routed to the loan-status tool.

Unsupported queries can be routed to the fallback path.

RESULT:
TASK 7 VALIDATION: PASSED


============================================================
TASK 8 — PERSISTENT JSON CONVERSATION MEMORY
============================================================

Objective:
Persist conversation memory and demonstrate multi-turn conversation
and fresh conversation reset.

Memory file:

conversation_memory.json

The agent uses thread-based conversation memory.

The same thread ID allows a conversation to continue using its
previous context.

A new thread ID starts a fresh conversation.

Validation demonstrated:

- Multi-turn conversation
- Persistent conversation state
- Fresh conversation reset

RESULT:
TASK 8 VALIDATION: PASSED


============================================================
TASK 9 — STRUCTURED OUTPUT VALIDATION
============================================================

Objective:
Validate every agent response against the declared structured
output schema.

Implementation:
Agent responses are validated against the project's JSON schema
before being accepted as valid output.

This prevents malformed responses from being returned as valid
agent results.

RESULT:
TASK 9 VALIDATION: PASSED


============================================================
TASK 10 — GUARDRAILS
============================================================

Objective:
Implement and demonstrate input-side and output-side guardrails.

Input guardrail 1:
PII masking

Sensitive user information such as account numbers, phone numbers
and email addresses is detected and masked before processing.

Input guardrail 2:
Prompt-injection detection

Deliberate prompt-injection input was tested and the guardrail
correctly fired.

Output guardrail:
Groundedness refusal

If an answer cannot be supported by retrieved knowledge, the agent
refuses to provide an unsupported answer.

Validation output:

Output groundedness refusal: FIRED

The PII masking and prompt-injection guardrails were also
deliberately tested.

RESULT:
TASK 10 VALIDATION: PASSED


============================================================
TASK 11 — FASTAPI BACKEND
============================================================

Objective:
Expose the support agent through a FastAPI backend using Pydantic
request and response models.

Working endpoints:

GET /health

POST /query

Health endpoint response:

{
    "status": "ok",
    "service": "cred-support-agent",
    "mock_llm": true
}

HTTP status:

200 OK

The /query endpoint was tested successfully through the FastAPI
Swagger interface.

The endpoint returned a structured response containing the agent
answer, route, sources and retrieval information.

RESULT:
TASK 11 VALIDATION: PASSED


============================================================
TASK 12 — JSONL LOGGING AND TRACE ID
============================================================

Objective:
Create JSONL request logs containing trace IDs and request timing.

Log file:

api_logs.jsonl

Each API request produces a JSONL record containing:

- timestamp
- trace_id
- endpoint
- query
- route
- status
- latency_ms

Example:

{
    "timestamp": "2026-09-10T12:42:19.917550+00:00",
    "trace_id": "generated-per-request",
    "endpoint": "/query",
    "query": "What are the KYC requirements?",
    "route": "rag",
    "status": "success",
    "latency_ms": 193.745
}

PII is masked before logging.

RESULT:
TASK 12 VALIDATION: PASSED


============================================================
TASK 13 — RAG TRIAD EVALUATION
============================================================

Objective:
Evaluate 15 RAG queries under MOCK_LLM using:

1. Context relevance
2. Groundedness
3. Answer relevance

Number of queries:

15

Final averages:

Average Context Relevance: 0.4502

Average Groundedness: 0.9368

Average Answer Relevance: 0.5747

The evaluator calculates the three metrics for every query and
then computes the final averages.

Validation output:

TASK 13 — FINAL AVERAGES

Number of queries: 15
Average Context Relevance: 0.4502
Average Groundedness: 0.9368
Average Answer Relevance: 0.5747

TASK 13 VALIDATION: PASSED

RESULT:
TASK 13 VALIDATION: PASSED


============================================================
TASK 14 — FASTMCP SERVER AND CLIENT
============================================================

Objective:
Implement an MCP server and client with an HTTP MCP endpoint and
at least two callable tools.

FastMCP version:

4.0.3

MCP server name:

Cred Support MCP

MCP HTTP endpoint:

http://127.0.0.1:8001/mcp

Two MCP tools were exposed:

1. get_loan_status
2. search_knowledge_base

MCP client validation:

Number of tools: 2

Tool ID: get_loan_status

Tool ID: search_knowledge_base

The client successfully connected to the MCP server.

The get_loan_status tool was successfully called using a valid
loan application record ID.

The search_knowledge_base tool was successfully called using a
banking knowledge-base query.

RESULT:
TASK 14 VALIDATION: PASSED


============================================================
TASK 15 — SQLITE LANGGRAPH CHECKPOINT
============================================================

Objective:
Demonstrate LangGraph checkpointing using SQLite and resume the
same workflow thread without re-executing completed nodes.

SQLite checkpoint database:

langgraph_checkpoint.db

The workflow uses the same thread ID for interruption and resume.

PHASE 1:

NODE 1 executed.

NODE 2 executed.

A simulated interruption occurred after NODE 2.

The checkpoint preserved the completed workflow state.

PHASE 2:

The same thread was resumed from the SQLite checkpoint.

NODE 3 executed.

NODE 1 and NODE 2 were not executed again.

Final events:

['node_1', 'node_2', 'node_3']

Validation:

PHASE 1 VALIDATION: PASSED

NODE 3: executed

TASK 15 VALIDATION: PASSED

RESULT:
TASK 15 VALIDATION: PASSED


============================================================
TASK 16 — TIMEOUTS AND RETRIES
============================================================

Objective:
Implement resilient workflow execution using per-node timeouts,
global timeout and retry policies.

Implemented resilience features:

- Per-node timeout
- Global workflow timeout
- Exponential backoff
- Maximum retry attempts
- Initial retry interval
- Maximum retry interval
- Jitter
- Simulated transient failure recovery

A simulated transient failure was introduced to test the retry
mechanism.

The retry mechanism successfully recovered from the simulated
transient failure.

PER-NODE TIMEOUT TEST:

A deliberately slow node exceeded the configured per-node timeout.

Result:

Per-node timeout test: PASSED

GLOBAL TIMEOUT TEST:

A deliberately slow workflow exceeded the configured global timeout.

Result:

Global timeout test: PASSED

Final validation:

TASK 16 VALIDATION: PASSED

RESULT:
TASK 16 VALIDATION: PASSED


============================================================
FINAL CAPSTONE VALIDATION SUMMARY
============================================================

Task 1  — Deterministic Dataset                  — PASSED
Task 2  — Knowledge Base                         — PASSED
Task 3  — Chunking, Embeddings and ChromaDB      — PASSED
Task 4  — Grounded Generation and Calibration    — PASSED
Task 5  — Precision@3 and Recall@3               — PASSED
Task 6  — Loan Application Status Tool            — PASSED
Task 7  — LangGraph Routing                      — PASSED
Task 8  — Persistent JSON Memory                 — PASSED
Task 9  — Structured Output Validation           — PASSED
Task 10 — Guardrails                             — PASSED
Task 11 — FastAPI + Pydantic                    — PASSED
Task 12 — JSONL Logging + Trace ID               — PASSED
Task 13 — 15-Query RAG Triad Evaluation          — PASSED
Task 14 — FastMCP Server + Client                — PASSED
Task 15 — SQLite LangGraph Checkpoint             — PASSED
Task 16 — Timeout + Retry Resilience              — PASSED


============================================================
FINAL PROJECT STATUS
============================================================

CAPSTONE TASKS COMPLETED: 16 / 16

VALIDATION STATUS: PASSED

LLM MODE: MOCK_LLM

EXTERNAL API KEYS REQUIRED FOR DEMONSTRATION: NO

The project has been implemented and validated across all four
capstone parts:

Part 1:
Dataset, knowledge base, chunking, embeddings, retrieval,
grounded generation and evaluation.

Part 2:
Loan-status tool, LangGraph routing, conversation memory,
structured outputs and guardrails.

Part 3:
FastAPI, Pydantic, JSONL logging, trace IDs and RAG triad
evaluation.

Part 4:
FastMCP, SQLite checkpointing, timeout handling and retry
resilience.


============================================================
PUBLIC REPOSITORY SECURITY
============================================================

The public GitHub repository must not contain private environment
files or local virtual-environment files.

The following must remain excluded from the public repository:

.env
.venv/

API keys must never be committed to GitHub.

The README.md provides instructions confirming that the project
runs using MOCK_LLM without external API keys.


============================================================
END OF CAPSTONE VALIDATION TRANSCRIPT
============================================================