"""
CRED SUPPORT AGENT - FASTAPI
----------------------------
Task 11: FastAPI endpoints with Pydantic.
Task 12: JSONL logging with trace ID, timing and PII masking.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import run_agent, mask_pii


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

LOG_FILE = PROJECT_ROOT / "api_logs.jsonl"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Cred Support Agent API",
    description="Banking and FinTech support agent API",
    version="1.0.0",
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str
    mock_llm: bool


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User support query",
    )

    thread_id: str = Field(
        default="default",
        min_length=1,
        description="Conversation thread identifier",
    )


class QueryResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]
    fallback: bool
    loan_record_id: str | None = None
    similarity: float | None = None


# ============================================================
# TASK 12 — JSONL LOGGING
# ============================================================

def write_jsonl_log(
    *,
    trace_id: str,
    endpoint: str,
    query: str,
    route: str | None,
    status: str,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """
    Write one structured JSON log entry.

    Important:
    - PII is masked before writing.
    - Logs are JSONL: one JSON object per line.
    - trace_id allows one request to be traced.
    - latency_ms records request execution time.
    """

    log_entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "trace_id": trace_id,

        "endpoint": endpoint,

        # Never write raw user PII to logs.
        "query": mask_pii(query),

        "route": route,

        "status": status,

        "latency_ms": round(
            latency_ms,
            3
        ),
    }

    if error is not None:
        log_entry["error"] = error

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                log_entry,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Health-check endpoint.

    Confirms that the API service is running.
    """

    return HealthResponse(
        status="ok",
        service="cred-support-agent",
        mock_llm=True,
    )


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_agent(
    request: QueryRequest,
):
    """
    Run one support-agent query.

    Task 12:
    - generates a unique trace ID
    - measures execution time
    - masks PII before logging
    - writes a JSONL record
    """

    trace_id = str(
        uuid.uuid4()
    )

    start_time = time.perf_counter()

    try:

        result = run_agent(
            query=request.query,
            thread_id=request.thread_id,
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        route = result.get(
            "route"
        )

        write_jsonl_log(
            trace_id=trace_id,
            endpoint="/query",
            query=request.query,
            route=route,
            status="success",
            latency_ms=latency_ms,
        )

        return QueryResponse(
            answer=result.get(
                "answer",
                "No answer generated.",
            ),

            route=route or "unknown",

            sources=result.get(
                "sources",
                [],
            ),

            fallback=result.get(
                "fallback",
                False,
            ),

            loan_record_id=result.get(
                "loan_record_id"
            ),

            similarity=result.get(
                "similarity"
            ),
        )

    except Exception as exc:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        write_jsonl_log(
            trace_id=trace_id,
            endpoint="/query",
            query=request.query,
            route=None,
            status="error",
            latency_ms=latency_ms,
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {exc}",
        )


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )