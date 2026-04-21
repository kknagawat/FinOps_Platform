"""
src/api/main.py
===============
Step 7 — FastAPI Platform Service

All 8 required endpoints:
  GET  /health
  POST /etl/run
  GET  /etl/status/{job_id}
  POST /agent/query
  GET  /agent/query/history
  GET  /analytics/dashboard
  GET  /governance/pending
  POST /governance/review/{id}
  POST /agent/query/validate

Technical requirements met:
  ✓ Pydantic v2 models (schemas.py) on every request and response
  ✓ Proper HTTP status codes (200/202/400/403/404/422/500)
  ✓ Structured error responses (ErrorResponse model)
  ✓ Dependency injection for DB connection, agent, RAG readiness
  ✓ CORS middleware
  ✓ Background tasks for ETL with job ID polling
  ✓ Governance layer wraps every /agent/query call
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AgentQueryRequest, AgentQueryResponse, AuditEntry,
    ClassificationEnum, DashboardKPIs, ETLJobResponse, ETLStatusResponse,
    ErrorResponse, HealthResponse, HistoryResponse, PendingReview,
    ReviewRequest, ReviewResponse, ValidateSQLRequest, ValidateSQLResponse,
)

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT / "finops.db"

# ── In-memory ETL job registry ────────────────────────────────────────────────
_etl_jobs:  dict[str, dict] = {}

# ── Module-level singletons ───────────────────────────────────────────────────
_agent     = None
_rag_ready = False


# =============================================================================
# Lifespan: initialise agent + RAG on startup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise agent and RAG index when the server starts."""
    global _agent, _rag_ready
    log.info("FinOps API starting up …")

    if DB_PATH.exists():
        try:
            from src.rag.retriever import get_retriever
            get_retriever()
            _rag_ready = True
            log.info("RAG index ready")
        except Exception as e:
            log.warning("RAG not ready at startup: %s", e)

        try:
            from src.agent.agent import FinOpsAgent
            _agent = FinOpsAgent(verbose=False)
            log.info("Agent ready")
        except Exception as e:
            log.warning("Agent not ready at startup: %s", e)
    else:
        log.warning("finops.db not found — run POST /etl/run to initialise")

    yield   # server runs here
    log.info("FinOps API shutting down")


# =============================================================================
# App factory
# =============================================================================

app = FastAPI(
    title       = "FinOps Analytics Platform API",
    description = (
        "AI-powered FinOps analytics with multi-tool LangChain agent, "
        "RAG-powered policy retrieval, and human-in-the-loop governance."
    ),
    version = "1.0.0",
    lifespan = lifespan,
    docs_url = "/docs",
    redoc_url = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# Error handler
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {"error": "Internal server error", "detail": str(exc)},
    )


# =============================================================================
# Dependencies
# =============================================================================

def get_db() -> sqlite3.Connection:
    """Yield a SQLite connection; close on exit."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_agent():
    if _agent is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Agent not initialised. Run POST /etl/run first.",
        )
    return _agent


def require_db():
    if not DB_PATH.exists():
        raise HTTPException(
            status_code = 503,
            detail      = "Database not found. Run POST /etl/run first.",
        )


# =============================================================================
# GET /health
# =============================================================================

@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "Health check — DB status, row counts, agent readiness",
)
def health_check(db: sqlite3.Connection = Depends(get_db)):
    """
    Returns platform health:
    - Database connectivity and row counts per table
    - Whether the LangChain agent is initialised
    - Whether the RAG FAISS index is loaded
    - Number of queries currently pending human review
    """
    tables = [
        "customers", "subscriptions", "transactions",
        "invoices", "support_tickets", "product_usage", "date_dim",
    ]
    row_counts: dict[str, int] = {}
    db_status = "connected"

    try:
        for t in tables:
            row_counts[t] = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    except Exception as e:
        db_status = f"error: {e}"

    pending = 0
    try:
        pending = db.execute(
            "SELECT COUNT(*) FROM approval_queue WHERE status='pending'"
        ).fetchone()[0]
    except Exception:
        pass

    return HealthResponse(
        status          = "ok" if db_status == "connected" else "degraded",
        db_status       = db_status,
        row_counts      = row_counts,
        agent_ready     = _agent is not None,
        rag_index_ready = _rag_ready,
        pending_reviews = pending,
    )


# =============================================================================
# POST /etl/run
# =============================================================================

def _run_etl(job_id: str) -> None:
    """Background ETL task: ingest → clean → load → build RAG index → init agent."""
    global _agent, _rag_ready
    _etl_jobs[job_id]["status"] = "running"
    t0 = time.perf_counter()

    try:
        from src.etl.ingestion import run_ingestion
        from src.etl.cleaning  import run_cleaning
        from src.etl.loader    import run_loader
        from src.rag.embedder  import build_and_save
        from src.rag.retriever import get_retriever
        from src.agent.agent   import FinOpsAgent

        raw     = run_ingestion()
        cleaned = run_cleaning(raw)
        result  = run_loader(cleaned)

        build_and_save()
        get_retriever()
        _rag_ready = True

        _agent = FinOpsAgent(verbose=False)

        _etl_jobs[job_id].update({
            "status":         "completed",
            "rows_processed": result["row_counts_loaded"],
            "errors":         result["issues"],
            "duration_s":     round(time.perf_counter() - t0, 2),
        })

    except Exception as e:
        log.error("ETL job %s failed: %s", job_id, e, exc_info=True)
        _etl_jobs[job_id].update({
            "status":     "failed",
            "errors":     [str(e)],
            "duration_s": round(time.perf_counter() - t0, 2),
        })


@app.post(
    "/etl/run",
    response_model = ETLJobResponse,
    status_code    = 202,
    tags           = ["ETL"],
    summary        = "Trigger full ETL pipeline (async)",
)
def run_etl(background_tasks: BackgroundTasks):
    """
    Starts the full ETL pipeline as a background task:
    1. Ingest raw files
    2. Clean and validate
    3. Load into SQLite
    4. Build RAG FAISS index
    5. Initialise the LangChain agent

    Returns immediately with a `job_id`. Poll `GET /etl/status/{job_id}` for progress.
    """
    job_id = str(uuid.uuid4())
    _etl_jobs[job_id] = {"status": "accepted", "job_id": job_id}
    background_tasks.add_task(_run_etl, job_id)
    return ETLJobResponse(
        job_id  = job_id,
        status  = "accepted",
        message = f"ETL started. Poll GET /etl/status/{job_id} for progress.",
    )


@app.get(
    "/etl/status/{job_id}",
    response_model = ETLStatusResponse,
    tags           = ["ETL"],
    summary        = "Poll ETL job status",
)
def etl_status(job_id: str):
    job = _etl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ETLStatusResponse(**{k: v for k, v in job.items() if k != "job_id"},
                              job_id=job_id)


# =============================================================================
# POST /agent/query
# =============================================================================

@app.post(
    "/agent/query",
    response_model = AgentQueryResponse,
    tags           = ["Agent"],
    summary        = "Submit a natural language question to the governed agent",
)
def agent_query(
    request: AgentQueryRequest,
    agent   = Depends(get_agent),
    _db     = Depends(require_db),
):
    """
    Accepts a natural language question and routes it through:
    1. **Safety classifier** — blocks mutating requests, queues PII/bulk requests
    2. **LangChain agent** — routes to the right tool (SQL, RAG, charts, etc.)
    3. **Output guardrails** — PII masking, row limits, confidence surfacing
    4. **Audit log** — every interaction is recorded

    **HTTP status codes:**
    - `200` — safe query, answered
    - `202` — queued for human review (`approval_required=true`)
    - `403` — blocked query (mutating SQL, data deletion, etc.)
    - `503` — agent not initialised
    """
    from src.governance.governance import governed_query
    from src.governance.classifier import Classification

    response = governed_query(agent, request.question, session_id=request.session_id)

    # Map governance classification to HTTP status
    if response.classification == Classification.BLOCKED.value:
        raise HTTPException(
            status_code = 403,
            detail      = response.answer,
        )

    http_status = 202 if response.approval_required else 200

    return JSONResponse(
        status_code = http_status,
        content     = AgentQueryResponse(
            answer                = response.answer,
            tools_used            = response.tools_used,
            generated_sql         = response.generated_sql,
            raw_results           = response.raw_results,
            sources               = response.sources,
            chart_path            = response.chart_path,
            classification        = response.classification,
            classification_reason = response.classification_reason,
            warnings              = response.warnings,
            execution_time_ms     = response.execution_time_ms,
            confidence            = response.confidence,
            approval_required     = response.approval_required,
            approval_id           = response.approval_id,
            audit_id              = response.audit_id,
        ).model_dump(),
    )


# =============================================================================
# GET /agent/query/history
# =============================================================================

@app.get(
    "/agent/query/history",
    response_model = HistoryResponse,
    tags           = ["Agent"],
    summary        = "Paginated audit trail with optional filters",
)
def query_history(
    limit:          int           = Query(20, ge=1, le=100, description="Rows per page"),
    offset:         int           = Query(0,  ge=0,        description="Pagination offset"),
    tool_used:      Optional[str] = Query(None, description="Filter by tool name"),
    classification: Optional[str] = Query(None, description="Filter: safe / requires_review / blocked"),
    _db             = Depends(require_db),
):
    """
    Returns the last N agent interactions with full audit detail.
    Supports pagination (limit/offset) and filtering by tool or classification.
    """
    from src.governance.audit import get_history
    total, entries = get_history(limit, offset, tool_used, classification)
    return HistoryResponse(
        total   = total,
        limit   = limit,
        offset  = offset,
        entries = [AuditEntry(**e) for e in entries],
    )


# =============================================================================
# GET /analytics/dashboard
# =============================================================================

@app.get(
    "/analytics/dashboard",
    response_model = DashboardKPIs,
    tags           = ["Analytics"],
    summary        = "Pre-computed KPI dashboard via direct SQL",
)
def dashboard(db: sqlite3.Connection = Depends(get_db)):
    """
    Returns live KPIs computed via direct SQL (not the agent):
    MRR, ARR, active customers, churn rate, avg ticket resolution time, NPS proxy.

    Direct SQL is used here (not the agent) so the dashboard is fast
    and deterministic — it never triggers a governance check or LLM call.
    """
    try:
        mrr = float(db.execute(
            "SELECT COALESCE(SUM(mrr),0) FROM subscriptions WHERE status='active'"
        ).fetchone()[0])

        active_customers = db.execute(
            "SELECT COUNT(*) FROM customers WHERE is_active=1"
        ).fetchone()[0]

        churned = db.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM subscriptions WHERE status='cancelled'"
        ).fetchone()[0]
        total_subs = db.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM subscriptions"
        ).fetchone()[0]
        churn_rate = round(100.0 * churned / total_subs, 2) if total_subs else 0.0

        avg_res = db.execute(
            """SELECT AVG((julianday(resolved_at)-julianday(created_at))*24)
               FROM support_tickets
               WHERE resolved_at IS NOT NULL AND created_at IS NOT NULL"""
        ).fetchone()[0] or 0.0

        avg_rating = db.execute(
            "SELECT AVG(rating) FROM support_tickets WHERE rating IS NOT NULL"
        ).fetchone()[0]
        nps = round((float(avg_rating) - 3.0) / 2.0 * 100, 1) if avg_rating else None

        return DashboardKPIs(
            mrr                     = round(mrr, 2),
            arr                     = round(mrr * 12, 2),
            active_customers        = active_customers,
            churn_rate_pct          = churn_rate,
            avg_ticket_resolution_h = round(float(avg_res), 2),
            nps_score               = nps,
            computed_at             = datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard computation failed: {e}")


# =============================================================================
# GET /governance/pending
# =============================================================================

@app.get(
    "/governance/pending",
    response_model = list[PendingReview],
    tags           = ["Governance"],
    summary        = "List queries pending human review",
)
def list_pending(_db = Depends(require_db)):
    """Returns all queries in the approval queue with status = 'pending'."""
    from src.governance.queue import list_pending as _list
    return [PendingReview(**r) for r in _list()]


# =============================================================================
# POST /governance/review/{id}
# =============================================================================

@app.post(
    "/governance/review/{review_id}",
    response_model = ReviewResponse,
    tags           = ["Governance"],
    summary        = "Approve or reject a pending query",
)
def process_review(
    review_id: int,
    request:   ReviewRequest,
    _db        = Depends(require_db),
):
    """
    Approve or reject a query that was held for human review.

    - `approve` → status set to 'approved'; the agent can execute the query
    - `reject`  → status set to 'rejected'; the request is discarded

    Returns 404 if the review ID doesn't exist.
    Returns 400 if the entry is not in 'pending' status.
    """
    from src.governance.queue import approve, reject

    try:
        if request.action.value == "approve":
            updated = approve(review_id, request.reviewer_notes, request.reviewer_id)
        else:
            updated = reject(review_id, request.reviewer_notes, request.reviewer_id)

        return ReviewResponse(
            id      = review_id,
            action  = request.action.value,
            status  = updated["status"],
            message = f"Review #{review_id} {updated['status']} successfully.",
        )
    except ValueError as e:
        status_code = 404 if "not found" in str(e) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# POST /agent/query/validate
# =============================================================================

@app.post(
    "/agent/query/validate",
    response_model = ValidateSQLResponse,
    tags           = ["Agent"],
    summary        = "Classify a SQL string without executing it",
)
def validate_sql(request: ValidateSQLRequest):
    """
    Classifies a raw SQL string as safe / requires_review / blocked
    without executing it. Useful for pre-checking queries before submission.
    """
    from src.governance.classifier import classify_with_sql
    result = classify_with_sql(question="", sql=request.sql)
    return ValidateSQLResponse(
        sql            = request.sql,
        classification = result.classification.value,
        reason         = result.reason,
    )


# =============================================================================
# Run directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )