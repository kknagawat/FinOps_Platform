"""
src/api/schemas.py
==================
Pydantic v2 request/response models for all FastAPI endpoints.
Every field has a description for automatic OpenAPI docs generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Shared enums
# =============================================================================

class ClassificationEnum(str, Enum):
    safe             = "safe"
    requires_review  = "requires_review"
    blocked          = "blocked"


class ApprovalAction(str, Enum):
    approve = "approve"
    reject  = "reject"


class JobStatus(str, Enum):
    accepted  = "accepted"
    running   = "running"
    completed = "completed"
    failed    = "failed"


# =============================================================================
# Request models
# =============================================================================

class AgentQueryRequest(BaseModel):
    question:   str           = Field(...,  description="Natural language question", min_length=3)
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    user_id:    Optional[str] = Field(None, description="User identifier for audit trail")

    model_config = {"json_schema_extra": {"example": {
        "question": "What is our current MRR?",
        "session_id": "session-abc123",
    }}}


class ValidateSQLRequest(BaseModel):
    sql: str = Field(..., description="SQL string to classify without executing")

    model_config = {"json_schema_extra": {"example": {
        "sql": "SELECT customer_id, SUM(amount) FROM transactions GROUP BY 1"
    }}}


class ReviewRequest(BaseModel):
    action:         ApprovalAction = Field(...,  description="approve or reject")
    reviewer_notes: Optional[str]  = Field(None, description="Reason for decision")
    reviewer_id:    Optional[str]  = Field(None, description="Reviewer identifier")

    model_config = {"json_schema_extra": {"example": {
        "action": "approve",
        "reviewer_notes": "Verified legitimate business request",
        "reviewer_id": "manager-001",
    }}}


# =============================================================================
# Response models
# =============================================================================

class HealthResponse(BaseModel):
    status:          str            = Field(..., description="ok or degraded")
    db_status:       str            = Field(..., description="connected or error message")
    row_counts:      dict[str, int] = Field(..., description="Row count per table")
    agent_ready:     bool           = Field(..., description="Whether the agent is initialised")
    rag_index_ready: bool           = Field(..., description="Whether the FAISS index is loaded")
    pending_reviews: int            = Field(..., description="Number of queries awaiting review")


class ETLJobResponse(BaseModel):
    job_id:  str = Field(..., description="Job ID for status polling")
    status:  str = Field(..., description="accepted")
    message: str = Field(..., description="Instructions for polling")


class ETLStatusResponse(BaseModel):
    job_id:         str
    status:         JobStatus
    rows_processed: Optional[dict[str, int]] = None
    errors:         Optional[list[str]]      = None
    duration_s:     Optional[float]          = None


class AgentQueryResponse(BaseModel):
    answer:               str
    tools_used:           list[str]
    generated_sql:        Optional[str]              = None
    raw_results:          Optional[Any]              = None
    sources:              Optional[list[dict]]        = None
    chart_path:           Optional[str]              = None
    classification:       ClassificationEnum
    classification_reason: str                       = ""
    warnings:             list[str]                  = Field(default_factory=list)
    execution_time_ms:    float
    confidence:           float
    approval_required:    bool                       = False
    approval_id:          Optional[int]              = None
    audit_id:             Optional[int]              = None


class AuditEntry(BaseModel):
    id:                int
    timestamp:         str
    user_question:     str
    tools_used:        Optional[str]
    generated_sql:     Optional[str]
    classification:    str
    result_summary:    Optional[str]
    execution_time_ms: Optional[float]
    approval_status:   str


class HistoryResponse(BaseModel):
    total:   int
    limit:   int
    offset:  int
    entries: list[AuditEntry]


class DashboardKPIs(BaseModel):
    mrr:                     float
    arr:                     float
    active_customers:        int
    churn_rate_pct:          float
    avg_ticket_resolution_h: float
    nps_score:               Optional[float]
    computed_at:             str


class PendingReview(BaseModel):
    id:             int
    timestamp:      str
    user_question:  str
    generated_sql:  Optional[str]
    classification: str
    reason:         Optional[str]
    status:         str


class ReviewResponse(BaseModel):
    id:      int
    action:  str
    status:  str
    message: str


class ValidateSQLResponse(BaseModel):
    sql:            str
    classification: ClassificationEnum
    reason:         str


class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None
    code:   Optional[str] = None