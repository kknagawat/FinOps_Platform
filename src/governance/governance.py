"""
src/governance/governance.py
============================
Step 6 — Governance Layer Orchestrator

This module is the single entry point for governed agent execution.
It wraps every agent call with the full governance pipeline:

    ┌─────────────────────────────────────────────────────────┐
    │  User question                                          │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  1. classify_question()   (pre-execution)               │
    │     BLOCKED       → return immediately, log, done       │
    │     REQUIRES_REVIEW → enqueue, return queue ID, done    │
    │     SAFE          → continue                            │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  2. agent.query(question)                               │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  3. classify_with_sql()   (post-SQL check)              │
    │     BLOCKED → discard result, return error              │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  4. apply_all() guardrails                              │
    │     - PII masking                                       │
    │     - Row limit                                         │
    │     - Confidence surfacing                              │
    │     - Reflection check                                  │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  5. log_interaction()  (audit log, always)              │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Return GovernedResponse to caller                      │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.governance.audit      import log_interaction
from src.governance.classifier import Classification, classify_question, classify_with_sql
from src.governance.guardrails import apply_all
from src.governance.queue      import enqueue

log = logging.getLogger(__name__)


# =============================================================================
# Response dataclass
# =============================================================================

@dataclass
class GovernedResponse:
    """
    The complete governed response returned to the caller (FastAPI, CLI, etc.)

    Fields
    ------
    answer            : final text shown to the user
    tools_used        : list of tool names invoked
    generated_sql     : SQL string (if any)
    raw_results       : data rows (PII-masked, row-limited)
    sources           : RAG source citations
    chart_path        : path to generated PNG (if any)
    classification    : safe / requires_review / blocked
    classification_reason : why this classification was assigned
    warnings          : list of guardrail / reflection warnings
    execution_time_ms : total latency
    confidence        : agent confidence score
    approval_required : True when queued for human review
    approval_id       : queue entry id (set when approval_required=True)
    audit_id          : audit log row id
    """
    answer:               str
    tools_used:           list[str]        = field(default_factory=list)
    generated_sql:        Optional[str]    = None
    raw_results:          Any              = None
    sources:              Any              = None
    chart_path:           Optional[str]    = None
    classification:       str             = "safe"
    classification_reason: str            = ""
    warnings:             list[str]       = field(default_factory=list)
    execution_time_ms:    float           = 0.0
    confidence:           float           = 1.0
    approval_required:    bool            = False
    approval_id:          Optional[int]   = None
    audit_id:             Optional[int]   = None


# =============================================================================
# Main governed query function
# =============================================================================

def governed_query(agent, question: str, session_id: Optional[str] = None) -> GovernedResponse:
    """
    Run a question through the full governance pipeline.

    Parameters
    ----------
    agent      : FinOpsAgent instance
    question   : user's natural language question
    session_id : optional conversation session ID

    Returns
    -------
    GovernedResponse — always returned, even for blocked/queued requests.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Pre-execution classification
    # ──────────────────────────────────────────────────────────────────────────
    pre_result = classify_question(question)
    log.info("Pre-classification: %s  [%s]",
             pre_result.classification.value, pre_result.rule_triggered)

    if pre_result.is_blocked:
        audit_id = log_interaction(
            user_question     = question,
            tools_used        = [],
            generated_sql     = None,
            classification    = Classification.BLOCKED.value,
            result_summary    = f"Blocked: {pre_result.reason}",
            execution_time_ms = 0.0,
            approval_status   = "blocked",
        )
        return GovernedResponse(
            answer               = (
                f"🚫 Your request has been blocked.\n\n"
                f"Reason: {pre_result.reason}\n\n"
                "This platform only supports read-only analytics operations. "
                "Data modification requests are never permitted."
            ),
            classification       = Classification.BLOCKED.value,
            classification_reason = pre_result.reason,
            audit_id             = audit_id,
        )

    if pre_result.requires_review:
        queue_id = enqueue(
            user_question  = question,
            classification = Classification.REQUIRES_REVIEW.value,
            reason         = pre_result.reason,
        )
        audit_id = log_interaction(
            user_question     = question,
            tools_used        = [],
            generated_sql     = None,
            classification    = Classification.REQUIRES_REVIEW.value,
            result_summary    = f"Queued for review: {pre_result.reason}",
            execution_time_ms = 0.0,
            approval_status   = "pending",
        )
        return GovernedResponse(
            answer = (
                f"⏳ Your request requires human review before it can be executed.\n\n"
                f"**Reason:** {pre_result.reason}\n\n"
                f"**Queue ID:** {queue_id}\n\n"
                "A reviewer has been notified. You can check the status via "
                f"`GET /governance/pending` or reference queue ID #{queue_id}."
            ),
            classification        = Classification.REQUIRES_REVIEW.value,
            classification_reason = pre_result.reason,
            approval_required     = True,
            approval_id           = queue_id,
            audit_id              = audit_id,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Execute the agent
    # ──────────────────────────────────────────────────────────────────────────
    raw_agent_result = agent.query(question, session_id=session_id)

    answer      = raw_agent_result.get("answer", "")
    tools_used  = raw_agent_result.get("tools_used", [])
    sql         = raw_agent_result.get("generated_sql")
    raw_results = raw_agent_result.get("raw_results")
    sources     = raw_agent_result.get("sources")
    chart_path  = raw_agent_result.get("chart_path")
    elapsed_ms  = raw_agent_result.get("execution_time_ms", 0.0)
    confidence  = raw_agent_result.get("confidence", 1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Post-SQL classification
    # ──────────────────────────────────────────────────────────────────────────
    final_class = pre_result
    if sql:
        post_result = classify_with_sql(question, sql)
        if post_result.classification.value != Classification.SAFE.value:
            final_class = post_result
            log.warning("Post-SQL classification upgrade: %s  [%s]",
                        post_result.classification.value, post_result.rule_triggered)

        if post_result.is_blocked:
            audit_id = log_interaction(
                user_question     = question,
                tools_used        = tools_used,
                generated_sql     = sql,
                classification    = Classification.BLOCKED.value,
                result_summary    = f"Post-SQL block: {post_result.reason}",
                execution_time_ms = elapsed_ms,
                approval_status   = "blocked",
            )
            return GovernedResponse(
                answer = (
                    f"🚫 The generated SQL was blocked after review.\n\n"
                    f"Reason: {post_result.reason}\n\n"
                    "The query was not executed. Please rephrase your question."
                ),
                tools_used            = tools_used,
                generated_sql         = sql,
                classification        = Classification.BLOCKED.value,
                classification_reason = post_result.reason,
                execution_time_ms     = elapsed_ms,
                audit_id              = audit_id,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Apply output guardrails
    # ──────────────────────────────────────────────────────────────────────────
    guarded = apply_all(
        answer      = answer,
        raw_results = raw_results,
        confidence  = confidence,
        question    = question,
        sql         = sql,
    )

    final_answer      = guarded["answer"]
    final_raw         = guarded["raw_results"]
    warnings          = guarded["warnings"]
    reflection        = guarded["reflection"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Audit log (always, regardless of outcome)
    # ──────────────────────────────────────────────────────────────────────────
    audit_id = log_interaction(
        user_question     = question,
        tools_used        = tools_used,
        generated_sql     = sql,
        classification    = final_class.classification.value,
        result_summary    = final_answer[:300],
        execution_time_ms = elapsed_ms,
        approval_status   = "not_required",
        reviewer_notes    = ("; ".join(reflection["internal"])
                             if reflection.get("internal") else None),
    )

    log.info("Governed query complete  [%s]  %.0f ms  tools=%s  audit_id=%d",
             final_class.classification.value, elapsed_ms, tools_used, audit_id)

    return GovernedResponse(
        answer                = final_answer,
        tools_used            = tools_used,
        generated_sql         = sql,
        raw_results           = final_raw,
        sources               = sources,
        chart_path            = chart_path,
        classification        = final_class.classification.value,
        classification_reason = final_class.reason,
        warnings              = warnings,
        execution_time_ms     = elapsed_ms,
        confidence            = confidence,
        audit_id              = audit_id,
    )