"""
src/governance/audit.py
=======================
Step 6 — Audit Log

Every agent interaction is recorded here — no exceptions.
The log is append-only: rows are never updated or deleted,
preserving a complete, tamper-evident history.

Schema (in finops.db / audit_log table)
----------------------------------------
id                INTEGER  PK AUTOINCREMENT
timestamp         TEXT     ISO-8601 UTC
user_question     TEXT     Original question from the user
tools_used        TEXT     JSON array of tool names called
generated_sql     TEXT     SQL string (if sql_query_tool was used)
classification    TEXT     safe / requires_review / blocked
result_summary    TEXT     First 300 chars of the agent answer
execution_time_ms REAL     End-to-end latency
approval_status   TEXT     not_required / pending / approved / rejected / blocked
reviewer_notes    TEXT     Notes from human reviewer (approval queue)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "finops.db"
MAX_SUMMARY_CHARS = 300
MAX_HISTORY_ROWS  = 1000


# =============================================================================
# Low-level DB helper
# =============================================================================

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# =============================================================================
# Write: log a single interaction
# =============================================================================

def log_interaction(
    user_question:     str,
    tools_used:        list[str],
    generated_sql:     Optional[str],
    classification:    str,
    result_summary:    str,
    execution_time_ms: float,
    approval_status:   str = "not_required",
    reviewer_notes:    Optional[str] = None,
) -> int:
    """
    Insert one row into audit_log.

    Parameters
    ----------
    user_question     : original NL question
    tools_used        : list of tool names that were called (JSON-serialised)
    generated_sql     : SQL string if sql_query_tool was used, else None
    classification    : safe / requires_review / blocked
    result_summary    : truncated answer text (first MAX_SUMMARY_CHARS chars)
    execution_time_ms : total latency including tool calls
    approval_status   : not_required / pending / approved / rejected / blocked
    reviewer_notes    : optional human reviewer comments

    Returns
    -------
    int : the new row id (useful for linking to approval_queue entries)
    """
    summary   = result_summary[:MAX_SUMMARY_CHARS] if result_summary else ""
    tools_json = json.dumps(tools_used)
    ts         = datetime.now(timezone.utc).isoformat()

    conn   = _conn()
    cursor = conn.execute(
        """
        INSERT INTO audit_log
            (timestamp, user_question, tools_used, generated_sql,
             classification, result_summary, execution_time_ms,
             approval_status, reviewer_notes)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (ts, user_question, tools_json, generated_sql,
         classification, summary, execution_time_ms,
         approval_status, reviewer_notes),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()

    log.debug("Audit log entry #%d  [%s]  %.0f ms  %s",
              row_id, classification, execution_time_ms, user_question[:60])
    return row_id


# =============================================================================
# Read: query history
# =============================================================================

def get_history(
    limit:        int           = 20,
    offset:       int           = 0,
    tool_filter:  Optional[str] = None,
    class_filter: Optional[str] = None,
) -> tuple[int, list[dict]]:
    """
    Retrieve paginated audit history with optional filters.

    Parameters
    ----------
    limit        : rows to return (max MAX_HISTORY_ROWS)
    offset       : pagination offset
    tool_filter  : return only rows where tools_used contains this name
    class_filter : return only rows with this classification value

    Returns
    -------
    (total_count, list_of_row_dicts)
    """
    limit = min(limit, MAX_HISTORY_ROWS)

    where_parts: list[str] = []
    params:      list[Any] = []

    if tool_filter:
        where_parts.append("tools_used LIKE ?")
        params.append(f"%{tool_filter}%")
    if class_filter:
        where_parts.append("classification = ?")
        params.append(class_filter)

    where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    conn  = _conn()
    total = conn.execute(
        f"SELECT COUNT(*) FROM audit_log {where}", params
    ).fetchone()[0]

    rows  = conn.execute(
        f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()

    return total, [dict(r) for r in rows]


def get_stats() -> dict:
    """Return aggregate statistics from the audit log."""
    conn  = _conn()
    total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    by_class = {}
    for row in conn.execute(
        "SELECT classification, COUNT(*) AS n FROM audit_log GROUP BY classification"
    ).fetchall():
        by_class[row["classification"]] = row["n"]

    avg_ms = conn.execute(
        "SELECT AVG(execution_time_ms) FROM audit_log"
    ).fetchone()[0] or 0.0

    top_tools_raw = conn.execute(
        "SELECT tools_used FROM audit_log WHERE tools_used != '[]' LIMIT 200"
    ).fetchall()
    conn.close()

    tool_counts: dict[str, int] = {}
    for row in top_tools_raw:
        try:
            tools = json.loads(row["tools_used"])
            for t in tools:
                tool_counts[t] = tool_counts.get(t, 0) + 1
        except Exception:
            pass

    top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_interactions": total,
        "by_classification":  by_class,
        "avg_latency_ms":     round(avg_ms, 1),
        "top_tools":          [{"tool": t, "uses": n} for t, n in top_tools],
    }