"""
src/governance/queue.py
=======================
Step 6 — Approval Queue

Queries classified as REQUIRES_REVIEW are held here until a human
approves or rejects them.

Queue lifecycle
---------------
  1. classify() → REQUIRES_REVIEW
  2. enqueue()  → status = 'pending'   (agent pauses, returns queue ID to user)
  3. Human calls approve() or reject() via API
  4. approve()  → status = 'approved'  (result may be executed)
     reject()   → status = 'rejected'  (request discarded with notes)

Schema (approval_queue in finops.db)
--------------------------------------
id              INTEGER  PK AUTOINCREMENT
timestamp       TEXT     when the request arrived
user_question   TEXT     original question
generated_sql   TEXT     SQL if already generated
classification  TEXT     always 'requires_review'
reason          TEXT     which rule triggered the review
status          TEXT     pending / approved / rejected
reviewer_notes  TEXT     set by human reviewer
reviewed_at     TEXT     ISO timestamp of review action
result_cache    TEXT     serialised result (for post-approval execution)
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


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# =============================================================================
# Enqueue
# =============================================================================

def enqueue(
    user_question:  str,
    classification: str,
    reason:         str,
    generated_sql:  Optional[str] = None,
    result_cache:   Optional[Any] = None,
) -> int:
    """
    Add a request to the approval queue with status = 'pending'.

    Parameters
    ----------
    user_question  : original NL question
    classification : typically 'requires_review'
    reason         : human-readable explanation of why review is needed
    generated_sql  : SQL that would run if approved
    result_cache   : pre-computed result to return on approval (optional)

    Returns
    -------
    int : the new queue entry id (returned to the user so they can track it)
    """
    ts    = datetime.now(timezone.utc).isoformat()
    cache = json.dumps(result_cache) if result_cache is not None else None

    conn   = _conn()
    cursor = conn.execute(
        """
        INSERT INTO approval_queue
            (timestamp, user_question, generated_sql, classification,
             reason, status, result_cache)
        VALUES (?,?,?,?,?,?,?)
        """,
        (ts, user_question, generated_sql, classification, reason, "pending", cache),
    )
    conn.commit()
    qid = cursor.lastrowid
    conn.close()

    log.info("Queued for review  id=%d  reason=%s", qid, reason)
    return qid


# =============================================================================
# List pending
# =============================================================================

def list_pending() -> list[dict]:
    """Return all entries with status = 'pending', newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM approval_queue WHERE status = 'pending' ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all(limit: int = 50) -> list[dict]:
    """Return all queue entries (pending + resolved), newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM approval_queue ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_entry(queue_id: int) -> Optional[dict]:
    """Fetch a single queue entry by id."""
    conn  = _conn()
    row   = conn.execute(
        "SELECT * FROM approval_queue WHERE id = ?", (queue_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# =============================================================================
# Review actions
# =============================================================================

def approve(
    queue_id:       int,
    reviewer_notes: Optional[str] = None,
    reviewer_id:    Optional[str] = None,
) -> dict:
    """
    Approve a pending queue entry.

    Sets status → 'approved' and stamps reviewed_at.
    The caller (governance.py) is responsible for executing the query
    if the entry has a result_cache or generated_sql.

    Returns the updated row as a dict.
    Raises ValueError if the entry doesn't exist or isn't pending.
    """
    return _update_status(queue_id, "approved", reviewer_notes, reviewer_id)


def reject(
    queue_id:       int,
    reviewer_notes: Optional[str] = None,
    reviewer_id:    Optional[str] = None,
) -> dict:
    """
    Reject a pending queue entry.

    Sets status → 'rejected'. The original request will not execute.
    reviewer_notes should explain why the request was denied.
    """
    return _update_status(queue_id, "rejected", reviewer_notes, reviewer_id)


def _update_status(
    queue_id:       int,
    new_status:     str,
    reviewer_notes: Optional[str],
    reviewer_id:    Optional[str],
) -> dict:
    conn = _conn()
    row  = conn.execute(
        "SELECT * FROM approval_queue WHERE id = ?", (queue_id,)
    ).fetchone()

    if not row:
        conn.close()
        raise ValueError(f"Queue entry {queue_id} not found")

    if row["status"] != "pending":
        conn.close()
        raise ValueError(
            f"Queue entry {queue_id} is already '{row['status']}' — cannot change"
        )

    notes = reviewer_notes
    if reviewer_id and reviewer_notes:
        notes = f"[{reviewer_id}] {reviewer_notes}"
    elif reviewer_id:
        notes = f"[{reviewer_id}]"

    conn.execute(
        """
        UPDATE approval_queue
        SET status = ?, reviewer_notes = ?, reviewed_at = ?
        WHERE id = ?
        """,
        (new_status, notes, datetime.now(timezone.utc).isoformat(), queue_id),
    )
    conn.commit()
    updated = dict(conn.execute(
        "SELECT * FROM approval_queue WHERE id = ?", (queue_id,)
    ).fetchone())
    conn.close()

    log.info("Queue entry %d → %s  notes=%s", queue_id, new_status, notes)
    return updated


# =============================================================================
# Stats
# =============================================================================

def queue_stats() -> dict:
    conn  = _conn()
    total = conn.execute("SELECT COUNT(*) FROM approval_queue").fetchone()[0]
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM approval_queue GROUP BY status"
    ).fetchall():
        by_status[row["status"]] = row["n"]
    conn.close()
    return {"total": total, "by_status": by_status}