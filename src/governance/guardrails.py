"""
src/governance/guardrails.py
============================
Step 6 — Output Guardrails

Applied to every agent response BEFORE it reaches the user.

Four guardrails
---------------
1. PII Masking
   Regex-replace email addresses and phone numbers in the response text
   and in any raw result rows. Masking happens at the output layer (not
   the SQL layer) to catch PII that leaks through JOINs.

2. Row Limit
   If raw_results contains more than MAX_RESULT_ROWS rows, truncate and
   attach a warning. Prevents accidental full-table dumps reaching the client.

3. Confidence Surfacing
   If the agent's self-reported confidence < LOW_CONFIDENCE_THRESHOLD,
   prepend a visible uncertainty notice to the answer.

4. Reflection Step (Bonus)
   A lightweight self-evaluation pass that checks:
     a. Does the generated SQL reference the right tables/concepts?
     b. Are numeric values in the result plausible (non-negative amounts)?
     c. Is the answer non-empty and substantive?
   If reflection flags issues, appends ⚠ notes to the answer.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RESULT_ROWS         = 1000
LOW_CONFIDENCE_THRESHOLD = 0.5

# ── PII patterns ──────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)
_PHONE_RE = re.compile(
    r"(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})"
)

# Minimum answer length to be considered substantive
MIN_ANSWER_LENGTH = 30


# =============================================================================
# Guardrail 1: PII Masking
# =============================================================================

def mask_email(email: str) -> str:
    """Mask an email: keep domain, replace local part with 'xx***'."""
    parts = email.split("@")
    if len(parts) != 2:
        return "***@***.***"
    local  = parts[0]
    masked = (local[:2] + "***") if len(local) > 2 else "***"
    return f"{masked}@{parts[1]}"


def mask_pii_in_text(text: str) -> str:
    """Replace emails and phone numbers in a plain-text string."""
    text = _EMAIL_RE.sub(lambda m: mask_email(m.group()), text)
    text = _PHONE_RE.sub("[PHONE REDACTED]", text)
    return text


def mask_pii_in_value(val: Any) -> Any:
    """Recursively mask PII in strings, dicts, and lists."""
    if isinstance(val, str):
        return mask_pii_in_text(val)
    if isinstance(val, dict):
        return {k: mask_pii_in_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [mask_pii_in_value(item) for item in val]
    return val


# =============================================================================
# Guardrail 2: Row Limit
# =============================================================================

def enforce_row_limit(raw_results: Any) -> tuple[Any, Optional[str]]:
    """
    Truncate raw_results to MAX_RESULT_ROWS if necessary.

    Returns
    -------
    (truncated_results, warning_message | None)
    """
    if not isinstance(raw_results, list):
        return raw_results, None

    if len(raw_results) > MAX_RESULT_ROWS:
        truncated = raw_results[:MAX_RESULT_ROWS]
        warning   = (
            f"⚠ Result truncated to {MAX_RESULT_ROWS} rows "
            f"({len(raw_results) - MAX_RESULT_ROWS} additional rows omitted). "
            "Apply filters or use pagination to retrieve the full result."
        )
        return truncated, warning

    return raw_results, None


# =============================================================================
# Guardrail 3: Confidence Surfacing
# =============================================================================

def confidence_warning(confidence: Optional[float]) -> Optional[str]:
    """
    Return a warning string if confidence is below the threshold.

    The agent sets confidence = 0.5 when it answered from memory (no tool
    called) and 0.3 when its answer contains error language.
    """
    if confidence is None:
        return None
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        pct = int(confidence * 100)
        return (
            f"⚠ Low confidence ({pct}%): the agent may not have selected the "
            "right tool for this question. Please verify the answer or "
            "rephrase your question for better routing."
        )
    return None


# =============================================================================
# Guardrail 4: Reflection Step (Bonus)
# =============================================================================

# Keywords that should appear in SQL when they appear in the question
_CONCEPT_MAP = {
    "revenue":     ["amount", "mrr", "total", "transactions", "invoices"],
    "customer":    ["customer", "customers"],
    "churn":       ["status", "cancelled", "churn", "subscriptions"],
    "ticket":      ["support_tickets", "ticket", "tickets"],
    "feature":     ["feature_name", "product_usage", "feature"],
    "subscription":["subscriptions", "subscription"],
    "invoice":     ["invoices", "invoice"],
    "mrr":         ["mrr", "subscriptions"],
    "sla":         ["resolution", "response", "sla"],
}

# Columns that should never be negative
_NON_NEGATIVE_COLS = {"amount", "mrr", "total", "revenue", "paid_amount", "subtotal"}


def reflect(
    question:   str,
    sql:        Optional[str],
    answer:     str,
    raw_results: Any,
) -> dict:
    """
    Self-evaluate the agent's output before returning it to the user.

    Checks
    ------
    1. SQL–question alignment: does the SQL reference tables relevant to the question?
    2. Numeric plausibility: no negative values in financial columns.
    3. Answer completeness: is the answer non-empty and substantive?

    Returns
    -------
    {
        "passed"  : bool,
        "notes"   : list[str],   (visible to user if failed)
        "internal": list[str],   (logged but not shown to user)
    }
    """
    notes:    list[str] = []
    internal: list[str] = []
    passed = True

    q_lower = question.lower()
    s_lower = (sql or "").lower()

    # ── Check 1: SQL–question concept alignment ───────────────────────────────
    if sql:
        for concept, sql_terms in _CONCEPT_MAP.items():
            if concept in q_lower and not any(t in s_lower for t in sql_terms):
                internal.append(
                    f"SQL may not address '{concept}' from the question "
                    f"(expected one of {sql_terms} in SQL)"
                )
                # Only surface this if it seems like a meaningful mismatch
                if concept in ("revenue", "churn", "ticket"):
                    passed = False
                    notes.append(
                        f"⚠ The generated SQL may not fully address '{concept}' — "
                        "please verify the result."
                    )

    # ── Check 2: Numeric plausibility ────────────────────────────────────────
    if isinstance(raw_results, list):
        for row in raw_results[:10]:  # check first 10 rows
            if isinstance(row, dict):
                for col, val in row.items():
                    if col.lower() in _NON_NEGATIVE_COLS and isinstance(val, (int, float)):
                        if val < 0:
                            passed = False
                            notes.append(
                                f"⚠ Negative value detected in '{col}': {val}. "
                                "This may indicate a data issue."
                            )
                            break  # one warning per result set is enough

    # ── Check 3: Answer completeness ─────────────────────────────────────────
    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        passed = False
        notes.append("⚠ The response appears incomplete. Please try rephrasing.")

    if not notes and passed:
        internal.append("✓ Reflection passed all checks")

    return {"passed": passed, "notes": notes, "internal": internal}


# =============================================================================
# Main apply function — run all guardrails in sequence
# =============================================================================

def apply_all(
    answer:      str,
    raw_results: Any,
    confidence:  Optional[float],
    question:    str             = "",
    sql:         Optional[str]   = None,
) -> dict:
    """
    Apply all four guardrails to an agent response.

    Parameters
    ----------
    answer      : the agent's text response
    raw_results : list of dicts from sql_query_tool (may be None)
    confidence  : agent confidence score (0–1)
    question    : original user question (needed for reflection)
    sql         : generated SQL string (needed for reflection)

    Returns
    -------
    {
        "answer"      : str   (possibly modified with warnings)
        "raw_results" : any   (possibly truncated + PII-masked)
        "warnings"    : list[str]
        "reflection"  : dict
    }
    """
    warnings: list[str] = []

    # ── 1. PII masking ────────────────────────────────────────────────────────
    answer      = mask_pii_in_text(answer)
    raw_results = mask_pii_in_value(raw_results)

    # ── 2. Row limit ─────────────────────────────────────────────────────────
    raw_results, row_warning = enforce_row_limit(raw_results)
    if row_warning:
        warnings.append(row_warning)

    # ── 3. Confidence ─────────────────────────────────────────────────────────
    conf_warning = confidence_warning(confidence)
    if conf_warning:
        warnings.append(conf_warning)

    # ── 4. Reflection ────────────────────────────────────────────────────────
    reflection = reflect(question, sql, answer, raw_results)
    if not reflection["passed"] and reflection["notes"]:
        warnings.extend(reflection["notes"])

    # Append warnings to the answer so the user sees them
    if warnings:
        answer += "\n\n" + "\n".join(warnings)

    return {
        "answer":      answer,
        "raw_results": raw_results,
        "warnings":    warnings,
        "reflection":  reflection,
    }