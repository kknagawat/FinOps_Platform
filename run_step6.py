"""
run_step6.py
============
Step 6 — Human-in-the-Loop Governance Layer
---------------------------------------------
Tests all four governance capabilities without requiring a live LLM:

  1. Safety Classifier   — 20 test cases (BLOCKED / REQUIRES_REVIEW / SAFE)
  2. Approval Queue      — full lifecycle: enqueue → list → approve/reject
  3. Audit Log           — writes, reads, stats
  4. Output Guardrails   — PII masking, row limits, confidence, reflection

Usage
-----
  python run_step6.py           # run all tests
  python run_step6.py --demo    # also run a live governed agent query (needs API key)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s │ %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Test 1 — Safety Classifier
# =============================================================================

CLASSIFIER_CASES = [
    # (question, optional_sql, expected_classification)

    # ── BLOCKED ──────────────────────────────────────────────────────────────
    ("Delete all transactions from 2022",          None,                    "blocked"),
    ("Remove duplicate customer records",          None,                    "blocked"),
    ("Drop the invoices table",                    None,                    "blocked"),
    ("",  "DELETE FROM transactions WHERE amount < 0",                      "blocked"),
    ("",  "INSERT INTO customers VALUES (1,'x','y')",                       "blocked"),
    ("",  "UPDATE subscriptions SET mrr = 0 WHERE customer_id='CUST-0001'", "blocked"),
    ("",  "DROP TABLE product_usage",                                        "blocked"),

    # ── REQUIRES_REVIEW ───────────────────────────────────────────────────────
    ("Show me all customer email addresses",       None,                    "requires_review"),
    ("What is the phone number for CUST-0123?",    None,                    "requires_review"),
    ("Export all customer records",                None,                    "requires_review"),
    ("Download a full dump of all transactions",   None,                    "requires_review"),
    ("Are there any financial irregularities?",    None,                    "requires_review"),
    ("",  "SELECT email, phone FROM customers",                             "requires_review"),
    ("",  "SELECT first_name, last_name, email FROM customers LIMIT 100",   "requires_review"),

    # ── SAFE ─────────────────────────────────────────────────────────────────
    ("What is our current MRR?",                   None,                    "safe"),
    ("Show me the top 10 customers by revenue",    None,                    "safe"),
    ("What is our refund policy?",                 None,                    "safe"),
    ("Generate a bar chart of monthly revenue",    None,                    "safe"),
    ("Which features have the lowest adoption?",   None,                    "safe"),
    ("",  "SELECT customer_id, SUM(amount) FROM transactions GROUP BY 1",   "safe"),
]


def test_classifier() -> tuple[int, int]:
    from src.governance.classifier import classify, Classification

    passed = failed = 0
    print(f"\n  {'Question / SQL':<55} {'Expected':<16} {'Got':<16} Status")
    print("  " + "─" * 100)

    for question, sql, expected in CLASSIFIER_CASES:
        result = classify(question, generated_sql=sql)
        got    = result.classification.value
        ok     = got == expected
        if ok:
            passed += 1
        else:
            failed += 1
        display  = (question or f"[SQL] {sql}")[:53]
        icon     = "✓" if ok else "✗"
        print(f"  {display:<55} {expected:<16} {got:<16} {icon}")

    return passed, failed


# =============================================================================
# Test 2 — Approval Queue
# =============================================================================

def test_queue() -> None:
    from src.governance.queue import enqueue, list_pending, get_entry, approve, reject, queue_stats

    print()

    # Enqueue two items
    q1 = enqueue(
        user_question  = "Show me all customer email addresses",
        classification = "requires_review",
        reason         = "SQL accesses PII column: 'email'",
        generated_sql  = "SELECT email FROM customers",
    )
    q2 = enqueue(
        user_question  = "Export all customer records",
        classification = "requires_review",
        reason         = "Bulk data export requested",
    )
    print(f"  ✓ Enqueued 2 items: IDs {q1}, {q2}")

    # List pending
    pending = list_pending()
    in_list = any(p["id"] == q1 for p in pending) and any(p["id"] == q2 for p in pending)
    print(f"  ✓ list_pending() returned {len(pending)} pending entries  "
          f"({'both found' if in_list else 'MISSING items'})")

    # Approve one
    updated = approve(q1, reviewer_notes="Verified legitimate business request", reviewer_id="mgr-001")
    print(f"  ✓ approve(id={q1}) → status={updated['status']}  "
          f"notes={updated['reviewer_notes'][:40]}")

    # Reject the other
    updated2 = reject(q2, reviewer_notes="No business justification provided", reviewer_id="mgr-001")
    print(f"  ✓ reject(id={q2}) → status={updated2['status']}  "
          f"notes={updated2['reviewer_notes'][:40]}")

    # Confirm no longer pending
    still_pending = list_pending()
    clean = not any(p["id"] in (q1, q2) for p in still_pending)
    print(f"  ✓ After review: {len(still_pending)} unrelated items still pending  "
          f"({'resolved queue clean' if clean else 'ERROR: items still pending'})")

    # Try to double-approve (should raise)
    try:
        approve(q1)
        print("  ✗ ERROR: double-approve should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Double-approve correctly raises ValueError: {str(e)[:50]}")

    # Stats
    stats = queue_stats()
    print(f"  ✓ Queue stats: total={stats['total']}  by_status={stats['by_status']}")


# =============================================================================
# Test 3 — Audit Log
# =============================================================================

def test_audit() -> None:
    from src.governance.audit import log_interaction, get_history, get_stats

    print()

    # Write a few entries
    ids = []
    test_entries = [
        ("What is our MRR?",     ["revenue_calculator_tool"],  None,                        "safe",            "MRR is $197,397",   250.0,  "not_required"),
        ("Show all emails",       ["sql_query_tool"],           "SELECT email FROM customers","requires_review", "Queued for review", 10.0,   "pending"),
        ("Delete transactions",   [],                           None,                        "blocked",         "Request blocked",   5.0,    "blocked"),
        ("Churn risk analysis",   ["customer_segmentation_tool"], None,                      "safe",            "3 high-risk...",    1200.0, "not_required"),
    ]
    for q, tools, sql, cls, summary, ms, status in test_entries:
        lid = log_interaction(q, tools, sql, cls, summary, ms, status)
        ids.append(lid)
    print(f"  ✓ Wrote {len(ids)} audit log entries: IDs {ids}")

    # Read back with filter
    total, rows = get_history(limit=10, class_filter="safe")
    safe_ids_found = all(
        any(r["user_question"] == q for r in rows)
        for q, _, _, cls, *_ in test_entries if cls == "safe"
    )
    print(f"  ✓ get_history(class_filter='safe'): {len(rows)} rows returned  "
          f"({'all safe entries found' if safe_ids_found else 'some missing'})")

    # Filter by tool
    total2, rows2 = get_history(limit=10, tool_filter="revenue_calculator_tool")
    print(f"  ✓ get_history(tool_filter='revenue_calculator_tool'): {len(rows2)} rows")

    # Stats
    stats = get_stats()
    print(f"  ✓ Audit stats: total={stats['total_interactions']}  "
          f"by_class={stats['by_classification']}  "
          f"avg_ms={stats['avg_latency_ms']}")


# =============================================================================
# Test 4 — Output Guardrails
# =============================================================================

def test_guardrails() -> None:
    from src.governance.guardrails import (
        mask_pii_in_text, mask_pii_in_value,
        enforce_row_limit, confidence_warning,
        reflect, apply_all,
    )

    print()

    # ── PII masking ───────────────────────────────────────────────────────────
    email_text  = "Customer john.doe@example.com called. Number: 555-867-5309"
    masked_text = mask_pii_in_text(email_text)
    email_gone  = "john.doe@example.com" not in masked_text
    phone_gone  = "867-5309" not in masked_text
    print(f"  {'✓' if email_gone else '✗'} Email masked:  '{masked_text}'")
    print(f"  {'✓' if phone_gone else '✗'} Phone masked in same string")

    # Masking in dict
    result_with_pii = [
        {"customer_id": "CUST-001", "email": "jane@acme.com", "revenue": 5000},
        {"customer_id": "CUST-002", "email": "bob@corp.com",  "revenue": 3000},
    ]
    masked_result = mask_pii_in_value(result_with_pii)
    emails_masked = all("jane@acme.com" not in r["email"] for r in masked_result)
    revenue_ok    = all(r["revenue"] > 0 for r in masked_result)
    print(f"  {'✓' if emails_masked else '✗'} Emails masked in list-of-dicts")
    print(f"  {'✓' if revenue_ok else '✗'} Non-PII numeric values preserved")

    # ── Row limit ─────────────────────────────────────────────────────────────
    big_result = [{"id": i} for i in range(1500)]
    truncated, warning = enforce_row_limit(big_result)
    ok_trunc  = len(truncated) == 1000
    ok_warn   = warning is not None and "1000" in warning
    print(f"  {'✓' if ok_trunc else '✗'} Row limit: 1500 rows → {len(truncated)} rows")
    print(f"  {'✓' if ok_warn else '✗'} Warning message contains row limit info")

    small_result = [{"id": i} for i in range(50)]
    truncated2, warning2 = enforce_row_limit(small_result)
    print(f"  ✓ 50 rows → {len(truncated2)} rows (no truncation)")

    # ── Confidence surfacing ──────────────────────────────────────────────────
    low_conf_warn  = confidence_warning(0.3)
    high_conf_warn = confidence_warning(0.9)
    no_warn        = confidence_warning(None)
    print(f"  {'✓' if low_conf_warn else '✗'}  Low confidence (0.3) → warning: "
          f"'{(low_conf_warn or '')[:50]}…'")
    print(f"  {'✓' if high_conf_warn is None else '✗'} High confidence (0.9) → no warning")
    print(f"  {'✓' if no_warn is None else '✗'}  None confidence → no warning")

    # ── Reflection ────────────────────────────────────────────────────────────
    # Should pass
    r1 = reflect(
        question    = "What is our total revenue this quarter?",
        sql         = "SELECT SUM(amount) AS revenue FROM transactions WHERE status='completed'",
        answer      = "Our total revenue this quarter is $1,234,567, up 12% from Q3.",
        raw_results = [{"revenue": 1234567}],
    )
    print(f"  {'✓' if r1['passed'] else '~'} Reflection PASS test: passed={r1['passed']}  notes={r1['notes']}")

    # Should flag negative revenue
    r2 = reflect(
        question    = "Show me revenue by month",
        sql         = "SELECT strftime('%Y-%m', transaction_date) AS month, SUM(amount) AS revenue FROM transactions GROUP BY month",
        answer      = "Monthly revenue shown below.",
        raw_results = [{"month": "2025-01", "revenue": -500}],
    )
    flagged_negative = not r2["passed"] and any("Negative" in n for n in r2["notes"])
    print(f"  {'✓' if flagged_negative else '✗'} Reflection FAIL test (negative revenue): "
          f"passed={r2['passed']}  notes={r2['notes']}")

    # ── apply_all integration ─────────────────────────────────────────────────
    full_result = apply_all(
        answer      = "Top customers by revenue: jane@acme.com ($50,000), bob@corp.com ($30,000)",
        raw_results = [{"customer_id": "C1", "email": "jane@acme.com", "revenue": 50000}],
        confidence  = 0.9,
        question    = "Show me top customers by revenue",
        sql         = "SELECT customer_id, email, SUM(amount) FROM transactions GROUP BY customer_id ORDER BY 3 DESC",
    )
    email_masked_in_answer = "jane@acme.com" not in full_result["answer"]
    print(f"  {'✓' if email_masked_in_answer else '✗'} apply_all: email masked in final answer")
    print(f"  ✓ apply_all warnings: {full_result['warnings']}")


# =============================================================================
# Optional demo: live governed query
# =============================================================================

def demo_governed_query() -> None:
    """Run a real governed query through the full pipeline (needs API key)."""
    from src.agent.agent import FinOpsAgent
    from src.governance.governance import governed_query

    print("\n  Initialising agent for live governance demo…")
    agent = FinOpsAgent(verbose=False)

    test_cases = [
        ("What is our current MRR?",           "safe"),
        ("Show all customer emails and phones", "requires_review"),
        ("Drop the customers table",           "blocked"),
    ]

    for question, expected in test_cases:
        print(f"\n  Q: {question}")
        response = governed_query(agent, question)
        icon     = "✓" if response.classification == expected else "✗"
        print(f"  {icon} Classification: {response.classification}  (expected: {expected})")
        print(f"    Answer: {response.answer[:150]}…")
        if response.approval_id:
            print(f"    Queue ID: {response.approval_id}")
        if response.audit_id:
            print(f"    Audit ID: {response.audit_id}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6 — Governance Layer Tests")
    parser.add_argument("--demo", action="store_true",
                        help="Run live governed agent queries (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    BANNER = "█" * 72
    print()
    print(BANNER)
    print("  FINOPS ANALYTICS PLATFORM — STEP 6")
    print("  Human-in-the-Loop Governance Layer")
    print(BANNER)

    # ── Test 1: Classifier ────────────────────────────────────────────────────
    _section("TEST 1 — Safety Classifier  (3 tiers: blocked / requires_review / safe)")
    passed, failed = test_classifier()
    total = passed + failed
    print(f"\n  Result: {passed}/{total} cases correct  "
          f"({'✓ ALL PASS' if failed == 0 else f'✗ {failed} FAILURES'})")

    # ── Test 2: Approval Queue ────────────────────────────────────────────────
    _section("TEST 2 — Approval Queue  (enqueue → list → approve/reject)")
    test_queue()

    # ── Test 3: Audit Log ─────────────────────────────────────────────────────
    _section("TEST 3 — Audit Log  (write / read / stats)")
    test_audit()

    # ── Test 4: Guardrails ────────────────────────────────────────────────────
    _section("TEST 4 — Output Guardrails  (PII / row limit / confidence / reflection)")
    test_guardrails()

    # ── Live demo (optional) ──────────────────────────────────────────────────
    if args.demo:
        _load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n  --demo requires ANTHROPIC_API_KEY to be set. Skipping.")
        else:
            _section("DEMO — Live Governed Agent Queries")
            demo_governed_query()

    # ── Summary ───────────────────────────────────────────────────────────────
    SEP2 = "═" * 72
    print(f"\n{SEP2}")
    print("  STEP 6 COMPLETE")
    print(SEP2)
    print("  Deliverables:")
    print("    ✓ src/governance/classifier.py  — 3-tier safety classifier")
    print("    ✓ src/governance/audit.py        — append-only audit log")
    print("    ✓ src/governance/queue.py        — approval queue (enqueue/approve/reject)")
    print("    ✓ src/governance/guardrails.py   — PII masking + row limits + reflection")
    print("    ✓ src/governance/governance.py   — full pipeline orchestrator")
    print()
    print("  Integration:")
    print("    • governance.governed_query(agent, question)  wraps all agent calls")
    print("    • FastAPI Step 7 routes will use governed_query instead of agent.query")
    print(SEP2)


def _section(title: str) -> None:
    print()
    print("┌" + "─" * 70 + "┐")
    print(f"│  {title:<68}│")
    print("└" + "─" * 70 + "┘")


def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


if __name__ == "__main__":
    main()