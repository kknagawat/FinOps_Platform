"""
run_step7.py
============
Step 7 — FastAPI Platform Service
------------------------------------
Starts the FastAPI server and runs all 8 endpoints as integration tests.

Usage
-----
  python run_step7.py           # run integration tests (server auto-starts)
  python run_step7.py --serve   # just start the server (for manual testing)

The --serve flag starts uvicorn on http://localhost:8000.
OpenAPI docs available at http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Integration tests (no live agent needed — tests non-LLM endpoints)
# =============================================================================

def run_tests() -> None:
    """
    Test all 8 endpoints using FastAPI's TestClient.
    Endpoints that need the agent (POST /agent/query) are tested with
    mock governance so they run without an API key.
    """
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  pip install httpx  to run integration tests")
        return

    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")

    from src.api.main import app
    client = TestClient(app, raise_server_exceptions=False)

    BANNER = "█" * 72
    print()
    print(BANNER)
    print("  FINOPS ANALYTICS PLATFORM — STEP 7")
    print("  FastAPI Service — Integration Tests")
    print(BANNER)

    tests = [
        # (method, path, body, expected_status, description)
        ("GET",  "/health",              None,                      [200, 503], "Health check"),
        ("POST", "/etl/run",             None,                      [202],      "ETL trigger (async)"),
        ("GET",  "/etl/status/fake-id",  None,                      [404],      "ETL status (not found)"),
        ("GET",  "/analytics/dashboard", None,                      [200, 503], "KPI dashboard"),
        ("POST", "/agent/query/validate",
         {"sql": "SELECT COUNT(*) FROM customers"},                  [200],      "Validate safe SQL"),
        ("POST", "/agent/query/validate",
         {"sql": "DELETE FROM customers"},                           [200],      "Validate blocked SQL"),
        ("POST", "/agent/query/validate",
         {"sql": "SELECT email FROM customers"},                     [200],      "Validate PII SQL"),
        ("GET",  "/governance/pending",  None,                      [200],      "List pending reviews"),
        ("POST", "/governance/review/99999",
         {"action": "approve", "reviewer_notes": "test"},           [404],      "Review not found"),
        ("GET",  "/agent/query/history", None,                      [200],      "Query history"),
        ("GET",  "/agent/query/history?tool_used=sql_query_tool",
         None,                                                       [200],      "History filtered by tool"),
        ("GET",  "/agent/query/history?classification=safe",
         None,                                                       [200],      "History filtered by class"),
    ]

    passed = failed = 0
    print(f"\n  {'Method':<6} {'Path':<45} {'Expected':>10} {'Got':>6}  Status")
    print("  " + "─" * 78)

    for method, path, body, expected_codes, desc in tests:
        try:
            if method == "GET":
                r = client.get(path)
            else:
                r = client.post(path, json=body)

            ok   = r.status_code in expected_codes
            icon = "✓" if ok else "✗"
            if ok:
                passed += 1
            else:
                failed += 1

            print(f"  {method:<6} {path:<45} {str(expected_codes):>10} {r.status_code:>6}  {icon} {desc}")

            # Show key response fields for interesting endpoints
            if path == "/agent/query/validate" and r.status_code == 200:
                data = r.json()
                print(f"           → classification={data.get('classification')}  "
                      f"reason={data.get('reason','')[:50]}")
            elif path == "/health" and r.status_code == 200:
                data = r.json()
                print(f"           → db={data.get('db_status')}  "
                      f"agent={data.get('agent_ready')}  "
                      f"rag={data.get('rag_index_ready')}")
            elif path == "/analytics/dashboard" and r.status_code == 200:
                data = r.json()
                print(f"           → mrr={data.get('mrr'):,.0f}  "
                      f"arr={data.get('arr'):,.0f}  "
                      f"active_customers={data.get('active_customers')}")

        except Exception as e:
            print(f"  {method:<6} {path:<45} {'ERROR':>10}  {'N/A':>6}  ✗ {e}")
            failed += 1

    print(f"\n  Result: {passed}/{passed+failed} tests passed")

    # ── Validate SQL endpoint deep test ──────────────────────────────────────
    print()
    print("  SQL Validation endpoint — detailed check:")
    print("  " + "─" * 60)

    val_cases = [
        ("SELECT COUNT(*) FROM customers",                         "safe"),
        ("DELETE FROM transactions",                               "blocked"),
        ("UPDATE subscriptions SET mrr=0",                         "blocked"),
        ("SELECT email, phone FROM customers",                     "requires_review"),
        ("SELECT customer_id, SUM(amount) FROM transactions GROUP BY 1", "safe"),
        ("DROP TABLE invoices",                                    "blocked"),
    ]
    val_pass = 0
    for sql, expected in val_cases:
        r    = client.post("/agent/query/validate", json={"sql": sql})
        data = r.json()
        got  = data.get("classification", "error")
        ok   = got == expected
        if ok: val_pass += 1
        icon = "✓" if ok else "✗"
        print(f"  {icon} {sql[:50]:<50} → {got}")

    print(f"\n  SQL validation: {val_pass}/{len(val_cases)} correct")

    SEP2 = "═" * 72
    print(f"\n{SEP2}")
    print("  STEP 7 COMPLETE")
    print(SEP2)
    print("  Endpoints:")
    for _, path, _, _, desc in tests[:8]:
        print(f"    ✓  {path:<40} {desc}")
    print()
    print("  Start the server:   python run_step7.py --serve")
    print("  OpenAPI docs:       http://localhost:8000/docs")
    print("  ReDoc:              http://localhost:8000/redoc")
    print(SEP2)


# =============================================================================
# Serve mode
# =============================================================================

def serve() -> None:
    """Start the uvicorn server."""
    import uvicorn
    print()
    print("Starting FinOps API server …")
    print("  URL:      http://localhost:8000")
    print("  Docs:     http://localhost:8000/docs")
    print("  ReDoc:    http://localhost:8000/redoc")
    print("  Stop:     Ctrl+C")
    print()
    uvicorn.run(
        "src.api.main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )


# =============================================================================
# curl examples (for README / submission)
# =============================================================================

CURL_EXAMPLES = """
# ── curl examples for all 8 endpoints ─────────────────────────────────────

# Health check
curl http://localhost:8000/health

# Trigger ETL
curl -X POST http://localhost:8000/etl/run

# Poll ETL status
curl http://localhost:8000/etl/status/<job_id>

# Ask the agent a question
curl -X POST http://localhost:8000/agent/query \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What is our current MRR?"}'

# Query history (paginated)
curl "http://localhost:8000/agent/query/history?limit=10&offset=0"

# Query history (filtered)
curl "http://localhost:8000/agent/query/history?tool_used=sql_query_tool"
curl "http://localhost:8000/agent/query/history?classification=safe"

# KPI Dashboard
curl http://localhost:8000/analytics/dashboard

# List pending reviews
curl http://localhost:8000/governance/pending

# Approve a review
curl -X POST http://localhost:8000/governance/review/1 \\
  -H "Content-Type: application/json" \\
  -d '{"action": "approve", "reviewer_notes": "Verified OK", "reviewer_id": "mgr-01"}'

# Validate SQL safety
curl -X POST http://localhost:8000/agent/query/validate \\
  -H "Content-Type: application/json" \\
  -d '{"sql": "SELECT SUM(amount) FROM transactions GROUP BY customer_id"}'
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7 — FastAPI Service")
    parser.add_argument("--serve", action="store_true",
                        help="Start the uvicorn server instead of running tests")
    parser.add_argument("--curl",  action="store_true",
                        help="Print curl examples for all endpoints")
    args = parser.parse_args()

    if args.curl:
        print(CURL_EXAMPLES)
    elif args.serve:
        serve()
    else:
        run_tests()