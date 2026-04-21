"""
run_step3.py
============
Step 3 — Advanced SQL Query Runner
--------------------------------------
Executes all 8 hand-written analytical queries from sql/queries.sql
against finops.db and prints:
  • Query name and description
  • Row count returned
  • First 5 rows of results
  • Column names
  • Execution time

Prerequisites
-------------
  finops.db must exist (run run_step2.py first).

Usage
-----
  python run_step3.py                    # run all 8 queries
  python run_step3.py --query Q3         # run a single query by label
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")
log = logging.getLogger(__name__)

DB_PATH    = PROJECT_ROOT / "finops.db"
QUERY_FILE = PROJECT_ROOT / "sql" / "queries.sql"


# =============================================================================
# Query definitions — each entry maps a label to its SQL and a description
# =============================================================================

QUERY_META = {
    "Q1": {
        "title":       "Revenue Cohort Analysis",
        "techniques":  "CTE + Window Functions + Conditional Aggregation",
        "description": "Cohort matrix: revenue and retention at M1/M2/M3/M6/M12 milestones.",
    },
    "Q2": {
        "title":       "MRR Movement Waterfall",
        "techniques":  "CTE + LAG() + CASE Classification",
        "description": "Monthly New / Expansion / Contraction / Churned / Net MRR movements.",
    },
    "Q3": {
        "title":       "Customer Health Score",
        "techniques":  "Multiple CTEs + Correlated Subquery + DENSE_RANK()",
        "description": "Composite 0–100 score per active customer; weighted across 4 dimensions.",
    },
    "Q4": {
        "title":       "Subscription Overlap Detection",
        "techniques":  "Self-JOIN + ROW_NUMBER() + Date Arithmetic",
        "description": "Customers with overlapping subs; overlap days and double-billed MRR.",
    },
    "Q5": {
        "title":       "Support Ticket Resolution Funnel",
        "techniques":  "NTILE(100) + Window Functions",
        "description": "P50/P95 response and resolution times, FCR and escalation rates per priority.",
    },
    "Q6": {
        "title":       "Revenue Concentration Risk (Pareto + HHI)",
        "techniques":  "Recursive CTE + Window Functions",
        "description": "Cumulative revenue %, Pareto segments, concentration risk flags, HHI index.",
    },
    "Q7": {
        "title":       "Product Feature Adoption Funnel",
        "techniques":  "Multiple JOINs + Conditional Aggregation (CASE inside COUNT)",
        "description": "Reach → Activation → Power User funnel rates per feature.",
    },
    "Q8": {
        "title":       "Churn Prediction Signals",
        "techniques":  "4-CTE Chain + DENSE_RANK()",
        "description": "Four-signal churn score per active customer: usage decline, ticket spike, rating drop, no login.",
    },
}


# =============================================================================
# SQL parser — split queries.sql into individual queries by Q-label comments
# =============================================================================

def parse_queries(sql_file: Path) -> dict[str, str]:
    """
    Parse queries.sql and return a dict {label: sql_text}.

    The file uses `-- Q1 — Title` style separator comments between queries.
    We split on lines starting with `-- Q[digit]` and assign each block
    to its label.
    """
    raw     = sql_file.read_text(encoding="utf-8")
    lines   = raw.splitlines()
    queries: dict[str, str] = {}
    current_label: str | None = None
    current_lines: list[str]  = []

    for line in lines:
        # Detect a new query header: "-- Q1 —" or "-- Q1 —" etc.
        stripped = line.strip()
        if stripped.startswith("-- Q") and len(stripped) > 4 and stripped[4].isdigit():
            # Flush the previous query
            if current_label and current_lines:
                sql = "\n".join(current_lines).strip()
                if sql:
                    queries[current_label] = sql
            # Start new label
            # Extract label: everything up to first space after "Q\d"
            parts = stripped[3:].split()          # e.g. ["Q1", "—", "Revenue…"]
            current_label = parts[0].rstrip("—")  # e.g. "Q1"
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the last query
    if current_label and current_lines:
        sql = "\n".join(current_lines).strip()
        if sql:
            queries[current_label] = sql

    return queries


# =============================================================================
# Query runner
# =============================================================================

def run_query(
    conn:  sqlite3.Connection,
    label: str,
    sql:   str,
    meta:  dict,
    max_sample_rows: int = 5,
) -> dict:
    """
    Execute a single query and return results + metadata.
    """
    SEP  = "─" * 72
    SEP2 = "═" * 72

    print(f"\n{SEP2}")
    print(f"  {label} — {meta['title']}")
    print(f"  Techniques : {meta['techniques']}")
    print(f"  Description: {meta['description']}")
    print(SEP2)

    t0 = time.perf_counter()
    try:
        cursor  = conn.execute(sql)
        rows    = cursor.fetchall()
        cols    = [d[0] for d in cursor.description] if cursor.description else []
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n  ✓  {len(rows):,} rows  │  {len(cols)} columns  │  {elapsed:.1f} ms\n")
        print(f"  Columns: {cols}\n")

        # Print sample rows
        if rows:
            sample = rows[:max_sample_rows]
            col_widths = [
                max(len(str(c)), max(len(str(r[i])) for r in sample))
                for i, c in enumerate(cols)
            ]
            col_widths = [min(w, 20) for w in col_widths]  # cap at 20 chars

            header = "  │ " + " │ ".join(
                str(c)[:20].ljust(col_widths[i]) for i, c in enumerate(cols)
            ) + " │"
            print(header)
            print("  " + SEP)

            for row in sample:
                line = "  │ " + " │ ".join(
                    str(v if v is not None else "NULL")[:20].ljust(col_widths[i])
                    for i, v in enumerate(row)
                ) + " │"
                print(line)

            if len(rows) > max_sample_rows:
                print(f"  ... {len(rows) - max_sample_rows} more rows not shown")

        return {
            "label":    label,
            "rows":     len(rows),
            "cols":     len(cols),
            "elapsed_ms": round(elapsed, 1),
            "error":    None,
        }

    except sqlite3.Error as e:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\n  ✗  ERROR: {e}\n")
        log.error("[%s] SQL error: %s", label, e)
        return {
            "label":    label,
            "rows":     0,
            "cols":     0,
            "elapsed_ms": round(elapsed, 1),
            "error":    str(e),
        }


# =============================================================================
# Main runner
# =============================================================================

def run_step3(target_query: str | None = None) -> None:
    BANNER = "█" * 72

    print()
    print(BANNER)
    print("  FINOPS ANALYTICS PLATFORM — STEP 3")
    print("  Advanced SQL Queries — 8 Analytical Patterns")
    print(BANNER)

    # Verify prerequisites
    if not DB_PATH.exists():
        log.error("finops.db not found at %s", DB_PATH)
        log.error("Run python run_step2.py first.")
        sys.exit(1)

    if not QUERY_FILE.exists():
        log.error("queries.sql not found at %s", QUERY_FILE)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Parse queries.sql
    queries = parse_queries(QUERY_FILE)
    log.info("Parsed %d queries from queries.sql", len(queries))

    # Filter to target query if specified
    if target_query:
        if target_query not in queries:
            log.error("Query %s not found. Available: %s", target_query, list(queries.keys()))
            sys.exit(1)
        run_labels = [target_query]
    else:
        run_labels = sorted(queries.keys())

    # Execute
    results = []
    t_total = time.perf_counter()

    for label in run_labels:
        if label not in queries:
            log.warning("No SQL found for %s — skipping", label)
            continue
        meta   = QUERY_META.get(label, {"title": label, "techniques": "", "description": ""})
        result = run_query(conn, label, queries[label], meta)
        results.append(result)

    conn.close()
    total_elapsed = time.perf_counter() - t_total

    # Final scorecard
    print("\n" + "═" * 72)
    print("  STEP 3 SUMMARY")
    print("═" * 72)
    print(f"\n  {'Query':<6} {'Title':<38} {'Rows':>7} {'ms':>6}  {'Status'}")
    print("  " + "─" * 64)
    for r in results:
        title  = QUERY_META.get(r["label"], {}).get("title", "")[:38]
        status = "✓ OK" if r["error"] is None else f"✗ {r['error'][:20]}"
        print(f"  {r['label']:<6} {title:<38} {r['rows']:>7,} {r['elapsed_ms']:>6.1f}  {status}")

    n_ok   = sum(1 for r in results if r["error"] is None)
    n_fail = len(results) - n_ok
    print(f"\n  {n_ok}/{len(results)} queries succeeded  │  {total_elapsed:.2f}s total")
    print()

    if n_fail:
        sys.exit(1)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Step 3 SQL queries")
    parser.add_argument(
        "--query",
        help="Run a single query by label, e.g. --query Q3",
        default=None,
    )
    args = parser.parse_args()
    run_step3(target_query=args.query)