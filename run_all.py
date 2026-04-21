"""
run_all.py
==========
Step 1 Master Orchestrator
---------------------------
Runs all four Step 1 deliverables in sequence:

  1. ingestion.py   — Load all 6 raw files into DataFrames
  2. profiling.py   — Profile raw data, save data_profile_report.csv
  3. cleaning.py    — Apply all cleaning transforms
  4. validation.py  — Cross-table integrity checks, save validation_report.csv

Usage
-----
From the project root:

    python run_all.py

The script is IDEMPOTENT — running it multiple times on the same raw files
always produces identical output files.

Output files
------------
  notebooks/data_profile_report.csv   — raw-data profiling report
  notebooks/validation_report.csv     — integrity validation results

The cleaned DataFrames are returned by run_step1() so they can be passed
directly into Step 2 (SQLite loading) without re-running ingestion.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path regardless of where this is called from
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.ingestion import run_ingestion
from src.etl.profiling import run_profiling
from src.etl.cleaning  import run_cleaning
from src.etl.validation import run_validation

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s │ %(message)s",
)
log = logging.getLogger(__name__)


def run_step1() -> dict:
    """
    Execute all four Step 1 deliverables end-to-end.

    Returns
    -------
    dict with keys:
        "raw_dataframes"     — dict[str, pd.DataFrame]   (from ingestion)
        "profiles"           — list[dict]                 (from profiling)
        "cleaned_dataframes" — dict[str, pd.DataFrame]   (from cleaning)
        "validation_results" — dict[str, Any]             (from validation)
        "elapsed_seconds"    — float
    """
    BANNER = "█" * 64
    t_start = time.perf_counter()

    print()
    print(BANNER)
    print("  FINOPS ANALYTICS PLATFORM — STEP 1")
    print("  Data Ingestion  │  Profiling  │  Cleaning  │  Validation")
    print(BANNER)
    print()

    # ── Deliverable 1: Ingestion ───────────────────────────────────────────────
    _section("DELIVERABLE 1 — Ingestion", 1)
    raw_dfs = run_ingestion()

    # ── Deliverable 2: Profiling (on RAW data) ─────────────────────────────────
    _section("DELIVERABLE 2 — Data Profiling (raw data)", 2)
    profiles = run_profiling(raw_dfs)

    # ── Deliverable 3: Cleaning ────────────────────────────────────────────────
    _section("DELIVERABLE 3 — Cleaning Pipeline", 3)
    cleaned_dfs = run_cleaning(raw_dfs)

    # ── Deliverable 4: Validation (on CLEANED data) ────────────────────────────
    _section("DELIVERABLE 4 — Cross-Table Validation (cleaned data)", 4)
    validation = run_validation(cleaned_dfs)

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _final_summary(raw_dfs, cleaned_dfs, validation, elapsed)

    return {
        "raw_dataframes":     raw_dfs,
        "profiles":           profiles,
        "cleaned_dataframes": cleaned_dfs,
        "validation_results": validation,
        "elapsed_seconds":    round(elapsed, 2),
    }


def _section(title: str, n: int) -> None:
    """Print a numbered section header."""
    print()
    print("┌" + "─" * 62 + "┐")
    print(f"│  {n}.  {title:<56}│")
    print("└" + "─" * 62 + "┘")
    print()


def _final_summary(raw_dfs, cleaned_dfs, validation, elapsed: float) -> None:
    """Print the final end-to-end summary."""
    SEP = "═" * 64
    print()
    print(SEP)
    print("  STEP 1 COMPLETE")
    print(SEP)
    print()

    # Table-level row counts: raw vs cleaned
    print(f"  {'Table':<22} {'Raw':>8} {'Cleaned':>9} {'Δ':>6}")
    print("  " + "─" * 50)
    for name in raw_dfs:
        r = len(raw_dfs[name])
        c = len(cleaned_dfs[name])
        delta = r - c
        mark = f"  (-{delta})" if delta else ""
        print(f"  {name:<22} {r:>8,} {c:>9,} {delta:>6,}")
    print()

    # Validation scorecard
    total_orphans = sum(
        v["orphan_count"] for v in validation["fk_orphans"].values()
    )
    total_nonpos  = sum(validation["positive_amounts"].values())
    n_overlap     = validation["subscription_overlaps"]["affected_customer_count"]
    null_pks_ok   = all(v == 0 for v in validation["null_pks"].values())

    print(f"  {'Check':<38} {'Result':>14}")
    print("  " + "─" * 54)
    print(f"  {'Null PKs':<38} {'✓ PASS' if null_pks_ok else '✗ FAIL':>14}")
    print(f"  {'FK orphans (total)':<38} {total_orphans:>14,}")
    print(f"  {'Non-positive amounts':<38} {total_nonpos:>14,}")
    print(f"  {'Customers w/ overlapping subs':<38} {n_overlap:>14,}")
    print()

    # Output files
    notebooks = Path(__file__).parent / "notebooks"
    print("  Output files:")
    for fname in ("data_profile_report.csv", "validation_report.csv"):
        p = notebooks / fname
        exists = "✓" if p.exists() else "✗"
        print(f"    {exists}  notebooks/{fname}")
    print()
    print(f"  Total elapsed: {elapsed:.2f}s")
    print(SEP)
    print()


if __name__ == "__main__":
    result = run_step1()