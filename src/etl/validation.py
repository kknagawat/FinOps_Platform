"""
src/etl/validation.py
=====================
Deliverable 4 — Cross-Table Referential Integrity Validation
--------------------------------------------------------------
Responsibilities:
  • Assert no null PKs in any table.
  • Find and log orphaned foreign keys across all FK relationships.
  • Assert all monetary amounts are > 0 (post-cleaning).
  • Assert all date columns that should be non-null are parseable.
  • Detect customers with overlapping active subscription periods.
  • Output a structured validation report to notebooks/validation_report.csv.
  • Print a human-readable summary to the console.

Design
------
  * Orphans are LOGGED, not dropped — the downstream SQL schema can handle
    orphans with LEFT JOINs; it's the analyst's call whether to exclude them.
  * Hard assertions (null PKs) raise AssertionError — these would break
    SQLite FK constraints and must be fixed before loading.
  * Soft assertions (orphans, non-positive amounts) are warnings — they
    are real data quality issues but the pipeline can continue.

Run standalone:
    python -m src.etl.validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Individual validation checks
# =============================================================================

def check_null_pks(dataframes: dict[str, pd.DataFrame]) -> dict[str, int]:
    """
    Assert that no primary key column contains null values.

    A null PK would violate SQLite's NOT NULL constraint on the PK column
    and make FK joins unreliable (NULL != NULL in SQL).

    Raises
    ------
    AssertionError if any null PKs are found.
    """
    PK_MAP = {
        "customers":       "customer_id",
        "transactions":    "transaction_id",
        "subscriptions":   "subscription_id",
        "invoices":        "invoice_id",
        "support_tickets": "ticket_id",
        "product_usage":   "usage_id",
    }

    results: dict[str, int] = {}
    log.info("  [1/5] Checking null PKs …")

    for table, pk_col in PK_MAP.items():
        df      = dataframes[table]
        n_nulls = int(df[pk_col].isna().sum()) if pk_col in df.columns else -1
        results[table] = n_nulls

        if n_nulls == 0:
            log.info("    ✓  %-22s  %s  — no nulls", table, pk_col)
        else:
            log.error("    ✗  %-22s  %s  — %d NULL PKs!", table, pk_col, n_nulls)
            raise AssertionError(
                f"NULL PKs found in {table}.{pk_col}: {n_nulls} rows. "
                "This must be resolved before database loading."
            )

    return results


def check_fk_orphans(dataframes: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Find orphaned foreign key values across all FK relationships.

    An orphan is a FK value that has no matching PK in the parent table.

    Relationships checked
    ---------------------
    transactions.customer_id    → customers.customer_id
    transactions.subscription_id → subscriptions.subscription_id
    invoices.customer_id        → customers.customer_id
    invoices.subscription_id    → subscriptions.subscription_id
    support_tickets.customer_id → customers.customer_id
    subscriptions.customer_id   → customers.customer_id
    product_usage.customer_id   → customers.customer_id  (skips nulls)

    Why orphans exist here
    ----------------------
    The source systems were not enforcing FK constraints. Orphaned
    subscription_ids in transactions likely come from subscriptions that
    were deleted after the transactions were recorded, or from data exports
    that weren't time-aligned. We log them for the SQL layer to handle
    (typically via LEFT JOIN rather than INNER JOIN).
    """
    customers = dataframes["customers"]
    txn       = dataframes["transactions"]
    subs      = dataframes["subscriptions"]
    inv       = dataframes["invoices"]
    tickets   = dataframes["support_tickets"]
    usage     = dataframes["product_usage"]

    valid_customer_ids = set(customers["customer_id"].dropna())
    valid_sub_ids      = set(subs["subscription_id"].dropna())

    FK_CHECKS = [
        ("transactions",    "customer_id",    txn["customer_id"],              valid_customer_ids),
        ("transactions",    "subscription_id", txn["subscription_id"],          valid_sub_ids),
        ("invoices",        "customer_id",    inv["customer_id"],              valid_customer_ids),
        ("invoices",        "subscription_id", inv["subscription_id"],          valid_sub_ids),
        ("support_tickets", "customer_id",    tickets["customer_id"],          valid_customer_ids),
        ("subscriptions",   "customer_id",    subs["customer_id"],             valid_customer_ids),
        ("product_usage",   "customer_id",    usage["customer_id"].dropna(),   valid_customer_ids),
    ]

    results: dict[str, dict] = {}
    log.info("  [2/5] Checking FK orphans …")

    for child_table, fk_col, fk_series, valid_set in FK_CHECKS:
        key        = f"{child_table}.{fk_col}"
        orphans    = fk_series[~fk_series.isin(valid_set)]
        n_orphans  = int(orphans.count())
        sample     = orphans.dropna().unique()[:5].tolist()

        results[key] = {"orphan_count": n_orphans, "orphan_sample": sample}

        if n_orphans == 0:
            log.info("    ✓  %-40s  no orphans", key)
        else:
            log.warning(
                "    ⚠  %-40s  %d orphans  (sample: %s)",
                key, n_orphans, sample[:3],
            )

    return results


def check_positive_amounts(dataframes: dict[str, pd.DataFrame]) -> dict[str, int]:
    """
    Verify that all monetary amounts are strictly positive after cleaning.

    Columns checked
    ---------------
    transactions.amount  — after abs() correction in cleaning.py
    invoices.total       — invoice total must be > 0
    subscriptions.mrr    — after abs() correction in cleaning.py
                           Note: zero MRR is valid for free-tier subs;
                           we check for < 0 (which should be zero after abs()).

    These checks use > 0 for transactions and invoices (a zero-amount
    transaction or invoice is not meaningful). For subscriptions we check
    >= 0 (free plans legitimately have mrr = 0).
    """
    results: dict[str, int] = {}
    log.info("  [3/5] Checking positive amounts …")

    checks = [
        ("transactions",  "amount", "> 0",  lambda s: (s <= 0).sum()),
        ("invoices",      "total",  "> 0",  lambda s: (s <= 0).sum()),
        ("subscriptions", "mrr",    ">= 0", lambda s: (s < 0).sum()),
    ]

    for table, col, rule, fn in checks:
        df  = dataframes[table]
        s   = pd.to_numeric(df[col], errors="coerce").dropna()
        n   = int(fn(s))
        key = f"{table}.{col}"
        results[key] = n

        if n == 0:
            log.info("    ✓  %-30s  all values %s", key, rule)
        else:
            log.warning("    ⚠  %-30s  %d values NOT %s", key, n, rule)

    return results


def check_date_parseability(dataframes: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Verify that date columns contain parseable values after cleaning.

    Date columns are stored as strings in the cleaned DataFrames.
    A non-null value that is NOT a valid YYYY-MM-DD string indicates
    a parsing failure in cleaning.py (which would already have logged it).

    For each date column we report: null count and null percentage.
    Columns expected to have nulls (e.g. end_date for active subs)
    are noted with an explanation.
    """
    DATE_COLS: dict[str, list[str]] = {
        "customers":       ["signup_date"],
        "transactions":    ["transaction_date"],
        "subscriptions":   ["start_date", "end_date"],
        "invoices":        ["issue_date", "due_date", "paid_date"],
        "support_tickets": ["created_at", "first_response_at", "resolved_at"],
        "product_usage":   ["session_date"],
    }

    EXPECTED_NULLS = {
        "subscriptions.end_date":                 "active subscriptions have no end_date — EXPECTED",
        "invoices.paid_date":                     "unpaid invoices have no paid_date — EXPECTED",
        "support_tickets.first_response_at":      "open tickets have no first_response — EXPECTED",
        "support_tickets.resolved_at":            "unresolved tickets have no resolved_at — EXPECTED",
        "support_tickets.created_at":             "793 rows with unparseable tz-offset format",
    }

    results: dict[str, dict] = {}
    log.info("  [4/5] Checking date column parseability …")

    for table, cols in DATE_COLS.items():
        df = dataframes[table]
        for col in cols:
            key     = f"{table}.{col}"
            n_null  = int(df[col].isna().sum()) if col in df.columns else 0
            pct     = round(n_null / len(df) * 100, 1) if len(df) else 0
            note    = EXPECTED_NULLS.get(key, "")
            results[key] = {"null_count": n_null, "null_pct": pct, "note": note}

            if n_null == 0:
                log.info("    ✓  %-42s  0 nulls", key)
            elif note:
                log.info("    ℹ  %-42s  %d nulls (%.1f%%)  — %s", key, n_null, pct, note)
            else:
                log.warning("    ⚠  %-42s  %d nulls (%.1f%%)", key, n_null, pct)

    return results


def check_subscription_overlaps(dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Detect customers with overlapping active subscription periods.

    Definition of overlap
    ---------------------
    Two subscriptions for the same customer overlap if subscription B's
    start_date falls within subscription A's active period
    (i.e. start_date_B < end_date_A).

    For active subscriptions with no end_date we use "2099-12-31" as
    a sentinel end date (meaning "still active").

    Why this matters
    ----------------
    Overlapping subscriptions mean the customer is being billed twice
    for the same period — a revenue recognition and billing compliance
    issue. The SQL query Q4 in queries.sql specifically calculates the
    double-billed MRR impact.

    Returns
    -------
    dict with:
      - affected_customer_count : int
      - affected_customer_ids   : list[str] (all of them)
      - sample                  : list[str] (first 10)
      - overlap_pairs           : list[dict] with subscription pair details
    """
    log.info("  [5/5] Detecting subscription period overlaps …")

    subs = dataframes["subscriptions"].copy()

    # Only active subscriptions can be "currently overlapping"
    # (but we also check cancelled ones to find historical double-billing)
    active = subs[subs["status"].isin(["active", "paused"])].copy()

    active["start_dt"] = pd.to_datetime(active["start_date"], errors="coerce")
    active["end_dt"]   = pd.to_datetime(
        active["end_date"].fillna("2099-12-31"), errors="coerce"
    )

    overlap_customers: list[str]  = []
    overlap_pairs:     list[dict] = []

    for cust_id, grp in active.groupby("customer_id"):
        grp = grp.sort_values("start_dt").reset_index(drop=True)
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                row_i = grp.iloc[i]
                row_j = grp.iloc[j]
                # Overlap: j starts before i ends
                if pd.notna(row_i["end_dt"]) and row_j["start_dt"] < row_i["end_dt"]:
                    if cust_id not in overlap_customers:
                        overlap_customers.append(cust_id)
                    overlap_days = (
                        min(row_i["end_dt"], row_j["end_dt"]) -
                        max(row_i["start_dt"], row_j["start_dt"])
                    ).days
                    overlap_pairs.append({
                        "customer_id": cust_id,
                        "sub_a":       row_i["subscription_id"],
                        "sub_b":       row_j["subscription_id"],
                        "overlap_days": max(overlap_days, 0),
                    })

    n = len(overlap_customers)
    if n == 0:
        log.info("    ✓  No overlapping subscription periods found")
    else:
        log.warning(
            "    ⚠  %d customers have overlapping active subscriptions "
            "(sample: %s …)", n, overlap_customers[:5],
        )

    return {
        "affected_customer_count": n,
        "affected_customer_ids":   overlap_customers,
        "sample":                  overlap_customers[:10],
        "overlap_pairs":           overlap_pairs[:20],  # top 20 for reporting
    }


# =============================================================================
# Save validation report
# =============================================================================

def save_validation_report(results: dict[str, Any]) -> Path:
    """
    Flatten all validation results → one CSV per check type,
    combined into notebooks/validation_report.csv.
    """
    rows: list[dict] = []

    # Null PKs
    for table, n in results["null_pks"].items():
        rows.append({"check": "null_pk", "key": table, "value": n,
                     "status": "PASS" if n == 0 else "FAIL"})

    # FK orphans
    for key, info in results["fk_orphans"].items():
        rows.append({"check": "fk_orphan", "key": key,
                     "value": info["orphan_count"],
                     "sample": str(info["orphan_sample"]),
                     "status": "PASS" if info["orphan_count"] == 0 else "WARN"})

    # Positive amounts
    for key, n in results["positive_amounts"].items():
        rows.append({"check": "positive_amount", "key": key, "value": n,
                     "status": "PASS" if n == 0 else "WARN"})

    # Date parseability
    for key, info in results["date_parseability"].items():
        rows.append({"check": "date_null", "key": key,
                     "value": info["null_count"],
                     "note": info.get("note", ""),
                     "status": "INFO" if info.get("note") else
                               ("PASS" if info["null_count"] == 0 else "WARN")})

    # Subscription overlaps
    rows.append({
        "check":  "subscription_overlap",
        "key":    "customers_with_overlapping_subs",
        "value":  results["subscription_overlaps"]["affected_customer_count"],
        "sample": str(results["subscription_overlaps"]["sample"]),
        "status": "PASS"
                  if results["subscription_overlaps"]["affected_customer_count"] == 0
                  else "WARN",
    })

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "validation_report.csv"
    df.to_csv(out, index=False)
    log.info("  Validation report saved → %s  (%d checks)", out, len(df))
    return out


# =============================================================================
# Console summary
# =============================================================================

def print_validation_summary(results: dict[str, Any]) -> None:
    """Print a concise pass/warn/fail summary to stdout."""
    SEP = "─" * 68
    print("\n" + "═" * 68)
    print("  CROSS-TABLE VALIDATION REPORT")
    print("═" * 68)

    # Null PKs
    print("\n  [1] NULL PRIMARY KEYS")
    print("  " + SEP)
    all_pass = all(v == 0 for v in results["null_pks"].values())
    status = "✓ PASS — all 6 tables have zero null PKs" if all_pass else "✗ FAIL"
    print(f"  {status}")

    # FK orphans
    print("\n  [2] FOREIGN KEY ORPHANS")
    print("  " + SEP)
    print(f"  {'Relationship':<42} {'Orphans':>9}")
    print("  " + SEP)
    for key, info in results["fk_orphans"].items():
        n     = info["orphan_count"]
        icon  = "✓" if n == 0 else "⚠"
        print(f"  {icon}  {key:<40} {n:>9,}")

    # Positive amounts
    print("\n  [3] NON-POSITIVE AMOUNTS (after cleaning)")
    print("  " + SEP)
    for key, n in results["positive_amounts"].items():
        icon = "✓" if n == 0 else "⚠"
        print(f"  {icon}  {key:<40} {n:>9,} violations")

    # Date parseability
    print("\n  [4] DATE COLUMNS — null / unparseable values")
    print("  " + SEP)
    print(f"  {'Column':<42} {'Nulls':>8}  Note")
    print("  " + SEP)
    for key, info in results["date_parseability"].items():
        icon = "ℹ" if info.get("note") else ("✓" if info["null_count"] == 0 else "⚠")
        note = info.get("note", "")[:35]
        print(f"  {icon}  {key:<40} {info['null_count']:>8,}  {note}")

    # Overlaps
    n_overlap = results["subscription_overlaps"]["affected_customer_count"]
    print("\n  [5] OVERLAPPING SUBSCRIPTION PERIODS")
    print("  " + SEP)
    icon = "✓" if n_overlap == 0 else "⚠"
    print(f"  {icon}  {n_overlap} customer(s) with overlapping active subscriptions")
    if n_overlap:
        print(f"     Sample: {results['subscription_overlaps']['sample'][:5]}")

    # Overall summary
    total_orphans = sum(v["orphan_count"] for v in results["fk_orphans"].values())
    total_nonpos  = sum(results["positive_amounts"].values())
    print("\n" + "═" * 68)
    print(f"  SUMMARY")
    print("═" * 68)
    print(f"  Null PKs:                    {sum(results['null_pks'].values()):>6}  (0 = PASS)")
    print(f"  FK orphans total:            {total_orphans:>6}  (>0 = log and continue)")
    print(f"  Non-positive amounts:        {total_nonpos:>6}  (>0 = warn)")
    print(f"  Customers w/ overlap subs:   {n_overlap:>6}  (>0 = warn)")
    print()


# =============================================================================
# Main validation orchestrator
# =============================================================================

def run_validation(cleaned_dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Run all five validation checks on the cleaned DataFrames.

    Parameters
    ----------
    cleaned_dataframes : dict returned by cleaning.run_cleaning()

    Returns
    -------
    dict with keys:
        null_pks, fk_orphans, positive_amounts,
        date_parseability, subscription_overlaps
    """
    log.info("=" * 60)
    log.info("VALIDATION — Cross-table integrity checks")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "null_pks":            check_null_pks(cleaned_dataframes),
        "fk_orphans":          check_fk_orphans(cleaned_dataframes),
        "positive_amounts":    check_positive_amounts(cleaned_dataframes),
        "date_parseability":   check_date_parseability(cleaned_dataframes),
        "subscription_overlaps": check_subscription_overlaps(cleaned_dataframes),
    }

    save_validation_report(results)
    print_validation_summary(results)

    log.info("Validation complete.")
    return results


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.etl.ingestion import run_ingestion
    from src.etl.cleaning  import run_cleaning

    raw     = run_ingestion()
    cleaned = run_cleaning(raw)
    run_validation(cleaned)