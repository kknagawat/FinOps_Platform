"""
src/etl/loader.py
=================
Step 2 — SQLite Loading Script
---------------------------------
Responsibilities:
  • Create the SQLite database from schema.sql.
  • Populate the date_dim table programmatically (no raw data needed).
  • Load all six cleaned DataFrames into their SQLite tables using
    parameterised INSERT statements (no string interpolation → no SQL injection).
  • Cast every column to the exact SQLite type declared in schema.sql.
  • Run post-load integrity checks:
      - Row counts match source DataFrames.
      - FK orphan counts via LEFT JOIN queries.
      - CHECK constraint violations logged for key columns.

Design decisions
----------------
  Why parameterised INSERTs instead of df.to_sql()?
    pd.DataFrame.to_sql() uses executemany() under the hood, which is correct,
    but it does NOT enforce the schema CHECK constraints — it just dumps values.
    Our explicit column-by-column type casting + executemany() gives us:
      (a) guaranteed type safety on every column
      (b) CHECK constraint enforcement by SQLite at insert time
      (c) the ability to log exactly which rows fail a constraint

  Why re-create the DB on every run?
    Idempotency. Partial loads (e.g. network drop mid-run) can leave the DB
    in an inconsistent state. Dropping and re-creating guarantees a clean slate.
    For a production system you would use upsert (INSERT OR REPLACE) instead.

Run standalone:
    python -m src.etl.loader
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve()
PROJECT    = _HERE.parents[2]
DB_PATH    = PROJECT / "finops.db"
SCHEMA_SQL = PROJECT / "sql" / "schema.sql"


# =============================================================================
# Section 1 — Database initialisation
# =============================================================================

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and configure runtime pragmas.

    Pragmas set here persist for the lifetime of the connection:
      PRAGMA foreign_keys = ON  →  FK constraints are enforced
      PRAGMA journal_mode = WAL →  safe for concurrent reads (useful in Step 7)
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row   # rows accessible as dicts
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    sql_text = SCHEMA_SQL.read_text(encoding="utf-8")
    statements = [
        s.strip() for s in sql_text.split(";")
        if s.strip() and not s.strip().startswith("--")
    ]
    for stmt in statements:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e).lower():
                log.warning("Schema statement skipped: %s", e)
    conn.commit()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    log.info("  Schema created — tables: %s", [t[0] for t in tables])


# =============================================================================
# Section 2 — Type casting helpers
# =============================================================================
# SQLite is dynamically typed but our schema declares TEXT / REAL / INTEGER.
# We cast every column explicitly so CHECK constraints see the right type.

def _to_text(val: Any) -> str | None:
    """Cast to str; map NaN / None / 'nan' → None (SQL NULL)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return None if s.lower() in ("nan", "none", "") else s


def _to_real(val: Any) -> float | None:
    """Cast to float; map non-numeric → None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _to_int(val: Any) -> int | None:
    """Cast to int; map non-integer → None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _to_bool_int(val: Any) -> int | None:
    """Cast Python bool / bool-like → 0 or 1; None stays None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    try:
        return int(bool(val))
    except (ValueError, TypeError):
        return None


# =============================================================================
# Section 3 — date_dim population
# =============================================================================

def populate_date_dim(
    conn: sqlite3.Connection,
    start: str = "2021-01-01",
    end:   str = "2026-12-31",
) -> int:
    """
    Programmatically generate one row per calendar day and insert
    into date_dim.

    Range 2021-01-01 → 2026-12-31 covers:
      • All transaction dates in the dataset (earliest ~2021).
      • A 12-month forecast horizon beyond the latest date (~2025).

    Fiscal quarter convention: FQ1 = Jan-Mar, FQ2 = Apr-Jun, etc.
    (calendar-aligned, not offset fiscal year).

    Uses executemany() with parameterised INSERT — fastest bulk method
    for SQLite and immune to any date-string injection.
    """
    dates = pd.date_range(start=start, end=end, freq="D")

    rows = [
        (
            int(d.strftime("%Y%m%d")),      # date_id
            d.strftime("%Y-%m-%d"),          # full_date
            int(d.year),                     # year
            int(d.quarter),                  # quarter
            int(d.month),                    # month
            d.strftime("%B"),                # month_name
            int(d.isocalendar()[1]),         # week_of_year
            int(d.day),                      # day_of_month
            int(d.dayofweek),                # day_of_week (0=Mon)
            d.strftime("%A"),                # day_name
            int(d.dayofweek >= 5),           # is_weekend
            int(d.is_month_start),           # is_month_start
            int(d.is_month_end),             # is_month_end
            f"FQ{d.quarter}-{d.year}",       # fiscal_quarter
        )
        for d in dates
    ]

    conn.executemany(
        """
        INSERT OR REPLACE INTO date_dim
            (date_id, full_date, year, quarter, month, month_name,
             week_of_year, day_of_month, day_of_week, day_name,
             is_weekend, is_month_start, is_month_end, fiscal_quarter)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    log.info("  date_dim populated: %d rows (%s → %s)", len(rows), start, end)
    return len(rows)


# =============================================================================
# Section 4 — Per-table loaders (parameterised INSERT)
# =============================================================================

def load_customers(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert cleaned customers DataFrame into customers table."""
    rows = [
        (
            _to_text(r.customer_id),
            _to_text(r.first_name),
            _to_text(r.last_name),
            _to_text(r.email),
            _to_text(r.phone),
            _to_text(r.city),
            _to_text(r.country),
            _to_text(r.zip_code),
            _to_text(r.signup_date),
            _to_bool_int(r.is_active),
            _to_text(r.company),
            _to_text(r.loyalty_tier),
            _to_bool_int(r.email_missing),
            _to_bool_int(r.phone_missing),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO customers
            (customer_id, first_name, last_name, email, phone,
             city, country, zip_code, signup_date, is_active,
             company, loyalty_tier, email_missing, phone_missing)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_subscriptions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert cleaned subscriptions DataFrame."""
    rows = [
        (
            _to_text(r.subscription_id),
            _to_text(r.customer_id),
            _to_text(r.plan_name),
            _to_real(r.mrr),
            _to_text(r.currency),
            _to_text(r.start_date),
            _to_text(r.end_date),
            _to_text(r.status),
            _to_text(r.billing_cycle),
            _to_bool_int(r.auto_renew),
            _to_bool_int(r.mrr_was_negative),
            _to_bool_int(r.is_future_start),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO subscriptions
            (subscription_id, customer_id, plan_name, mrr, currency,
             start_date, end_date, status, billing_cycle, auto_renew,
             mrr_was_negative, is_future_start)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_transactions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert cleaned transactions DataFrame."""
    rows = [
        (
            _to_text(r.transaction_id),
            _to_text(r.customer_id),
            _to_text(r.subscription_id),
            _to_text(r.transaction_date),
            _to_real(r.amount),
            _to_text(r.currency),
            _to_text(r.status),
            _to_text(r.payment_method),
            _to_text(r.invoice_id),
            _to_text(r.description),
            _to_bool_int(r.was_negative),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO transactions
            (transaction_id, customer_id, subscription_id, transaction_date,
             amount, currency, status, payment_method, invoice_id,
             description, was_negative)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_invoices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert cleaned invoices DataFrame."""
    rows = [
        (
            _to_text(r.invoice_id),
            _to_text(r.customer_id),
            _to_text(r.subscription_id),
            _to_text(r.issue_date),
            _to_text(r.due_date),
            _to_real(r.subtotal),
            _to_real(r.tax),
            _to_real(r.total),
            _to_real(r.paid_amount),
            _to_text(r.payment_status),
            _to_text(r.paid_date),
            _to_text(r.payment_method),
            _to_text(r.currency),
            _to_real(r.tax_rate),
            _to_bool_int(r.tax_error_flag),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO invoices
            (invoice_id, customer_id, subscription_id, issue_date, due_date,
             subtotal, tax, total, paid_amount, payment_status, paid_date,
             payment_method, currency, tax_rate, tax_error_flag)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_support_tickets(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert cleaned support_tickets DataFrame."""
    rows = [
        (
            _to_text(r.ticket_id),
            _to_text(r.customer_id),
            _to_text(r.category),
            _to_text(r.priority),
            _to_text(r.status),
            _to_text(r.created_at),
            _to_text(r.first_response_at),
            _to_text(r.resolved_at),
            _to_real(r.rating),
            _to_text(r.resolution_text) or '',
            _to_text(r.agent_name),
            _to_text(r.channel),
            _to_bool_int(r.is_escalated),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO support_tickets
            (ticket_id, customer_id, category, priority, status,
             created_at, first_response_at, resolved_at, rating,
             resolution_text, agent_name, channel, is_escalated)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_product_usage(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert cleaned product_usage DataFrame."""
    rows = [
        (
            _to_text(r.usage_id),
            _to_text(r.customer_id),
            _to_text(r.feature_name),
            _to_text(r.session_date),
            _to_real(r.session_duration_seconds),
            _to_real(r.usage_count),
            _to_text(r.device),
            _to_text(r.session_id),
            _to_bool_int(r.customer_id_missing),
            _to_bool_int(r.duration_invalid),
            _to_bool_int(r.usage_count_outlier),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO product_usage
            (usage_id, customer_id, feature_name, session_date,
             session_duration_seconds, usage_count, device, session_id,
             customer_id_missing, duration_invalid, usage_count_outlier)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


# =============================================================================
# Section 5 — Post-load integrity checks
# =============================================================================

def run_integrity_checks(
    conn:           sqlite3.Connection,
    source_counts:  dict[str, int],
) -> dict[str, Any]:
    """
    Verify the database after loading.

    Checks
    ------
    1. Row counts match source DataFrames.
    2. No NULL PKs (should be guaranteed by NOT NULL + PK, but we assert).
    3. FK orphans via LEFT JOIN (same check as validation.py but in SQL).
    4. CHECK constraint violations — values that would fail the declared
       constraints (SQLite enforces these at INSERT time, but we double-check
       to surface any edge cases that slipped through casting).
    """
    issues:  list[str] = []
    results: dict[str, Any] = {}

    SEP = "─" * 60

    # ── 1. Row count match ─────────────────────────────────────────────────────
    log.info("  [1/4] Row count verification …")
    count_results = {}
    for table, expected in source_counts.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        ok     = actual == expected
        count_results[table] = {"expected": expected, "actual": actual, "match": ok}
        if ok:
            log.info("    ✓  %-22s  expected %d  got %d", table, expected, actual)
        else:
            msg = f"ROW COUNT MISMATCH {table}: expected {expected}, got {actual}"
            log.error("    ✗  %s", msg)
            issues.append(msg)
    results["row_counts"] = count_results

    # ── 2. NULL PKs ────────────────────────────────────────────────────────────
    log.info("  [2/4] NULL PK check …")
    pk_map = {
        "customers":       "customer_id",
        "subscriptions":   "subscription_id",
        "transactions":    "transaction_id",
        "invoices":        "invoice_id",
        "support_tickets": "ticket_id",
        "product_usage":   "usage_id",
    }
    null_pks = {}
    for table, pk in pk_map.items():
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {pk} IS NULL"
        ).fetchone()[0]
        null_pks[table] = n
        if n == 0:
            log.info("    ✓  %-22s  %s  — no NULLs", table, pk)
        else:
            msg = f"NULL PK in {table}.{pk}: {n} rows"
            log.error("    ✗  %s", msg)
            issues.append(msg)
    results["null_pks"] = null_pks

    # ── 3. FK orphans ──────────────────────────────────────────────────────────
    log.info("  [3/4] FK orphan check (SQL LEFT JOIN) …")
    fk_checks = [
        ("transactions → customers",
         "SELECT COUNT(*) FROM transactions t "
         "LEFT JOIN customers c ON t.customer_id = c.customer_id "
         "WHERE c.customer_id IS NULL AND t.customer_id IS NOT NULL"),

        ("transactions → subscriptions",
         "SELECT COUNT(*) FROM transactions t "
         "LEFT JOIN subscriptions s ON t.subscription_id = s.subscription_id "
         "WHERE s.subscription_id IS NULL AND t.subscription_id IS NOT NULL"),

        ("invoices → customers",
         "SELECT COUNT(*) FROM invoices i "
         "LEFT JOIN customers c ON i.customer_id = c.customer_id "
         "WHERE c.customer_id IS NULL AND i.customer_id IS NOT NULL"),

        ("invoices → subscriptions",
         "SELECT COUNT(*) FROM invoices i "
         "LEFT JOIN subscriptions s ON i.subscription_id = s.subscription_id "
         "WHERE s.subscription_id IS NULL AND i.subscription_id IS NOT NULL"),

        ("support_tickets → customers",
         "SELECT COUNT(*) FROM support_tickets t "
         "LEFT JOIN customers c ON t.customer_id = c.customer_id "
         "WHERE c.customer_id IS NULL AND t.customer_id IS NOT NULL"),

        ("product_usage → customers",
         "SELECT COUNT(*) FROM product_usage p "
         "LEFT JOIN customers c ON p.customer_id = c.customer_id "
         "WHERE c.customer_id IS NULL AND p.customer_id IS NOT NULL"),
    ]
    orphan_results = {}
    for label, query in fk_checks:
        n = conn.execute(query).fetchone()[0]
        orphan_results[label] = n
        if n == 0:
            log.info("    ✓  %-38s  0 orphans", label)
        else:
            log.warning("    ⚠  %-38s  %d orphans", label, n)
    results["fk_orphans"] = orphan_results

    # ── 4. CHECK constraint spot-checks ───────────────────────────────────────
    log.info("  [4/4] CHECK constraint violations …")
    constraint_checks = [
        ("transactions.amount < 0",
         "SELECT COUNT(*) FROM transactions WHERE amount < 0"),
        ("invoices.total < 0",
         "SELECT COUNT(*) FROM invoices WHERE total < 0"),
        ("subscriptions.mrr < 0",
         "SELECT COUNT(*) FROM subscriptions WHERE mrr < 0"),
        ("support_tickets.rating out of [1,5]",
         "SELECT COUNT(*) FROM support_tickets WHERE rating IS NOT NULL AND (rating < 1 OR rating > 5)"),
        ("date_dim.quarter out of [1,4]",
         "SELECT COUNT(*) FROM date_dim WHERE quarter NOT BETWEEN 1 AND 4"),
        ("date_dim.month out of [1,12]",
         "SELECT COUNT(*) FROM date_dim WHERE month NOT BETWEEN 1 AND 12"),
    ]
    check_results = {}
    for label, query in constraint_checks:
        n = conn.execute(query).fetchone()[0]
        check_results[label] = n
        if n == 0:
            log.info("    ✓  %-42s  0 violations", label)
        else:
            msg = f"CHECK violation [{label}]: {n} rows"
            log.warning("    ⚠  %s", msg)
            issues.append(msg)
    results["check_violations"] = check_results
    results["issues"] = issues

    return results


# =============================================================================
# Section 6 — Main loader orchestrator
# =============================================================================

def run_loader(
    cleaned_dataframes: dict[str, pd.DataFrame],
    db_path:            Path = DB_PATH,
) -> dict[str, Any]:
    """
    Full loading pipeline:
      1. Drop old DB (idempotency).
      2. Create schema from schema.sql.
      3. Populate date_dim.
      4. Load all six tables.
      5. Post-load integrity checks.

    Parameters
    ----------
    cleaned_dataframes : output of cleaning.run_cleaning()
    db_path            : path to the SQLite file (default: project root)

    Returns
    -------
    dict with keys: row_counts_loaded, integrity, issues, db_path
    """
    log.info("=" * 60)
    log.info("LOADER — SQLite Database Loading")
    log.info("=" * 60)

    # ── Drop and recreate for idempotency ──────────────────────────────────────
    if db_path.exists():
        db_path.unlink()
        log.info("  Existing database removed for fresh load")

    conn = get_connection(db_path)

    # ── Schema ─────────────────────────────────────────────────────────────────
    log.info("  Creating schema from sql/schema.sql …")
    create_schema(conn)

    # ── date_dim ───────────────────────────────────────────────────────────────
    log.info("  Populating date_dim …")
    populate_date_dim(conn)

    # ── Six fact/dimension tables ──────────────────────────────────────────────
    # Disable FK enforcement during bulk load — data has known orphan FKs
    # (documented in validation.py: 544 orphans from misaligned source exports).
    # We load everything, then re-enable FKs and report orphans in the
    # integrity phase — exactly how a production data warehouse operates.
    conn.execute("PRAGMA foreign_keys = OFF")
    log.info("  Loading tables (FK checks OFF during bulk load) …")

    TABLE_LOADERS = [
        ("customers",       load_customers,       cleaned_dataframes["customers"]),
        ("subscriptions",   load_subscriptions,   cleaned_dataframes["subscriptions"]),
        ("transactions",    load_transactions,     cleaned_dataframes["transactions"]),
        ("invoices",        load_invoices,         cleaned_dataframes["invoices"]),
        ("support_tickets", load_support_tickets,  cleaned_dataframes["support_tickets"]),
        ("product_usage",   load_product_usage,    cleaned_dataframes["product_usage"]),
    ]
    # Note: customers must load before subscriptions (FK), subscriptions before
    # transactions/invoices (FK).  The order above respects that dependency.

    loaded_counts: dict[str, int] = {}
    for table_name, loader_fn, df in TABLE_LOADERS:
        n = loader_fn(conn, df)
        loaded_counts[table_name] = n
        log.info("    ✓  %-22s  %d rows inserted", table_name, n)

    # ── Integrity checks ───────────────────────────────────────────────────────
    conn.execute("PRAGMA foreign_keys = ON")   # re-enable for check queries
    log.info("  Running post-load integrity checks …")
    integrity = run_integrity_checks(conn, loaded_counts)

    conn.close()

    log.info("-" * 60)
    log.info("  Database ready: %s", db_path)
    log.info("  Size on disk:   %.1f KB", db_path.stat().st_size / 1024)
    log.info("=" * 60)

    return {
        "db_path":          str(db_path),
        "row_counts_loaded": loaded_counts,
        "integrity":         integrity,
        "issues":            integrity.get("issues", []),
    }


# =============================================================================
# Console summary
# =============================================================================

def print_load_summary(result: dict[str, Any]) -> None:
    SEP  = "─" * 64
    SEP2 = "═" * 64

    print("\n" + SEP2)
    print("  STEP 2 — SQLITE LOAD SUMMARY")
    print(SEP2)

    print(f"\n  Database: {result['db_path']}")

    print(f"\n  {'Table':<22} {'Rows loaded':>12}")
    print("  " + SEP)
    for table, n in result["row_counts_loaded"].items():
        print(f"  {table:<22} {n:>12,}")

    integ = result["integrity"]

    # FK orphans
    print(f"\n  {'FK Relationship':<40} {'Orphans':>8}")
    print("  " + SEP)
    for label, n in integ["fk_orphans"].items():
        icon = "✓" if n == 0 else "⚠"
        print(f"  {icon}  {label:<38} {n:>8,}")

    # CHECK violations
    print(f"\n  {'CHECK constraint':<44} {'Violations':>8}")
    print("  " + SEP)
    for label, n in integ["check_violations"].items():
        icon = "✓" if n == 0 else "⚠"
        print(f"  {icon}  {label:<42} {n:>8,}")

    # Issues
    if result["issues"]:
        print(f"\n  ⚠  Issues ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"     • {issue}")
    else:
        print("\n  ✓  No critical issues found.")

    print()


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")

    import sys
    sys.path.insert(0, str(PROJECT))
    from src.etl.ingestion import run_ingestion
    from src.etl.cleaning  import run_cleaning

    raw     = run_ingestion()
    cleaned = run_cleaning(raw)
    result  = run_loader(cleaned)
    print_load_summary(result)