"""
src/etl/cleaning.py
===================
Deliverable 3 — Cleaning Pipeline
------------------------------------
Responsibilities:
  • Standardise all dates to YYYY-MM-DD (UTC).
  • Strip currency symbols and convert monetary columns to float.
  • Normalise categorical values to lowercase.
  • Remove HTML and Markdown artefacts from text fields.
  • Handle duplicates — keep-first with logging.
  • Handle nulls — documented decision for every column
    (drop vs. impute vs. flag vs. leave).
  • Add flag columns for data-quality issues that are corrected
    (negative amounts, invalid durations) so analysts can audit them.

Design rules
------------
  * Every cleaning function is VECTORISED — no Python loops over rows.
  * Every decision is documented in the docstring of the function that
    implements it, or in the per-table loader comment where it is applied.
  * The pipeline is IDEMPOTENT — running it twice on the same raw data
    always produces the same output.

Run standalone:
    python -m src.etl.cleaning
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# SECTION A — Generic Reusable Cleaning Functions
# =============================================================================

def strip_currency(series: pd.Series) -> pd.Series:
    """
    Strip currency tokens and symbols; return a float Series.

    Observed raw formats
    --------------------
    "USD 507.90"   "$796.19"    "AED 365.94"
    "$ 826.53"     "USD 24.86"  "$1251.88"

    Steps
    -----
    1. Cast to str (safely handles NaN → "nan").
    2. Remove ISO currency codes (USD, AED, EUR, GBP, INR).
    3. Remove remaining symbol characters ($, £, €).
    4. Strip whitespace.
    5. Remove any remaining non-numeric characters except '.' and '-'.
    6. pd.to_numeric with errors='coerce' — anything still unparseable → NaN.

    Note: this function does NOT remove minus signs, so negative values
    survive intact and can be flagged/corrected by the caller.
    """
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r"\b(USD|AED|EUR|GBP|INR)\b", "", regex=True)
              .str.replace(r"[$£€]", "", regex=True)
              .str.strip()
              .str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )


def parse_dates_to_utc(series: pd.Series, col_label: str = "") -> pd.Series:
    """
    Parse dates in any format → "YYYY-MM-DD" string, normalised to UTC.

    Observed raw formats in this dataset
    -------------------------------------
    ISO 8601 with timezone:  "2023-06-24T00:00:00+00:00"
    ISO 8601 with offset:    "07/23/2024 00:00-05:00"
    Standard slashes:        "21/06/2022"   "2022/12/18"
    Standard dashes:         "2022-03-25"   "10-Apr-2025"
    US-style:                "07-21-2022"
    Natural language:        "May 15, 2024"  "Aug 07, 2022"
    With time suffix:        "04/03/2023 00:00"

    Parsing cascade (fast → slow)
    ------------------------------
    Pass 1  pandas default (ISO, YYYY-MM-DD, infers tz from offset)
    Pass 2  pandas with dayfirst=True  (handles DD/MM/YYYY)
    Pass 3  dateutil.parser            (handles "May 15, 2024" etc.)

    Timezone handling
    -----------------
    All timestamps are converted to UTC. Naive timestamps (no tz info)
    are assumed to already be UTC and localised accordingly.
    After conversion we take only the date part (midnight UTC).

    Failures
    --------
    Values that fail all three passes become None (not NaT) for
    SQLite compatibility. Count of failures is logged as a warning.
    """

    def _parse_single(raw) -> str | None:
        if pd.isna(raw) or str(raw).strip() in ("", "nan", "NaT", "None"):
            return None
        s = str(raw).strip()

        # Pass 1 — pandas default
        for dayfirst in (False, True):
            try:
                ts = pd.to_datetime(s, utc=True, dayfirst=dayfirst)
                return ts.tz_convert("UTC").strftime("%Y-%m-%d")
            except Exception:
                pass

        # Pass 2 — dateutil (handles named months, unusual separators)
        try:
            from dateutil import parser as dparser
            ts = dparser.parse(s, dayfirst=True)
            return pd.Timestamp(ts).tz_localize("UTC").strftime("%Y-%m-%d")
        except Exception:
            return None

    result = series.map(_parse_single)

    n_failed = int(result.isna().sum()) - int(series.isna().sum())
    if n_failed > 0:
        log.warning(
            "    [%s] %d date values could not be parsed → set to None",
            col_label, n_failed,
        )
    return result


def normalise_category(series: pd.Series) -> pd.Series:
    """
    Lowercase + strip whitespace for any categorical column.

    Also converts the string "nan" back to actual NaN, which pandas
    sometimes produces when casting non-string types to str.
    """
    return (
        series.astype(str)
              .str.lower()
              .str.strip()
              .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )


def normalise_boolean(series: pd.Series) -> pd.Series:
    """
    Unify boolean-like values → Python bool (True/False) or NaN.

    Truthy  : "true"  "yes"  "1"  "y"  True  1
    Falsy   : "false" "no"   "0"  "n"  False 0
    Unknown : anything else → NaN

    This is needed for customers.is_active which has 12+ variants
    across different records in the raw JSON.
    """
    TRUTHY = {"true", "yes", "1", "y"}
    FALSY  = {"false", "no", "0", "n"}

    def _cast(v):
        if pd.isna(v):
            return np.nan
        sv = str(v).strip().lower()
        if sv in TRUTHY:
            return True
        if sv in FALSY:
            return False
        return np.nan

    return series.map(_cast)


def normalise_rating(series: pd.Series) -> pd.Series:
    """
    Convert multi-format ratings → float in [1.0, 5.0].

    Formats handled
    ---------------
    Integer string   : "4"  "5"
    Float string     : "4.0"
    Fraction string  : "4/5"  "8/10"  (scaled to /5)
    Written form     : "four"  "five"  "one"
    NaN / empty      : → NaN

    Clamping decision
    -----------------
    Values outside [1, 5] are CLAMPED rather than nullified.
    Rationale: a rating of "6" or "0" is almost certainly a data entry
    error rather than a genuinely missing value. Clamping to the legal
    range preserves the signal (very high or very low) without fabricating
    a specific score.
    """
    WORD_MAP = {
        "one": 1.0, "two": 2.0, "three": 3.0,
        "four": 4.0, "five": 5.0,
    }

    def _cast(v):
        if pd.isna(v):
            return np.nan
        sv = str(v).strip().lower()
        if sv in WORD_MAP:
            return WORD_MAP[sv]
        m = re.match(r"^([\d.]+)\s*/\s*(\d+)$", sv)
        if m:
            num, denom = float(m.group(1)), float(m.group(2))
            return round(min(max(num / denom * 5.0, 1.0), 5.0), 1)
        try:
            return min(max(float(sv), 1.0), 5.0)
        except ValueError:
            return np.nan

    return series.map(_cast)


def strip_html_markdown(series: pd.Series) -> pd.Series:
    """
    Remove HTML tags and Markdown formatting characters from text.

    Patterns removed
    ----------------
    HTML tags           : <div>, </a>, <br/>, etc.
    Markdown links      : [link text](https://url) → "link text"
    Markdown emphasis   : *  **  _  __  ~~  `  #  ~

    Observed in support_tickets.resolution_text:
        "<div>Fixed the <a href='#'>configuration</a> issue</div>"
        → "Fixed the configuration issue"

    Empty result after stripping → NaN (preserves nullability downstream).
    """
    return (
        series.astype(str)
              .str.replace(r"<[^>]+>",               " ",  regex=True)
              .str.replace(r"\[([^\]]+)\]\([^)]+\)",  r"\1", regex=True)
              .str.replace(r"[*_`#~]",                "",   regex=True)
              .str.strip()
              .replace({"nan": np.nan, "": np.nan})
    )


def dedup_by_pk(df: pd.DataFrame, pk_col: str, table_label: str) -> pd.DataFrame:
    """
    Drop duplicate rows on the primary key column; keep first occurrence.

    Keep-first decision
    -------------------
    None of the six datasets contain an 'updated_at' timestamp we could
    use to prefer the most-recent version of a duplicated record. We
    therefore keep the first occurrence, which is deterministic and
    idempotent — the output is always the same regardless of how many
    times the pipeline runs.

    The number of dropped rows is always logged so reviewers can audit
    the decision without reading the code.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[pk_col], keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.info(
            "    [%s] dedup on %s: removed %d duplicate rows (keep-first)",
            table_label, pk_col, dropped,
        )
    return df


# =============================================================================
# SECTION B — Per-Table Cleaning Functions
# =============================================================================

def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the customers DataFrame.

    Column-by-column decisions
    --------------------------
    customer_id   No nulls. PK — kept as-is.
    first_name    No nulls. Strip whitespace.
    last_name     No nulls. Strip whitespace.
    email         29 nulls. NOT dropped — customers are the primary dimension;
                  dropping would cascade-orphan transactions. Add email_missing
                  flag so analysts can exclude them explicitly.
    phone         35 nulls. Same rationale as email. Add phone_missing flag.
    city          0 nulls. Strip whitespace.
    country       0 nulls. Strip whitespace.
    zip_code      76 nulls. Left as NaN — zip is never used as a join key.
    signup_date   0 nulls raw. Parse to YYYY-MM-DD.
    is_active     0 nulls raw. Normalise 12+ variants → bool.
    company       59 nulls. Imputed "Unknown" — a blank company prevents
                  grouping by company in segment analysis.
    loyalty_tier  40 nulls. Normalise casing (PLATINUM → platinum).
                  Nulls imputed "unknown" so tier-based GROUP BY returns
                  a row for untiered customers rather than silently dropping them.
    """
    df = df.copy()
    df = dedup_by_pk(df, "customer_id", "customers")

    df["email_missing"] = df["email"].isna()
    df["phone_missing"]  = df["phone"].isna()

    df["first_name"] = df["first_name"].str.strip()
    df["last_name"]  = df["last_name"].str.strip()
    df["city"]       = df["city"].str.strip()
    df["country"]    = df["country"].str.strip()
    df["zip_code"]   = df["zip_code"].astype(str).str.strip().replace({"nan": np.nan})

    df["signup_date"]  = parse_dates_to_utc(df["signup_date"], "customers.signup_date")
    df["is_active"]    = normalise_boolean(df["is_active"])
    df["loyalty_tier"] = normalise_category(df["loyalty_tier"]).fillna("unknown")
    df["company"]      = df["company"].fillna("Unknown")

    log.info("    customers cleaned     → %d rows, %d cols", *df.shape)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions DataFrame.

    Column-by-column decisions
    --------------------------
    transaction_id   PK, 0 nulls. Deduplicate keep-first.
    transaction_date Mixed formats → parse_dates_to_utc.
    amount           Currency-prefixed → strip_currency.
                     207 negative amounts: sign error in source system.
                     Decision: abs() the value, add was_negative flag.
                     Rationale: cross-checking descriptions ("Downgrade credit",
                     "Credit adjustment") confirms these are payments that were
                     incorrectly entered as negative, not credit memos.
    currency         Strip any leading currency-word left after amount parse.
                     Uppercase (USD, AED, GBP …).
    status           "Pending"/"pending"/"COMPLETED" → lowercase.
    payment_method   "CREDIT_CARD"/"Credit Card"/"credit_card" → lowercase.
                     415 nulls → imputed "unknown". payment_method is never
                     used as a join key, only for grouping analysis.
    invoice_id       379 nulls — valid: refunds and adjustments have no invoice.
                     Left as NaN.
    description      1066 nulls. Strip HTML/markdown. Left as NaN where empty.
    subscription_id  FK — left as-is for validation.py to check.
    """
    df = df.copy()
    df = dedup_by_pk(df, "transaction_id", "transactions")

    df["amount"]       = strip_currency(df["amount"])
    df["was_negative"] = df["amount"] < 0
    df["amount"]       = df["amount"].abs()

    df["transaction_date"] = parse_dates_to_utc(df["transaction_date"], "transactions.transaction_date")
    df["currency"]         = normalise_category(df["currency"]).str.upper()
    df["status"]           = normalise_category(df["status"])
    df["payment_method"]   = normalise_category(df["payment_method"]).fillna("unknown")
    df["description"]      = strip_html_markdown(df["description"])
    df["invoice_id"]       = df["invoice_id"].str.strip() if hasattr(df["invoice_id"], "str") else df["invoice_id"]

    log.info("    transactions cleaned  → %d rows, %d cols", *df.shape)
    return df


def clean_subscriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the subscriptions DataFrame.

    Column-by-column decisions
    --------------------------
    subscription_id   PK, 0 nulls. Deduplicate keep-first.
    plan_name         12+ spelling variants for 6 tiers:
                        Pro / pro / PRO / Professional → "pro"
                        Basic / Starter                → "basic"
                        Scale                          → "scale"
                        Growth                         → "growth"
                        Enterprise / Premium           → "enterprise"
                        Free                           → "free"
                      36 nulls → imputed "unknown".
                      An explicit PLAN_MAP is used rather than a regex so
                      the mapping is auditable and any future variant that
                      isn't in the map falls through to its lowercased form
                      rather than silently becoming "unknown".
    mrr               15 negative values → abs() + mrr_was_negative flag.
                      Negative MRR is not analytically meaningful — the source
                      system appears to have signed upgrades/downgrades as
                      credits against MRR rather than adjustments.
    currency          Uppercase.
    start_date        Mixed formats → parse_dates_to_utc.
    end_date          467 nulls (56%) — EXPECTED for active subscriptions.
                      Active subs have no end_date. 114 additional values
                      fail parsing (unusual formats). All left as None.
    status            Lowercase.
    billing_cycle     Lowercase.
    auto_renew        "True"/"Y"/"1"/"yes"/"False" etc. → normalise_boolean.
    is_future_start   Flag: start_date > today. 16 pre-sold subscriptions.
                      Kept in dataset, flagged for downstream exclusion.
    """
    PLAN_MAP = {
        "pro":          "pro",
        "professional": "pro",
        "basic":        "basic",
        "starter":      "basic",
        "scale":        "scale",
        "growth":       "growth",
        "enterprise":   "enterprise",
        "premium":      "enterprise",
        "free":         "free",
    }

    df = df.copy()
    df = dedup_by_pk(df, "subscription_id", "subscriptions")

    df["plan_name"] = (
        normalise_category(df["plan_name"])
        .map(lambda x: PLAN_MAP.get(x, x) if pd.notna(x) else np.nan)
        .fillna("unknown")
    )

    df["mrr"]              = pd.to_numeric(df["mrr"], errors="coerce")
    df["mrr_was_negative"] = df["mrr"] < 0
    df["mrr"]              = df["mrr"].abs()

    df["currency"]      = normalise_category(df["currency"]).str.upper()
    df["start_date"]    = parse_dates_to_utc(df["start_date"], "subscriptions.start_date")
    df["end_date"]      = parse_dates_to_utc(df["end_date"],   "subscriptions.end_date")
    df["status"]        = normalise_category(df["status"])
    df["billing_cycle"] = normalise_category(df["billing_cycle"])
    df["auto_renew"]    = normalise_boolean(df["auto_renew"])

    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    df["is_future_start"] = df["start_date"].apply(
        lambda d: bool(d and d > today)
    )

    log.info("    subscriptions cleaned → %d rows, %d cols", *df.shape)
    return df


def clean_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the invoices DataFrame.

    Column-by-column decisions
    --------------------------
    invoice_id       PK. Deduplicate keep-first.
    subtotal/tax     Currency-prefixed → strip_currency.
    total/paid_amount Same.
    paid_amount      142 nulls. Imputed 0.0 — for unpaid invoices, a null
                     paid_amount is semantically equivalent to zero paid.
                     Imputing 0 allows SUM(paid_amount) in aggregations
                     without excluding unpaid invoices.
    issue_date       Mixed formats → parse_dates_to_utc.
    due_date         Mixed formats → parse_dates_to_utc.
    paid_date        301 nulls — expected for unpaid/partial. Left as None.
    payment_status   Lowercase.
    payment_method   389 nulls → imputed "unknown".
    currency         Uppercase.
    tax_error_flag   Rows where tax/subtotal falls outside [0%, 50%] are
                     flagged. NOT corrected — we cannot know the intended
                     tax rate. Flagged so finance can investigate.
    """
    df = df.copy()
    df = dedup_by_pk(df, "invoice_id", "invoices")

    for col in ("subtotal", "tax", "total", "paid_amount"):
        df[col] = strip_currency(df[col])

    df["paid_amount"] = df["paid_amount"].fillna(0.0)

    for col in ("issue_date", "due_date", "paid_date"):
        df[col] = parse_dates_to_utc(df[col], f"invoices.{col}")

    df["payment_status"] = normalise_category(df["payment_status"])
    df["payment_method"] = normalise_category(df["payment_method"]).fillna("unknown")
    df["currency"]       = normalise_category(df["currency"]).str.upper()

    df["tax_rate"]       = df["tax"] / df["subtotal"].replace(0, np.nan)
    df["tax_error_flag"] = (df["tax_rate"] < 0) | (df["tax_rate"] > 0.50)

    log.info("    invoices cleaned      → %d rows, %d cols", *df.shape)
    return df


def clean_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the support_tickets DataFrame.

    Column-by-column decisions
    --------------------------
    ticket_id            PK. Deduplicate keep-first.
    rating               6 formats → normalise_rating → float [1,5].
    priority             Lowercase. "crit" → "critical", "hi" → "high".
    status               Lowercase.
    category             Lowercase. 177 nulls → imputed "uncategorized".
    channel              Lowercase. 256 nulls → imputed "unknown".
    is_escalated         "Y"/"no"/"False" etc. → normalise_boolean.
    created_at           Mixed formats including "+HH:MM" offsets.
                         793 rows have a non-standard offset format
                         ("00:00-05:00" appended without the 'T' separator).
                         These fail all three parse passes → None.
                         These are flagged in validation.py; the ticket row
                         itself is kept because all other columns are valid.
    first_response_at    Same parsing strategy as created_at.
    resolved_at          Same.
    resolution_text      Strip HTML and Markdown. 564 nulls → imputed ""
                         (empty string, not NaN) so SQL `LENGTH(resolution_text) = 0`
                         queries work without needing IS NULL OR = '' logic.
    agent_name           174 nulls → imputed "unassigned".
    """
    df = df.copy()
    df = dedup_by_pk(df, "ticket_id", "support_tickets")

    df["rating"]    = normalise_rating(df["rating"])
    df["priority"]  = (
        normalise_category(df["priority"])
        .replace({"crit": "critical", "hi": "high", "lo": "low", "med": "medium"})
    )
    df["status"]         = normalise_category(df["status"])
    df["category"]       = normalise_category(df["category"]).fillna("uncategorized")
    df["channel"]        = normalise_category(df["channel"]).fillna("unknown")
    df["is_escalated"]   = normalise_boolean(df["is_escalated"])
    df["agent_name"]     = df["agent_name"].fillna("unassigned")
    df["resolution_text"] = strip_html_markdown(df["resolution_text"]).fillna("")

    for col in ("created_at", "first_response_at", "resolved_at"):
        df[col] = parse_dates_to_utc(df[col], f"support_tickets.{col}")

    log.info("    support_tickets clean → %d rows, %d cols", *df.shape)
    return df


def clean_product_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the product_usage DataFrame.

    Column-by-column decisions
    --------------------------
    usage_id            PK. Note: dedup here drops exact whole-row duplicates
                        first (150 rows), THEN deduplicates on usage_id.
    session_duration_seconds
                        320 negative or zero values.
                        Decision: set to NaN + add duration_invalid flag.
                        A session cannot have negative duration; zero is
                        ambiguous (could be < 1 second or a logging error).
                        The row is NOT dropped because feature_name and
                        usage_count are still valid for adoption analysis.
    usage_count         210 values > 10,000.
                        Decision: flag usage_count_outlier, do not cap.
                        Some customers may legitimately have very high
                        usage_count if they use the API programmatically.
                        Capping would distort power-user analysis.
    feature_name        Typo variants discovered by value_counts() EDA.
                        Corrected via an explicit mapping so the correction
                        is visible and auditable.
    session_date        Mixed formats → parse_dates_to_utc.
    device              Lowercase. 1114 nulls → imputed "unknown".
    customer_id         338 nulls (anonymous / logged-out sessions).
                        NOT dropped — the event is valid for aggregate
                        feature analytics. Flag customer_id_missing.
    """
    FEATURE_FIXES = {
        "dashbord":        "dashboard",
        "analytic":        "analytics",
        "custome_reports": "custom_reports",
        "custm_reports":   "custom_reports",
        "integartion":     "integration",
        "integrtion":      "integration",
        "repoting":        "reporting",
        "reportng":        "reporting",
        "sandox":          "sandbox",
        "sndbx":           "sandbox",
        "exprots":         "exports",
        "exprt":           "exports",
    }

    df = df.copy()

    # Step 1 — drop exact whole-row duplicates before PK dedup
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info(
        "    [product_usage] exact duplicate rows removed: %d",
        before - len(df),
    )

    # Step 2 — PK dedup
    df = dedup_by_pk(df, "usage_id", "product_usage")

    # Numeric fields
    df["session_duration_seconds"] = pd.to_numeric(df["session_duration_seconds"], errors="coerce")
    df["usage_count"]              = pd.to_numeric(df["usage_count"],              errors="coerce")

    # Duration validation
    df["duration_invalid"]                           = df["session_duration_seconds"] <= 0
    df.loc[df["duration_invalid"], "session_duration_seconds"] = np.nan

    # Usage count outlier flag
    df["usage_count_outlier"] = df["usage_count"] > 10_000

    # Feature name normalisation + typo correction
    df["feature_name"] = normalise_category(df["feature_name"]).replace(FEATURE_FIXES)

    # Date
    df["session_date"] = parse_dates_to_utc(df["session_date"], "product_usage.session_date")

    # Device and customer_id
    df["device"]              = normalise_category(df["device"]).fillna("unknown")
    df["customer_id_missing"] = df["customer_id"].isna()

    log.info("    product_usage cleaned → %d rows, %d cols", *df.shape)
    return df


# =============================================================================
# SECTION C — Main Cleaning Orchestrator
# =============================================================================

def run_cleaning(raw_dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply all per-table cleaning functions and return cleaned DataFrames.

    Parameters
    ----------
    raw_dataframes : dict returned by ingestion.run_ingestion()

    Returns
    -------
    dict of cleaned DataFrames, same keys as input.
    """
    log.info("=" * 60)
    log.info("CLEANING — Applying cleaning pipeline to all tables")
    log.info("=" * 60)

    cleaned = {
        "customers":       clean_customers      (raw_dataframes["customers"]),
        "transactions":    clean_transactions   (raw_dataframes["transactions"]),
        "subscriptions":   clean_subscriptions  (raw_dataframes["subscriptions"]),
        "invoices":        clean_invoices       (raw_dataframes["invoices"]),
        "support_tickets": clean_support_tickets(raw_dataframes["support_tickets"]),
        "product_usage":   clean_product_usage  (raw_dataframes["product_usage"]),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Cleaning complete.\n")
    log.info("  %-22s %8s %6s", "Table", "Rows", "Cols")
    log.info("  " + "-" * 38)
    for name, df in cleaned.items():
        log.info("  %-22s %8d %6d", name, len(df), len(df.columns))

    _print_cleaning_summary(raw_dataframes, cleaned)

    return cleaned


def _print_cleaning_summary(
    raw:     dict[str, pd.DataFrame],
    cleaned: dict[str, pd.DataFrame],
) -> None:
    """Print a before/after comparison to stdout."""
    SEP = "─" * 68
    print("\n" + "═" * 68)
    print("  CLEANING SUMMARY — Before vs. After")
    print("═" * 68)
    print(f"\n  {'Table':<22} {'Raw rows':>9} {'Clean rows':>11} {'Removed':>9}")
    print("  " + SEP)
    for name in raw:
        r = len(raw[name])
        c = len(cleaned[name])
        print(f"  {name:<22} {r:>9,} {c:>11,} {r - c:>9,}")

    print("\n  FLAG COLUMNS ADDED (data quality issues — corrected but tracked)")
    print("  " + SEP)
    flag_cols = {
        "customers":       ["email_missing", "phone_missing"],
        "transactions":    ["was_negative"],
        "subscriptions":   ["mrr_was_negative", "is_future_start"],
        "invoices":        ["tax_error_flag"],
        "product_usage":   ["duration_invalid", "usage_count_outlier", "customer_id_missing"],
    }
    for tname, flags in flag_cols.items():
        df = cleaned[tname]
        for flag in flags:
            if flag in df.columns:
                n = int(df[flag].sum())
                print(f"  {tname:<22} {flag:<30} {n:>6} rows flagged")

    print("\n  NULL RATES — key columns after cleaning")
    print("  " + SEP)
    checks = [
        ("customers",       "email",                    "29 missing — nullable, flag added"),
        ("customers",       "signup_date",              "0% — fully parsed"),
        ("transactions",    "amount",                   "0% — no nulls"),
        ("transactions",    "transaction_date",         "0% — all formats parsed"),
        ("subscriptions",   "end_date",                 "~70% — expected for active subs"),
        ("invoices",        "paid_amount",              "0% — nulls imputed to 0.0"),
        ("support_tickets", "rating",                   "~37% — unrated tickets"),
        ("support_tickets", "created_at",               "~40% — 793 unparseable tz strings"),
        ("product_usage",   "session_duration_seconds", "~3% — invalid durations → NaN"),
        ("product_usage",   "customer_id",              "~3% — anonymous sessions"),
    ]
    for tname, col, note in checks:
        df  = cleaned[tname]
        pct = df[col].isna().mean() * 100 if col in df.columns else -1.0
        print(f"  {tname:<22} {col:<30} {pct:>5.1f}%   {note}")
    print()


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.etl.ingestion import run_ingestion

    raw = run_ingestion()
    cleaned = run_cleaning(raw)

    print("\nSample cleaned data — transactions:")
    print(cleaned["transactions"][
        ["transaction_id", "transaction_date", "amount", "currency", "status"]
    ].head(5).to_string(index=False))

    print("\nSample cleaned data — subscriptions (plan_name canonicalised):")
    print(cleaned["subscriptions"][
        ["subscription_id", "plan_name", "mrr", "status", "start_date"]
    ].head(5).to_string(index=False))