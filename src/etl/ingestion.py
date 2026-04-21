"""
src/etl/ingestion.py
====================
Deliverable 1 — Data Ingestion
-------------------------------
Responsibilities:
  • Load all six raw files into pandas DataFrames.
  • Flatten the nested customers JSON structure into a tabular DataFrame.
  • Handle mixed encodings and BOM (Byte Order Mark) markers.
  • Return a dict of {table_name: DataFrame} with ZERO modifications to the
    values — the raw data is preserved exactly as found on disk.
    Cleaning happens in cleaning.py.

Why keep ingestion separate from cleaning?
  Separating load from transform makes each step independently testable.
  If a new raw file arrives we re-run only this file; cleaning rules
  do not need to change.

Run standalone:
    python -m src.etl.ingestion
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ── Resolve data directory relative to this file ─────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# =============================================================================
# Individual loaders — one function per source file
# =============================================================================

def load_transactions() -> pd.DataFrame:
    """
    Load transactions.csv → DataFrame (all columns as str).

    Encoding note
    -------------
    We read with encoding="utf-8-sig" which transparently strips the UTF-8
    BOM (\xef\xbb\xbf) if present. Without this, the first column header
    would appear as '\\ufefftransaction_id' on systems that emit BOM.

    dtype=str
    ---------
    We load everything as strings so we can inspect the raw values in the
    profiling step before any type coercion causes silent data loss
    (e.g. pd.read_csv auto-parsing "USD 507.90" as NaN).
    """
    path = DATA_DIR / "transactions.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    log.info("  Loaded transactions.csv          → %d rows, %d cols", *df.shape)
    return df


def load_invoices() -> pd.DataFrame:
    """
    Load invoices.csv → DataFrame (all columns as str).

    Same encoding and dtype strategy as transactions.
    invoices.csv also contains currency-prefixed monetary values and
    mixed date formats — preserving as str lets cleaning.py handle them.
    """
    path = DATA_DIR / "invoices.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    log.info("  Loaded invoices.csv              → %d rows, %d cols", *df.shape)
    return df


def load_subscriptions() -> pd.DataFrame:
    """
    Load subscriptions.csv → DataFrame (all columns as str).

    Notable raw issue: the 'mrr' column has numeric values but some rows
    contain negative numbers — loaded as str so cleaning.py can flag and
    correct them explicitly rather than silently losing the sign info.
    """
    path = DATA_DIR / "subscriptions.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    log.info("  Loaded subscriptions.csv         → %d rows, %d cols", *df.shape)
    return df


def load_support_tickets() -> pd.DataFrame:
    """
    Load support_tickets.csv → DataFrame (all columns as str).

    The 'rating' column contains mixed types in the raw file:
    "4", "4/5", "four", "4.0" — loading as str prevents pandas
    from guessing a dtype and converting "four" to NaN automatically.
    """
    path = DATA_DIR / "support_tickets.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    log.info("  Loaded support_tickets.csv       → %d rows, %d cols", *df.shape)
    return df


def load_product_usage() -> pd.DataFrame:
    """
    Load product_usage.csv → DataFrame (all columns as str).

    This is the largest file (~10k rows). session_duration_seconds
    has negative and zero values that would be silently accepted as
    valid floats if we let pandas infer types — loaded as str to
    allow explicit validation in the cleaning step.
    """
    path = DATA_DIR / "product_usage.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    log.info("  Loaded product_usage.csv         → %d rows, %d cols", *df.shape)
    return df


def load_customers() -> pd.DataFrame:
    """
    Load customers_raw.json → flat tabular DataFrame.

    The JSON has two levels of nesting that must be flattened:

        {
          "customer_id": "CUST-0224",
          "name":    { "first": "Kenneth", "last": "Young" },   ← nested dict
          "address": { "city": "Hong Kong", "country": "China", "zip": "73828" }
        }

    pd.json_normalize() handles this in one call, producing columns:
        name.first, name.last, address.city, address.country, address.zip

    We immediately rename these to clean snake_case names so the rest
    of the pipeline never has to deal with dots in column names.

    Encoding note
    -------------
    json.load() reads the file in text mode; Python's default UTF-8
    decoder handles standard UTF-8 files. We pass encoding="utf-8-sig"
    explicitly to strip BOM if present.

    is_active variance
    ------------------
    The raw JSON encodes is_active as bool, string, and int
    simultaneously ("true", "yes", False, 1, "0", "N" …).
    We load all values as-is; normalise_boolean() in cleaning.py
    unifies them to True/False/NaN.
    """
    path = DATA_DIR / "customers_raw.json"

    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)

    # json_normalize flattens nested dicts using '.' as separator
    df = pd.json_normalize(raw)

    # Rename dotted nested keys → readable snake_case
    df.rename(columns={
        "name.first":      "first_name",
        "name.last":       "last_name",
        "address.city":    "city",
        "address.country": "country",
        "address.zip":     "zip_code",
    }, inplace=True)

    log.info("  Loaded customers_raw.json        → %d rows, %d cols", *df.shape)
    return df


# =============================================================================
# Main ingestion function
# =============================================================================

def run_ingestion() -> dict[str, pd.DataFrame]:
    """
    Load all six raw files and return a dict of DataFrames.

    Returns
    -------
    {
        "customers":       pd.DataFrame,
        "transactions":    pd.DataFrame,
        "subscriptions":   pd.DataFrame,
        "invoices":        pd.DataFrame,
        "support_tickets": pd.DataFrame,
        "product_usage":   pd.DataFrame,
    }

    All values are raw strings — no cleaning has been applied.
    """
    log.info("=" * 60)
    log.info("INGESTION — Loading all six raw files")
    log.info("=" * 60)

    dataframes = {
        "customers":       load_customers(),
        "transactions":    load_transactions(),
        "subscriptions":   load_subscriptions(),
        "invoices":        load_invoices(),
        "support_tickets": load_support_tickets(),
        "product_usage":   load_product_usage(),
    }

    log.info("-" * 60)
    log.info("Ingestion complete — %d tables loaded", len(dataframes))

    # Quick shape summary
    log.info("\n  %-22s %8s %6s", "Table", "Rows", "Cols")
    log.info("  " + "-" * 38)
    for name, df in dataframes.items():
        log.info("  %-22s %8d %6d", name, len(df), len(df.columns))

    return dataframes


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")
    dfs = run_ingestion()

    print("\nRaw column names per table:")
    for name, df in dfs.items():
        print(f"  {name}: {list(df.columns)}")

    print("\nSample raw values — transactions (first 3 rows):")
    print(dfs["transactions"][["transaction_id", "transaction_date", "amount", "currency", "status"]].head(3).to_string(index=False))

    print("\nSample raw values — customers is_active column:")
    print("  Unique values:", dfs["customers"]["is_active"].unique()[:10].tolist())