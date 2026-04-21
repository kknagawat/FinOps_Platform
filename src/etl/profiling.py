"""
src/etl/profiling.py
====================
Deliverable 2 — Data Profiling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PK_MAP: dict[str, str] = {
    "customers":       "customer_id",
    "transactions":    "transaction_id",
    "subscriptions":   "subscription_id",
    "invoices":        "invoice_id",
    "support_tickets": "ticket_id",
    "product_usage":   "usage_id",
}


def _profile_column(series: pd.Series) -> dict[str, Any]:
    total  = len(series)
    n_null = int(series.isna().sum())
    n_notna = total - n_null

    info: dict[str, Any] = {
        "dtype":      str(series.dtype),
        "total_rows": total,
        "null_count": n_null,
        "null_pct":   round(n_null / total * 100, 2) if total else 0.0,
        "n_unique":   int(series.nunique(dropna=True)),
    }

    numeric_attempt = pd.to_numeric(series, errors="coerce")
    coercion_rate   = numeric_attempt.notna().sum() / n_notna if n_notna else 0

    if pd.api.types.is_numeric_dtype(series) or coercion_rate > 0.80:
        s = numeric_attempt.dropna() if not pd.api.types.is_numeric_dtype(series) else series.dropna()
        if len(s):
            desc = s.describe()
            info.update({
                "mean":        round(float(desc.get("mean", np.nan)), 4),
                "std":         round(float(desc.get("std",  np.nan)), 4),
                "min":         round(float(desc.get("min",  np.nan)), 4),
                "p50":         round(float(desc.get("50%",  np.nan)), 4),
                "max":         round(float(desc.get("max",  np.nan)), 4),
                "column_kind": "numeric",
            })
        else:
            info["column_kind"] = "numeric_all_null"
    else:
        top5 = series.dropna().value_counts(dropna=True).head(5)
        info.update({
            "top_values":  str(top5.to_dict()),
            "column_kind": "categorical",
        })

    return info


def profile_table(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    pk_col     = PK_MAP.get(table_name)
    n_dup_rows = int(df.duplicated().sum())
    n_dup_pk   = int(df.duplicated(subset=[pk_col]).sum()) if pk_col and pk_col in df.columns else 0

    profile: dict[str, Any] = {
        "table":          table_name,
        "row_count":      len(df),
        "col_count":      len(df.columns),
        "duplicate_rows": n_dup_rows,
        "duplicate_pk":   n_dup_pk,
        "columns":        {},
    }
    for col in df.columns:
        profile["columns"][col] = _profile_column(df[col])

    return profile


def save_profile_csv(profiles: list[dict]) -> Path:
    rows: list[dict] = []
    for p in profiles:
        for col_name, col_stats in p["columns"].items():
            row: dict[str, Any] = {
                "table":          p["table"],
                "row_count":      p["row_count"],
                "col_count":      p["col_count"],
                "duplicate_rows": p["duplicate_rows"],
                "duplicate_pk":   p["duplicate_pk"],
                "column":         col_name,
            }
            row.update(col_stats)
            rows.append(row)

    report_df = pd.DataFrame(rows)
    out_path  = OUTPUT_DIR / "data_profile_report.csv"
    report_df.to_csv(out_path, index=False)
    log.info("  Profile CSV saved → %s", out_path)
    return out_path


def print_profile_summary(profiles: list[dict]) -> None:
    SEP = "─" * 68
    print("\n" + "═" * 68)
    print("  DATA PROFILING SUMMARY  (raw data, before cleaning)")
    print("═" * 68)

    for p in profiles:
        print(f"\n  TABLE: {p['table'].upper()}")
        print(f"  {'Rows':<20} {p['row_count']:>8,}")
        print(f"  {'Cols':<20} {p['col_count']:>8}")
        print(f"  {'Duplicate rows':<20} {p['duplicate_rows']:>8,}")
        print(f"  {'Duplicate PKs':<20} {p['duplicate_pk']:>8,}")
        print()
        print(f"  {'Column':<28} {'Dtype':<10} {'Nulls':>7} {'Null%':>7} {'Unique':>8}")
        print("  " + SEP)
        for col, s in p["columns"].items():
            print(
                f"  {col:<28} {s['dtype']:<10} "
                f"{s['null_count']:>7,} {s['null_pct']:>6.1f}% {s['n_unique']:>8,}"
            )
        print()


def run_profiling(dataframes: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Main entry point — profile all DataFrames, save CSV, print summary.
    Called by run_all.py.
    """
    log.info("=" * 60)
    log.info("PROFILING — Analysing all tables")
    log.info("=" * 60)

    profiles = [
        profile_table(df, name)
        for name, df in dataframes.items()
    ]

    save_profile_csv(profiles)
    print_profile_summary(profiles)

    log.info("Profiling complete.")
    return profiles