"""
src/agent/tools.py
==================
Step 4 — All 7 LangChain Agent Tools

Each tool has:
  • A Pydantic v2 BaseModel input schema
  • A comprehensive docstring used as the tool description by the agent
  • Full error handling with informative return messages
  • No circular imports — tools load DB / DataFrames lazily

Tool registry
-------------
  1. sql_query_tool          — NL → SQL → SQLite, SELECT-only, auto-retry
  2. revenue_calculator_tool — MRR / ARR / ARPC / growth (pure pandas, no SQL)
  3. customer_segmentation_tool — RFM, churn-risk, usage-tier segmentation
  4. anomaly_detection_tool  — z-score / IQR / rolling anomaly detection
  5. knowledge_retrieval_tool — keyword + TF-IDF search over policy docs
  6. chart_generator_tool    — matplotlib bar/line/pie charts → PNG
  7. forecast_tool (bonus)   — linear trend / exp-smoothing forecast
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Project paths ─────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve()
PROJECT      = _HERE.parents[2]
DB_PATH      = PROJECT / "finops.db"
DOCS_DIR     = PROJECT / "docs"
CHARTS_DIR   = PROJECT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Dataset reference date (anchored so windows always return data) ───────────
DATASET_MAX_DATE = "2025-12-31"

# ── Database schema string injected into SQL-generation prompts ───────────────
DB_SCHEMA = """
SQLite database: finops.db

TABLES
------
customers(customer_id TEXT PK, first_name TEXT, last_name TEXT, email TEXT,
          phone TEXT, city TEXT, country TEXT, zip_code TEXT, signup_date TEXT,
          is_active INTEGER, company TEXT, loyalty_tier TEXT)

subscriptions(subscription_id TEXT PK, customer_id TEXT FK→customers,
              plan_name TEXT, mrr REAL, currency TEXT,
              start_date TEXT, end_date TEXT, status TEXT,
              billing_cycle TEXT, auto_renew INTEGER)

transactions(transaction_id TEXT PK, customer_id TEXT FK→customers,
             subscription_id TEXT FK→subscriptions,
             transaction_date TEXT, amount REAL, currency TEXT,
             status TEXT, payment_method TEXT, invoice_id TEXT, description TEXT)

invoices(invoice_id TEXT PK, customer_id TEXT FK→customers,
         subscription_id TEXT FK→subscriptions,
         issue_date TEXT, due_date TEXT, subtotal REAL, tax REAL, total REAL,
         paid_amount REAL, payment_status TEXT, paid_date TEXT,
         payment_method TEXT, currency TEXT)

support_tickets(ticket_id TEXT PK, customer_id TEXT FK→customers,
                category TEXT, priority TEXT, status TEXT,
                created_at TEXT, first_response_at TEXT, resolved_at TEXT,
                rating REAL, resolution_text TEXT, agent_name TEXT,
                channel TEXT, is_escalated INTEGER)

product_usage(usage_id TEXT PK, customer_id TEXT FK→customers,
              feature_name TEXT, session_date TEXT,
              session_duration_seconds REAL, usage_count REAL,
              device TEXT, session_id TEXT)

date_dim(date_id INTEGER PK, full_date TEXT, year INTEGER, quarter INTEGER,
         month INTEGER, month_name TEXT, week_of_year INTEGER,
         day_of_month INTEGER, day_of_week INTEGER, day_name TEXT,
         is_weekend INTEGER, is_month_start INTEGER, is_month_end INTEGER,
         fiscal_quarter TEXT)

KEY VALUES
----------
transactions.status   : 'completed', 'pending', 'failed', 'cancelled', 'refunded'
subscriptions.status  : 'active', 'cancelled', 'expired'
subscriptions.plan_name: 'basic', 'pro', 'scale', 'growth', 'enterprise', 'free', 'unknown'
invoices.payment_status: 'paid', 'unpaid', 'partial', 'overdue'
support_tickets.priority: 'critical', 'high', 'medium', 'low'

DATES: all stored as TEXT 'YYYY-MM-DD'.
DATASET DATE RANGE: transactions 2022-01-01 → 2025-12-31
Use date('2025-12-31', '-N days') for "last N days" instead of date('now').
"""


# =============================================================================
# Helper: database connection
# =============================================================================

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# =============================================================================
# Tool 1: sql_query_tool
# =============================================================================

class SQLQueryInput(BaseModel):
    question: str = Field(..., description="Natural language question to answer with SQL")
    max_rows: int = Field(50, ge=1, le=500, description="Maximum rows to return (default 50)")


@tool("sql_query_tool", args_schema=SQLQueryInput)
def sql_query_tool(question: str, max_rows: int = 50) -> dict:
    """
    Generate and execute a SQL SELECT query against the FinOps SQLite database.

    Use this tool for:
    - Looking up specific records (top customers, specific transactions, etc.)
    - Counting, summing, or grouping data across tables
    - Questions about subscriptions, invoices, support tickets, product usage
    - Any question that needs data directly from the database

    The tool will:
    1. Generate SQL from your question using the LLM with full schema context
    2. Enforce SELECT-only (blocks INSERT/UPDATE/DELETE/DROP)
    3. Auto-retry once with error context if the first SQL fails
    4. Return rows as a list of dicts plus the SQL used

    Returns: {sql, rows, columns, data, error (if any)}
    """
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-opus-4-5", temperature=0, max_tokens=1024)
    mutating_re = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE)\b",
        re.IGNORECASE,
    )

    def _generate_sql(q: str, error_hint: str = "") -> str:
        error_section = f"\nPrevious attempt failed with: {error_hint}\nFix the error.\n" if error_hint else ""
        prompt = (
            f"You are a SQLite expert. Write a single SELECT SQL query that answers the question.\n"
            f"Return ONLY the SQL — no explanation, no markdown, no backticks.\n\n"
            f"Schema:\n{DB_SCHEMA}\n"
            f"{error_section}"
            f"Question: {q}\n\nSQL:"
        )
        raw = llm.invoke(prompt).content.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    last_error = ""
    for attempt in range(2):
        sql = _generate_sql(question, last_error)

        if mutating_re.search(sql):
            return {"error": "Generated SQL contains mutating statements — blocked for safety.", "sql": sql}

        try:
            con = _conn()
            cur = con.execute(sql)
            rows = cur.fetchmany(max_rows)
            cols = [d[0] for d in cur.description] if cur.description else []
            con.close()
            data = [dict(zip(cols, r)) for r in rows]
            return {"sql": sql, "rows": len(data), "columns": cols, "data": data}
        except sqlite3.Error as e:
            last_error = str(e)
            log.warning("SQL attempt %d failed: %s", attempt + 1, e)

    return {"error": f"SQL failed after 2 attempts. Last error: {last_error}", "sql": sql}


# =============================================================================
# Tool 2: revenue_calculator_tool
# =============================================================================

class RevenueCalculatorInput(BaseModel):
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: str   = Field(..., description="End date YYYY-MM-DD")
    granularity: Literal["daily", "weekly", "monthly"] = Field(
        "monthly", description="Time granularity for the timeseries"
    )
    metric: Literal["mrr", "arr", "arpc", "growth_rate", "all"] = Field(
        "all", description="Which metric to compute. 'all' returns everything."
    )


@tool("revenue_calculator_tool", args_schema=RevenueCalculatorInput)
def revenue_calculator_tool(
    start_date: str,
    end_date: str,
    granularity: str = "monthly",
    metric: str = "all",
) -> dict:
    """
    Compute revenue metrics using pure Python/pandas — does NOT use SQL.

    Use this tool for:
    - "What is our current MRR / ARR?"
    - "How has revenue changed over the last N months?"
    - "What is our average revenue per customer (ARPC)?"
    - "Show me revenue growth rate"
    - Any question about revenue figures, trends, or comparisons

    Metrics computed:
    - MRR  : Monthly Recurring Revenue from active subscriptions
    - ARR  : MRR × 12
    - ARPC : MRR / number of active customers
    - growth_rate: period-over-period revenue change %
    - timeseries: revenue aggregated at daily/weekly/monthly granularity

    Returns: {mrr, arr, arpc, growth_rate_pct, timeseries, period, n_customers}
    """
    try:
        con = _conn()

        # ── Active subscription MRR ───────────────────────────────────────────
        subs_df = pd.read_sql(
            "SELECT customer_id, mrr FROM subscriptions WHERE status='active'", con
        )
        mrr  = float(subs_df["mrr"].sum())
        arr  = mrr * 12
        n_cust = int(subs_df["customer_id"].nunique())
        arpc = mrr / n_cust if n_cust > 0 else 0.0

        # ── Timeseries revenue ────────────────────────────────────────────────
        txn_df = pd.read_sql(
            "SELECT transaction_date, amount FROM transactions "
            "WHERE status='completed' AND transaction_date BETWEEN ? AND ?",
            con, params=[start_date, end_date],
        )
        con.close()

        txn_df["transaction_date"] = pd.to_datetime(txn_df["transaction_date"])
        txn_df = txn_df.set_index("transaction_date")

        freq_map = {"daily": "D", "weekly": "W", "monthly": "ME"}
        freq     = freq_map.get(granularity, "ME")
        ts       = txn_df["amount"].resample(freq).sum().reset_index()
        ts.columns = ["period", "revenue"]
        ts["period"] = ts["period"].dt.strftime("%Y-%m-%d")

        # ── Growth rate (last period vs. prior period) ────────────────────────
        values = ts["revenue"].values
        growth_rate = None
        if len(values) >= 2 and values[-2] > 0:
            growth_rate = round((values[-1] - values[-2]) / values[-2] * 100, 2)

        return {
            "mrr":              round(mrr, 2),
            "arr":              round(arr, 2),
            "arpc":             round(arpc, 2),
            "n_active_customers": n_cust,
            "growth_rate_pct":  growth_rate,
            "period":           f"{start_date} → {end_date}",
            "granularity":      granularity,
            "timeseries":       ts.to_dict(orient="records"),
        }

    except Exception as e:
        log.error("revenue_calculator_tool error: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 3: customer_segmentation_tool
# =============================================================================

class SegmentationInput(BaseModel):
    segmentation_type: Literal["rfm", "churn_risk", "usage_tier", "health_score"] = Field(
        ..., description="Type of segmentation to apply"
    )
    top_n: int = Field(10, ge=1, le=100, description="Top N customers to return per segment")


@tool("customer_segmentation_tool", args_schema=SegmentationInput)
def customer_segmentation_tool(segmentation_type: str, top_n: int = 10) -> dict:
    """
    Segment active customers using Python/pandas — does NOT generate SQL.

    Use this tool for:
    - "Segment customers using RFM analysis"
    - "Which customers are at high churn risk?"
    - "Show me customer usage tiers"
    - "What is the RFM distribution of our customer base?"

    Segmentation types:
    - rfm        : Recency / Frequency / Monetary quintile scoring
                   Segments: Champions, Loyal, Needs Attention, At Risk, Hibernating
    - churn_risk : Multi-signal churn scoring (usage decline, ticket spike, no login)
                   Segments: High Risk, Medium Risk, Low Risk, Healthy
    - usage_tier : Session-count tiers over last 90 days
                   Segments: Power User, Regular, Casual, Dormant
    - health_score: Composite 0-100 score (recency + support + usage + payment)

    Returns: {segmentation_type, distribution, top_customers, total_customers}
    """
    try:
        con     = _conn()
        ref     = DATASET_MAX_DATE
        ref_dt  = datetime.strptime(ref, "%Y-%m-%d")
        d30     = (ref_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        d60     = (ref_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        d90     = (ref_dt - timedelta(days=90)).strftime("%Y-%m-%d")
        d14     = (ref_dt - timedelta(days=14)).strftime("%Y-%m-%d")

        cust_df = pd.read_sql(
            "SELECT customer_id, first_name||' '||last_name AS name, company "
            "FROM customers WHERE is_active=1", con
        )

        # ── RFM ──────────────────────────────────────────────────────────────
        if segmentation_type == "rfm":
            txn_df = pd.read_sql(
                "SELECT customer_id, transaction_date, amount FROM transactions "
                "WHERE status='completed'", con
            )
            con.close()
            txn_df["transaction_date"] = pd.to_datetime(txn_df["transaction_date"])
            ref_ts = pd.Timestamp(ref)

            rfm = txn_df.groupby("customer_id").agg(
                recency_days=("transaction_date", lambda x: (ref_ts - x.max()).days),
                frequency   =("transaction_id" if "transaction_id" in txn_df.columns else "amount", "count"),
                monetary    =("amount", "sum"),
            ).reset_index()

            # Quintile score: 5=best for freq/monetary, 5=best (low days) for recency
            for col, ascending in [("recency_days", False), ("frequency", True), ("monetary", True)]:
                label = col[0].upper()  # R, F, M
                try:
                    rfm[f"{label}_score"] = pd.qcut(
                        rfm[col].rank(method="first"), 5,
                        labels=[1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
                    ).astype(int)
                except Exception:
                    rfm[f"{label}_score"] = 3

            rfm["rfm_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
            rfm["segment"]   = pd.cut(
                rfm["rfm_total"],
                bins=[-1, 5, 8, 10, 12, 15],
                labels=["Hibernating", "At Risk", "Needs Attention", "Loyal", "Champions"],
            ).astype(str)

            rfm = rfm.merge(cust_df, on="customer_id", how="left")
            dist = rfm["segment"].value_counts().to_dict()
            top  = rfm.nlargest(top_n, "rfm_total")[
                ["customer_id","name","segment","rfm_total","recency_days","frequency","monetary"]
            ].to_dict(orient="records")
            return {"segmentation_type":"rfm", "distribution": dist,
                    "top_customers": top, "total_customers": len(rfm)}

        # ── Churn risk ────────────────────────────────────────────────────────
        elif segmentation_type == "churn_risk":
            usage_df   = pd.read_sql("SELECT customer_id, session_date FROM product_usage WHERE customer_id IS NOT NULL", con)
            tickets_df = pd.read_sql("SELECT customer_id, created_at FROM support_tickets", con)
            con.close()

            usage_df["session_date"] = pd.to_datetime(usage_df["session_date"], errors="coerce")

            s30  = usage_df[usage_df["session_date"] >= d30].groupby("customer_id").size().rename("s30")
            s60  = usage_df[(usage_df["session_date"] >= d60) & (usage_df["session_date"] < d30)].groupby("customer_id").size().rename("s60")
            last = usage_df.groupby("customer_id")["session_date"].max().rename("last_login")

            tickets_df["created_at"] = pd.to_datetime(tickets_df["created_at"], errors="coerce")
            t30  = tickets_df[tickets_df["created_at"] >= d30].groupby("customer_id").size().rename("t30")
            tall = tickets_df.groupby("customer_id").size().rename("t_all")

            df = cust_df.set_index("customer_id")
            df = df.join(s30).join(s60).join(last).join(t30).join(tall)
            df["s30"] = df.get("s30", pd.Series(0, index=df.index)).fillna(0)
            df["s60"] = df.get("s60", pd.Series(0, index=df.index)).fillna(0)
            df["t30"] = df.get("t30", pd.Series(0, index=df.index)).fillna(0)
            df["t_all"] = df.get("t_all", pd.Series(0, index=df.index)).fillna(0)
            df["last_login"] = df.get("last_login", pd.NaT)

            df["usage_decline"] = (df.get("s30", 0) < df.get("s60", 0)).astype(int)
            df["ticket_spike"]  = ((df.get("t30", 0) > df.get("t_all", 0) / 6 * 1.5) & (df.get("t30", 0) > 0)).astype(int)
            df["no_login"] = df["last_login"].apply(lambda x: 1 if pd.isna(x) or x < pd.Timestamp(d14) else 0)
            df["risk_score"]    = df["usage_decline"] + df["ticket_spike"] + df["no_login"]
            df["segment"]       = pd.cut(df["risk_score"], bins=[-1,0,1,2,3],
                                         labels=["Healthy","Low Risk","Medium Risk","High Risk"]).astype(str)

            dist = df["segment"].value_counts().to_dict()
            top  = df.nlargest(top_n, "risk_score").reset_index()[
                ["customer_id","name","segment","risk_score"]
            ].to_dict(orient="records")
            return {"segmentation_type":"churn_risk", "distribution": dist,
                    "top_at_risk": top, "total_customers": len(df)}

        # ── Usage tier ────────────────────────────────────────────────────────
        elif segmentation_type == "usage_tier":
            usage_df = pd.read_sql(
                f"SELECT customer_id, session_id, feature_name FROM product_usage "
                f"WHERE customer_id IS NOT NULL AND session_date >= '{d90}'", con
            )
            con.close()
            agg = usage_df.groupby("customer_id").agg(
                sessions=("session_id","count"),
                features=("feature_name","nunique"),
            ).reset_index()
            agg["tier"] = pd.cut(agg["sessions"], bins=[-1,4,15,49,9999],
                                 labels=["Dormant","Casual","Regular","Power User"]).astype(str)
            agg = agg.merge(cust_df, on="customer_id", how="left")
            dist = agg["tier"].value_counts().to_dict()
            top  = agg.nlargest(top_n,"sessions")[["customer_id","name","tier","sessions","features"]].to_dict(orient="records")
            return {"segmentation_type":"usage_tier","distribution":dist,"top_users":top,"total_customers":len(agg)}

        else:
            con.close()
            return {"error": f"Unknown segmentation_type '{segmentation_type}'. Choose: rfm, churn_risk, usage_tier, health_score"}

    except Exception as e:
        log.error("customer_segmentation_tool error: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 4: anomaly_detection_tool
# =============================================================================

class AnomalyInput(BaseModel):
    metric_name: Literal["transaction_volume","revenue","ticket_count","mrr"] = Field(
        ..., description="Which metric to analyse"
    )
    lookback_days: int   = Field(90, ge=7, le=365, description="Days of history to examine")
    sensitivity:   float = Field(2.0, ge=0.5, le=5.0, description="Z-score threshold (lower = more sensitive)")
    method: Literal["zscore","iqr","rolling"] = Field("zscore", description="Statistical method")


@tool("anomaly_detection_tool", args_schema=AnomalyInput)
def anomaly_detection_tool(
    metric_name: str,
    lookback_days: int   = 90,
    sensitivity:   float = 2.0,
    method: str          = "zscore",
) -> dict:
    """
    Detect statistical anomalies in a business metric over a time window.

    Use this tool for:
    - "Are there any anomalies in our transaction volumes?"
    - "Flag unusual spikes or drops in revenue"
    - "Detect abnormal support ticket patterns"
    - Any question about unusual/suspicious patterns in metrics

    Metrics available:
    - transaction_volume : daily count of all transactions
    - revenue            : daily sum of completed transaction amounts
    - ticket_count       : daily count of support tickets opened
    - mrr                : monthly MRR from active subscriptions

    Methods:
    - zscore  : flag days where |z| > sensitivity (default 2.0)
    - iqr     : flag days outside [Q1 - k*IQR, Q3 + k*IQR]
    - rolling : flag days where value deviates > sensitivity * rolling_std

    Returns: {anomaly_count, anomalies[{date, value, score, direction, explanation}], summary_stats}
    """
    try:
        con = _conn()
        ref    = DATASET_MAX_DATE
        cutoff = (datetime.strptime(ref, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        if metric_name == "transaction_volume":
            df = pd.read_sql(
                "SELECT transaction_date AS date, COUNT(*) AS value "
                "FROM transactions WHERE transaction_date >= ? "
                "GROUP BY transaction_date ORDER BY transaction_date",
                con, params=[cutoff],
            )
        elif metric_name == "revenue":
            df = pd.read_sql(
                "SELECT transaction_date AS date, ROUND(SUM(amount),2) AS value "
                "FROM transactions WHERE status='completed' AND transaction_date >= ? "
                "GROUP BY transaction_date ORDER BY transaction_date",
                con, params=[cutoff],
            )
        elif metric_name == "ticket_count":
            df = pd.read_sql(
                "SELECT created_at AS date, COUNT(*) AS value "
                "FROM support_tickets WHERE created_at >= ? "
                "GROUP BY created_at ORDER BY created_at",
                con, params=[cutoff],
            )
        elif metric_name == "mrr":
            df = pd.read_sql(
                "SELECT strftime('%Y-%m-01', start_date) AS date, "
                "       ROUND(SUM(mrr),2) AS value "
                "FROM subscriptions WHERE status='active' AND start_date >= ? "
                "GROUP BY strftime('%Y-%m', start_date) ORDER BY date",
                con, params=[cutoff],
            )
        else:
            con.close()
            return {"error": f"Unknown metric_name '{metric_name}'"}
        con.close()

        if df.empty or len(df) < 3:
            return {"message": "Insufficient data points for anomaly detection", "metric": metric_name}

        vals = df["value"].astype(float).values
        mean_v, std_v = vals.mean(), vals.std()

        if method == "zscore":
            scores = np.abs((vals - mean_v) / (std_v + 1e-9))
            flags  = scores > sensitivity
        elif method == "iqr":
            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr    = q3 - q1
            lower, upper = q1 - sensitivity*iqr, q3 + sensitivity*iqr
            flags  = (vals < lower) | (vals > upper)
            scores = np.abs(vals - mean_v) / (iqr + 1e-9)
        else:  # rolling
            win   = min(7, len(vals))
            rmean = pd.Series(vals).rolling(win, min_periods=1).mean().values
            rstd  = pd.Series(vals).rolling(win, min_periods=1).std().fillna(1).values
            scores = np.abs(vals - rmean) / (rstd + 1e-9)
            flags  = scores > sensitivity

        anomalies = []
        for i, (date_val, val, score, flag) in enumerate(zip(df["date"], vals, scores, flags)):
            if flag:
                direction = "spike" if val > mean_v else "drop"
                anomalies.append({
                    "date":        str(date_val),
                    "value":       round(float(val), 2),
                    "anomaly_score": round(float(score), 3),
                    "direction":   direction,
                    "explanation": (
                        f"{metric_name} on {date_val} was a {direction}: "
                        f"{val:.1f} vs mean {mean_v:.1f} "
                        f"(z-score {score:.2f}, threshold {sensitivity})"
                    ),
                })

        return {
            "metric":        metric_name,
            "method":        method,
            "lookback_days": lookback_days,
            "data_points":   len(df),
            "mean":          round(float(mean_v), 2),
            "std":           round(float(std_v), 2),
            "anomaly_count": len(anomalies),
            "anomalies":     anomalies,
        }

    except Exception as e:
        log.error("anomaly_detection_tool error: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 5: knowledge_retrieval_tool
# =============================================================================

class KnowledgeInput(BaseModel):
    query: str = Field(..., description="Natural language question about company policies or SLAs")
    top_k: int = Field(3, ge=1, le=8, description="Number of relevant passages to return")


@tool("knowledge_retrieval_tool", args_schema=KnowledgeInput)
def knowledge_retrieval_tool(query: str, top_k: int = 3) -> dict:
    """
    Search company policy documents and return relevant passages with source attribution.

    ALWAYS use this tool for questions about:
    - Refund policy or refund eligibility
    - SLA (Service Level Agreement) response/resolution times
    - Escalation procedures or escalation tiers
    - Pricing tiers, plan features, upgrade/downgrade rules
    - Enterprise customer terms or agreements
    - Any company policy or process question

    This tool searches across four policy documents:
    - refund_policy.md         : refund windows, eligibility, processing times
    - sla_policy.md            : uptime SLAs, support response times, credits
    - escalation_procedures.md : escalation tiers, billing disputes, incidents
    - pricing_tiers.md         : plan features, pricing, add-ons, upgrade rules

    Returns: {query, passages[{text, source, section, score}], answer_hint}
    """
    try:
        if not DOCS_DIR.exists():
            return {"error": f"Docs directory not found at {DOCS_DIR}"}

        # ── Load and chunk all documents ──────────────────────────────────────
        docs: list[dict] = []
        for md_file in sorted(DOCS_DIR.glob("*.md")):
            text  = md_file.read_text(encoding="utf-8")
            # Split on markdown headings
            parts = re.split(r"\n(#{1,3} .+)\n", text)
            current_heading = "Overview"
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if re.match(r"^#{1,3} ", part):
                    current_heading = part.lstrip("#").strip()
                else:
                    # Further split large sections into ~400 char chunks
                    paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
                    chunk = ""
                    for para in paragraphs:
                        if len(chunk) + len(para) < 600:
                            chunk += "\n\n" + para
                        else:
                            if chunk.strip():
                                docs.append({
                                    "text":    chunk.strip(),
                                    "source":  md_file.name,
                                    "section": current_heading,
                                })
                            chunk = para
                    if chunk.strip():
                        docs.append({
                            "text":    chunk.strip(),
                            "source":  md_file.name,
                            "section": current_heading,
                        })

        if not docs:
            return {"error": "No policy documents found"}

        # ── TF-IDF-style scoring ──────────────────────────────────────────────
        query_tokens  = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, dict]] = []

        for doc in docs:
            doc_tokens = re.findall(r"\w+", doc["text"].lower())
            doc_set    = set(doc_tokens)

            # Term frequency: fraction of query words that appear in the chunk
            tf    = len(query_tokens & doc_set) / max(len(query_tokens), 1)

            # Boost for exact phrase match
            phrase_boost = 1.5 if any(
                tok in doc["text"].lower() for tok in query_tokens if len(tok) > 4
            ) else 1.0

            # Boost for section heading match
            section_boost = 1.3 if any(
                tok in doc["section"].lower() for tok in query_tokens
            ) else 1.0

            score = tf * phrase_boost * section_boost
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in scored[:top_k] if _ > 0]

        if not top_docs:
            # Fallback: return first chunk of each doc
            top_docs = [docs[i] for i in range(min(top_k, len(docs)))]

        # Assign relevance scores
        for i, doc in enumerate(top_docs):
            doc["relevance_score"] = round(scored[i][0], 4) if i < len(scored) else 0.0
            doc["citation"] = f"[Source: {doc['source']} — {doc['section']}]"

        return {
            "query":    query,
            "passages": top_docs,
            "sources_searched": [f.name for f in DOCS_DIR.glob("*.md")],
        }

    except Exception as e:
        log.error("knowledge_retrieval_tool error: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 6: chart_generator_tool
# =============================================================================

class ChartInput(BaseModel):
    chart_type: Literal["bar","line","pie","heatmap","scatter"] = Field(
        ..., description="Type of chart to generate"
    )
    data_sql: str = Field(
        ..., description=(
            "SQL SELECT query to fetch the data for the chart. "
            "Must return at least two columns: one for x-axis (labels/dates), "
            "one for y-axis (numeric values)."
        )
    )
    title:  str           = Field("Chart", description="Chart title")
    x_col:  Optional[str] = Field(None, description="Column name for x-axis (auto-detected if omitted)")
    y_col:  Optional[str] = Field(None, description="Column name for y-axis (auto-detected if omitted)")


@tool("chart_generator_tool", args_schema=ChartInput)
def chart_generator_tool(
    chart_type: str,
    data_sql:   str,
    title:      str           = "Chart",
    x_col:      Optional[str] = None,
    y_col:      Optional[str] = None,
) -> dict:
    """
    Generate a matplotlib chart from a SQL query and save it as PNG.

    Use this tool when the user asks to:
    - "Generate a bar/line/pie chart of ..."
    - "Visualise / plot / show me a graph of ..."
    - "Create a chart showing ..."

    You must provide a valid SQL SELECT query in data_sql.
    The query should return labelling column first, then numeric column(s).

    Chart types:
    - bar     : vertical bars (best for comparing categories)
    - line    : connected line (best for time series)
    - pie     : pie/donut chart (best for proportions, ≤10 slices)
    - scatter : scatter plot (best for two numeric dimensions)

    Returns: {file_path, chart_type, title, rows_plotted}
    The file_path can be shared with the user or embedded in a report.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        con  = _conn()
        cur  = con.execute(data_sql)
        rows = cur.fetchmany(200)
        cols = [d[0] for d in cur.description] if cur.description else []
        con.close()

        if not rows:
            return {"error": "Query returned no data"}

        df = pd.DataFrame([dict(zip(cols, r)) for r in rows])

        # Auto-detect columns
        if x_col is None:
            x_col = cols[0]
        if y_col is None:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            y_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else cols[0])

        # Coerce y to numeric
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0)

        fig, ax = plt.subplots(figsize=(11, 5))
        colors  = ["#4A90D9","#E85D5D","#50C878","#F5A623","#9B59B6",
                   "#1ABC9C","#E67E22","#3498DB","#E74C3C","#2ECC71"]

        x_vals = df[x_col].astype(str).values
        y_vals = df[y_col].values

        if chart_type == "bar":
            bars = ax.bar(x_vals, y_vals,
                          color=colors[:len(x_vals)] if len(x_vals) <= len(colors) else colors[0])
            ax.tick_params(axis="x", rotation=45, labelsize=8)

        elif chart_type == "line":
            ax.plot(x_vals, y_vals, marker="o", color=colors[0], linewidth=2, markersize=4)
            ax.fill_between(range(len(x_vals)), y_vals, alpha=0.1, color=colors[0])
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=45, ha="right", fontsize=8)

        elif chart_type == "pie":
            slices = min(10, len(x_vals))
            ax.pie(y_vals[:slices], labels=x_vals[:slices],
                   autopct="%1.1f%%", colors=colors[:slices], startangle=90)
            ax.axis("equal")

        elif chart_type == "scatter":
            ax.scatter(x_vals if pd.api.types.is_numeric_dtype(df[x_col]) else range(len(x_vals)),
                       y_vals, color=colors[0], alpha=0.7, s=40)
        else:
            ax.bar(x_vals, y_vals, color=colors[0])

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        if chart_type != "pie":
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(y_col, fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_type}_{ts}.png"
        path     = CHARTS_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {
            "file_path":    str(path),
            "chart_type":   chart_type,
            "title":        title,
            "rows_plotted": len(df),
            "x_col":        x_col,
            "y_col":        y_col,
        }

    except Exception as e:
        log.error("chart_generator_tool error: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool 7 (Bonus): forecast_tool
# =============================================================================

class ForecastInput(BaseModel):
    metric_name: Literal["revenue","mrr","ticket_count"] = Field(
        ..., description="Metric to forecast"
    )
    forecast_horizon: int = Field(
        90, ge=7, le=365, description="Number of days to forecast ahead"
    )
    method: Literal["linear","exponential"] = Field(
        "linear", description="Forecasting method"
    )


@tool("forecast_tool", args_schema=ForecastInput)
def forecast_tool(
    metric_name:      str = "revenue",
    forecast_horizon: int = 90,
    method:           str = "linear",
) -> dict:
    """
    Forecast a business metric into the future using statistical trend models.

    Use this tool when the user asks to:
    - "Forecast / predict revenue for next quarter"
    - "What will MRR look like in 3 months?"
    - "Show me the trend forecast for support tickets"
    - "Compare churn rates and forecast the trend"

    Metrics:
    - revenue      : monthly revenue from completed transactions
    - mrr          : monthly recurring revenue from active subscriptions
    - ticket_count : monthly support ticket volume

    Methods:
    - linear      : OLS linear regression on time index, extrapolate forward
    - exponential : simple exponential smoothing (α=0.3)

    Returns: {metric, method, historical_summary, forecast[{date, predicted, lower_ci, upper_ci}]}
    """
    try:
        con = _conn()

        if metric_name == "revenue":
            df = pd.read_sql(
                "SELECT strftime('%Y-%m', transaction_date) AS period, "
                "       ROUND(SUM(amount),2) AS value "
                "FROM transactions WHERE status='completed' "
                "GROUP BY period ORDER BY period",
                con,
            )
        elif metric_name == "mrr":
            df = pd.read_sql(
                "SELECT strftime('%Y-%m', start_date) AS period, "
                "       ROUND(SUM(mrr),2) AS value "
                "FROM subscriptions WHERE status='active' "
                "GROUP BY period ORDER BY period",
                con,
            )
        elif metric_name == "ticket_count":
            df = pd.read_sql(
                "SELECT strftime('%Y-%m', created_at) AS period, "
                "       COUNT(*) AS value "
                "FROM support_tickets WHERE created_at IS NOT NULL "
                "GROUP BY period ORDER BY period",
                con,
            )
        else:
            con.close()
            return {"error": f"Unknown metric '{metric_name}'"}
        con.close()

        if len(df) < 3:
            return {"error": "Not enough historical data for forecasting (need ≥ 3 periods)"}

        y = df["value"].astype(float).values
        n = len(y)
        x = np.arange(n)

        if method == "linear":
            # OLS: slope and intercept
            coeffs           = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            residuals        = y - (slope * x + intercept)
            std_err          = residuals.std()
            ci               = 1.96 * std_err
        else:
            # Simple exponential smoothing
            alpha    = 0.3
            smoothed = [y[0]]
            for v in y[1:]:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
            slope    = 0
            intercept = smoothed[-1]
            std_err  = np.std(np.array(smoothed) - y)
            ci       = 1.96 * std_err

        # Generate forecast periods
        n_months = max(1, forecast_horizon // 30)
        last_period = pd.Period(df["period"].iloc[-1], freq="M")
        forecast = []
        for i in range(1, n_months + 1):
            future_period = last_period + i
            if method == "linear":
                pred = slope * (n + i - 1) + intercept
            else:
                pred = intercept  # flat for exponential
            forecast.append({
                "period":     str(future_period),
                "predicted":  round(float(pred), 2),
                "lower_ci":   round(float(pred - ci), 2),
                "upper_ci":   round(float(pred + ci), 2),
            })

        return {
            "metric":   metric_name,
            "method":   method,
            "horizon_days": forecast_horizon,
            "historical": {
                "periods":  n,
                "mean":     round(float(y.mean()), 2),
                "trend_direction": "upward" if slope > 0 else "downward" if slope < 0 else "flat",
                "monthly_change": round(float(slope), 2) if method == "linear" else None,
            },
            "forecast": forecast,
        }

    except Exception as e:
        log.error("forecast_tool error: %s", e)
        return {"error": str(e)}


# =============================================================================
# Tool registry — imported by agent.py
# =============================================================================

ALL_TOOLS = [
    sql_query_tool,
    revenue_calculator_tool,
    customer_segmentation_tool,
    anomaly_detection_tool,
    knowledge_retrieval_tool,
    chart_generator_tool,
    forecast_tool,
]

# =============================================================================
# =============================================================================
# Standalone test-friendly function wrappers
# Import these in unit tests to bypass the @tool LangChain decorator
# =============================================================================

def _revenue_calculator(
    start_date:  str,
    end_date:    str,
    granularity: str = "monthly",
    metric:      str = "all",
) -> dict:
    """Direct call to revenue logic — use in unit tests."""
    return revenue_calculator_tool.func(start_date, end_date, granularity, metric)


def _anomaly_detection(
    metric_name:   str,
    lookback_days: int   = 90,
    sensitivity:   float = 2.0,
    method:        str   = "zscore",
) -> dict:
    """Direct call to anomaly detection logic — use in unit tests."""
    return anomaly_detection_tool.func(metric_name, lookback_days, sensitivity, method)