# FinOps Analytics Platform

An AI-powered FinOps analytics platform for a B2B SaaS company. Orchestrates data engineering, advanced SQL analytics, a multi-tool LangChain agent, RAG-powered policy retrieval, human-in-the-loop governance, and a FastAPI REST service into a single governed, composable platform.

---

## Architecture Overview

```
Raw Files (data/)
│  customers_raw.json   transactions.csv   invoices.csv
│  subscriptions.csv    support_tickets.csv   product_usage.csv
│
▼
┌─────────────────────────────────────────────────────────┐
│  Step 1 — ETL Pipeline  (src/etl/)                      │
│                                                         │
│  ingestion.py  →  cleaning.py  →  profiling.py          │
│                       │                                 │
│                  validation.py  (cross-table FK checks) │
└───────────────────────┬─────────────────────────────────┘
                        │  cleaned DataFrames
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2 — SQLite  (finops.db)                           │
│                                                         │
│  schema.sql  ←─  loader.py  ←─  DataFrames             │
│  Star schema: fact tables + customers + subscriptions   │
│  + date_dim + audit_log + approval_queue                │
└───────────┬─────────────────────┬───────────────────────┘
            │                     │
     sql/queries.sql          finops.db
     (8 analytical queries)       │
                                  ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4 — Multi-Tool LangChain Agent  (src/agent/)      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Agent Router  (Claude claude-opus-4-5)  │   │
│  └──┬───────┬───────┬───────┬───────┬───────┬──────┘   │
│     │       │       │       │       │       │           │
│  sql_  rev_ seg_  anom_  know_  chart_ fore_           │
│  query calc ment  det    retr   gen    cast             │
│     │       │       │       │       │       │           │
│     └───────┴───────┘       │       └───────┘           │
│         SQLite          RAG Pipeline             pandas  │
└─────────────────────────┬───────────────────────────────┘
                          │
               ┌──────────▼──────────┐
               │  Step 5 — RAG       │
               │  src/rag/           │
               │                     │
               │  chunker.py         │
               │  (heading-aware,    │
               │   400-char chunks)  │
               │       ↓             │
               │  embedder.py        │
               │  (all-MiniLM-L6-v2  │
               │   384-dim FAISS)    │
               │       ↓             │
               │  retriever.py       │
               │  (FAISS 70%         │
               │   + BM25 30%)       │
               │                     │
               │  docs/              │
               │  ├ refund_policy.md │
               │  ├ sla_policy.md    │
               │  ├ escalation_      │
               │  │  procedures.md  │
               │  └ pricing_tiers.md│
               └─────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Step 6 — Governance Layer  (src/governance/)           │
│                                                         │
│  classifier.py → [safe | requires_review | blocked]     │
│                          │                              │
│                   requires_review                       │
│                          │                              │
│                   queue.py (approval_queue table)       │
│                          │                              │
│  audit.py ← every interaction → audit_log table         │
│                                                         │
│  guardrails.py                                          │
│  ├ PII masking (emails, phones)                         │
│  ├ Row limit (max 1000)                                 │
│  ├ Confidence surfacing                                 │
│  └ Reflection step (SQL ↔ question alignment check)     │
│                                                         │
│  governance.py (orchestrator — wraps all agent calls)  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Step 7 — FastAPI  (src/api/)                           │
│                                                         │
│  GET  /health                                           │
│  POST /etl/run              GET /etl/status/{job_id}    │
│  POST /agent/query          GET /agent/query/history    │
│  GET  /analytics/dashboard                              │
│  GET  /governance/pending                               │
│  POST /governance/review/{id}                           │
│  POST /agent/query/validate                             │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
finops_platform/
├── data/                          # Raw data files (6 files)
├── docs/                          # Policy markdown files (4 files)
├── rag_index/                     # FAISS index + metadata (auto-generated)
├── charts/                        # Generated chart PNGs (auto-generated)
├── src/
│   ├── etl/
│   │   ├── ingestion.py           # Load all 6 raw files into DataFrames
│   │   ├── cleaning.py            # All cleaning transforms (vectorised)
│   │   ├── profiling.py           # Data profiling → CSV report
│   │   ├── validation.py          # Cross-table FK integrity checks
│   │   └── loader.py              # Load DataFrames → SQLite
│   ├── agent/
│   │   ├── tools.py               # 7 LangChain tools with Pydantic schemas
│   │   └── agent.py               # LangGraph agent with MemorySaver
│   ├── rag/
│   │   ├── chunker.py             # Heading-aware markdown chunking
│   │   ├── embedder.py            # sentence-transformers + FAISS indexing
│   │   └── retriever.py           # Hybrid search (FAISS + BM25)
│   ├── governance/
│   │   ├── classifier.py          # 3-tier safety classifier
│   │   ├── audit.py               # Append-only audit log
│   │   ├── queue.py               # Approval queue lifecycle
│   │   ├── guardrails.py          # PII masking, row limits, reflection
│   │   └── governance.py          # Full pipeline orchestrator
│   └── api/
│       ├── schemas.py             # Pydantic v2 request/response models
│       └── main.py                # FastAPI app — 8 endpoints
├── sql/
│   ├── schema.sql                 # DDL — star schema, constraints, indexes
│   └── queries.sql                # 8 hand-written analytical queries
├── notebooks/
│   └── data_profile_report.csv    # Auto-generated data profiling report
├── tests/
│   └── test_suite.py              # pytest unit + integration tests
├── run_step1.py                   # Step 1 orchestrator
├── run_step2.py                   # Step 2 orchestrator
├── run_step3.py                   # Step 3 query runner
├── run_step4.py                   # Step 4 agent runner (12 test questions)
├── run_step5.py                   # Step 5 RAG pipeline builder
├── run_step6.py                   # Step 6 governance tests
├── run_step7.py                   # Step 7 API tests / server
├── agent_evaluation.json          # 12 test question results
├── README.md
├── requirements.txt
└── .env.example
```

---

## Setup Instructions

### 1. Requirements

- Python 3.10 or higher
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### 2. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd finops_platform
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` embedding model (~90MB). Requires internet access.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
```

### 5. Run each step in order

```bash
# Step 1: Ingest and clean data
python run_step1.py

# Step 2: Load into SQLite
python run_step2.py

# Step 3: Validate SQL queries
python run_step3.py

# Step 4: Run agent against 12 test questions (needs API key)
python run_step4.py

# Step 5: Build RAG index
python run_step5.py

# Step 6: Test governance layer
python run_step6.py

# Step 7: Start the API server
python run_step7.py --serve
# Then open: http://127.0.0.1:8000/docs
```

### 6. Run tests

```bash
pytest tests/test_suite.py -v
```

---

## API Documentation

### GET /health
```bash
curl http://127.0.0.1:8000/health
```
```json
{
  "status": "ok",
  "db_status": "connected",
  "row_counts": {"customers": 510, "transactions": 5020},
  "agent_ready": true,
  "rag_index_ready": true,
  "pending_reviews": 0
}
```

### POST /etl/run
```bash
curl -X POST http://127.0.0.1:8000/etl/run
```
```json
{"job_id": "abc-123", "status": "accepted", "message": "Poll GET /etl/status/abc-123"}
```

### GET /etl/status/{job_id}
```bash
curl http://127.0.0.1:8000/etl/status/abc-123
```

### POST /agent/query
```bash
curl -X POST http://127.0.0.1:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our current MRR?"}'
```
```json
{
  "answer": "Current MRR is $197,397...",
  "tools_used": ["revenue_calculator_tool"],
  "classification": "safe",
  "execution_time_ms": 1823,
  "confidence": 0.9
}
```

### GET /agent/query/history
```bash
# Last 20 queries
curl "http://127.0.0.1:8000/agent/query/history?limit=20&offset=0"

# Filter by tool
curl "http://127.0.0.1:8000/agent/query/history?tool_used=sql_query_tool"

# Filter by classification
curl "http://127.0.0.1:8000/agent/query/history?classification=safe"
```

### GET /analytics/dashboard
```bash
curl http://127.0.0.1:8000/analytics/dashboard
```
```json
{
  "mrr": 197397.0,
  "arr": 2368764.0,
  "active_customers": 248,
  "churn_rate_pct": 18.4,
  "avg_ticket_resolution_h": 14.7,
  "nps_score": 22.0
}
```

### GET /governance/pending
```bash
curl http://127.0.0.1:8000/governance/pending
```

### POST /governance/review/{id}
```bash
# Approve
curl -X POST http://127.0.0.1:8000/governance/review/1 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve", "reviewer_notes": "Verified business need", "reviewer_id": "mgr-01"}'

# Reject
curl -X POST http://127.0.0.1:8000/governance/review/1 \
  -H "Content-Type: application/json" \
  -d '{"action": "reject", "reviewer_notes": "No business justification"}'
```

### POST /agent/query/validate
```bash
curl -X POST http://127.0.0.1:8000/agent/query/validate \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT SUM(amount) FROM transactions WHERE status=\"completed\""}'
```
```json
{"sql": "...", "classification": "safe", "reason": "Read-only analytics query"}
```

---

## Design Decisions

### Schema Design
The database uses a **star schema** with `transactions`, `invoices`, `support_tickets`, and `product_usage` as fact tables, and `customers`, `subscriptions`, and `date_dim` as dimensions. Star schema was chosen over pure snowflake because it minimises JOIN depth for analytical queries — most questions only need 2-3 tables. The `date_dim` table (2,191 rows for 2021–2026) enables fiscal-period grouping, month-start cohort bucketing, and weekend filtering without `strftime()` calls in every query. Indexes are placed on every FK column, every date column, and low-cardinality GROUP BY columns (status, priority, plan_name) — 20 indexes total, each justified by a comment in schema.sql.

### Tool Design Rationale
Each tool is scoped to a **single responsibility** so the agent's routing decision is unambiguous:
- `revenue_calculator_tool` operates on DataFrames (not SQL) to guarantee consistent results regardless of SQL generation errors. Revenue figures are too important to leave to LLM SQL generation.
- `sql_query_tool` handles open-ended lookups where the structure of the answer isn't predetermined. It generates SQL via Claude, enforces SELECT-only, and retries once with the error context if the first attempt fails.
- `knowledge_retrieval_tool` is the exclusive entry point for all policy questions. Isolating RAG from SQL prevents the agent from hallucinating policy facts from training data.
- `customer_segmentation_tool` implements RFM scoring in Python because the quintile-based segmentation logic is cleaner in pandas than in SQL, and it runs faster on pre-loaded DataFrames.

### Chunking Strategy
Policy documents are chunked by **heading boundary** (##/### markers) rather than fixed-size. Each heading in the policy docs corresponds to one complete policy rule (e.g. "### Enterprise Plans" = all enterprise refund terms). Splitting at headings ensures each chunk is a self-contained unit that answers a specific policy question. Target size is 400 characters (the average section length), with 50-character overlap between adjacent chunks in the same section to avoid cutting mid-sentence at paragraph boundaries. This produces 28 chunks across 4 documents.

### Governance Classification Logic
The classifier is **rule-based (regex), not LLM-based**, for three reasons: speed (microseconds vs. 500ms+ for an LLM call), determinism (same input always produces the same classification), and auditability (every rule is explicit and version-controlled). The classifier runs **twice**: pre-execution (on the raw question) and post-SQL (on the generated SQL), because a safe-sounding question can produce dangerous SQL ("show me all data" → `SELECT * FROM customers`). BLOCKED rules are checked before REQUIRES_REVIEW to prevent an ambiguous request from being accidentally approved.

### Agent Prompt Design
The system prompt uses **explicit tool routing rules** rather than relying on tool descriptions alone. Without explicit rules, the agent routes "What is our MRR?" to `sql_query_tool` (generates `SUM(mrr)` SQL) about 30% of the time instead of `revenue_calculator_tool`. With explicit rules ("revenue metrics → ALWAYS use revenue_calculator_tool"), routing accuracy improves to near-100%. The DB schema is injected into the SQL generation prompt (not the agent system prompt) to keep the agent context concise while giving the SQL tool the column-level detail it needs.

---

## Challenges and Solutions

### Challenge 1: Heterogeneous Date Formats Across All 6 Files
Every file used different date formats: `2022/12/18`, `23/08/2022`, `May 15, 2024`, `2023-06-24T00:00:00+00:00`, `07/23/2024 00:00-05:00`. A single `pd.to_datetime(infer_datetime_format=True)` parsed ~85% correctly but silently mislabelled `06/07/2023` as June 7 instead of July 6 (DD/MM vs. MM/DD ambiguity).

**Solution:** A three-pass cascade: (1) pandas ISO inference, (2) pandas with `dayfirst=True`, (3) `dateutil.parser` for natural-language dates. All parsed timestamps are immediately normalised to UTC midnight to eliminate timezone ambiguity. ~793 support ticket timestamps with a non-standard format (`00:00-05:00` suffix) failed all three passes and are set to None with a warning — the ticket row is retained because all other columns are valid.

### Challenge 2: LangChain 1.x Removed AgentExecutor
The original agent used `from langchain.agents import AgentExecutor, create_tool_calling_agent`, which raised `ImportError` on LangChain 1.x because `AgentExecutor` was removed entirely. The `langchain.agents` module in 1.x only exports `AgentState` and a few utilities.

**Solution:** Migrated to `langgraph.prebuilt.create_react_agent` with `MemorySaver` and `thread_id`. LangGraph is the official replacement and ships with LangChain 1.x. `MemorySaver` + `thread_id` provides equivalent conversation memory: all turns with the same `thread_id` share history, so "show me the same data for Q2" works correctly across turns.

### Challenge 3: SQLite FK Violations During Bulk Load
The source data contains 544 orphaned foreign keys — transactions and invoices referencing subscription IDs that were deleted from the source system after the transactions were recorded. Loading with `PRAGMA foreign_keys = ON` caused `sqlite3.IntegrityError: FOREIGN KEY constraint failed` on the first orphaned row, stopping the entire load.

**Solution:** Disable FK enforcement with `PRAGMA foreign_keys = OFF` during bulk load, then re-enable it and run explicit LEFT JOIN orphan queries as part of the post-load integrity check. This mirrors how production data warehouses work: load everything, then report referential issues for the data steward. Orphans are logged with their counts and sample values so they can be investigated without blocking the pipeline.

---

## Deployment

**Live URL:** `https://finops-platform.onrender.com/docs`

*(Replace with your actual Render URL after deploying)*

**Interactive API docs:** `https://finops-analytics-platform.onrender.com/docs`

### Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — no manual config needed
5. Add environment variable in Render dashboard:
   - Key: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...`
6. Click **Deploy**

Build time is ~3-5 minutes (installs dependencies, runs ETL, builds RAG index).