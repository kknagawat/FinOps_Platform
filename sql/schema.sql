-- =============================================================================
-- sql/schema.sql
-- FinOps Analytics Platform — SQLite Database Schema
-- =============================================================================
-- Schema pattern : Star schema
--   Fact tables  : transactions, invoices, support_tickets, product_usage
--   Dimensions   : customers, subscriptions, date_dim
--
-- Star vs Snowflake decision
-- --------------------------
-- A pure star schema (all dimensions denormalised directly off facts) was
-- chosen over snowflake because:
--   1. SQLite is used for analytics (read-heavy), not OLTP. Fewer JOINs
--      means faster GROUP BY and aggregation queries.
--   2. The dimension tables are small (customers ~510, subscriptions ~824)
--      so denormalisation does not waste meaningful storage.
--   3. The 8 analytical queries (queries.sql) all benefit from direct
--      fact→dimension joins without intermediate bridge tables.
--
-- SQLite FK note
-- --------------
-- SQLite does NOT enforce FK constraints by default.
-- You MUST run:  PRAGMA foreign_keys = ON;
-- before any INSERT/UPDATE/DELETE to activate enforcement.
-- The loader.py script does this automatically on every connection.
-- =============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;       -- write-ahead logging: better concurrency
PRAGMA synchronous   = NORMAL;   -- safe + faster than FULL for analytics use


-- =============================================================================
-- DIMENSION: date_dim
-- =============================================================================
-- Purpose : Time-intelligence dimension. Pre-computed date attributes so
--           analytical queries can GROUP BY year/quarter/month/week without
--           calling strftime() on every row — which is non-sargable and
--           prevents index use.
-- Populated: Programmatically by loader.py (2021-01-01 → 2027-12-31).
-- =============================================================================

DROP TABLE IF EXISTS date_dim;

CREATE TABLE date_dim (
    -- Integer surrogate key in YYYYMMDD format (e.g. 20240115).
    -- Used as a JOIN key from fact tables when date_id columns are added.
    date_id         INTEGER  PRIMARY KEY,

    full_date       TEXT     NOT NULL UNIQUE,   -- "YYYY-MM-DD" — the natural join key
    year            INTEGER  NOT NULL CHECK (year  BETWEEN 2000 AND 2100),
    quarter         INTEGER  NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month           INTEGER  NOT NULL CHECK (month  BETWEEN 1 AND 12),
    month_name      TEXT     NOT NULL,           -- "January" … "December"
    week_of_year    INTEGER  NOT NULL CHECK (week_of_year BETWEEN 1 AND 53),
    day_of_month    INTEGER  NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_week     INTEGER  NOT NULL CHECK (day_of_week  BETWEEN 0 AND 6),  -- 0=Mon
    day_name        TEXT     NOT NULL,           -- "Monday" … "Sunday"
    is_weekend      INTEGER  NOT NULL DEFAULT 0 CHECK (is_weekend IN (0, 1)),
    is_month_start  INTEGER  NOT NULL DEFAULT 0 CHECK (is_month_start IN (0, 1)),
    is_month_end    INTEGER  NOT NULL DEFAULT 0 CHECK (is_month_end   IN (0, 1)),
    fiscal_quarter  TEXT     NOT NULL            -- e.g. "FQ1-2024"
);

-- Index on full_date: every fact→date_dim JOIN uses this column
CREATE INDEX IF NOT EXISTS idx_date_dim_full_date     ON date_dim (full_date);
-- Indexes for common GROUP BY patterns in analytical queries
CREATE INDEX IF NOT EXISTS idx_date_dim_year_month    ON date_dim (year, month);
CREATE INDEX IF NOT EXISTS idx_date_dim_fiscal_qtr    ON date_dim (fiscal_quarter);


-- =============================================================================
-- DIMENSION: customers
-- =============================================================================
-- Central dimension. Every fact table has a customer_id FK pointing here.
-- =============================================================================

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id     TEXT     PRIMARY KEY  NOT NULL,
    first_name      TEXT     NOT NULL,
    last_name       TEXT     NOT NULL,
    email           TEXT,                -- nullable: 29 customers have no email (flagged)
    phone           TEXT,                -- nullable: 32 customers have no phone (flagged)
    city            TEXT,
    country         TEXT,
    zip_code        TEXT,
    signup_date     TEXT,                -- "YYYY-MM-DD"
    is_active       INTEGER  CHECK (is_active IN (0, 1, NULL)),
    company         TEXT     NOT NULL DEFAULT 'Unknown',
    loyalty_tier    TEXT     NOT NULL DEFAULT 'unknown',

    -- Data-quality flag columns (added by cleaning.py)
    email_missing   INTEGER  NOT NULL DEFAULT 0 CHECK (email_missing IN (0, 1)),
    phone_missing   INTEGER  NOT NULL DEFAULT 0 CHECK (phone_missing IN (0, 1))
);

-- Justification: is_active is the most-filtered column in churn/retention
-- queries — every "active customer" WHERE clause uses this index.
CREATE INDEX IF NOT EXISTS idx_customers_is_active   ON customers (is_active);

-- Justification: cohort analysis (Q1) groups by strftime('%Y-%m', signup_date).
-- This index lets SQLite resolve the range scan without a full table scan.
CREATE INDEX IF NOT EXISTS idx_customers_signup_date ON customers (signup_date);

-- Justification: geographic segmentation and country-level revenue queries.
CREATE INDEX IF NOT EXISTS idx_customers_country     ON customers (country);

-- Justification: loyalty-tier grouping for RFM and segmentation queries.
CREATE INDEX IF NOT EXISTS idx_customers_loyalty     ON customers (loyalty_tier);


-- =============================================================================
-- DIMENSION: subscriptions
-- =============================================================================
-- Also acts as a mini-fact for MRR waterfall (Q2): each row carries mrr.
-- =============================================================================

DROP TABLE IF EXISTS subscriptions;

CREATE TABLE subscriptions (
    subscription_id TEXT     PRIMARY KEY  NOT NULL,
    customer_id     TEXT     NOT NULL
                             REFERENCES customers (customer_id)
                             ON DELETE RESTRICT,

    plan_name       TEXT     NOT NULL DEFAULT 'unknown',
    mrr             REAL     NOT NULL DEFAULT 0
                             CHECK (mrr >= 0),           -- negative MRR corrected in cleaning
    currency        TEXT     NOT NULL DEFAULT 'USD',
    start_date      TEXT,                                -- "YYYY-MM-DD"
    end_date        TEXT,                                -- NULL for active subscriptions
    status          TEXT     NOT NULL DEFAULT 'unknown',
    billing_cycle   TEXT,
    auto_renew      INTEGER  CHECK (auto_renew IN (0, 1, NULL)),

    -- Data-quality flag columns
    mrr_was_negative  INTEGER NOT NULL DEFAULT 0 CHECK (mrr_was_negative IN (0, 1)),
    is_future_start   INTEGER NOT NULL DEFAULT 0 CHECK (is_future_start  IN (0, 1))
);

-- Justification: customer_id is the join key for every fact→subscription
-- lookup and for subscription overlap detection (Q4 self-join).
CREATE INDEX IF NOT EXISTS idx_subs_customer_id  ON subscriptions (customer_id);

-- Justification: status is used in WHERE clauses in almost every query
-- (e.g. WHERE status = 'active' for MRR queries).
CREATE INDEX IF NOT EXISTS idx_subs_status       ON subscriptions (status);

-- Justification: plan_name is the GROUP BY key in churn-by-plan (Q12)
-- and MRR waterfall (Q2) queries.
CREATE INDEX IF NOT EXISTS idx_subs_plan_name    ON subscriptions (plan_name);

-- Justification: start_date range scans for MRR waterfall date windows.
CREATE INDEX IF NOT EXISTS idx_subs_start_date   ON subscriptions (start_date);

-- Justification: compound index for overlap detection self-join (Q4)
-- which filters on customer_id AND compares start/end dates.
CREATE INDEX IF NOT EXISTS idx_subs_cust_dates   ON subscriptions (customer_id, start_date, end_date);


-- =============================================================================
-- FACT: transactions
-- =============================================================================

DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    transaction_id   TEXT  PRIMARY KEY  NOT NULL,
    customer_id      TEXT  NOT NULL
                           REFERENCES customers (customer_id)
                           ON DELETE RESTRICT,
    subscription_id  TEXT  REFERENCES subscriptions (subscription_id)
                           ON DELETE RESTRICT,         -- nullable: some txns have no sub

    transaction_date TEXT,                             -- "YYYY-MM-DD"
    amount           REAL  NOT NULL
                           CHECK (amount >= 0),        -- negative amounts corrected + flagged
    currency         TEXT,
    status           TEXT,
    payment_method   TEXT  NOT NULL DEFAULT 'unknown',
    invoice_id       TEXT,                             -- nullable: refunds have no invoice
    description      TEXT,

    -- Data-quality flag
    was_negative     INTEGER NOT NULL DEFAULT 0 CHECK (was_negative IN (0, 1))
);

-- Justification: customer_id is the GROUP BY key for LTV, cohort (Q1),
-- health score (Q3), and churn signal (Q8) queries.
CREATE INDEX IF NOT EXISTS idx_txn_customer_id   ON transactions (customer_id);

-- Justification: transaction_date is filtered in every time-series query
-- and in the cohort assignment CTE (Q1).
CREATE INDEX IF NOT EXISTS idx_txn_date          ON transactions (transaction_date);

-- Justification: status = 'completed' filter appears in virtually every
-- revenue query — this index avoids full table scans.
CREATE INDEX IF NOT EXISTS idx_txn_status        ON transactions (status);

-- Justification: compound index for cohort revenue lookups (Q1) which
-- filter on customer_id AND transaction_date simultaneously.
CREATE INDEX IF NOT EXISTS idx_txn_cust_date     ON transactions (customer_id, transaction_date);


-- =============================================================================
-- FACT: invoices
-- =============================================================================

DROP TABLE IF EXISTS invoices;

CREATE TABLE invoices (
    invoice_id       TEXT  PRIMARY KEY  NOT NULL,
    customer_id      TEXT  NOT NULL
                           REFERENCES customers (customer_id)
                           ON DELETE RESTRICT,
    subscription_id  TEXT  REFERENCES subscriptions (subscription_id)
                           ON DELETE RESTRICT,

    issue_date       TEXT,
    due_date         TEXT,
    subtotal         REAL  CHECK (subtotal  >= 0),
    tax              REAL  CHECK (tax       >= 0),
    total            REAL  CHECK (total     >= 0),
    paid_amount      REAL  NOT NULL DEFAULT 0
                           CHECK (paid_amount >= 0),
    payment_status   TEXT,
    paid_date        TEXT,                             -- NULL for unpaid invoices
    payment_method   TEXT  NOT NULL DEFAULT 'unknown',
    currency         TEXT,

    -- Data-quality flags
    tax_rate         REAL,
    tax_error_flag   INTEGER NOT NULL DEFAULT 0 CHECK (tax_error_flag IN (0, 1))
);

-- Justification: customer_id JOIN for payment reliability score (Q3).
CREATE INDEX IF NOT EXISTS idx_inv_customer_id       ON invoices (customer_id);

-- Justification: subscription_id JOIN for MRR-vs-invoice reconciliation.
CREATE INDEX IF NOT EXISTS idx_inv_subscription_id   ON invoices (subscription_id);

-- Justification: payment_status WHERE clauses (paid / unpaid / partial)
-- appear in payment reliability component of Q3 health score.
CREATE INDEX IF NOT EXISTS idx_inv_payment_status    ON invoices (payment_status);

-- Justification: issue_date for time-series revenue analysis.
CREATE INDEX IF NOT EXISTS idx_inv_issue_date        ON invoices (issue_date);

-- Justification: compound index for on-time payment check (Q3) which
-- needs both customer_id and date comparison in a single scan.
CREATE INDEX IF NOT EXISTS idx_inv_cust_issue        ON invoices (customer_id, issue_date);


-- =============================================================================
-- FACT: support_tickets
-- =============================================================================

DROP TABLE IF EXISTS support_tickets;

CREATE TABLE support_tickets (
    ticket_id          TEXT  PRIMARY KEY  NOT NULL,
    customer_id        TEXT  NOT NULL
                             REFERENCES customers (customer_id)
                             ON DELETE RESTRICT,

    category           TEXT  NOT NULL DEFAULT 'uncategorized',
    priority           TEXT,
    status             TEXT,
    created_at         TEXT,
    first_response_at  TEXT,
    resolved_at        TEXT,
    rating             REAL  CHECK (rating IS NULL OR (rating BETWEEN 1.0 AND 5.0)),
    resolution_text    TEXT  NOT NULL DEFAULT '',
    agent_name         TEXT  NOT NULL DEFAULT 'unassigned',
    channel            TEXT  NOT NULL DEFAULT 'unknown',
    is_escalated       INTEGER CHECK (is_escalated IN (0, 1, NULL))
);

-- Justification: customer_id JOIN for support component of health score (Q3)
-- and churn signal ticket-spike detection (Q8).
CREATE INDEX IF NOT EXISTS idx_tkt_customer_id  ON support_tickets (customer_id);

-- Justification: priority GROUP BY in resolution funnel query (Q5)
-- and SLA compliance checks.
CREATE INDEX IF NOT EXISTS idx_tkt_priority     ON support_tickets (priority);

-- Justification: created_at for time-window recency checks in Q8 churn
-- signals (last 30 days vs prior 30 days ticket frequency).
CREATE INDEX IF NOT EXISTS idx_tkt_created_at   ON support_tickets (created_at);

-- Justification: status WHERE clause for open/closed/resolved filtering.
CREATE INDEX IF NOT EXISTS idx_tkt_status       ON support_tickets (status);

-- Justification: compound index for Q8 churn signal which groups by
-- customer_id and filters by created_at date range simultaneously.
CREATE INDEX IF NOT EXISTS idx_tkt_cust_date    ON support_tickets (customer_id, created_at);


-- =============================================================================
-- FACT: product_usage
-- =============================================================================

DROP TABLE IF EXISTS product_usage;

CREATE TABLE product_usage (
    usage_id                   TEXT  PRIMARY KEY  NOT NULL,
    customer_id                TEXT  REFERENCES customers (customer_id)
                                     ON DELETE RESTRICT,   -- nullable: anonymous sessions

    feature_name               TEXT  NOT NULL,
    session_date               TEXT,
    session_duration_seconds   REAL  CHECK (
                                   session_duration_seconds IS NULL
                                   OR session_duration_seconds >= 0
                               ),
    usage_count                REAL  CHECK (usage_count IS NULL OR usage_count >= 0),
    device                     TEXT  NOT NULL DEFAULT 'unknown',
    session_id                 TEXT,

    -- Data-quality flags
    duration_invalid           INTEGER NOT NULL DEFAULT 0 CHECK (duration_invalid    IN (0, 1)),
    usage_count_outlier        INTEGER NOT NULL DEFAULT 0 CHECK (usage_count_outlier IN (0, 1)),
    customer_id_missing        INTEGER NOT NULL DEFAULT 0 CHECK (customer_id_missing IN (0, 1))
);

-- Justification: customer_id is the join key for usage component of
-- health score (Q3) and churn signal usage-decline detection (Q8).
CREATE INDEX IF NOT EXISTS idx_usage_customer_id   ON product_usage (customer_id);

-- Justification: feature_name GROUP BY in adoption funnel query (Q7).
CREATE INDEX IF NOT EXISTS idx_usage_feature_name  ON product_usage (feature_name);

-- Justification: session_date for time-window filters in Q7 (last 30 days
-- power users) and Q8 (no login in last 14 days).
CREATE INDEX IF NOT EXISTS idx_usage_session_date  ON product_usage (session_date);

-- Justification: compound index for Q8 "no login in 14 days" check which
-- needs both customer_id and MAX(session_date) in one efficient scan.
CREATE INDEX IF NOT EXISTS idx_usage_cust_date     ON product_usage (customer_id, session_date);


-- =============================================================================
-- GOVERNANCE: audit_log  (used in Step 6)
-- =============================================================================

DROP TABLE IF EXISTS audit_log;

CREATE TABLE audit_log (
    id                 INTEGER  PRIMARY KEY  AUTOINCREMENT,
    timestamp          TEXT     NOT NULL,
    user_question      TEXT     NOT NULL,
    tools_used         TEXT,
    generated_sql      TEXT,
    classification     TEXT     NOT NULL DEFAULT 'safe',
    result_summary     TEXT,
    execution_time_ms  REAL,
    approval_status    TEXT     NOT NULL DEFAULT 'not_required',
    reviewer_notes     TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp      ON audit_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_classification ON audit_log (classification);


-- =============================================================================
-- GOVERNANCE: approval_queue  (used in Step 6)
-- =============================================================================

DROP TABLE IF EXISTS approval_queue;

CREATE TABLE approval_queue (
    id              INTEGER  PRIMARY KEY  AUTOINCREMENT,
    timestamp       TEXT     NOT NULL,
    user_question   TEXT     NOT NULL,
    generated_sql   TEXT,
    classification  TEXT     NOT NULL DEFAULT 'requires_review',
    reason          TEXT,
    status          TEXT     NOT NULL DEFAULT 'pending',
    reviewer_notes  TEXT,
    reviewed_at     TEXT,
    result_cache    TEXT
);

CREATE INDEX IF NOT EXISTS idx_queue_status    ON approval_queue (status);
CREATE INDEX IF NOT EXISTS idx_queue_timestamp ON approval_queue (timestamp);