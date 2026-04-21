# FinOps Analytics Platform — ER Diagram
## Star Schema (with one snowflake arm)

```mermaid
erDiagram

    %% ── DIMENSIONS ──────────────────────────────────────────

    date_dim {
        INTEGER date_id       PK
        TEXT    full_date     UK
        INTEGER year
        INTEGER quarter
        INTEGER month
        TEXT    month_name
        INTEGER week_of_year
        INTEGER day_of_month
        INTEGER day_of_week
        TEXT    day_name
        INTEGER is_weekend
        INTEGER is_month_start
        INTEGER is_month_end
        TEXT    fiscal_quarter
    }

    customers {
        TEXT    customer_id   PK
        TEXT    first_name
        TEXT    last_name
        TEXT    email
        TEXT    phone
        TEXT    city
        TEXT    country
        TEXT    zip_code
        TEXT    signup_date
        INTEGER is_active
        TEXT    company
        TEXT    loyalty_tier
        INTEGER email_missing
        INTEGER phone_missing
    }

    subscriptions {
        TEXT    subscription_id  PK
        TEXT    customer_id      FK
        TEXT    plan_name
        REAL    mrr
        TEXT    currency
        TEXT    start_date
        TEXT    end_date
        TEXT    status
        TEXT    billing_cycle
        INTEGER auto_renew
        INTEGER mrr_was_negative
        INTEGER is_future_start
    }

    %% ── FACTS ───────────────────────────────────────────────

    transactions {
        TEXT    transaction_id   PK
        TEXT    customer_id      FK
        TEXT    subscription_id  FK
        TEXT    transaction_date FK
        REAL    amount
        TEXT    currency
        TEXT    status
        TEXT    payment_method
        TEXT    invoice_id
        TEXT    description
        INTEGER was_negative
    }

    invoices {
        TEXT    invoice_id       PK
        TEXT    customer_id      FK
        TEXT    subscription_id  FK
        TEXT    issue_date       FK
        TEXT    due_date
        REAL    subtotal
        REAL    tax
        REAL    total
        REAL    paid_amount
        TEXT    payment_status
        TEXT    paid_date
        TEXT    payment_method
        TEXT    currency
        REAL    tax_rate
        INTEGER tax_error_flag
    }

    support_tickets {
        TEXT    ticket_id         PK
        TEXT    customer_id       FK
        TEXT    category
        TEXT    priority
        TEXT    status
        TEXT    created_at        FK
        TEXT    first_response_at
        TEXT    resolved_at
        REAL    rating
        TEXT    resolution_text
        TEXT    agent_name
        TEXT    channel
        INTEGER is_escalated
    }

    product_usage {
        TEXT    usage_id                  PK
        TEXT    customer_id               FK
        TEXT    feature_name
        TEXT    session_date              FK
        REAL    session_duration_seconds
        REAL    usage_count
        TEXT    device
        TEXT    session_id
        INTEGER customer_id_missing
        INTEGER duration_invalid
        INTEGER usage_count_outlier
    }

    %% ── RELATIONSHIPS ────────────────────────────────────────

    customers        ||--o{ subscriptions   : "has"
    customers        ||--o{ transactions    : "makes"
    customers        ||--o{ invoices        : "receives"
    customers        ||--o{ support_tickets : "raises"
    customers        ||--o{ product_usage   : "generates"

    subscriptions    ||--o{ transactions    : "billed via"
    subscriptions    ||--o{ invoices        : "invoiced as"

    date_dim         ||--o{ transactions    : "on date"
    date_dim         ||--o{ invoices        : "issued on"
    date_dim         ||--o{ support_tickets : "created on"
    date_dim         ||--o{ product_usage   : "session on"
```

## Schema Design Notes

### Why Star (not pure Snowflake)?
Star schema minimises JOIN depth. Most analytical queries only need
2–3 tables. A full snowflake (separate plan_dim, billing_cycle_dim,
country_dim) would add hops with no performance gain in SQLite.

### The one Snowflake arm
`subscriptions` sits between `customers` and the financial fact tables.
A customer can hold multiple subscriptions; each transaction and invoice
belongs to exactly one subscription. This models SaaS billing correctly:
the subscription is the billing unit, not the customer.

### date_dim
Avoids `strftime()` scattered across every query. Analysts join on
`full_date` and filter/group on `year`, `quarter`, `month` columns.
Covers 2021-01-01 → 2026-12-31 (2,192 rows).

### Fact table grains
| Table            | Grain                               |
|------------------|-------------------------------------|
| transactions     | One row per payment attempt         |
| invoices         | One row per billing document        |
| support_tickets  | One row per support interaction     |
| product_usage    | One row per user session event      |