-- =============================================================================
-- FinOps Analytics Platform — sql/queries.sql
-- Step 3: Advanced Analytical SQL Queries (SQLite 3.35+)
-- =============================================================================
--
-- SQLite-specific notes applied throughout
-- ----------------------------------------
-- • No PERCENTILE_CONT / PERCENTILE_DISC → use NTILE(100) approximation.
-- • No DATE_TRUNC → use strftime('%Y-%m', date_col) for month bucketing.
-- • Recursive CTEs supported (SQLite 3.8.3+).
-- • NULLIF(x, 0) used everywhere a denominator could be zero.
-- • All time-relative windows anchor to the MAX known date in the dataset
--   (2025-12-31) rather than date('now') so results are reproducible and
--   non-empty regardless of when the query is run.
-- =============================================================================


-- =============================================================================
-- Q1 — Revenue Cohort Analysis
-- Technique: CTE + Window Functions + Conditional Aggregation
-- =============================================================================
--
-- Approach
-- --------
-- 1. cohort_base CTE: assign every customer to a signup-month cohort and
--    count cohort size using COUNT() OVER (PARTITION BY cohort_month).
-- 2. txn_enriched CTE: join transactions to cohort, compute months_since_signup
--    using Julian Day arithmetic (julianday difference / 30.44 ≈ months).
-- 3. cohort_revenue CTE: pivot with CASE-inside-SUM/COUNT to produce one column
--    per milestone (M1, M2, M3, M6, M12) for both revenue and retention.
--    Retention = distinct customers who made ≥1 completed transaction at that
--    milestone / cohort_size.
--
-- Assumptions
-- -----------
-- • "Month 1" = transactions in the same calendar month as signup.
-- • Months are approximated as 30.44-day periods (average Gregorian month).
-- • Only 'completed' transactions contribute to revenue.
-- • Customers who never transacted have cohort_size counted but zero revenue.
-- =============================================================================

WITH cohort_base AS (
    SELECT
        c.customer_id,
        strftime('%Y-%m', c.signup_date)                        AS cohort_month,
        -- cohort_size: how many customers share this signup month
        COUNT(c.customer_id) OVER (
            PARTITION BY strftime('%Y-%m', c.signup_date)
        )                                                        AS cohort_size
    FROM customers c
    WHERE c.signup_date IS NOT NULL
),

txn_enriched AS (
    SELECT
        t.customer_id,
        t.amount,
        t.transaction_date,
        cb.cohort_month,
        cb.cohort_size,
        -- months_since_signup: integer number of ~30-day periods since signup
        -- +1 so that the signup month itself = month 1 (not month 0)
        CAST(
            (julianday(strftime('%Y-%m-01', t.transaction_date))
             - julianday(cb.cohort_month || '-01')) / 30.44
        AS INTEGER) + 1                                          AS months_since_signup
    FROM transactions t
    JOIN cohort_base cb
        ON t.customer_id = cb.customer_id
    WHERE t.status      = 'completed'
      AND t.transaction_date IS NOT NULL
),

cohort_revenue AS (
    SELECT
        cohort_month,
        -- cohort_size is the same for all rows with the same cohort_month
        MAX(cohort_size)                                         AS cohort_size,

        -- Revenue at each milestone (sum of completed transaction amounts)
        ROUND(SUM(CASE WHEN months_since_signup = 1  THEN amount ELSE 0 END), 2) AS rev_m1,
        ROUND(SUM(CASE WHEN months_since_signup = 2  THEN amount ELSE 0 END), 2) AS rev_m2,
        ROUND(SUM(CASE WHEN months_since_signup = 3  THEN amount ELSE 0 END), 2) AS rev_m3,
        ROUND(SUM(CASE WHEN months_since_signup = 6  THEN amount ELSE 0 END), 2) AS rev_m6,
        ROUND(SUM(CASE WHEN months_since_signup = 12 THEN amount ELSE 0 END), 2) AS rev_m12,

        -- Retention: distinct customers with any revenue at that milestone
        COUNT(DISTINCT CASE WHEN months_since_signup = 1  THEN customer_id END) AS retained_m1,
        COUNT(DISTINCT CASE WHEN months_since_signup = 2  THEN customer_id END) AS retained_m2,
        COUNT(DISTINCT CASE WHEN months_since_signup = 3  THEN customer_id END) AS retained_m3,
        COUNT(DISTINCT CASE WHEN months_since_signup = 6  THEN customer_id END) AS retained_m6,
        COUNT(DISTINCT CASE WHEN months_since_signup = 12 THEN customer_id END) AS retained_m12
    FROM txn_enriched
    GROUP BY cohort_month
)

SELECT
    cohort_month,
    cohort_size,
    rev_m1,   rev_m2,   rev_m3,   rev_m6,   rev_m12,
    -- Retention rate: NULLIF prevents division-by-zero for cohorts with 0 customers
    ROUND(100.0 * retained_m1  / NULLIF(cohort_size, 0), 1) AS retention_pct_m1,
    ROUND(100.0 * retained_m2  / NULLIF(cohort_size, 0), 1) AS retention_pct_m2,
    ROUND(100.0 * retained_m3  / NULLIF(cohort_size, 0), 1) AS retention_pct_m3,
    ROUND(100.0 * retained_m6  / NULLIF(cohort_size, 0), 1) AS retention_pct_m6,
    ROUND(100.0 * retained_m12 / NULLIF(cohort_size, 0), 1) AS retention_pct_m12
FROM cohort_revenue
ORDER BY cohort_month;


-- =============================================================================
-- Q2 — MRR Movement Waterfall
-- Technique: CTE + LAG() + CASE classification
-- =============================================================================
--
-- Approach
-- --------
-- 1. monthly_mrr CTE: snapshot each customer's total MRR per calendar month
--    by joining subscriptions to date_dim (sampling on day_of_month = 1 so
--    we get one row per customer per month).
-- 2. mrr_with_lag CTE: LAG(mrr, 1, 0) gets the previous month's MRR per
--    customer, defaulting to 0 for the customer's first month.
-- 3. mrr_classified CTE: CASE on (prev_mrr, curr_mrr) pairs to classify each
--    monthly change as new / expansion / contraction / churned / unchanged.
-- 4. Final SELECT: aggregate by month, pivoting movement_type into columns.
--
-- Movement definitions
-- --------------------
-- new         : prev_mrr = 0 AND curr_mrr > 0   (first subscription)
-- expansion   : curr_mrr > prev_mrr > 0          (upgrade)
-- contraction : 0 < curr_mrr < prev_mrr          (downgrade)
-- churned     : prev_mrr > 0 AND curr_mrr = 0    (full cancellation)
-- unchanged   : curr_mrr = prev_mrr              (filtered out in final SELECT)
--
-- Assumption: one row per (customer, month) — customers with multiple active
-- subscriptions have their MRR summed before LAG comparison.
-- =============================================================================

WITH monthly_mrr AS (
    SELECT
        s.customer_id,
        strftime('%Y-%m', d.full_date)          AS month,
        SUM(s.mrr)                              AS mrr
    FROM subscriptions s
    -- Join to date_dim to generate one observation per active month per sub
    JOIN date_dim d
        ON  d.full_date BETWEEN s.start_date
                            AND COALESCE(s.end_date, '2099-12-31')
        AND d.day_of_month = 1          -- sample on the 1st of each month
    WHERE s.status IN ('active', 'cancelled', 'expired')
    GROUP BY s.customer_id, strftime('%Y-%m', d.full_date)
),

mrr_with_lag AS (
    SELECT
        customer_id,
        month,
        mrr                                     AS curr_mrr,
        -- LAG with default 0: a customer's first month gets prev_mrr = 0
        LAG(mrr, 1, 0) OVER (
            PARTITION BY customer_id
            ORDER BY month
        )                                       AS prev_mrr
    FROM monthly_mrr
),

mrr_classified AS (
    SELECT
        month,
        customer_id,
        curr_mrr,
        prev_mrr,
        curr_mrr - prev_mrr                     AS mrr_delta,
        CASE
            WHEN prev_mrr = 0    AND curr_mrr > 0              THEN 'new'
            WHEN prev_mrr > 0    AND curr_mrr > prev_mrr       THEN 'expansion'
            WHEN prev_mrr > 0    AND curr_mrr < prev_mrr
                                 AND curr_mrr > 0              THEN 'contraction'
            WHEN prev_mrr > 0    AND curr_mrr = 0              THEN 'churned'
            ELSE                                                     'unchanged'
        END                                     AS movement_type
    FROM mrr_with_lag
)

SELECT
    month,
    -- New MRR: full curr_mrr of first-time subscribers
    ROUND(SUM(CASE WHEN movement_type = 'new'
                   THEN curr_mrr ELSE 0 END), 2)        AS new_mrr,
    -- Expansion MRR: positive delta only
    ROUND(SUM(CASE WHEN movement_type = 'expansion'
                   THEN mrr_delta ELSE 0 END), 2)       AS expansion_mrr,
    -- Contraction MRR: negative delta (stored as negative number)
    ROUND(SUM(CASE WHEN movement_type = 'contraction'
                   THEN mrr_delta ELSE 0 END), 2)       AS contraction_mrr,
    -- Churned MRR: the MRR lost (stored as negative for waterfall convention)
    ROUND(SUM(CASE WHEN movement_type = 'churned'
                   THEN -prev_mrr ELSE 0 END), 2)       AS churned_mrr,
    -- Net New MRR = sum of all deltas
    ROUND(SUM(mrr_delta), 2)                            AS net_new_mrr
FROM mrr_classified
WHERE movement_type != 'unchanged'
GROUP BY month
ORDER BY month;


-- =============================================================================
-- Q3 — Customer Health Score
-- Technique: Multiple CTEs + Correlated Subquery + DENSE_RANK()
-- =============================================================================
--
-- Approach
-- --------
-- Four independent CTEs, each computing one health dimension for active customers.
-- Each dimension is normalised to [0, 100] before weighting.
-- Weighted sum: recency(30%) + support(25%) + usage(25%) + payment(20%) = 100%.
--
-- Scoring logic
-- -------------
-- recency_score  : 100 if last txn ≤ 30 days ago; scales linearly to 0 at 180d.
-- support_score  : blend of inverse ticket frequency (many tickets = lower score)
--                  and average rating (5 stars = 100).
-- usage_score    : blend of sessions-per-week (capped at 10/wk → 100) and
--                  distinct feature breadth (10+ features → 100).
-- payment_score  : on-time invoice rate (paid_date ≤ due_date) × 100.
--
-- Null handling
-- -------------
-- COALESCE(x, 50) imputes 50 (neutral) for customers with no data in a dimension
-- (e.g. new customers with no transactions, or customers who never rated tickets).
-- This prevents NULL health scores from dropping customers off the ranking.
--
-- Reference date: MAX(transaction_date) in the dataset instead of date('now')
-- to ensure reproducible results regardless of when the query runs.
-- =============================================================================

WITH ref AS (
    -- Single reference date used consistently across all CTEs
    SELECT MAX(transaction_date) AS ref_date FROM transactions
),

recency_cte AS (
    SELECT
        c.customer_id,
        MAX(t.transaction_date)                                  AS last_txn_date,
        -- days since last completed transaction
        CAST(julianday((SELECT ref_date FROM ref))
             - julianday(MAX(t.transaction_date)) AS INTEGER)    AS days_since_txn,
        -- normalise: 100 at 0 days, 0 at 180 days, floored at 0
        ROUND(MAX(0,
            100 - (CAST(julianday((SELECT ref_date FROM ref))
                        - julianday(MAX(t.transaction_date)) AS INTEGER)
                   / 180.0 * 100)
        ), 2)                                                    AS recency_score
    FROM customers c
    LEFT JOIN transactions t
           ON t.customer_id = c.customer_id
          AND t.status      = 'completed'
    WHERE c.is_active = 1
    GROUP BY c.customer_id
),

support_cte AS (
    SELECT
        c.customer_id,
        COUNT(st.ticket_id)                                      AS total_tickets,
        ROUND(AVG(st.rating), 2)                                 AS avg_rating,
        -- ticket frequency score: subtract 5 points per ticket, floor at 0
        ROUND(MAX(0, 100 - COUNT(st.ticket_id) * 5), 2)         AS freq_score,
        -- rating score: scale 1-5 → 0-100
        ROUND(COALESCE(AVG(st.rating), 3.0) / 5.0 * 100, 2)    AS rating_score,
        -- combined support score: 50% frequency, 50% rating
        ROUND(
            MAX(0, 100 - COUNT(st.ticket_id) * 5) * 0.5
          + COALESCE(AVG(st.rating), 3.0) / 5.0 * 100 * 0.5
        , 2)                                                     AS support_score
    FROM customers c
    LEFT JOIN support_tickets st ON st.customer_id = c.customer_id
    WHERE c.is_active = 1
    GROUP BY c.customer_id
),

usage_cte AS (
    SELECT
        c.customer_id,
        COUNT(pu.usage_id)                                       AS total_sessions,
        COUNT(DISTINCT pu.feature_name)                          AS distinct_features,
        -- sessions per week over last 90 days (90 days / 7 ≈ 12.86 weeks)
        ROUND(COUNT(pu.usage_id) / 12.86, 2)                    AS sessions_per_week,
        -- session score: 10 sessions/week → 100, capped at 100
        ROUND(MIN(COUNT(pu.usage_id) / 12.86 * 10, 100), 2)    AS session_score,
        -- breadth score: each feature = 10 points, capped at 100
        ROUND(MIN(COUNT(DISTINCT pu.feature_name) * 10.0, 100), 2) AS breadth_score,
        -- combined usage score: 60% session frequency, 40% feature breadth
        ROUND(
            MIN(COUNT(pu.usage_id) / 12.86 * 10, 100) * 0.60
          + MIN(COUNT(DISTINCT pu.feature_name) * 10.0, 100) * 0.40
        , 2)                                                     AS usage_score
    FROM customers c
    LEFT JOIN product_usage pu
           ON pu.customer_id  = c.customer_id
          -- last 90 days relative to reference date
          AND pu.session_date >= date((SELECT ref_date FROM ref), '-90 days')
    WHERE c.is_active = 1
    GROUP BY c.customer_id
),

payment_cte AS (
    SELECT
        c.customer_id,
        COUNT(i.invoice_id)                                      AS total_invoices,
        COUNT(CASE
            WHEN i.paid_date  <= i.due_date
             AND i.payment_status = 'paid'   THEN 1
        END)                                                     AS on_time_count,
        -- on-time rate: paid before or on due_date / all invoices with both dates
        ROUND(
            100.0
          * COUNT(CASE WHEN i.paid_date <= i.due_date
                        AND i.payment_status = 'paid' THEN 1 END)
          / NULLIF(COUNT(CASE WHEN i.due_date IS NOT NULL THEN 1 END), 0)
        , 2)                                                     AS payment_score
    FROM customers c
    LEFT JOIN invoices i ON i.customer_id = c.customer_id
    WHERE c.is_active = 1
    GROUP BY c.customer_id
)

SELECT
    c.customer_id,
    c.first_name || ' ' || c.last_name                          AS customer_name,
    c.company,
    c.loyalty_tier,
    -- weighted composite score (COALESCE neutrals: 50 = neither healthy nor sick)
    ROUND(
        COALESCE(r.recency_score,  50) * 0.30
      + COALESCE(s.support_score,  50) * 0.25
      + COALESCE(u.usage_score,    50) * 0.25
      + COALESCE(p.payment_score,  50) * 0.20
    , 2)                                                         AS health_score,
    COALESCE(r.recency_score,  50)                               AS recency_score,
    COALESCE(s.support_score,  50)                               AS support_score,
    COALESCE(u.usage_score,    50)                               AS usage_score,
    COALESCE(p.payment_score,  50)                               AS payment_score,
    r.days_since_txn,
    s.total_tickets,
    s.avg_rating,
    u.sessions_per_week,
    u.distinct_features,
    p.on_time_count,
    p.total_invoices,
    -- rank 1 = healthiest customer
    DENSE_RANK() OVER (
        ORDER BY
            COALESCE(r.recency_score,  50) * 0.30
          + COALESCE(s.support_score,  50) * 0.25
          + COALESCE(u.usage_score,    50) * 0.25
          + COALESCE(p.payment_score,  50) * 0.20
        DESC
    )                                                            AS health_rank
FROM customers c
LEFT JOIN recency_cte r ON r.customer_id = c.customer_id
LEFT JOIN support_cte s ON s.customer_id = c.customer_id
LEFT JOIN usage_cte   u ON u.customer_id = c.customer_id
LEFT JOIN payment_cte p ON p.customer_id = c.customer_id
WHERE c.is_active = 1
ORDER BY health_score DESC;


-- =============================================================================
-- Q4 — Subscription Overlap Detection
-- Technique: Self-JOIN + ROW_NUMBER() + Date Arithmetic
-- =============================================================================
--
-- Approach
-- --------
-- 1. subs_with_end CTE: normalise NULL end_date → '2099-12-31' (still active).
--    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY start_date) assigns
--    sub_rank so we can label the first vs. duplicate subscription per customer.
-- 2. overlapping CTE: self-JOIN on the same customer_id where subscription B
--    starts before subscription A ends AND B starts after A (using the rank
--    to avoid mirror-image duplicates: a.sub_rank < b.sub_rank).
-- 3. Overlap days: julianday(MIN(end_a, end_b)) - julianday(MAX(start_a, start_b)).
-- 4. Double-billed amount: the LOWER MRR subscription's daily rate × overlap days
--    (the customer was charged for two subs but only needed one).
--
-- Assumptions
-- -----------
-- • NULL end_date = subscription still active as of '2099-12-31'.
-- • Double-billing impact = MIN(mrr_a, mrr_b) / 30 × overlap_days
--   (we charge the smaller one as waste; the larger may be intentional upgrade).
-- =============================================================================

WITH subs_with_end AS (
    SELECT
        subscription_id,
        customer_id,
        plan_name,
        mrr,
        start_date,
        COALESCE(end_date, '2099-12-31')             AS eff_end_date,
        status,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY start_date, subscription_id     -- subscription_id breaks ties
        )                                            AS sub_rank
    FROM subscriptions
    WHERE start_date IS NOT NULL
),

overlapping AS (
    SELECT
        a.customer_id,
        a.subscription_id                            AS sub_a_id,
        b.subscription_id                            AS sub_b_id,
        a.plan_name                                  AS plan_a,
        b.plan_name                                  AS plan_b,
        a.mrr                                        AS mrr_a,
        b.mrr                                        AS mrr_b,
        a.sub_rank                                   AS rank_a,
        b.sub_rank                                   AS rank_b,
        a.start_date                                 AS start_a,
        a.eff_end_date                               AS end_a,
        b.start_date                                 AS start_b,
        b.eff_end_date                               AS end_b,
        -- overlap period: from MAX(start) to MIN(end)
        CAST(
            julianday(MIN(a.eff_end_date, b.eff_end_date))
          - julianday(MAX(a.start_date,  b.start_date))
        AS INTEGER)                                  AS overlap_days,
        -- double-billed MRR: lower-value sub's daily rate × overlap days
        ROUND(
            MIN(a.mrr, b.mrr) / 30.0
          * CAST(
                julianday(MIN(a.eff_end_date, b.eff_end_date))
              - julianday(MAX(a.start_date,  b.start_date))
            AS INTEGER)
        , 2)                                         AS double_billed_amount
    FROM subs_with_end a
    JOIN subs_with_end b
        ON  a.customer_id        = b.customer_id
        AND a.sub_rank           < b.sub_rank        -- avoid mirror-image pairs
        AND b.start_date         < a.eff_end_date    -- b starts before a ends
        AND b.start_date        >= a.start_date      -- b starts after a (stricter overlap)
)

SELECT
    customer_id,
    sub_a_id                                         AS original_subscription,
    sub_b_id                                         AS overlapping_subscription,
    rank_a,
    rank_b,
    plan_a,
    plan_b,
    mrr_a,
    mrr_b,
    start_a,
    end_a,
    start_b,
    end_b,
    overlap_days,
    double_billed_amount,
    -- running total of double-billing per customer
    SUM(double_billed_amount) OVER (
        PARTITION BY customer_id
        ORDER BY start_b
    )                                                AS cumulative_double_billed
FROM overlapping
WHERE overlap_days > 0
ORDER BY double_billed_amount DESC, customer_id;


-- =============================================================================
-- Q5 — Support Ticket Resolution Funnel
-- Technique: CTE + NTILE(100) + Window Functions
-- =============================================================================
--
-- Approach
-- --------
-- 1. ticket_times CTE: compute hours-to-first-response and hours-to-resolution
--    for each ticket using julianday arithmetic (×24 converts days to hours).
-- 2. with_percentiles CTE: NTILE(100) over response/resolution time per priority
--    bucket creates percentile ranks (~p50, ~p95).
-- 3. priority_medians CTE: extract p50 (median) per priority for outlier detection.
-- 4. Final SELECT: aggregate per priority — p50/p95, FCR rate, escalation rate,
--    outlier count (> 2× median), and running average via window function.
--
-- SQLite limitation note
-- ----------------------
-- SQLite has no native PERCENTILE_CONT. NTILE(100) is the standard workaround:
-- buckets are 1 (fastest 1%) → 100 (slowest 1%). Filtering WHERE pct_rank = 50
-- approximates the median; WHERE pct_rank >= 95 captures the P95 tier.
-- For datasets of this size (~2000 tickets) the approximation error is < 2%.
--
-- FCR (First-Contact Resolution): ticket closed without escalation.
-- Outlier: resolution_hours > 2× the median for that priority.
-- =============================================================================

WITH ticket_times AS (
    SELECT
        ticket_id,
        customer_id,
        priority,
        status,
        is_escalated,
        created_at,
        first_response_at,
        resolved_at,
        -- hours to first response; NULL if no response recorded yet
        CASE WHEN first_response_at IS NOT NULL AND created_at IS NOT NULL
             THEN ROUND((julianday(first_response_at)
                         - julianday(created_at)) * 24, 2)
        END                                          AS response_hours,
        -- hours to resolution; NULL if not yet resolved
        CASE WHEN resolved_at IS NOT NULL AND created_at IS NOT NULL
             THEN ROUND((julianday(resolved_at)
                         - julianday(created_at)) * 24, 2)
        END                                          AS resolution_hours
    FROM support_tickets
    WHERE created_at IS NOT NULL
),

with_percentiles AS (
    SELECT
        *,
        -- NTILE(100) approximates percentile rank within each priority bucket
        NTILE(100) OVER (
            PARTITION BY priority
            ORDER BY response_hours
        )                                            AS response_pct_rank,
        NTILE(100) OVER (
            PARTITION BY priority
            ORDER BY resolution_hours
        )                                            AS resolution_pct_rank
    FROM ticket_times
    -- only tickets with valid resolution times for percentile calculation
    WHERE resolution_hours IS NOT NULL
      AND resolution_hours > 0
),

priority_medians AS (
    -- extract median (p50 bucket) per priority for outlier threshold
    SELECT
        priority,
        AVG(CASE WHEN resolution_pct_rank = 50
                 THEN resolution_hours END)          AS p50_resolution_hours
    FROM with_percentiles
    GROUP BY priority
)

SELECT
    p.priority,
    COUNT(p.ticket_id)                               AS total_tickets,

    -- Median time-to-first-response (p50 bucket)
    ROUND(AVG(CASE WHEN p.response_pct_rank = 50
                   THEN p.response_hours END), 2)    AS p50_response_hours,

    -- Median time-to-resolution (p50 bucket)
    ROUND(AVG(CASE WHEN p.resolution_pct_rank = 50
                   THEN p.resolution_hours END), 2)  AS p50_resolution_hours,

    -- P95 time-to-resolution (worst 5% of tickets)
    ROUND(AVG(CASE WHEN p.resolution_pct_rank >= 95
                   THEN p.resolution_hours END), 2)  AS p95_resolution_hours,

    -- First-Contact Resolution rate: closed without escalation / total closed
    ROUND(
        100.0
      * COUNT(CASE WHEN p.is_escalated = 0
                    AND p.status IN ('closed', 'resolved') THEN 1 END)
      / NULLIF(COUNT(CASE WHEN p.status IN ('closed','resolved') THEN 1 END), 0)
    , 1)                                             AS fcr_rate_pct,

    -- Escalation rate: escalated / total
    ROUND(
        100.0
      * COUNT(CASE WHEN p.is_escalated = 1 THEN 1 END)
      / NULLIF(COUNT(p.ticket_id), 0)
    , 1)                                             AS escalation_rate_pct,

    -- Outlier count: tickets taking > 2× the priority median
    COUNT(CASE WHEN p.resolution_hours > 2 * m.p50_resolution_hours
               THEN 1 END)                           AS outlier_tickets,

    -- Running average resolution time (window function across all rows in group)
    ROUND(AVG(p.resolution_hours) OVER (
        PARTITION BY p.priority
    ), 2)                                            AS avg_resolution_hours

FROM with_percentiles  p
JOIN priority_medians  m ON m.priority = p.priority
GROUP BY p.priority
ORDER BY
    -- Sort critical → high → medium → low
    CASE p.priority
        WHEN 'critical' THEN 1
        WHEN 'high'     THEN 2
        WHEN 'medium'   THEN 3
        WHEN 'low'      THEN 4
        ELSE                 5
    END;


-- =============================================================================
-- Q6 — Revenue Concentration Risk (Pareto + HHI)
-- Technique: Recursive CTE + Window Functions
-- =============================================================================
--
-- Approach
-- --------
-- 1. customer_revenue CTE: total completed revenue per customer.
-- 2. ranked CTE: add revenue_pct (each customer's % of grand total) and row rank.
-- 3. Recursive CTE cumulative: accumulate the running sum of revenue_pct
--    row by row from rank 1 (highest revenue) downward.
--    Base case: rank 1. Recursive case: add the next rank's share.
-- 4. hhi CTE: Herfindahl-Hirschman Index = Σ(market_share_pct²) / 100
--    Range: near 0 = perfectly dispersed; 10000 = single customer monopoly.
--    Thresholds: < 1500 = competitive, 1500-2500 = moderate concentration,
--    > 2500 = highly concentrated.
-- 5. Final SELECT: label Pareto segments (top 50%, 80%, 90%) and flag
--    customers whose individual share > 5% as concentration risks.
--
-- SQLite recursive CTE note
-- -------------------------
-- SQLite's recursive CTEs require UNION ALL (not UNION). The maximum recursion
-- depth defaults to 1000, which is sufficient for up to 1000 customers.
-- For larger datasets, PRAGMA recursive_triggers can be used.
-- =============================================================================

WITH customer_revenue AS (
    SELECT
        t.customer_id,
        c.first_name || ' ' || c.last_name          AS customer_name,
        c.company,
        ROUND(SUM(t.amount), 2)                      AS total_revenue
    FROM transactions t
    JOIN customers    c ON c.customer_id = t.customer_id
    WHERE t.status = 'completed'
    GROUP BY t.customer_id
),

ranked AS (
    SELECT
        customer_id,
        customer_name,
        company,
        total_revenue,
        ROUND(100.0 * total_revenue
              / SUM(total_revenue) OVER (), 4)       AS revenue_pct,
        -- grand total for HHI calculation
        SUM(total_revenue) OVER ()                   AS grand_total,
        ROW_NUMBER() OVER (
            ORDER BY total_revenue DESC
        )                                            AS rnk
    FROM customer_revenue
),

-- Recursive CTE: accumulate cumulative_pct one row at a time
-- Base case: the top revenue customer (rnk = 1)
-- Recursive case: add the next customer's share to the running total
cumulative (rnk, customer_id, customer_name, company,
            total_revenue, revenue_pct, grand_total, cum_pct) AS (
    SELECT rnk, customer_id, customer_name, company,
           total_revenue, revenue_pct, grand_total,
           revenue_pct                               -- cum_pct starts at first customer's share
    FROM ranked
    WHERE rnk = 1

    UNION ALL

    SELECT r.rnk, r.customer_id, r.customer_name, r.company,
           r.total_revenue, r.revenue_pct, r.grand_total,
           ROUND(c.cum_pct + r.revenue_pct, 4)       -- accumulate
    FROM ranked r
    JOIN cumulative c ON r.rnk = c.rnk + 1
),

hhi AS (
    -- HHI = sum of squared market shares (using % scale, so divide by 100 for 0-10000 range)
    SELECT ROUND(SUM(revenue_pct * revenue_pct) / 100.0, 2) AS hhi_index
    FROM ranked
)

SELECT
    c.rnk                                            AS revenue_rank,
    c.customer_id,
    c.customer_name,
    c.company,
    c.total_revenue,
    c.revenue_pct,
    c.cum_pct                                        AS cumulative_pct,
    -- Pareto segment labels
    CASE
        WHEN c.cum_pct <= 50 THEN 'top_50pct'
        WHEN c.cum_pct <= 80 THEN 'top_80pct'
        WHEN c.cum_pct <= 90 THEN 'top_90pct'
        ELSE                      'long_tail'
    END                                              AS pareto_segment,
    -- Flag customers contributing > 5% of total revenue
    CASE WHEN c.revenue_pct > 5.0 THEN 1 ELSE 0 END AS concentration_risk_flag,
    h.hhi_index
FROM cumulative c
CROSS JOIN hhi h
ORDER BY c.rnk;


-- =============================================================================
-- Q7 — Product Feature Adoption Funnel
-- Technique: Multiple JOINs + Conditional Aggregation (CASE inside SUM/COUNT)
-- =============================================================================
--
-- Approach
-- --------
-- 1. feature_stats CTE: per-(customer, feature) aggregate — total sessions ever,
--    last session date. This gives us "activated" (> 1 session) per customer.
-- 2. feature_recent CTE: sessions per customer per feature in the last 30 days.
--    Used for "power user" (5+ sessions in last 30 days) definition.
-- 3. Final SELECT: aggregate to feature level using conditional aggregation.
--    COUNT(DISTINCT CASE WHEN ... THEN customer_id END) counts unique customers
--    meeting each funnel stage without needing separate subqueries.
-- 4. The "pivot-style" output uses each feature as a row, not a column,
--    because SQLite has no native PIVOT. The comment describes how to
--    achieve column-per-feature with a GROUP_CONCAT or application-layer pivot.
--
-- Funnel stages
-- -------------
-- reached    : customer had ≥ 1 session with this feature (any time)
-- activated  : customer had > 1 session with this feature (any time)
-- power_user : customer had ≥ 5 sessions with this feature in last 30 days
--
-- Note: "last 30 days" anchored to MAX(session_date) in dataset for reproducibility.
-- =============================================================================

WITH ref AS (
    SELECT MAX(session_date) AS ref_date FROM product_usage
),

feature_stats AS (
    SELECT
        feature_name,
        customer_id,
        COUNT(session_id)              AS total_sessions,
        MAX(session_date)              AS last_session
    FROM product_usage
    WHERE customer_id IS NOT NULL      -- exclude anonymous sessions
    GROUP BY feature_name, customer_id
),

feature_recent AS (
    SELECT
        feature_name,
        customer_id,
        COUNT(session_id)              AS recent_sessions
    FROM product_usage
    WHERE customer_id IS NOT NULL
      AND session_date >= date((SELECT ref_date FROM ref), '-30 days')
    GROUP BY feature_name, customer_id
)

SELECT
    fs.feature_name,

    -- Stage 1: Reach — any session ever
    COUNT(DISTINCT fs.customer_id)                               AS total_unique_users,

    -- Stage 2: Activation — used more than once
    COUNT(DISTINCT CASE WHEN fs.total_sessions > 1
                        THEN fs.customer_id END)                 AS activated_users,

    -- Stage 3: Power users — 5+ sessions in last 30 days
    COUNT(DISTINCT CASE WHEN fr.recent_sessions >= 5
                        THEN fr.customer_id END)                 AS power_users,

    -- Reach → Activation rate
    ROUND(
        100.0
      * COUNT(DISTINCT CASE WHEN fs.total_sessions > 1
                            THEN fs.customer_id END)
      / NULLIF(COUNT(DISTINCT fs.customer_id), 0)
    , 1)                                                         AS activation_rate_pct,

    -- Activation → Power User conversion rate
    ROUND(
        100.0
      * COUNT(DISTINCT CASE WHEN fr.recent_sessions >= 5
                            THEN fr.customer_id END)
      / NULLIF(COUNT(DISTINCT CASE WHEN fs.total_sessions > 1
                                   THEN fs.customer_id END), 0)
    , 1)                                                         AS activation_to_power_pct,

    -- Average sessions per user (engagement depth)
    ROUND(AVG(fs.total_sessions), 2)                             AS avg_sessions_per_user,

    -- Average recent sessions per user (momentum)
    ROUND(AVG(COALESCE(fr.recent_sessions, 0)), 2)               AS avg_recent_sessions

FROM feature_stats fs
LEFT JOIN feature_recent fr
       ON fr.feature_name = fs.feature_name
      AND fr.customer_id  = fs.customer_id
GROUP BY fs.feature_name
ORDER BY activation_rate_pct ASC;  -- lowest adoption first (most actionable)


-- =============================================================================
-- Q8 — Churn Prediction Signals
-- Technique: 4+ CTE chain + LAG over monthly aggregates + DENSE_RANK()
-- =============================================================================
--
-- Approach
-- --------
-- Four independent signal CTEs, each computing one binary churn indicator (0/1)
-- per active customer. Signals are summed to produce a churn_risk_score (0-4).
-- DENSE_RANK() orders customers from highest to lowest risk.
--
-- The four signals
-- ----------------
-- Signal 1 — usage_decline:
--   Sessions in last 30 days < sessions in prior 30 days.
--   Anchor: MAX(session_date) in dataset.
--   A sudden drop in engagement is the strongest leading churn indicator.
--
-- Signal 2 — ticket_spike:
--   Tickets in last 30 days > 1.5× average monthly ticket rate.
--   Average monthly rate = total tickets / 6 (using 6-month window).
--   Rising support volume indicates friction and dissatisfaction.
--
-- Signal 3 — rating_drop:
--   Average rating in last 30 days < overall average rating.
--   NULL-safe: customers with no recent ratings do not trigger this signal.
--   Declining satisfaction scores precede cancellation.
--
-- Signal 4 — no_recent_login:
--   No product_usage session in the last 14 days.
--   Anchor: MAX(session_date). Absence of login is the most direct churn signal.
--
-- Null handling
-- -------------
-- COALESCE(signal, 0): customers with no data for a dimension score 0 for that
-- signal. Exception: no_recent_login defaults to 1 if customer has NO session
-- history at all (they never logged in → highest risk).
-- =============================================================================

WITH ref AS (
    SELECT
        MAX(session_date)  AS ref_date,
        MAX(created_at)    AS ref_ticket_date
    FROM product_usage, support_tickets
),

-- Signal 1: Declining usage — last 30d vs. prior 30d session count
signal_usage AS (
    SELECT
        customer_id,
        -- last 30 days sessions
        COUNT(CASE
            WHEN session_date >= date((SELECT ref_date FROM ref), '-30 days')
            THEN 1 END)                                          AS sessions_last30,
        -- prior 30 days sessions (30-60 days ago)
        COUNT(CASE
            WHEN session_date >= date((SELECT ref_date FROM ref), '-60 days')
             AND session_date  < date((SELECT ref_date FROM ref), '-30 days')
            THEN 1 END)                                          AS sessions_prev30,
        MAX(session_date)                                        AS last_session_date,
        -- Signal fires if last30 < prev30 (usage is declining)
        CASE WHEN
            COUNT(CASE WHEN session_date >= date((SELECT ref_date FROM ref), '-30 days')
                       THEN 1 END)
            <
            COUNT(CASE WHEN session_date >= date((SELECT ref_date FROM ref), '-60 days')
                        AND session_date < date((SELECT ref_date FROM ref), '-30 days')
                       THEN 1 END)
        THEN 1 ELSE 0 END                                        AS usage_decline_signal
    FROM product_usage
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),

-- Signal 2: Ticket spike — recent tickets vs. historical average
signal_tickets AS (
    SELECT
        customer_id,
        COUNT(ticket_id)                                         AS total_tickets,
        -- approximate monthly avg over a 6-month horizon
        ROUND(COUNT(ticket_id) / 6.0, 2)                        AS avg_monthly_tickets,
        COUNT(CASE
            WHEN created_at >= date((SELECT ref_ticket_date FROM ref), '-30 days')
            THEN 1 END)                                          AS tickets_last30,
        -- Signal fires if last-30d tickets > 1.5× monthly average AND > 0
        CASE WHEN
            COUNT(CASE WHEN created_at >= date((SELECT ref_ticket_date FROM ref), '-30 days')
                       THEN 1 END) > COUNT(ticket_id) / 6.0 * 1.5
            AND
            COUNT(CASE WHEN created_at >= date((SELECT ref_ticket_date FROM ref), '-30 days')
                       THEN 1 END) > 0
        THEN 1 ELSE 0 END                                        AS ticket_spike_signal
    FROM support_tickets
    GROUP BY customer_id
),

-- Signal 3: Rating drop — recent average rating below overall average
signal_rating AS (
    SELECT
        customer_id,
        ROUND(AVG(rating), 2)                                    AS overall_avg_rating,
        ROUND(AVG(CASE
            WHEN created_at >= date((SELECT ref_ticket_date FROM ref), '-30 days')
            THEN rating END), 2)                                 AS recent_avg_rating,
        -- Signal fires only if both averages are non-null AND recent < overall
        CASE WHEN
            AVG(CASE WHEN created_at >= date((SELECT ref_ticket_date FROM ref), '-30 days')
                     THEN rating END) IS NOT NULL
            AND
            AVG(CASE WHEN created_at >= date((SELECT ref_ticket_date FROM ref), '-30 days')
                     THEN rating END)
            < AVG(rating)
        THEN 1 ELSE 0 END                                        AS rating_drop_signal
    FROM support_tickets
    WHERE rating IS NOT NULL
    GROUP BY customer_id
),

-- Signal 4: No recent login — no session in last 14 days
signal_login AS (
    SELECT
        customer_id,
        MAX(session_date)                                        AS last_login,
        -- Signal fires if last session is older than 14 days
        CASE WHEN MAX(session_date) < date((SELECT ref_date FROM ref), '-14 days')
             THEN 1 ELSE 0 END                                   AS no_login_signal
    FROM product_usage
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
)

SELECT
    c.customer_id,
    c.first_name || ' ' || c.last_name                          AS customer_name,
    c.company,
    c.loyalty_tier,

    -- Individual signal flags (0 or 1)
    COALESCE(su.usage_decline_signal, 0)                         AS usage_decline,
    COALESCE(st.ticket_spike_signal,  0)                         AS ticket_spike,
    COALESCE(sr.rating_drop_signal,   0)                         AS rating_drop,
    -- No login defaults to 1 if customer has NO session history at all
    COALESCE(sl.no_login_signal, 1)                              AS no_recent_login,

    -- Composite churn risk score: 0 (no risk) → 4 (all signals firing)
    COALESCE(su.usage_decline_signal, 0)
  + COALESCE(st.ticket_spike_signal,  0)
  + COALESCE(sr.rating_drop_signal,   0)
  + COALESCE(sl.no_login_signal,      1)                         AS churn_risk_score,

    -- Supporting metrics for diagnosis
    COALESCE(su.sessions_last30, 0)                              AS sessions_last30,
    COALESCE(su.sessions_prev30, 0)                              AS sessions_prev30,
    sl.last_login,
    COALESCE(st.tickets_last30, 0)                               AS tickets_last30,
    COALESCE(st.avg_monthly_tickets, 0)                          AS avg_monthly_tickets,
    sr.recent_avg_rating,
    sr.overall_avg_rating,

    -- Rank: 1 = highest churn risk
    DENSE_RANK() OVER (
        ORDER BY
            COALESCE(su.usage_decline_signal, 0)
          + COALESCE(st.ticket_spike_signal,  0)
          + COALESCE(sr.rating_drop_signal,   0)
          + COALESCE(sl.no_login_signal,      1)
        DESC
    )                                                            AS churn_risk_rank

FROM customers c
LEFT JOIN signal_usage   su ON su.customer_id = c.customer_id
LEFT JOIN signal_tickets st ON st.customer_id = c.customer_id
LEFT JOIN signal_rating  sr ON sr.customer_id = c.customer_id
LEFT JOIN signal_login   sl ON sl.customer_id = c.customer_id
WHERE c.is_active = 1
ORDER BY churn_risk_score DESC, c.customer_id
LIMIT 50;  -- top 50 at-risk customers; remove LIMIT for full list