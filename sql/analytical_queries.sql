-- ============================================================================
-- Veriscan — Analytical Queries for Snowflake
-- Demonstrates connectivity between the application layer and the cloud
-- data warehouse.  Run in a Snowflake worksheet after ingestion.
-- ============================================================================

USE DATABASE GRAPHGUARD_DB;
USE SCHEMA PUBLIC;

-- ============================================================================
-- Query 1: Fraud Rate by Transaction Category
-- Purpose : Identify which spending categories carry the highest fraud risk.
-- ============================================================================
SELECT
    f.CATEGORY,
    COUNT(*)                                                   AS total_txns,
    SUM(CASE WHEN s.RISK_LEVEL IN ('High','Critical') THEN 1 ELSE 0 END) AS high_risk_txns,
    ROUND(high_risk_txns / NULLIF(total_txns, 0) * 100, 2)    AS fraud_rate_pct,
    ROUND(AVG(s.COMBINED_RISK_SCORE), 4)                       AS avg_risk_score
FROM TRANSACTION_FEATURES f
JOIN FRAUD_SCORES s ON f.TRANSACTION_ID = s.TRANSACTION_ID
GROUP BY f.CATEGORY
ORDER BY fraud_rate_pct DESC;

-- ============================================================================
-- Query 2: High-Risk User Profile Summary
-- Purpose : Rank users by cumulative risk to prioritize security reviews.
-- ============================================================================
SELECT
    a.USER_ID,
    a.AVG_RISK,
    a.MAX_RISK,
    a.HIGH_RISK_COUNT,
    a.RECOMMENDED_SECURITY_LEVEL,
    COUNT(s.SCORE_ID)              AS total_scored_txns,
    ROUND(AVG(s.COMBINED_RISK_SCORE), 4) AS avg_combined_score
FROM AUTH_PROFILES a
LEFT JOIN FRAUD_SCORES s ON a.USER_ID = s.USER_ID
GROUP BY a.USER_ID, a.AVG_RISK, a.MAX_RISK, a.HIGH_RISK_COUNT,
         a.RECOMMENDED_SECURITY_LEVEL
ORDER BY a.MAX_RISK DESC
LIMIT 20;

-- ============================================================================
-- Query 3: Spending Velocity Anomaly Detection
-- Purpose : Surface transactions with abnormal burst patterns (velocity > 5
--           within 1 hour) that may indicate card-testing attacks.
-- ============================================================================
SELECT
    f.TRANSACTION_ID,
    f.USER_ID,
    f.AMOUNT,
    f.TXN_COUNT_1H,
    f.TXN_COUNT_24H,
    f.TXN_COUNT_7D,
    s.COMBINED_RISK_SCORE,
    s.RISK_LEVEL
FROM TRANSACTION_FEATURES f
JOIN FRAUD_SCORES s ON f.TRANSACTION_ID = s.TRANSACTION_ID
WHERE f.TXN_COUNT_1H > 5
ORDER BY f.TXN_COUNT_1H DESC, s.COMBINED_RISK_SCORE DESC
LIMIT 50;

-- ============================================================================
-- Query 4: Merchant Risk Aggregation
-- Purpose : Identify merchants with disproportionately high risk transactions
--           for merchant-level monitoring.
-- ============================================================================
SELECT
    r.MERCHANT_NAME,
    COUNT(*)                                                        AS total_txns,
    SUM(CASE WHEN s.RISK_LEVEL IN ('High','Critical') THEN 1 ELSE 0 END) AS flagged_txns,
    ROUND(flagged_txns / NULLIF(total_txns, 0) * 100, 2)           AS flag_rate_pct,
    ROUND(AVG(r.AMOUNT), 2)                                        AS avg_amount,
    ROUND(MAX(s.COMBINED_RISK_SCORE), 4)                            AS max_risk
FROM RAW_TRANSACTIONS r
JOIN FRAUD_SCORES s ON r.TRANSACTION_ID = s.TRANSACTION_ID
GROUP BY r.MERCHANT_NAME
HAVING flagged_txns > 0
ORDER BY flag_rate_pct DESC
LIMIT 25;

-- ============================================================================
-- Query 5: Geographic Fraud Distribution
-- Purpose : Map fraud hotspots by splitting LOCATION into city/state.
-- ============================================================================
SELECT
    SPLIT_PART(r.LOCATION, ',', 2)  AS state,
    COUNT(*)                        AS total_txns,
    SUM(CASE WHEN s.RISK_LEVEL IN ('High','Critical') THEN 1 ELSE 0 END) AS flagged,
    ROUND(AVG(s.COMBINED_RISK_SCORE), 4) AS avg_risk,
    ROUND(AVG(f.LOCATION_ENTROPY), 4)    AS avg_location_entropy
FROM RAW_TRANSACTIONS r
JOIN FRAUD_SCORES s         ON r.TRANSACTION_ID = s.TRANSACTION_ID
JOIN TRANSACTION_FEATURES f ON r.TRANSACTION_ID = f.TRANSACTION_ID
GROUP BY state
ORDER BY flagged DESC
LIMIT 15;

-- ============================================================================
-- Query 6: Time-of-Day Risk Heatmap
-- Purpose : Reveal temporal patterns — which hours carry the most risk.
-- ============================================================================
SELECT
    f.HOUR_OF_DAY,
    f.IS_WEEKEND,
    COUNT(*)                                                        AS txn_count,
    SUM(CASE WHEN s.RISK_LEVEL IN ('High','Critical') THEN 1 ELSE 0 END) AS high_risk_count,
    ROUND(AVG(s.COMBINED_RISK_SCORE), 4)                            AS avg_risk,
    ROUND(AVG(f.AMOUNT), 2)                                        AS avg_amount
FROM TRANSACTION_FEATURES f
JOIN FRAUD_SCORES s ON f.TRANSACTION_ID = s.TRANSACTION_ID
GROUP BY f.HOUR_OF_DAY, f.IS_WEEKEND
ORDER BY f.HOUR_OF_DAY;

-- ============================================================================
-- Query 7: Pipeline Health Monitoring
-- Purpose : Audit the data pipeline for failures or performance regressions.
-- ============================================================================
SELECT
    STAGE,
    STATUS,
    COUNT(*)                          AS run_count,
    ROUND(AVG(DURATION_MS), 2)        AS avg_duration_ms,
    MAX(DURATION_MS)                  AS max_duration_ms,
    SUM(RECORDS_PROCESSED)            AS total_records,
    SUM(CASE WHEN STATUS = 'failed' THEN 1 ELSE 0 END) AS failure_count
FROM PIPELINE_RUNS
GROUP BY STAGE, STATUS
ORDER BY STAGE, STATUS;

-- ============================================================================
-- Query 8: Category Risk Weight Alignment Check
-- Purpose : Compare heuristic category risk weights against actual observed
--           fraud rates to validate or recalibrate the scoring model.
-- ============================================================================
SELECT
    f.CATEGORY,
    f.CATEGORY_RISK_WEIGHT                                          AS heuristic_weight,
    COUNT(*)                                                        AS total_txns,
    SUM(CASE WHEN s.RISK_LEVEL IN ('High','Critical') THEN 1 ELSE 0 END) AS observed_flags,
    ROUND(observed_flags / NULLIF(total_txns, 0), 4)                AS observed_flag_rate,
    ROUND(observed_flag_rate - f.CATEGORY_RISK_WEIGHT, 4)           AS weight_delta
FROM TRANSACTION_FEATURES f
JOIN FRAUD_SCORES s ON f.TRANSACTION_ID = s.TRANSACTION_ID
GROUP BY f.CATEGORY, f.CATEGORY_RISK_WEIGHT
ORDER BY ABS(weight_delta) DESC;
