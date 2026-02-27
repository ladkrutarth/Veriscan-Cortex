-- ============================================================================
-- Veriscan — Snowflake Schema DDL
-- Creates the database, schema, warehouse, and all five core tables.
-- Run this script in a Snowflake worksheet or via SnowSQL.
-- ============================================================================

-- 1. Infrastructure Setup
CREATE DATABASE IF NOT EXISTS GRAPHGUARD_DB;
USE DATABASE GRAPHGUARD_DB;

CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

CREATE WAREHOUSE IF NOT EXISTS GRAPH_GUARD_DATABASE
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME  = TRUE;
USE WAREHOUSE GRAPH_GUARD_DATABASE;

-- ============================================================================
-- 2. Core Tables
-- ============================================================================

-- 2a. Raw transaction records ingested from the Kaggle fraud-detection dataset.
CREATE TABLE IF NOT EXISTS RAW_TRANSACTIONS (
    TRANSACTION_ID    VARCHAR(50)   PRIMARY KEY,
    USER_ID           VARCHAR(20)   NOT NULL,
    MERCHANT_NAME     VARCHAR(100),
    CATEGORY          VARCHAR(50),
    AMOUNT            FLOAT,
    LOCATION          VARCHAR(100),
    TRANSACTION_DATE  TIMESTAMP_NTZ,
    STATUS            VARCHAR(20),
    INGESTED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 2b. Computed features used by the ML fraud model (19 signals).
CREATE TABLE IF NOT EXISTS TRANSACTION_FEATURES (
    TRANSACTION_ID         VARCHAR(50)   PRIMARY KEY,
    USER_ID                VARCHAR(20)   NOT NULL,
    AMOUNT                 FLOAT,
    AMOUNT_ZSCORE          FLOAT,
    IS_HIGH_VALUE          BOOLEAN,
    USER_AVG_AMOUNT        FLOAT,
    USER_STD_AMOUNT        FLOAT,
    USER_TOTAL_TRANSACTIONS INT,
    TXN_COUNT_1H           INT,
    TXN_COUNT_24H          INT,
    TXN_COUNT_7D           INT,
    CATEGORY               VARCHAR(50),
    CATEGORY_RISK_WEIGHT   FLOAT,
    HOUR_OF_DAY            INT,
    DAY_OF_WEEK            INT,
    IS_WEEKEND             BOOLEAN,
    LOCATION               VARCHAR(100),
    IS_NEW_LOCATION        BOOLEAN,
    LOCATION_ENTROPY       FLOAT,
    COMPUTED_AT            TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 2c. Hybrid fraud scores combining heuristics + Random Forest predictions.
CREATE TABLE IF NOT EXISTS FRAUD_SCORES (
    SCORE_ID               VARCHAR(50)   PRIMARY KEY,
    TRANSACTION_ID         VARCHAR(50)   NOT NULL,
    USER_ID                VARCHAR(20)   NOT NULL,
    ZSCORE_FLAG            FLOAT,
    VELOCITY_FLAG          FLOAT,
    CATEGORY_RISK_SCORE    FLOAT,
    GEOGRAPHIC_RISK_SCORE  FLOAT,
    ISOLATION_FOREST_SCORE FLOAT,
    COMBINED_RISK_SCORE    FLOAT,
    RISK_LEVEL             VARCHAR(20),
    RECOMMENDATION         VARCHAR(200),
    SCORED_AT              TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 2d. Dynamic authentication profiles per user (security-level recommendation).
CREATE TABLE IF NOT EXISTS AUTH_PROFILES (
    USER_ID                     VARCHAR(20)  PRIMARY KEY,
    AVG_RISK                    FLOAT,
    MAX_RISK                    FLOAT,
    HIGH_RISK_COUNT             INT,
    RECOMMENDED_SECURITY_LEVEL  VARCHAR(20),
    NUM_QUESTIONS               INT,
    UPLOADED_AT                 TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 2e. Pipeline execution audit log for observability.
CREATE TABLE IF NOT EXISTS PIPELINE_RUNS (
    RUN_ID              VARCHAR(50),
    TIMESTAMP           VARCHAR(100),
    STAGE               VARCHAR(50),
    STATUS              VARCHAR(20),
    RECORDS_PROCESSED   INT,
    DURATION_MS         FLOAT,
    ERROR_MESSAGE       VARCHAR(500)
);

-- ============================================================================
-- 3. Useful Views
-- ============================================================================

-- 3a. Enriched transaction view joining raw data with features and scores.
CREATE OR REPLACE VIEW ENRICHED_TRANSACTIONS AS
SELECT
    r.TRANSACTION_ID,
    r.USER_ID,
    r.MERCHANT_NAME,
    r.CATEGORY,
    r.AMOUNT,
    r.LOCATION,
    r.TRANSACTION_DATE,
    f.AMOUNT_ZSCORE,
    f.IS_HIGH_VALUE,
    f.TXN_COUNT_1H,
    f.TXN_COUNT_24H,
    f.LOCATION_ENTROPY,
    s.COMBINED_RISK_SCORE,
    s.RISK_LEVEL,
    s.RECOMMENDATION
FROM RAW_TRANSACTIONS r
LEFT JOIN TRANSACTION_FEATURES f ON r.TRANSACTION_ID = f.TRANSACTION_ID
LEFT JOIN FRAUD_SCORES s         ON r.TRANSACTION_ID = s.TRANSACTION_ID;

-- 3b. User risk dashboard view.
CREATE OR REPLACE VIEW USER_RISK_DASHBOARD AS
SELECT
    a.USER_ID,
    a.AVG_RISK,
    a.MAX_RISK,
    a.HIGH_RISK_COUNT,
    a.RECOMMENDED_SECURITY_LEVEL,
    COUNT(s.SCORE_ID)                              AS TOTAL_SCORED_TXNS,
    AVG(s.COMBINED_RISK_SCORE)                     AS AVG_COMBINED_SCORE,
    SUM(CASE WHEN s.RISK_LEVEL = 'Critical' THEN 1 ELSE 0 END) AS CRITICAL_TXNS
FROM AUTH_PROFILES a
LEFT JOIN FRAUD_SCORES s ON a.USER_ID = s.USER_ID
GROUP BY a.USER_ID, a.AVG_RISK, a.MAX_RISK, a.HIGH_RISK_COUNT,
         a.RECOMMENDED_SECURITY_LEVEL;
