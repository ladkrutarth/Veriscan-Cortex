# Veriscan — Phase 2 Report
## Product System Integration & Reproducibility

> **Course:** CS 5588 — Data Science Capstone  
> **Team:** Veriscan  
> **Date:** February 27, 2026  
> **GitHub:** *(insert repository link)*

---

## 1. Product & Stakeholder System Brief

### Product Name
**Veriscan** — Advanced Fraud Detection & Private AI Intelligence Dashboard

### Stakeholder Users
- **Fraud Analysts** — Investigate flagged transactions and approve/deny holds.
- **Security Operations (SecOps)** — Monitor authentication risk levels and enforce adaptive verification.
- **Compliance Officers** — Audit CFPB complaint trends and ensure regulatory adherence.
- **Executive Leadership** — Review macro-level risk dashboards for portfolio-wide decision making.

### Decision Workflow Supported
Veriscan supports a **Transaction Risk Triage** workflow:

1. Raw transactions are ingested from the Kaggle fraud-detection dataset.
2. A 19-signal feature engineering pipeline computes statistical and behavioral indicators.
3. A hybrid scoring engine (Random Forest + heuristic overrides) assigns a risk level to each transaction.
4. Dynamic authentication adapts security challenges based on cumulative user risk.
5. An Agentic AI analyst (GuardAgent) enables natural-language investigation of risk profiles.

### Core Value Proposition
- **Privacy-first AI** — All intelligence runs locally on Apple Silicon (no cloud API calls for inference).
- **Hybrid Intelligence** — Combines 98%-accurate ML with rule-based heuristic overrides for explainability.
- **Autonomous Investigation** — The GuardAgent achieves 100% tool-selection accuracy for analyst queries.

### Key Risks if System Produces Incorrect Outputs
| Risk | Impact | Mitigation |
|------|--------|------------|
| False negatives (missed fraud) | Financial losses, customer harm | Heuristic overrides for extreme z-scores and entropy |
| False positives (blocking legit) | Customer friction, revenue loss | Tiered risk levels (Low/Medium/High/Critical) with human review |
| RAG hallucination | Misleading compliance analysis | Confidence scoring and source attribution in results |
| Model drift | Degraded accuracy over time | Pipeline logging, periodic retraining on fresh data |

---

## 2. Data & Knowledge Infrastructure

### Dataset Sources

| Dataset | Source | Modality | Records | Ownership |
|---------|--------|----------|---------|-----------|
| Fraud Transactions | Kaggle (kartik2112/fraud-detection) | Structured CSV | ~1.3M (train+test) | Public, CC0 |
| CFPB Complaints | Consumer Financial Protection Bureau | Structured CSV | 142K | Public, US Gov |
| Pipeline Logs | System-generated | Structured CSV | Ongoing | Internal |

### Multimodal Ingestion Pipeline

1. **Raw CSV Adapter** (`scripts/load_kaggle_data.py`): Normalizes Kaggle columns → internal schema (9 columns). Supports configurable sampling with balanced fraud ratios.
2. **Feature Engineering** (`scripts/feature_engineering.py`): Computes 19 statistical signals (amount z-score, velocity windows, location entropy, category risk).
3. **CFPB Ingestion** (`models/rag_engine_local.py`): Parses complaint narratives into ChromaDB vector store for semantic retrieval.
4. **Snowflake Upload** (`scripts/upload_all_to_snowflake.py`): Batch-uploads all outputs to 5 Snowflake tables with TRUNCATE + write_pandas.

### Data Preprocessing & Sampling Strategy
- Top-N most active cardholders are selected as user cohort (configurable, default N=10).
- Balanced sampling with configurable fraud ratio (default 15%) to prevent class imbalance.
- Category names are cleaned via a mapping table (14 Kaggle categories → 12 display names).

### Data Inventory

| File | Location | Rows | Columns |
|------|----------|------|---------|
| `fraudTrain_sampled.csv` | `dataset/csv_data/` | 50,000 | 23 |
| `fraudTest_sampled.csv` | `dataset/csv_data/` | 50,000 | 23 |
| `cfpb_credit_card_sampled.csv` | `dataset/csv_data/` | 50,000 | 18 |
| `features_output.csv` | `dataset/csv_data/` | ~10K | 20 |
| `processed_fraud_train.csv` | `dataset/csv_data/` | ~10K | 23 |
| `pipeline_logs.csv` | `dataset/csv_data/` | Ongoing | 7 |

---

## 3. Retrieval, Intelligence, and Modeling Pipeline

### Chunking / Indexing Strategy
- **Transaction documents**: Each row → 1 natural-language sentence describing merchant, category, amount, and location
- **CFPB complaints**: Issue + narrative concatenated; batch-indexed in groups of 100
- **Vector store**: ChromaDB with `all-MiniLM-L6-v2` sentence-transformer embeddings (local, no API)

### Retrieval Architecture
**Hybrid Retrieval** with two pathways:
1. **Dense retrieval**: Semantic similarity search via ChromaDB (cosine distance, top-5 results)
2. **Deterministic routing**: Keyword-based intent router in GuardAgent bypasses LLM for known query patterns (user IDs, risk lookups)

### Embedding & Model Configurations

| Component | Model | Details |
|-----------|-------|---------|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, local sentence-transformers |
| LLM | `Meta-Llama-3-8B-Instruct` | 4-bit quantized, MLX-LM on Apple Silicon |
| Fraud Classifier | Random Forest | 98.18% accuracy, trained on 19 engineered features |
| Anomaly Detection | Isolation Forest | Integrated into fraud scoring pipeline |

### Example Grounded Outputs

**Task 1 — User Risk Investigation:**
> *Query:* "Investigate risk for USER_001"  
> *GuardAgent routes* → `get_user_risk_profile(user_id="USER_001")`  
> *Output:* Returns avg_risk, max_risk, high_risk_count, transaction history, and recommended security level with supporting evidence from scored transactions.

**Task 2 — CFPB Compliance Query:**
> *Query:* "What are common credit card fraud complaints?"  
> *GuardAgent routes* → `query_rag(question="credit card fraud complaints")`  
> *Output:* Returns top-5 semantically similar CFPB complaint narratives with confidence scores, enabling compliance officers to identify recurring patterns.

---

## 4. Application / Decision Interface Integration

### Streamlit Dashboard
The `streamlit_app.py` application provides a **5-tab interface**:

| Tab | Purpose |
|-----|---------|
| **Market Dashboard** | Macro-level fraud metrics, category distributions, Plotly visualizations |
| **AI Fraud Prediction** | Per-transaction ML scoring with the 98% Random Forest model |
| **Dynamic Authentication** | Adaptive identity verification based on cumulative user risk |
| **CFPB Intelligence** | Consumer complaint analysis and trend exploration |
| **Agentic Analyst** | Natural-language GuardAgent interface for autonomous investigation |

### Evidence & Reasoning Display
- Risk badges (Low / Medium / High / Critical) with color-coded CSS indicators
- Feature contribution breakdown showing which of the 19 signals drove the score
- RAG results include confidence scores and source metadata

### Query Logging & Interaction Tracking
- `pipeline_logs.csv` records every pipeline execution with: run_id, timestamp, stage, status, records_processed, duration_ms, error_message
- Pipeline events logged via `log_pipeline_event()` in `scripts/feature_engineering.py`

### Deployment
- Launch: `streamlit run streamlit_app.py`
- Accessible at `http://localhost:8501`

---

## 5. Snowflake Data Platform Integration

### Snowflake Schema

**Database:** `GRAPHGUARD_DB`  
**Schema:** `PUBLIC`  
**Warehouse:** `GRAPH_GUARD_DATABASE` (XSMALL, auto-suspend 60s)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `RAW_TRANSACTIONS` | Source data | TRANSACTION_ID (PK), USER_ID, AMOUNT, CATEGORY, LOCATION |
| `TRANSACTION_FEATURES` | Engineered features | 19 signals including AMOUNT_ZSCORE, velocity counts, LOCATION_ENTROPY |
| `FRAUD_SCORES` | ML + heuristic scores | COMBINED_RISK_SCORE, RISK_LEVEL, RECOMMENDATION |
| `AUTH_PROFILES` | User security profiles | AVG_RISK, MAX_RISK, RECOMMENDED_SECURITY_LEVEL |
| `PIPELINE_RUNS` | Audit trail | STAGE, STATUS, RECORDS_PROCESSED, DURATION_MS |

**Views:**
- `ENRICHED_TRANSACTIONS` — Joins raw data with features and scores for unified analysis
- `USER_RISK_DASHBOARD` — Aggregated user-level risk summary

### Ingestion Scripts
- `scripts/upload_all_to_snowflake.py` — Orchestrates full upload: database setup → table creation → CSV ingestion via `write_pandas`
- `scripts/ingest_config.yaml` — Configuration file with Snowflake credentials (supports environment variable overrides)

### Example Analytical Queries
See `sql/analytical_queries.sql` for 8 production-ready queries:
1. Fraud rate by category
2. High-risk user profile summary
3. Spending velocity anomaly detection
4. Merchant risk aggregation
5. Geographic fraud distribution
6. Time-of-day risk heatmap
7. Pipeline health monitoring
8. Category risk weight calibration

### How Snowflake Supports the Workflow
- **Analytics**: Enables SQL-based fraud pattern discovery across the full dataset at scale
- **Modeling**: Feature tables in Snowflake serve as the ground truth for model retraining
- **Application**: The Streamlit dashboard can query Snowflake views for real-time risk summaries
- **Auditing**: Pipeline run logs provide observability into data freshness and processing health

---

## 6. Reproducibility & Deployment Plan

### Environment Configuration

```bash
# 1. Clone the repository
git clone <repository-url>
cd Hands-On-Week-4

# 2. Create and activate a Python environment
conda create -n veriscan python=3.10 -y
conda activate veriscan

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set Snowflake credentials
export SNOWFLAKE_ACCOUNT="your-account"
export SNOWFLAKE_USER="your-user"
export SNOWFLAKE_PASSWORD="your-password"
```

### Model / Dataset Versioning
- **Dataset**: Kaggle `kartik2112/fraud-detection` (static, versioned via download date)
- **Trained model**: `models/fraud_model_rf.joblib` (serialized via joblib, reproducible with fixed `random_state=42`)
- **Encoders**: `models/encoders.joblib` (label encoders for categorical features)
- **Vector store**: `.chroma_db_local/` (rebuilt on demand via `rag_engine_local.py`)

### Execution Instructions

```bash
# Step 1: Prepare the dataset (download fraudTrain.csv to dataset/csv_data/)
python scripts/load_kaggle_data.py --input dataset/csv_data/fraudTrain.csv --sample 10000

# Step 2: Run feature engineering
python scripts/feature_engineering.py

# Step 3: Train the fraud model
python models/train_fraud_model.py

# Step 4: (Optional) Upload to Snowflake
python scripts/upload_all_to_snowflake.py

# Step 5: Launch the dashboard
streamlit run streamlit_app.py
```

### Deployment / Scaling Considerations
- **Local deployment**: Optimized for Apple Silicon (M1/M2/M3) with MLX-LM acceleration
- **Cloud deployment**: Streamlit Cloud or AWS EC2 (requires GPU for LLM inference at scale)
- **Scaling**: Snowflake warehouse can be right-sized (XSMALL → XLARGE) based on query load
- **Security**: Credentials managed via environment variables; `.env` and secrets excluded in `.gitignore`

---

## 7. GitHub Repository Structure

```
Hands-On-Week-4/
├── streamlit_app.py                    # Main 5-tab dashboard
├── requirements.txt                    # Python dependencies
├── Phase-2-Report.md                   # This report
├── CONTRIBUTIONS.md                    # Team contribution breakdown
├── README.md                           # Project documentation & quick start
│
├── models/                             # Intelligence & Modeling Layer
│   ├── guard_agent_local.py            # Agentic AI (GuardAgent, 100% accuracy)
│   ├── rag_engine_local.py             # Local RAG engine (ChromaDB + MiniLM)
│   ├── local_llm.py                    # MLX-LM wrapper for Llama-3
│   ├── train_fraud_model.py            # Random Forest training pipeline
│   ├── habit_model.py                  # User behavior profiling
│   ├── compare_ml_models.py            # Model benchmarking suite
│   ├── evaluate_agent_local.py         # Agent accuracy evaluation
│   ├── evaluate_rag_local.py           # RAG precision evaluation
│   ├── fraud_model_rf.joblib           # Trained model artifact
│   └── encoders.joblib                 # Label encoders
│
├── scripts/                            # Data Pipeline Layer
│   ├── load_kaggle_data.py             # Kaggle dataset adapter
│   ├── feature_engineering.py          # 19-signal feature pipeline
│   ├── upload_all_to_snowflake.py      # Snowflake batch uploader
│   ├── ingest_csv_to_snowflake.py      # CSV → Snowflake ingestion
│   ├── ingest_config.yaml              # Connection config
│   ├── prepare_fraud_data.py           # Data preparation utilities
│   └── expand_dataset.py              # Dataset expansion tool
│
├── sql/                                # Snowflake SQL Layer
│   ├── create_tables.sql               # DDL: 5 tables + 2 views
│   └── analytical_queries.sql          # 8 analytical queries
│
├── dataset/csv_data/                   # Data Store
│   ├── fraudTrain.csv                  # Kaggle training set
│   ├── fraudTest.csv                   # Kaggle test set
│   ├── cfpb_credit_card.csv            # CFPB complaints
│   ├── features_output.csv             # Computed features
│   └── pipeline_logs.csv              # Pipeline audit trail
│
└── docs/
    └── architecture_diagram.png        # System architecture diagram
```

---

## 8. Individual Contribution Statement

See `CONTRIBUTIONS.md` for full details.

| Member | Contribution Description | Contribution % |
|--------|--------------------------|:--------------:|
| Member 1 | Data ingestion pipeline, Snowflake schema design, pipeline logging | 25% |
| Member 2 | Feature engineering (19 signals), fraud model (RF 98%), habit profiling | 25% |
| Member 3 | Streamlit dashboard (5 tabs), authentication UI, CSS aesthetics | 25% |
| Member 4 | SQL analytical queries, architecture diagram, documentation, deployment | 25% |

> **Note:** Update member names before final submission. Contributions must reflect actual technical work with corresponding commit evidence.
