# CONTRIBUTIONS.md — Veriscan Team Responsibilities

> **Note:** Update team member names before final submission.

## Team Members & Responsibilities

| Team Member | Role | Components Owned |
|-------------|------|------------------|
| **Member 1** | Data Engineer | Data ingestion scripts, Snowflake schema design, pipeline logging |
| **Member 2** | ML Engineer | Feature engineering, fraud scoring model (Random Forest + rules), user habit profiling |
| **Member 3** | Application Developer | Streamlit dashboard, authentication UI, visualization, accessibility |
| **Member 4** | DevOps / Analytics | SQL analytical queries, deployment, monitoring, documentation |

---

### Member 1 — Data Engineer
**Files owned:**
- `scripts/load_kaggle_data.py` — Data adapter & preprocessing
- `scripts/upload_all_to_snowflake.py` — CSV to Snowflake ingestion pipeline
- `sql/create_tables.sql` — Snowflake DDL for all data tables
- `pipeline_logs.csv` — pipeline execution log

**Responsibilities:**
- Designed the raw data ingestion pipeline with validation steps.
- Created Snowflake table schemas for transaction and metadata storage.
- Implemented robust logging to track pipeline execution and data integrity.
- Ensured reproducibility with config-driven connections.

---

### Member 2 — ML Engineer
**Files owned:**
- `scripts/feature_engineering.py` — feature computation pipeline (19 signals)
- `models/train_fraud_model.py` — training pipeline for Random Forest model
- `models/habit_model.py` — user behavior profiling and habit similarity scoring
- `models/fraud_model_rf.joblib` — trained fraud detection model

**Responsibilities:**
- Designed 19 transaction features (amount, velocity, geographic, time, category).
- Built a hybrid fraud model combining statistical heuristics and machine learning.
- Developed the user habit profiling system for historical behavior comparison.
- Implemented anomaly detection based on multi-dimensional user profile similarity.

---

### Member 3 — Application Developer
**Files owned:**
- `streamlit_app.py` — main Streamlit application (4-tab system)
- `requirements.txt` — project dependencies

**Responsibilities:**
- Built the Streamlit dashboard with dedicated tabs for Market Dash, AI Fraud ML, Auth, and CFPB Intel.
- Developed the context-aware identity verification interface.
- Implemented accessible "Soft Light-Glass" CSS aesthetics for a premium UI experience.
- Integrated interactive Plotly visualizations for macro and micro-level data analysis.

---

### Member 4 — DevOps / Analytics
**Files owned:**
- `sql/analytical_queries.sql` — analytical SQL queries
- `docs/architecture_diagram.png` — system architecture diagram
- `README.md` — project documentation
- `CONTRIBUTIONS.md` — this file

**Responsibilities:**
- Wrote analytical queries for spending patterns, anomaly detection, and merchant risk.
- Created the architecture diagram showing end-to-end data pipeline.
- Documented system workflow, component details, and implementation extensions.
- Set up monitoring and ensured the repository follows best practices.
