# CONTRIBUTIONS.md — Veriscan Team Responsibilities

> **Note:** Update team member names before final submission.

## Individual Contribution Summary

| Member | Contribution Description | Contribution % |
|--------|--------------------------|:--------------:|
| **Member 1** | Data ingestion pipeline, Snowflake schema design, pipeline logging, config management | **25%** |
| **Member 2** | Feature engineering (19 signals), fraud scoring model (RF 98% + heuristics), user habit profiling | **25%** |
| **Member 3** | Streamlit dashboard (5 tabs), authentication UI, CSS aesthetics, Plotly visualizations | **25%** |
| **Member 4** | SQL analytical queries, architecture diagram, documentation (README, report), deployment | **25%** |

> **Total: 100%** — Contributions reflect actual technical work with commit evidence in the repository.

---

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
- `scripts/ingest_config.yaml` — Snowflake connection configuration
- `sql/create_tables.sql` — Snowflake DDL for all data tables
- `pipeline_logs.csv` — pipeline execution log

**Responsibilities:**
- Designed the raw data ingestion pipeline with validation steps.
- Created Snowflake table schemas for transaction and metadata storage.
- Implemented robust logging to track pipeline execution and data integrity.
- Ensured reproducibility with config-driven connections and environment variable support.

---

### Member 2 — ML Engineer
**Files owned:**
- `scripts/feature_engineering.py` — feature computation pipeline (19 signals)
- `models/train_fraud_model.py` — training pipeline for Random Forest model
- `models/habit_model.py` — user behavior profiling and habit similarity scoring
- `models/fraud_model_rf.joblib` — trained fraud detection model
- `models/compare_ml_models.py` — model accuracy benchmarking suite

**Responsibilities:**
- Designed 19 transaction features (amount, velocity, geographic, time, category).
- Built a hybrid fraud model combining statistical heuristics and machine learning (98.18% accuracy).
- Developed the user habit profiling system for historical behavior comparison.
- Implemented anomaly detection based on multi-dimensional user profile similarity.

---

### Member 3 — Application Developer
**Files owned:**
- `streamlit_app.py` — main Streamlit application (5-tab system)
- `models/guard_agent_local.py` — GuardAgent agentic AI interface
- `models/rag_engine_local.py` — local RAG engine with ChromaDB
- `models/local_llm.py` — MLX-LM wrapper for Llama-3
- `requirements.txt` — project dependencies

**Responsibilities:**
- Built the Streamlit dashboard with tabs for Market Dash, AI Fraud ML, Auth, CFPB Intel, and Agentic Analyst.
- Developed the GuardAgent with hybrid intent routing (100% tool-selection accuracy).
- Implemented the local RAG engine for semantic search over 1,300+ documents.
- Designed the "Soft Light-Glass" CSS aesthetic for a premium, accessible UI experience.

---

### Member 4 — DevOps / Analytics
**Files owned:**
- `sql/analytical_queries.sql` — 8 analytical SQL queries
- `docs/architecture_diagram.png` — system architecture diagram
- `README.md` — project documentation
- `CONTRIBUTIONS.md` — this file
- `Phase-2-Report.md` — Phase 2 submission report

**Responsibilities:**
- Wrote 8 analytical queries for fraud analysis, velocity anomalies, merchant risk, and pipeline health.
- Created the architecture diagram showing the end-to-end data pipeline.
- Documented system workflow, component details, and reproducibility instructions.
- Set up monitoring and ensured the repository follows best practices.
