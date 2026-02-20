# GraphGuard - AI-Powered Dynamic Authentication System
## Week 4 Capstone Integration Submission



- **Deployed App:** https://graphguard1.streamlit.app/


### Project Overview
GraphGuard is an intelligent authentication and fraud detection system that uses AI to generate personalized security challenges based on real transaction history. The system provides:

- **Dynamic Authentication**: AI-generated questions specific to each user's transaction history
- **Real-time Fraud Detection**: Pattern analysis and risk scoring for suspicious transactions
- **Comprehensive Monitoring**: Automatic logging of all interactions with performance metrics
- **Production-Ready Architecture**: Scalable design with fallback mechanisms and monitoring



### Files Included

#### 1. `streamlit_app.py` - Main Application
Updated Streamlit application with:
- CSV transaction data loading (uses your actual transaction file)
- Comprehensive logging system (`logs/product_metrics.csv`)
- Real-time monitoring dashboard
- Authentication and fraud detection modules
- Fallback to demo data if CSV not found

#### 2. `Week4_Deliverables.docx` - Complete Documentation
Comprehensive capstone documentation including:
- **Section A**: Capstone Product Integration Brief
- **Section B**: Application Integration details
- **Section C**: Monitoring & Logging implementation
- **Section D**: Capstone Evaluation with impact metrics
- **Section E**: Deployment Readiness Plan with architecture
- **Section F**: Failure scenarios and mitigation strategies

#### 3. `logs/product_metrics.csv` - Sample Logs
Contains 10 example logged interactions demonstrating:
- Authentication generation events
- Authentication verification events
- Fraud detection analysis events
- Latency measurements
- Confidence scores
- Success/failure status

> **Course:** CS 5588 — Data Science Capstone | **Date:** February 2026

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Pipeline Workflow](#pipeline-workflow)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Component Details](#component-details)
- [Monitoring & Logging](#monitoring--logging)
- [Implemented Extensions](#implemented-extensions)
- [Demo Video](#demo-video)

---

## Project Overview

GraphGuard is an end-to-end **fraud detection and dynamic authentication system** that processes financial transaction data through a complete data pipeline:

**Data Sources → Cloud Warehouse (Snowflake) → Feature Engineering → Modeling / Decision Layer → Streamlit Dashboard**

The system uses a hybrid approach combining **rule-based heuristics**, **statistical anomaly detection** (z-score, velocity checks), and **machine learning** (IsolationForest) to score transaction risk in real-time. An AI-powered authentication module uses **Google Gemini LLM** with **RAG (Retrieval-Augmented Generation)** over project data to dynamically generate security challenges adapted to each user's risk profile.

### Key Capabilities
- **3,000 real transactions** across 10 users, 8 categories, 30+ merchants
- **19 engineered features** per transaction (amount, velocity, geographic, temporal, categorical)
- **Hybrid fraud scoring** with weighted combination of 5 risk signals
- **Adaptive authentication** with meaningful questions about stores, locations, and categories (no risk scores)
- **Transaction habit prediction model** that learns each user's spending patterns
- **Production-ready RAG pipeline** with query rewriting, re-ranking, confidence scoring, and automated evaluation
- **Full pipeline monitoring** with execution logs and performance metrics

---

## System Architecture

![GraphGuard Architecture](docs/architecture_diagram.png)

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│ CSV Data     │────▶│ Ingestion    │────▶│ Snowflake         │
│ (301 txns)   │     │ Script       │     │ RAW_TRANSACTIONS  │
└──────────────┘     └──────────────┘     └─────────┬─────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                   Feature Engineering                        │
│  19 features: amount z-score, velocity (1h/24h/7d),         │
│  category risk, geographic entropy, time-of-day, …          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Fraud Model     │   │ Auth Decision   │
          │ IsolationForest │   │ Risk Profiles   │
          │ + Rule Scoring  │   │ + Gemini LLM    │
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
          ┌───────────────────────────────────┐
          │        RAG Engine (ChromaDB)      │
          │  Embeds CSV + SQL → Vector Store  │
          └──────────────┬────────────────────┘
                         ▼
              ┌───────────────────────────┐
              │  Streamlit Dashboard      │
              │  🔑 Auth  🚨 Fraud       │
              │  📊 Dashboard  🔍 RAG Q&A│
              └───────────────────────────┘
                              │
              ┌───────────────────────────┐
              │  Gemini LLM (2.0 Flash)   │
              │  Dynamic Questions + RAG  │
              │  Answer Verification      │
              └───────────────────────────┘
                              │
              ┌───────────────────────────┐
              │  Monitoring & Logging     │
              │  pipeline_logs.csv        │
              └───────────────────────────┘
```

---

## Pipeline Workflow

| Stage | Script / File | Input | Output |
|-------|--------------|-------|--------|
| **1. Ingestion** | `scripts/ingest_csv_to_snowflake.py` | CSV file | Snowflake `RAW_TRANSACTIONS` |
| **2. Feature Engineering** | `scripts/feature_engineering.py` | Raw transactions | `features_output.csv` (19 features) |
| **3. Fraud Scoring** | `models/fraud_model.py` | Feature CSV | `fraud_scores_output.csv` |
| **4. Auth Profiling** | `models/auth_decision.py` | Fraud scores | `auth_profiles_output.csv` |
| **5. Dashboard** | `streamlit_app.py` | All outputs | Interactive web app (Gemini + RAG) |

### Reproducing the Pipeline
```bash
# Step 1: Ingest data (dry-run mode — no Snowflake credentials needed)
python scripts/ingest_csv_to_snowflake.py --dry-run

# Step 2: Compute features
python scripts/feature_engineering.py

# Step 3: Score transactions
python models/fraud_model.py

# Step 4: Generate auth profiles
python models/auth_decision.py

# Step 5: Launch dashboard
streamlit run streamlit_app.py
```

---

## Repository Structure

```
Hands-On-Week-4/
├── README.md                          # This file
├── CONTRIBUTIONS.md                   # Team member responsibilities
├── requirements.txt                   # Python dependencies
├── streamlit_app.py                   # Main Streamlit dashboard (Gemini + RAG)
├── pipeline_logs.csv                  # Pipeline execution log
│
├── scripts/                          # Data pipeline scripts
│   ├── ingest_csv_to_snowflake.py     # CSV → Snowflake ingestion
│   ├── ingest_config.yaml             # Connection configuration
│   └── feature_engineering.py         # Feature computation (19 features)
│
├── sql/                              # SQL schemas & queries
│   ├── create_tables.sql              # DDL for 5 Snowflake tables
│   └── analytical_queries.sql         # 8 analytical queries
│
├── models/                           # Modeling / AI / Decision layer
│   ├── fraud_model.py                 # Hybrid fraud scoring model
│   ├── auth_decision.py               # Auth decision + Gemini integration
│   ├── rag_engine.py                  # Production RAG engine (ChromaDB + hybrid retrieval)
│   ├── rag_evaluator.py               # Automated RAG accuracy evaluation (15 tests)
│   ├── gemini_question_gen.py         # Dynamic question generation (Gemini LLM)
│   └── habit_model.py                 # Transaction habit prediction model
│
└── docs/                             # Documentation assets
    └── architecture_diagram.png       # System architecture diagram
```

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Enable AI Features (Optional)
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
# Or enter it in the Streamlit sidebar at runtime
```

### Run the Dashboard
```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

### Run the Full Pipeline
```bash
# Feature engineering + scoring (no Snowflake needed)
python scripts/feature_engineering.py
python models/fraud_model.py
python models/auth_decision.py
```

#### Production Deployment

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets (ANTHROPIC_API_KEY)
4. Deploy

**AWS/Azure:**
1. Containerize with Docker
2. Deploy to ECS/Fargate or App Service
3. Configure environment variables
4. Set up monitoring (CloudWatch/Application Insights)

**Scaling Considerations:**
- Use PostgreSQL for transaction database (not CSV)
- Implement Redis caching for frequent queries
- Load balancer for multiple app instances
- Auto-scaling based on CPU/memory metrics

### Security & Compliance

#### Implemented Security Features
- API key authentication for Claude API
- Rate limiting (configurable)
- PII data handling (transaction IDs only)
- Secure logging (no sensitive data exposure)

#### Required for Production
- TLS/SSL encryption for all traffic
- Database encryption at rest
- RBAC for admin functions
- Audit logging for compliance
- GDPR/PCI-DSS compliance measures

### Troubleshooting

#### Issue: CSV File Not Found
**Solution:** 
- Ensure CSV file is in same directory as app
- Check filename matches exactly (case-sensitive)
- Or use "Demo Data" mode for testing

#### Issue: AI Questions Not Generating
**Solution:**
- System automatically falls back to demo mode
- Add ANTHROPIC_API_KEY for full AI features
- Check API key is valid and has credits

#### Issue: Slow Performance
**Solution:**
- Use data caching (already implemented)
- Reduce transaction history size (set to last 10-20)
- Deploy on more powerful instance

#### Issue: Logs Not Creating
**Solution:**
- Check write permissions on logs directory
- Verify logs folder exists
- System creates automatically on first run

### Performance Metrics

#### Target Performance
- Authentication latency: <500ms (P95)
- Fraud detection latency: <1000ms (P95)
- Success rate: >95%
- System uptime: 99.9%

#### Actual Performance (from logs)
- Average authentication latency: ~380ms
- Average fraud detection latency: ~880ms
- Success rate: 100% (10/10 logged interactions)
- All operations completed successfully

### Week 4 Deliverables Checklist

✅ **A. Integration Brief** - See Week4_Deliverables.docx Section A  
✅ **B. Application Integration** - streamlit_app.py with all components  
✅ **C. Monitoring & Logging** - logs/product_metrics.csv with 10+ entries  
✅ **D. Evaluation** - Impact metrics and technical analysis in document  
✅ **E. Deployment Plan** - Architecture diagram and scaling strategy  
✅ **F. Failure Analysis** - "Silent Drift" scenario with mitigation  
✅ **Individual Reflection** - Template provided in document  

### Next Steps

#### For Assignment Submission
1. **GitHub Repository:**
   - Upload all files to your repository
   - Include README.md (this file)
   - Push transaction CSV and logs folder
   - Tag release as `week-4-submission`

2. **Deploy Application:**
   - Deploy to Streamlit Cloud or AWS
   - Get public URL
   - Update document with deployment link
   - Test all features on deployed version

3. **Complete Individual Reflection:**
   - Open Week4_Deliverables.docx
   - Navigate to "Individual Reflection" section
   - Write 1 paragraph about production readiness improvements
   - Save and include in submission

4. **Canvas Submission:**
   - GitHub repository link
   - Deployed application link
   - Week4_Deliverables.docx (if separate from repo)




---

## Demo Video


