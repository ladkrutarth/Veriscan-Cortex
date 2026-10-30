# CLAUDE.md — Veriscan-Cortex

Privacy-first fraud intelligence platform (CS 5588 capstone). Streamlit dashboard →
FastAPI gateway → specialised agents → on-device AI (MLX Llama-3, LLaVA, MiniLM,
ChromaDB) + local CSV / Snowflake data. **No user data or model calls may leave the machine.**

## Run

```bash
pip install -r requirements.txt
./run_api.sh                              # FastAPI on :8000 (reload watches src/ only)
./run_dashboard.sh --server.port 8502     # Streamlit, separate terminal
```

- Requires macOS on Apple Silicon (MLX). The app currently runs on the system Python 3.9, so keep
  code 3.9-compatible (`Optional[...]`, not `X | None`, in Pydantic models). First run downloads Llama-3 (~4.9 GB).
- Config is env-only: see `.env.example`; read it through `core.settings.get_settings()`.
- API docs: http://127.0.0.1:8000/docs. Health: `GET /api/health`.
- Don't widen uvicorn `--reload-dir` beyond `src/`: edits to `app/` would restart the API
  and the dashboard reports "Backend API is not available".

## Layout and imports

- `src/` is the import root (`PYTHONPATH=src`). Imports are flat:
  `from models.local_llm import LocalLLM`, `from agents.react_loop import ReactLoop`,
  `from api.schemas import ...`. Never write `from src.…`.
- `src/paths.py` owns every data/log/Chroma path. Import paths from it; don't hard-code them.
- `src/api/`: `main.py` (`create_app()` factory), `lifespan.py` (loads models into `state.py`'s
  `AppState`), `routers/` (one module per feature; paths unchanged), `doc_chat.py` (grounded
  document chat), `uploads.py` (validated per-session storage), `middleware.py` (request IDs,
  access log, safe 500s), `schemas.py` (Pydantic models).
- `src/core/`: `settings.py` (typed env config) and `logging_config.py` (`setup_logging`,
  `redact_session`).
- `src/agents/` — `base.py` (`BaseAgent`, `AgentResult`, `AgentAction`), `react_loop.py`
  (bounded ReAct with JSON actions + keyword fallback), `tool_registry.py` (tool schemas +
  HITL gating), `trajectory_log.py`, advisor / orchestrator / specialists / Spending DNA.
- `src/models/` — LLM & vision wrappers, guard (security) agent, RAG engines,
  `rag_relevance.py`, data tools, demo `auth_store.py`.
- `app/streamlit_app.py` — UI only. It talks to the API over HTTP (`VERISCAN_API_URL`) and
  must never import or load models. Root `streamlit_app.py` is a backward-compatible shim.
- `scripts/` ETL + synthetic data generators; `sql/` Snowflake DDL & queries;
  `configs/ingest_config.yaml`; `docs/VERISCAN-CORTEX-FINAL-REPORT.md` is the capstone report.

## Conventions

- New agents implement `BaseAgent.run(query, session_id) -> AgentResult` and use
  `ToolRegistry` + `ReactLoop`. Keep `max_steps` / `max_tool_calls` bounded.
- Routes that call MLX must stay `async def` (MLX must run on the main thread, not a worker).
- Services come from `AppState` (`Depends(get_state)`); use `state.require(...)` for 503s.
- Log with `logging.getLogger(__name__)`; never log raw session IDs (use `redact_session`),
  passwords, document text or full prompts.
- Sensitive tools (e.g. `challenge_auth`) must stay HITL-gated: only run with explicit
  `approved=True` from the user, never from model output.
- Everything uploaded, indexed or remembered is scoped to `session_id`. Every ChromaDB
  query and every file under `data/user_uploads/` must filter on the caller's session.
- When changing an API response, update `schemas.py`, the route and the Streamlit client together.
- Match the surrounding style: type hints, `from __future__ import annotations`, small
  docstrings. Prefer `logging` over new `print()` calls.

## Environment flags

| Variable | Effect |
|---|---|
| `VERISCAN_FAST_MODE=1` | Use TinyLlama 1.1B instead of Llama-3-8B (fast dev) |
| `VERISCAN_LLM_MODEL` | Override the text model ID |
| `VERISCAN_SKIP_VISION=1` | Don't load LLaVA |
| `ENABLE_AGENTIC_ADVISOR=0` | Disable the advisor ReAct loop (on by default) |
| `ENABLE_MULTI_AGENT_ADVISOR=1` | Route the advisor through the orchestrator + specialists |
| `DECEPTION_LLM_DISABLED=1` / `DECEPTION_LLM_MODEL` | Decoy (deception) responder: templates only / model override |
| `VERISCAN_TRAJECTORY_LOG=0` | Stop writing `logs/trajectories.jsonl` |
| `VERISCAN_EVAL_KEYWORD_ONLY=1` | Agent eval scores the keyword fallback only (no LLM) |
| `VERISCAN_API_URL` | API base URL used by the dashboard |
| `SNOWFLAKE_ACCOUNT` / `SNOWFLAKE_USER` / `SNOWFLAKE_PASSWORD` | Snowflake credentials |

## Testing / evaluation

No pytest suite exists yet. `tests/` holds offline evaluation scripts:

```bash
PYTHONPATH=src python tests/evaluate_rag_local.py
PYTHONPATH=src VERISCAN_EVAL_KEYWORD_ONLY=1 python tests/evaluate_agent_local.py   # fast, no LLM
PYTHONPATH=src python tests/evaluate_advisor_agent_local.py
PYTHONPATH=src python src/models/evaluate_multimodal_rag_local.py
```

Unit/API tests (pytest, ~1 s, no models; CI runs them on 3.9 and 3.11):

```bash
pip install -r requirements-dev.txt
python -m pytest            # config in pyproject.toml
ruff check . && ruff format --check .
```

New tests go in `tests/unit/` and `tests/api/`. Use the fakes in `tests/conftest.py` (`FakeLLM`)
and `tests/api/conftest.py` (`FakeMultimodalRAG`, `state`, `client`, `uploads_dir`).

## Never commit

Secrets (`.env`, `.streamlit/secrets.toml`, real values in `configs/ingest_config.yaml`),
`*.joblib` model binaries, `.chroma_db*/`, `data/user_uploads/`, the full datasets
(`fraudTrain.csv`, `fraudTest.csv`, `cfpb_credit_card.csv`), or `logs/*.jsonl|csv`.
Only `*_sampled.csv` datasets are committed.

## Known production gaps

- `models/auth_store.py` is a demo: in-memory users with a static-salt PBKDF2.
- The server never validates `session_id` against sessions it issued. Uploads without one
  still fall back to a shared `"global"` session (session IDs are now format-validated).
- `/api/user/{user_id}/risk` and `/api/dna/profile/{user_id}` don't check the caller (IDOR).
- No Docker. `requirements.txt` is unpinned. MLX does not run in Linux containers.
- No trained fraud model exists (`paths.MODEL_DIR` is empty), so scoring always uses the
  heuristic in `models/agent_tools_data.py`; no training script produces the model.
- `app/streamlit_app.py` breaks the UI-only rule: it imports `FinancialAdvisorAgent` and
  `MultimodalRAG` directly, indexes uploads in-process, and calls a non-existent
  `/api/llm/generate`. Move these onto the API (`/api/rag/upload` already returns index details).
- Large modules still to split: `agents/financial_advisor_agent.py` (~1.5k lines),
  `app/streamlit_app.py` (~2k), `models/multimodal_rag.py` (~0.9k).
- Deception/ADDF routing is documented in comments (`DeceptionRouter`) but not wired in:
  login always returns `diverted=False`. Only `FastDeceptionLLM` exists.
- Agents still read some env flags via `os.environ` directly; migrate them to `core.settings`.

Subagents in `.claude/agents/` cover these areas: `security-auditor`, `test-engineer`,
`api-hardening`, `llm-eval-engineer`, `devops-engineer`, `ml-data-engineer`,
`observability-engineer`, `agent-developer`.
