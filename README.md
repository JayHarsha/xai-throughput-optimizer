# XAI Throughput Optimizer

**Asynchronous Pipeline for Explainable AI — NCI Dublin MSc AI Practicum**

Dissertation project demonstrating that synchronous XAI explanation generation is architecturally incompatible with concurrent real-world usage, and that an asynchronous pipeline resolves this without sacrificing explanation quality.

---

## Architecture

```
Browser / Locust
     │
     ▼
FastAPI Gateway  (4 Gunicorn/Uvicorn workers)
     │
     ├── /predict/baseline  ── LightGBM inference only (control group)
     │
     ├── /predict/synch     ── LightGBM + SHAP + Counterfactuals (blocking)
     │
     └── /predict/asynch    ── LightGBM + basic SHAP (Tier 1, returns immediately)
                                        │
                                        └── Celery task → Redis → 3 Worker containers
                                                 └── SHAP interactions + Counterfactuals (Tier 2)
```

**Stack:** Python 3.11 · FastAPI · LightGBM · SHAP TreeExplainer · Celery · Redis · Docker Compose

**Dataset:** LendingClub loan data (2007–2018, 2.26M records, 85 features after engineering)

---

## Endpoints

| Endpoint | Description | Used for |
|---|---|---|
| `POST /predict/baseline` | Pure inference — no XAI | Control group / latency floor |
| `POST /predict/synch` | Synchronous XAI (blocking) | Synchronous baseline comparison |
| `POST /predict/asynch` | Async Tier-1 + Celery Tier-2 | Main dissertation system |
| `GET /result/{task_id}` | Poll Celery task status | Async deep-analysis polling |
| `GET /metrics/live` | Live p50/p95 per endpoint | Dashboard metrics |
| `GET /` | Interactive demo dashboard | Browser UI |

---

## Quick Start

```bash
# Start all containers (API, 3 Celery workers, Redis)
docker compose up --build

# Open dashboard
open http://localhost:8000
```

---

## Load Testing

Run all 15 experiments automatically (3 architectures × 5 concurrency levels, one run each):

```powershell
.\scripts\benchmark_pipelines.ps1
```

Or manually for a single experiment:

```bash
locust -f tests/locust_baseline.py --host=http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 60s \
       --csv=results/locust_baseline_50u_run1 --headless
```

CSV files are written to `results/` and must follow the pattern `locust_<arch>_<N>u_run<R>_stats.csv` for the analysis script to pick them up.

---

## Statistical Validation

After collecting CSV results, fill in the latency arrays in [analysis/mann_whitney.py](analysis/mann_whitney.py) and run:

```bash
python analysis/mann_whitney.py
```

Tests performed (Mann-Whitney U, non-parametric, α = 0.05):
- Sync XAI latency > No-XAI latency → confirms XAI overhead is real
- Sync XAI latency > Async Tier-1 latency → confirms async superiority
- Async Tier-1 latency ≈ No-XAI latency → confirms async perceived speed matches inference-only

---

## Project Structure

```
xai-throughput-optimizer/
├── src/
│   ├── api/main.py              # FastAPI app + all endpoints
│   ├── worker/
│   │   ├── celery_app.py        # Celery + Redis config
│   │   └── tasks.py             # Async XAI task definition
│   └── ml/
│       └── explainer.py         # SHAP TreeExplainer + counterfactuals
├── tests/
│   ├── locust_asynch.py         # Async pipeline load test
│   ├── locust_synch.py          # Sync XAI load test
│   └── locust_baseline.py       # No-XAI control group load test
├── analysis/
│   └── mann_whitney.py          # Statistical significance testing
├── artifacts/                   # Trained model + preprocessor (not committed)
├── dashboard.html               # Interactive browser demo
└── docker-compose.yml
```
