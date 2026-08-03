# XAI Throughput Optimizer

A credit-risk microservice that measures what post-hoc explainability actually costs when
you put it on a live request path — and an architecture that makes it affordable.

MSc AI practicum, National College of Ireland.

## The problem

SHAP explanations are expensive. Compute them inline and a 10 ms prediction becomes a
multi-second request that collapses under concurrent load. This project quantifies that
collapse and tests a fix: return the decision and its main drivers immediately (**Tier-1**),
and defer the heavy interaction values and counterfactual recourse to background workers
(**Tier-2**), retrieved later by polling.

Three architectures are benchmarked under identical conditions — the only variable is
*where* the explanation is computed.

## Results

75 runs on AWS: 3 architectures × 5 concurrency levels × 5 repeats × 60 s, uniform 30 s
client SLA. p95 is the median across the 5 repeat runs.

| Users | No XAI | Synchronous XAI | Asynchronous XAI |
|---:|---:|---:|---:|
| 1 | 18 ms | 830 ms | 26 ms |
| 10 | 22 ms | 4,500 ms | 93 ms |
| 25 | 29 ms | 18,000 ms | 310 ms |
| 50 | 32 ms | ≥30,000 ms *(SLA-clipped)* | 2,100 ms |

At 50 concurrent users — throughput **23.3 / 2.7 / 12.6** req/s and failure rates
**0.0% / 10.7% / 5.7%** for control / synchronous / asynchronous.

**Findings.** Synchronous XAI inflates median p95 tail latency 173× over the control and
fails one request in ten at 50 users. Asynchronous decoupling cuts p95 by 98%, sustains
4.7× the throughput, and halves the failure rate. But a pre-registered TOST equivalence
test **rejects** equivalence with the no-XAI floor: Tier-1 SHAP still runs on the request
path, costing 7–29 ms through 10 users. Asynchronous XAI is *affordable, not free*.

The asynchronous arm is **bimodal** at 25–50 users — most runs sit near the median, a
minority stall completely — so both median and mean are reported.

## Architecture

```
                    ┌──────────────────────────────────────────┐
   client ─────────►│ FastAPI  (Gunicorn, 4 Uvicorn workers)   │
                    │                                          │
                    │  /predict/baseline  inference only       │
                    │  /predict/synch     inference + T1 + T2  │
                    │  /predict/asynch    inference + T1       │
                    └───────────────┬──────────────────────────┘
   ◄─ decision + Tier-1 + task_id ──┘
                                    │ Tier-2 job (Medium/High risk only)
                                    ▼
                            ┌───────────────┐
                            │ Redis  queue  │
                            └───────┬───────┘
                                    │ workers pull
                                    ▼
                            ┌───────────────────────────────┐
                            │ 3 workers  (0.5 vCPU each)    │
                            │ SHAP interactions +           │
                            │ counterfactual recourse       │
                            └───────┬───────────────────────┘
                                    │ store result
                                    ▼
                            ┌───────────────┐
   client ── GET /result/{task_id} ─┤ Redis  result │
   ◄─────── deep explanation ───────└───────────────┘
```

Tier-2 is gated to Medium/High risk applicants (default probability ≥ 0.40) — 73% of the
workload pool. Workers are CPU-capped so background explanation work can never starve the
interactive tier.

## Quick start

Requires **Python 3.10** (containers) or **3.11** (notebooks) — `scikit-learn==1.3.2`
has no wheels for 3.12+.

```bash
docker compose up --build -d     # API + 3 workers + Redis
docker compose ps                # all healthy?
```

Open <http://localhost:8000> for the demo dashboard. Model artefacts are committed, so
this runs without retraining.

```bash
pytest tests/ -v                 # smoke-test the three arms against the running API
```

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict/baseline` | Inference only — control arm and latency floor |
| `POST /predict/synch` | Inference + Tier-1 + Tier-2, all inline |
| `POST /predict/asynch` | Inference + Tier-1; Tier-2 queued, returns `task_id` |
| `GET /result/{task_id}` | Collect the Tier-2 explanation once workers finish |
| `GET /queue/depth` | Redis backlog — drives the benchmark's drain protocol |
| `GET /metrics/live` | Indicative latencies. Per-worker, so it sees ~¼ of traffic — never quote it |
| `GET /` | Demo dashboard |

## Benchmarking

Run from a load generator **in the same VPC**, targeting the API's private IP — Wi-Fi
jitter is comparable to the effect being measured.

```bash
./scripts/benchmark_pipelines.sh http://<API_PRIVATE_IP>:8000 aws_results
```

75 runs, roughly 2–2.5 hours. Between asynchronous runs the harness polls `/queue/depth`
and waits for three consecutive empty readings, so one run's backlog cannot contaminate
the next.

## Analysis

```bash
python analysis/mann_whitney.py aws_results
```

The results directory is a required argument — a bare run used to analyse the wrong
dataset silently. Outputs `summary_table.csv`, `mann_whitney_report.txt`,
`failure_breakdown.txt` and the figures under `<dir>/plots/`.

Tests: **H1** and **H2** by one-sided Mann-Whitney U on per-run p95s, pooled and per-level
with Holm–Bonferroni. **H3** by TOST equivalence on log-latencies with a pre-registered
1.5× margin — a non-significant two-sided test is not evidence of equivalence.

Two results directories, not interchangeable:

- `aws_results/` — **the reported dataset.** In-VPC load generator, Locust 2.45.0
- `local_pilot_results/` — early Wi-Fi pilot, Locust 2.19.0. Provenance only

## Layout

```
src/api/main.py          FastAPI app, all 7 endpoints
src/ml/explainer.py      SHAP Tier-1 / Tier-2 + counterfactual search
src/worker/tasks.py      Tier-2 background task
tests/payloads.py        Deterministic 500-applicant pool (seed 24245411)
tests/locust_*.py        One load test per architecture
scripts/                 75-run benchmark suite + drain protocol
analysis/mann_whitney.py Statistics, summary table and figures — single source of truth
notebooks/               01 training · 02 workload pool · 03 figures
aws_results/             The reported dataset (75 runs)
artifacts/               LightGBM model + preprocessor
```

## Model

LightGBM (ROC-AUC **0.7404**), chosen over Random Forest (0.7224) and XGBoost (0.7399) —
and, decisively, a tree ensemble compatible with polynomial-time TreeSHAP. Trained on
2.26M LendingClub loans (2007–2018) enriched with three FRED macroeconomic series, leaving
1,345,350 closed loans after removing indeterminate outcomes.

Reproducing the model requires the raw Kaggle dataset (see the Configuration Manual);
it is not committed. Everything else runs from a fresh clone.

## Limitations

Single node, single model, single dataset. The `m7i-flex` instance class has burstable
CPU. Tier-2 completion times are right-censored at the 60 s run boundary, and censored
from below at 2 s by the polling instrumentation. The workload is closed-loop, so the
synchronous penalty is a conservative estimate.
