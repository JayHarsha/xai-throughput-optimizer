import os
import random
import sys

from locust import HttpUser, task, between

# locust is launched as `locust -f tests/locust_baseline.py` from the repo root,
# so tests/ itself is not on sys.path — add it to import the shared payload pool.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from payloads import PAYLOADS  # noqa: E402


class BaselineUser(HttpUser):
    """
    Hits /predict/baseline — pure LightGBM inference with no SHAP or counterfactuals.
    Run this to establish the XAI-free latency floor for dissertation benchmarking.

    Each request samples a different applicant from the deterministic 500-payload
    pool (tests/payloads.py) so the workload isn't a single repeated input.

    Usage:
        locust -f tests/locust_baseline.py --host=http://localhost:8000 \
               --users 50 --spawn-rate 5 --run-time 60s --csv=results/locust_baseline_50u_run1
    """
    wait_time = between(1, 3)

    @task
    def submit_baseline_inference(self):
        with self.client.post(
            "/predict/baseline",
            json=random.choice(PAYLOADS),
            catch_response=True,
            timeout=30,  # generous enough to avoid clipping the true tail under heavy queueing contention
            headers={"Connection": "close"}  # avoid keep-alive reuse races under high request-rate load
        ) as response:
            if response.status_code == 200:
                data = response.json()
                latency = data.get("api_latency_ms", 0)
                if latency > 0:
                    response.success()
                else:
                    response.failure("No latency reported")
            else:
                response.failure(f"HTTP {response.status_code}")
