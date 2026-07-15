import os
import random
import sys

from locust import HttpUser, task, between

# locust is launched as `locust -f tests/locust_synch.py` from the repo root,
# so tests/ itself is not on sys.path — add it to import the shared payload pool.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from payloads import PAYLOADS  # noqa: E402


class SyncBaselineUser(HttpUser):
    """
    Hits /predict/synch — LightGBM + full SHAP + counterfactuals computed inline,
    blocking the request. The synchronous-XAI comparison arm.

    Each request samples a different applicant from the deterministic 500-payload
    pool (tests/payloads.py) so the workload isn't a single repeated input.
    """
    wait_time = between(1, 3)

    @task
    def submit_loan_application_sync(self):
        with self.client.post(
            "/predict/synch",
            json=random.choice(PAYLOADS),
            catch_response=True,
            timeout=30,  # SAME 30s client patience as baseline/asynch arms — a per-arm
                         # timeout would bias failure-rate comparisons (a 120s client
                         # makes sync look failure-free merely by waiting 4x longer)
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
