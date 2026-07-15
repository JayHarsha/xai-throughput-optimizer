import os
import random
import sys
import time

import gevent
from locust import HttpUser, task, between

# locust is launched as `locust -f tests/locust_asynch.py` from the repo root,
# so tests/ itself is not on sys.path — add it to import the shared payload pool.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from payloads import PAYLOADS  # noqa: E402


class CreditRiskUser(HttpUser):
    # Simulates a human user waiting 1 to 3 seconds between clicks
    wait_time = between(1, 3)

    @task
    def submit_loan_application(self):
        task_id = None
        # Varied applicant per request: Low-risk ones legitimately skip Tier-2,
        # so the Celery workload becomes a realistic Medium/High-risk fraction
        # instead of the former 100%-trigger single payload.
        payload = random.choice(PAYLOADS)
        with self.client.post(
            "/predict/asynch", json=payload,
            catch_response=True, timeout=30,
            headers={"Connection": "close"}  # avoid keep-alive reuse races under high request-rate load
        ) as response:
            if response.status_code == 200:
                data = response.json()
                tier = data.get("risk_tier", "")
                task_id = data.get("deep_analysis", {}).get("task_id")
                # Low risk legitimately skips Celery — not a failure
                if tier == "Low" or task_id:
                    response.success()
                else:
                    response.failure("No task_id returned for non-Low risk applicant")
            else:
                response.failure(f"HTTP {response.status_code}")

        # For Medium/High risk applicants, measure how long Tier-2 (SHAP interactions +
        # counterfactuals) actually takes to complete under load — not just the fast
        # Tier-1 acknowledgment above. Runs in its own greenlet so it doesn't block this
        # user from submitting its next request at the normal wait_time cadence.
        if task_id:
            gevent.spawn(self._poll_tier2_completion, task_id)

    def _poll_tier2_completion(self, task_id):
        start = time.time()
        max_wait = 120      # give up after 2 minutes and record it as a timed-out poll
        poll_interval = 2   # first retry gap; backs off 1.5x per poll (capped at 10s) so
                            # poll traffic stays light even when Tier-2 queues up under load

        gevent.sleep(2)     # Tier-2 takes >2s even unloaded; skip the guaranteed-pending first poll

        while True:
            elapsed_s = time.time() - start
            if elapsed_s > max_wait:
                self.environment.events.request.fire(
                    request_type="POLL",
                    name="tier2_completion",
                    response_time=elapsed_s * 1000,
                    response_length=0,
                    exception=TimeoutError(f"Tier-2 not completed within {max_wait}s"),
                )
                return

            try:
                # catch_response + unconditional success(): the polls are measurement
                # instrumentation, not user traffic — a transient poll error just means
                # "retry", so it must never be counted as an architecture failure.
                # Only the synthetic POLL tier2_completion event above carries pass/fail.
                with self.client.get(
                    f"/result/{task_id}",
                    name="/result/[task_id]",
                    timeout=10,
                    catch_response=True,
                ) as r:
                    r.success()
                    if r.status_code == 200:
                        status = r.json().get("status")
                        if status == "Completed":
                            self.environment.events.request.fire(
                                request_type="POLL",
                                name="tier2_completion",
                                response_time=(time.time() - start) * 1000,
                                response_length=len(r.content),
                                exception=None,
                            )
                            return
                        if status == "Failed":
                            self.environment.events.request.fire(
                                request_type="POLL",
                                name="tier2_completion",
                                response_time=(time.time() - start) * 1000,
                                response_length=0,
                                exception=RuntimeError("Celery task reported Failed"),
                            )
                            return
                        # otherwise still "Processing..." — keep polling
            except Exception:
                pass  # transient poll error; just retry on the next loop

            gevent.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 10)