"""
Smoke test — verifies all three prediction endpoints return 200 with expected fields.
Requires the API to be running at http://127.0.0.1:8000 (docker compose up -d).
Run with: pytest tests/test_baseline.py -v
"""
import requests
import pytest

BASE_URL = "http://127.0.0.1:8000"

PAYLOAD = {
    "loan_amnt": 27000.0, "term": 36.0, "int_rate": 12.99,
    "installment": 1213.0, "grade": "C", "sub_grade": "C2",
    "emp_length": "5 years", "home_ownership": "RENT",
    "annual_inc": 62000.0, "verification_status": "Verified",
    "purpose": "debt_consolidation", "addr_state": "IL",
    "dti": 22.50, "delinq_2yrs": 0.0, "fico_range_low": 680.0,
    "fico_range_high": 684.0, "inq_last_6mths": 1.0,
    "mths_since_last_delinq": 999.0, "mths_since_last_record": 999.0,
    "open_acc": 10.0, "pub_rec": 0.0, "revol_bal": 12500.0,
    "revol_util": 65.0, "total_acc": 20.0,
    "collections_12_mths_ex_med": 0.0, "mths_since_last_major_derog": 999.0,
    "policy_code": 1.0, "application_type": "Individual",
    "acc_now_delinq": 0.0, "tot_coll_amt": 0.0, "tot_cur_bal": 45000.0,
    "open_acc_6m": 1.0, "open_act_il": 2.0, "open_il_12m": 1.0,
    "open_il_24m": 2.0, "mths_since_rcnt_il": 12.0, "total_bal_il": 12000.0,
    "il_util": 70.0, "open_rv_12m": 2.0, "open_rv_24m": 3.0,
    "max_bal_bc": 6000.0, "all_util": 68.0, "total_rev_hi_lim": 28400.0,
    "inq_fi": 1.0, "total_cu_tl": 1.0, "inq_last_12m": 2.0,
    "acc_open_past_24mths": 5.0, "avg_cur_bal": 4500.0,
    "bc_open_to_buy": 4500.0, "bc_util": 55.0,
    "chargeoff_within_12_mths": 0.0, "delinq_amnt": 0.0,
    "mo_sin_old_il_acct": 72.0, "mo_sin_old_rev_tl_op": 120.0,
    "mo_sin_rcnt_rev_tl_op": 6.0, "mo_sin_rcnt_tl": 6.0,
    "mort_acc": 0.0, "mths_since_recent_bc": 12.0,
    "mths_since_recent_bc_dlq": 999.0, "mths_since_recent_inq": 6.0,
    "mths_since_recent_revol_delinq": 999.0, "num_accts_ever_120_pd": 0.0,
    "num_actv_bc_tl": 3.0, "num_actv_rev_tl": 6.0, "num_bc_sats": 4.0,
    "num_bc_tl": 7.0, "num_il_tl": 6.0, "num_op_rev_tl": 8.0,
    "num_rev_accts": 14.0, "num_rev_tl_bal_gt_0": 6.0, "num_sats": 10.0,
    "num_tl_120dpd_2m": 0.0, "num_tl_30dpd": 0.0, "num_tl_90g_dpd_24m": 0.0,
    "num_tl_op_past_12m": 3.0, "pct_tl_nvr_dlq": 100.0,
    "percent_bc_gt_75": 25.0, "pub_rec_bankruptcies": 0.0, "tax_liens": 0.0,
    "tot_hi_cred_lim": 65000.0, "total_bal_ex_mort": 45000.0,
    "total_bc_limit": 15000.0, "total_il_high_credit_limit": 20000.0,
    "macro_unemployment_rate": 2.5, "macro_cpi": 238.5, "macro_fed_funds": 1.0,
}


def test_baseline_endpoint():
    r = requests.post(f"{BASE_URL}/predict/baseline", json=PAYLOAD, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "default_probability" in data
    assert "risk_tier" in data
    assert data["mode"] == "baseline_no_xai"
    assert data["api_latency_ms"] > 0


def test_synch_endpoint():
    r = requests.post(f"{BASE_URL}/predict/synch", json=PAYLOAD, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert "default_probability" in data
    assert "risk_tier" in data
    assert data["mode"] == "synchronous_xai"
    assert data["api_latency_ms"] > 0


def test_asynch_endpoint():
    r = requests.post(f"{BASE_URL}/predict/asynch", json=PAYLOAD, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "default_probability" in data
    assert "risk_tier" in data
    assert "deep_analysis" in data
    assert data["api_latency_ms"] > 0


def test_metrics_endpoint():
    r = requests.get(f"{BASE_URL}/metrics/live", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert all(k in data for k in ("baseline", "synch", "asynch"))
