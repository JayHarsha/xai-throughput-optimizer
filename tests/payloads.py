"""
Shared loan-application payload pool for the three Locust load tests.

Every benchmark request used to send one identical hard-coded applicant, which
(a) exercised a single code path / risk tier 100% of the time and (b) let a
reviewer argue the workload was unrealistically cache-friendly. This module
deterministically generates N_PAYLOADS varied-but-realistic applicants instead
(fixed RNG seed, so every benchmark run replays the exact same payload pool).

Categorical values are restricted to high-frequency LendingClub categories so
the fitted preprocessor is guaranteed to have seen them all.
"""

import random

N_PAYLOADS = 500
_SEED = 24245411  # student id — fixed so the pool is identical across runs/arms

# The original benchmark applicant — kept as the template for fields we don't vary.
BASE_PAYLOAD = {
    "loan_amnt": 27000.0,
    "term": 36.0,
    "int_rate": 12.99,
    "installment": 1213.0,
    "grade": "C",
    "sub_grade": "C2",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 62000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "IL",
    "dti": 22.50,
    "delinq_2yrs": 0.0,
    "fico_range_low": 680.0,
    "fico_range_high": 684.0,
    "inq_last_6mths": 1.0,
    "mths_since_last_delinq": 999.0,
    "mths_since_last_record": 999.0,
    "open_acc": 10.0,
    "pub_rec": 0.0,
    "revol_bal": 12500.0,
    "revol_util": 65.0,
    "total_acc": 20.0,
    "collections_12_mths_ex_med": 0.0,
    "mths_since_last_major_derog": 999.0,
    "policy_code": 1.0,
    "application_type": "Individual",
    "acc_now_delinq": 0.0,
    "tot_coll_amt": 0.0,
    "tot_cur_bal": 45000.0,
    "open_acc_6m": 1.0,
    "open_act_il": 2.0,
    "open_il_12m": 1.0,
    "open_il_24m": 2.0,
    "mths_since_rcnt_il": 12.0,
    "total_bal_il": 12000.0,
    "il_util": 70.0,
    "open_rv_12m": 2.0,
    "open_rv_24m": 3.0,
    "max_bal_bc": 6000.0,
    "all_util": 68.0,
    "total_rev_hi_lim": 28400.0,
    "inq_fi": 1.0,
    "total_cu_tl": 1.0,
    "inq_last_12m": 2.0,
    "acc_open_past_24mths": 5.0,
    "avg_cur_bal": 4500.0,
    "bc_open_to_buy": 4500.0,
    "bc_util": 55.0,
    "chargeoff_within_12_mths": 0.0,
    "delinq_amnt": 0.0,
    "mo_sin_old_il_acct": 72.0,
    "mo_sin_old_rev_tl_op": 120.0,
    "mo_sin_rcnt_rev_tl_op": 6.0,
    "mo_sin_rcnt_tl": 6.0,
    "mort_acc": 0.0,
    "mths_since_recent_bc": 12.0,
    "mths_since_recent_bc_dlq": 999.0,
    "mths_since_recent_inq": 6.0,
    "mths_since_recent_revol_delinq": 999.0,
    "num_accts_ever_120_pd": 0.0,
    "num_actv_bc_tl": 3.0,
    "num_actv_rev_tl": 6.0,
    "num_bc_sats": 4.0,
    "num_bc_tl": 7.0,
    "num_il_tl": 6.0,
    "num_op_rev_tl": 8.0,
    "num_rev_accts": 14.0,
    "num_rev_tl_bal_gt_0": 6.0,
    "num_sats": 10.0,
    "num_tl_120dpd_2m": 0.0,
    "num_tl_30dpd": 0.0,
    "num_tl_90g_dpd_24m": 0.0,
    "num_tl_op_past_12m": 3.0,
    "pct_tl_nvr_dlq": 100.0,
    "percent_bc_gt_75": 25.0,
    "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0,
    "tot_hi_cred_lim": 65000.0,
    "total_bal_ex_mort": 45000.0,
    "total_bc_limit": 15000.0,
    "total_il_high_credit_limit": 20000.0,
    "macro_unemployment_rate": 2.5,
    "macro_cpi": 238.5,
    "macro_fed_funds": 1.0,
}

# Interest-rate bands and typical FICO midpoints per LendingClub grade —
# keeps grade / rate / FICO mutually consistent instead of independently random.
_GRADE_PROFILES = {
    "A": {"rate": (5.3, 9.0),   "fico": (720, 820)},
    "B": {"rate": (8.0, 12.5),  "fico": (690, 760)},
    "C": {"rate": (11.0, 16.5), "fico": (665, 720)},
    "D": {"rate": (15.0, 21.5), "fico": (645, 700)},
    "E": {"rate": (19.0, 27.0), "fico": (630, 680)},
}

_EMP_LENGTHS = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                "6 years", "7 years", "8 years", "9 years", "10+ years"]
_HOME = ["RENT", "MORTGAGE", "OWN"]
_VERIFICATION = ["Verified", "Source Verified", "Not Verified"]
_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other",
             "major_purchase", "medical", "small_business", "car"]
_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "NJ",
           "VA", "MI", "AZ", "MA", "WA"]


def _installment(loan_amnt: float, annual_rate_pct: float, term_months: int) -> float:
    r = annual_rate_pct / 100.0 / 12.0
    n = term_months
    return round(loan_amnt * r * (1 + r) ** n / ((1 + r) ** n - 1), 2)


def _make_payload(rng: random.Random) -> dict:
    p = dict(BASE_PAYLOAD)

    grade = rng.choice(list(_GRADE_PROFILES))
    prof = _GRADE_PROFILES[grade]
    term = rng.choice([36.0, 60.0])
    loan = float(rng.randrange(2000, 40001, 500))
    rate = round(rng.uniform(*prof["rate"]), 2)
    fico_low = float(rng.randrange(prof["fico"][0], prof["fico"][1], 5))
    income = float(rng.randrange(24000, 180001, 1000))

    p["grade"] = grade
    p["sub_grade"] = f"{grade}{rng.randint(1, 5)}"
    p["term"] = term
    p["loan_amnt"] = loan
    p["int_rate"] = rate
    p["installment"] = _installment(loan, rate, int(term))
    p["fico_range_low"] = fico_low
    p["fico_range_high"] = fico_low + 4.0
    p["annual_inc"] = income
    p["emp_length"] = rng.choice(_EMP_LENGTHS)
    p["home_ownership"] = rng.choice(_HOME)
    p["verification_status"] = rng.choice(_VERIFICATION)
    p["purpose"] = rng.choice(_PURPOSES)
    p["addr_state"] = rng.choice(_STATES)
    p["mort_acc"] = 0.0 if p["home_ownership"] == "RENT" else float(rng.randint(1, 4))

    p["dti"] = round(rng.uniform(3.0, 38.0), 2)
    p["revol_bal"] = float(rng.randrange(500, 60001, 100))
    p["revol_util"] = round(rng.uniform(5.0, 95.0), 1)
    p["open_acc"] = float(rng.randint(3, 25))
    p["total_acc"] = p["open_acc"] + float(rng.randint(2, 25))
    p["delinq_2yrs"] = float(rng.choices([0, 1, 2], weights=[80, 15, 5])[0])
    p["inq_last_6mths"] = float(rng.choices([0, 1, 2, 3], weights=[45, 30, 15, 10])[0])
    p["pub_rec"] = float(rng.choices([0, 1], weights=[92, 8])[0])
    p["pub_rec_bankruptcies"] = p["pub_rec"]
    p["mths_since_last_delinq"] = 999.0 if p["delinq_2yrs"] == 0 else float(rng.randint(3, 60))
    p["tot_cur_bal"] = float(rng.randrange(5000, 400001, 1000))
    p["avg_cur_bal"] = round(p["tot_cur_bal"] / max(p["open_acc"], 1), 2)
    p["bc_util"] = round(rng.uniform(5.0, 95.0), 1)
    p["percent_bc_gt_75"] = round(rng.uniform(0.0, 100.0), 1)
    p["pct_tl_nvr_dlq"] = round(rng.uniform(85.0, 100.0), 1)
    p["total_rev_hi_lim"] = float(rng.randrange(5000, 120001, 100))
    p["bc_open_to_buy"] = float(rng.randrange(0, 40001, 100))
    p["max_bal_bc"] = float(rng.randrange(500, 25001, 100))
    p["all_util"] = round(rng.uniform(20.0, 90.0), 1)
    p["il_util"] = round(rng.uniform(20.0, 95.0), 1)
    p["total_bal_il"] = float(rng.randrange(0, 80001, 500))
    p["tot_hi_cred_lim"] = float(rng.randrange(10000, 500001, 1000))
    p["total_bal_ex_mort"] = float(rng.randrange(2000, 200001, 500))
    p["total_bc_limit"] = float(rng.randrange(2000, 80001, 500))
    p["total_il_high_credit_limit"] = float(rng.randrange(0, 120001, 500))

    return p


def _build_pool() -> list[dict]:
    rng = random.Random(_SEED)
    return [_make_payload(rng) for _ in range(N_PAYLOADS)]


PAYLOADS = _build_pool()
