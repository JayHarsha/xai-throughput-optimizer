"""
Human-readable labels for all LendingClub features.
Used to clean up scikit-learn ColumnTransformer prefixed names
(num__fico_range_low → "FICO Score (Low)") in API output.
"""

# Numerical feature labels
FEATURE_LABELS: dict[str, str] = {
    # ── Core loan terms ────────────────────────────────────────────
    "loan_amnt":                        "Loan Amount",
    "term":                             "Loan Term (months)",
    "int_rate":                         "Interest Rate (%)",
    "installment":                      "Monthly Payment",

    # ── Creditworthiness ──────────────────────────────────────────
    "fico_range_low":                   "FICO Score (Low)",
    "fico_range_high":                  "FICO Score (High)",
    "pub_rec":                          "Public Derogatory Records",
    "pub_rec_bankruptcies":             "Bankruptcies",
    "tax_liens":                        "Tax Liens",
    "acc_now_delinq":                   "Currently Delinquent Accounts",
    "delinq_2yrs":                      "Delinquencies (2 Years)",
    "delinq_amnt":                      "Delinquent Amount",
    "mths_since_last_delinq":           "Months Since Last Delinquency",
    "mths_since_last_record":           "Months Since Last Public Record",
    "mths_since_last_major_derog":      "Months Since Major Derogatory Record",
    "num_accts_ever_120_pd":            "Accounts Ever 120+ Days Late",
    "num_tl_120dpd_2m":                 "Accounts 120+ Days Late (2 Months)",
    "num_tl_30dpd":                     "Accounts 30+ Days Late",
    "num_tl_90g_dpd_24m":               "Accounts 90+ Days Late (24 Months)",
    "chargeoff_within_12_mths":         "Charge-offs (12 Months)",
    "collections_12_mths_ex_med":       "Non-Medical Collections (12 Months)",
    "tot_coll_amt":                     "Total Collection Amount",
    "pct_tl_nvr_dlq":                   "% Accounts Never Delinquent",
    "inq_last_6mths":                   "Credit Inquiries (Last 6 Months)",
    "inq_last_12m":                     "Credit Inquiries (Last 12 Months)",
    "inq_fi":                           "Finance Company Inquiries",
    "mths_since_recent_inq":            "Months Since Most Recent Inquiry",

    # ── Income & debt load ────────────────────────────────────────
    "annual_inc":                       "Annual Income",
    "dti":                              "Debt-to-Income Ratio (%)",

    # ── Revolving credit ──────────────────────────────────────────
    "revol_bal":                        "Revolving Credit Balance",
    "revol_util":                       "Revolving Credit Utilisation (%)",
    "total_rev_hi_lim":                 "Total Revolving Credit Limit",
    "num_rev_accts":                    "Total Revolving Accounts",
    "num_rev_tl_bal_gt_0":              "Revolving Accounts with Balance",
    "num_actv_rev_tl":                  "Active Revolving Accounts",
    "num_op_rev_tl":                    "Open Revolving Accounts",
    "open_rv_12m":                      "Revolving Accounts Opened (12 Months)",
    "open_rv_24m":                      "Revolving Accounts Opened (24 Months)",
    "mo_sin_old_rev_tl_op":             "Age of Oldest Revolving Account (months)",
    "mo_sin_rcnt_rev_tl_op":            "Age of Newest Revolving Account (months)",
    "mths_since_recent_revol_delinq":   "Months Since Revolving Delinquency",

    # ── Bank card ─────────────────────────────────────────────────
    "bc_util":                          "Bank Card Utilisation (%)",
    "bc_open_to_buy":                   "Bank Card Available Credit",
    "max_bal_bc":                       "Max Bank Card Balance",
    "total_bc_limit":                   "Total Bank Card Limit",
    "num_bc_tl":                        "Total Bank Card Accounts",
    "num_bc_sats":                      "Satisfactory Bank Card Accounts",
    "num_actv_bc_tl":                   "Active Bank Card Accounts",
    "percent_bc_gt_75":                 "% Bank Cards Over 75% Utilised",
    "mths_since_recent_bc":             "Months Since Most Recent Bank Card",
    "mths_since_recent_bc_dlq":         "Months Since Bank Card Delinquency",

    # ── Instalment loans ──────────────────────────────────────────
    "open_act_il":                      "Active Instalment Loans",
    "open_il_12m":                      "Instalment Loans Opened (12 Months)",
    "open_il_24m":                      "Instalment Loans Opened (24 Months)",
    "mths_since_rcnt_il":               "Months Since Recent Instalment Loan",
    "total_bal_il":                     "Total Instalment Loan Balance",
    "il_util":                          "Instalment Loan Utilisation (%)",
    "total_il_high_credit_limit":       "Total Instalment Credit Limit",
    "num_il_tl":                        "Total Instalment Accounts",
    "mo_sin_old_il_acct":               "Age of Oldest Instalment Account (months)",

    # ── Mortgage ──────────────────────────────────────────────────
    "mort_acc":                         "Mortgage Accounts",
    "total_bal_ex_mort":                "Total Balance (excl. Mortgage)",

    # ── Overall credit mix ────────────────────────────────────────
    "open_acc":                         "Open Credit Lines",
    "total_acc":                        "Total Credit Lines",
    "open_acc_6m":                      "New Credit Lines (6 Months)",
    "acc_open_past_24mths":             "Accounts Opened (24 Months)",
    "num_tl_op_past_12m":               "Accounts Opened (12 Months)",
    "num_sats":                         "Satisfactory Accounts",
    "all_util":                         "Overall Credit Utilisation (%)",
    "avg_cur_bal":                      "Average Account Balance",
    "tot_cur_bal":                      "Total Current Balance",
    "tot_hi_cred_lim":                  "Total Credit Limit",
    "total_cu_tl":                      "Active Tradelines",
    "mo_sin_rcnt_tl":                   "Age of Newest Account (months)",
    "policy_code":                      "Policy Code",

    # ── Macroeconomic indicators ──────────────────────────────────
    "macro_unemployment_rate":          "Unemployment Rate (%)",
    "macro_cpi":                        "Consumer Price Index",
    "macro_fed_funds":                  "Federal Funds Rate (%)",
}

# Categorical column base labels (value appended at runtime)
CATEGORICAL_LABELS: dict[str, str] = {
    "grade":               "Credit Grade",
    "sub_grade":           "Credit Sub-Grade",
    "emp_length":          "Employment Length",
    "home_ownership":      "Home Ownership",
    "verification_status": "Income Verification",
    "purpose":             "Loan Purpose",
    "addr_state":          "State",
    "application_type":    "Application Type",
}


def humanize_feature(raw_name: str) -> str:
    """
    Convert a scikit-learn ColumnTransformer feature name to a human label.

    Examples:
      num__fico_range_low     → "FICO Score (Low)"
      num__dti                → "Debt-to-Income Ratio (%)"
      cat__grade_C            → "Credit Grade: C"
      cat__home_ownership_RENT → "Home Ownership: RENT"
    """
    if raw_name.startswith("num__"):
        key = raw_name[5:]
        return FEATURE_LABELS.get(key, _fallback(key))

    if raw_name.startswith("cat__"):
        rest = raw_name[5:]
        # Match against known categorical columns (longest match first to avoid
        # ambiguity between e.g. "grade" and "sub_grade")
        for col in sorted(CATEGORICAL_LABELS, key=len, reverse=True):
            prefix = col + "_"
            if rest.startswith(prefix):
                value = rest[len(prefix):]
                return f"{CATEGORICAL_LABELS[col]}: {value}"
        return _fallback(rest)

    # No prefix — try direct lookup then fallback
    return FEATURE_LABELS.get(raw_name, _fallback(raw_name))


def _fallback(name: str) -> str:
    """Last-resort: replace underscores with spaces and title-case."""
    return name.replace("_", " ").title()
