import shap
import joblib
import os
import numpy as np
from src.ml.feature_labels import humanize_feature

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'lightgbm_credit_model.joblib')

print("Loading Model into SHAP Explainer...")
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
print("SHAP Explainer ready.")

# ... (existing imports and model loading) ...

def compute_main_effects(X_processed, feature_names):
    """Computes base SHAP values for immediate regulatory compliance."""
    shap_values = explainer.shap_values(X_processed)
    main_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    
    main_impacts = sorted(list(zip(feature_names, main_vals)), key=lambda x: abs(x[1]), reverse=True)
    
    total_main = len(main_impacts)
    hidden_main = total_main - 5 if total_main > 5 else 0
    top_main = [{"feature": humanize_feature(f), "impact": round(float(v), 4)} for f, v in main_impacts[:5]]
    
    return {
        "primary_risk_factors": top_main,
        "hidden_primary_factors": hidden_main
    }

def compute_shap_interactions(X_processed, feature_names):
    """Computes the full SHAP interaction matrix asynchronously."""
    interaction_values = explainer.shap_interaction_values(X_processed)
    matrix = interaction_values[1][0] if isinstance(interaction_values, list) else interaction_values[0]
        
    interactions = []
    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = matrix[i][j] * 2 
            if abs(val) > 0.05: 
                interactions.append({
                    "feature_pair": f"{humanize_feature(feature_names[i])} × {humanize_feature(feature_names[j])}",
                    "impact": round(float(val), 4)
                })
                
    interactions_sorted = sorted(interactions, key=lambda x: abs(x["impact"]), reverse=True)
    
    total_interactions = len(interactions_sorted)
    hidden_interactions = total_interactions - 5 if total_interactions > 5 else 0
    top_interactions = interactions_sorted[:5]
    
    return {
        "complex_interactions": top_interactions,
        "hidden_complex_interactions": hidden_interactions
    }


import copy

def get_counterfactuals(original_payload, preprocessor, model, current_probability):
    # Only Medium/High risk (>= 0.40) need counterfactuals — matches tier boundary
    if current_probability < 0.40:
        return {"status": "Not required. Applicant is low risk (< 40% default probability)."}

    print("Initiating Counterfactual Grid Search...")
    
    # ==========================================
    # THE 3 OPTIMIZATIONS FOR ~2 SECOND LATENCY
    # ==========================================
    
    # 1. Only test the term they actually requested (cuts iterations by 50%)
    test_terms = [36, 60]  # [original_payload.get("term", 36.0)]
    
    # 2. Only test the extremes of debt payoff to save loops
    debt_paydown_options = [0.0, 2500.0, 5000.0] 
    
    original_loan_amnt = original_payload["loan_amnt"]
    original_revol_bal = original_payload["revol_bal"]
    
    min_loan_amnt = 1000.0
    
    # 3. Take larger steps down ($1,500 instead of $500)
    step_size = 1500.0 
    
    successful_variations = []
    iterations = 0

    current_test_amnt = original_loan_amnt 
    
    while current_test_amnt >= min_loan_amnt:
        for paydown in debt_paydown_options:
            for term in test_terms:
                
                # RULE 1: No Dumb Advice (Don't ask them to pay more than they are borrowing)
                if paydown > 0 and paydown >= current_test_amnt:
                    continue
                    
                iterations += 1
                
                simulated_payload = copy.deepcopy(original_payload)
                simulated_payload["loan_amnt"] = current_test_amnt
                simulated_payload["term"] = term
                simulated_payload["installment"] = original_payload["installment"] * (current_test_amnt / original_loan_amnt)
                simulated_payload["revol_bal"] = max(0, original_revol_bal - paydown)
                
                import pandas as pd
                df_sim = pd.DataFrame([simulated_payload])
                X_sim = preprocessor.transform(df_sim)
                sim_prob = model.predict_proba(X_sim)[0][1]
                
                if sim_prob <= 0.50:
                    # RULE 2: Diverse Options Only (Don't append the exact same loan amount twice)
                    already_exists = any(v['suggested_loan_amount'] == current_test_amnt for v in successful_variations)
                    
                    if not already_exists:
                        successful_variations.append({
                            "suggested_loan_amount": current_test_amnt,
                            "suggested_term": term,
                            "suggested_installment": round(simulated_payload["installment"], 2),
                            "required_debt_paydown": paydown,
                            "new_simulated_risk": round(float(sim_prob), 4),
                            "compliance_note": "Approval is contingent on paying down existing revolving balances." if paydown > 0 else "Modifying your application to these values could lower your risk profile."
                        })
                    
                    if len(successful_variations) >= 3:
                        return {
                            "actionable_paths": successful_variations,
                            "simulations_run": iterations
                        }
                        
        current_test_amnt -= step_size

    return {
        "status": "No viable counterfactuals found without changing fixed income/debt metrics.",
        "simulations_run": iterations
    }