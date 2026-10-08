"""
Step 9: Test Explainability
"""

import numpy as np
from explainability import SepsisExplainer

SEQ_DIR = "processed_data/sequences"

print("Loading test data...")
X_test = np.load(f"{SEQ_DIR}/X_test.npy")
y_test = np.load(f"{SEQ_DIR}/y_test.npy")
mask_test = np.load(f"{SEQ_DIR}/mask_test.npy")
pid_test = np.load(f"{SEQ_DIR}/pid_test.npy", allow_pickle=True)

print("Loading model + building explainer (one-time setup)...")
explainer = SepsisExplainer()

patient_true = np.array([y_test[i][mask_test[i] == 1].max() for i in range(len(pid_test))])

sepsis_idx = np.where(patient_true == 1)[0][:3]
non_sepsis_idx = np.where(patient_true == 0)[0][:2]
sample_idx = list(sepsis_idx) + list(non_sepsis_idx)

for i in sample_idx:
    pid = pid_test[i]
    true_label = "SEPSIS" if patient_true[i] == 1 else "No Sepsis"
    risk_score, reasons = explainer.explain(X_test[i], mask_test[i], top_k=5)

    if risk_score >= 0.6:
        bucket = "HIGH"
    elif risk_score >= 0.3:
        bucket = "MEDIUM"
    else:
        bucket = "LOW"

    print(f"\n{'='*60}")
    print(f"Patient {pid}  |  Actual outcome: {true_label}")
    print(f"Risk Score: {risk_score:.3f}  ->  {bucket} RISK")
    print("Top reasons:")
    for r in reasons:
        print(f"  - {r}")

print(f"\n{'='*60}")
print("Step 9 complete. If these reasons look clinically sensible,")
print("we're ready for Step 10 (Risk Bucketing) and Step 11 (Dashboard).")
