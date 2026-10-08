"""
Step 10: Risk Bucketing (High / Medium / Low)
"""

import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import precision_recall_curve, f1_score
from lstm_model import elementwise_bce

SEQ_DIR = "processed_data/sequences"
MODEL_DIR = "models"

print("Loading validation data...")
X_val = np.load(f"{SEQ_DIR}/X_val.npy")
y_val = np.load(f"{SEQ_DIR}/y_val.npy")
mask_val = np.load(f"{SEQ_DIR}/mask_val.npy")

print("Loading model...")
model = keras.models.load_model(
    f"{MODEL_DIR}/best_model.keras", compile=False,
    custom_objects={"elementwise_bce": elementwise_bce},
)

print("Predicting on validation set...")
val_prob = model.predict(X_val, batch_size=64, verbose=1)

real_mask_val = mask_val == 1
y_true_val = y_val[real_mask_val]
y_prob_val = val_prob[real_mask_val]

precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_prob_val)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
threshold_high = thresholds[best_idx]

print(f"\nHIGH threshold (max F1): {threshold_high:.4f}")
print(f"  -> Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, "
      f"F1: {f1_scores[best_idx]:.3f}")

target_recall = 0.85
valid_idx = np.where(recalls[:-1] >= target_recall)[0]
if len(valid_idx) > 0:
    medium_idx = valid_idx[np.argmax(thresholds[valid_idx])]
    threshold_medium = thresholds[medium_idx]
else:
    threshold_medium = thresholds.min()

print(f"\nMEDIUM threshold (recall >= {target_recall}): {threshold_medium:.4f}")

if threshold_medium >= threshold_high:
    threshold_medium = threshold_high * 0.5
    print(f"  (adjusted down to stay below HIGH threshold: {threshold_medium:.4f})")

thresholds_dict = {
    "high": float(threshold_high),
    "medium": float(threshold_medium),
}
with open(f"{MODEL_DIR}/risk_thresholds.json", "w") as f:
    json.dump(thresholds_dict, f, indent=2)
print(f"\nSaved thresholds to {MODEL_DIR}/risk_thresholds.json: {thresholds_dict}")


def bucket(prob):
    if prob >= threshold_high:
        return "HIGH"
    elif prob >= threshold_medium:
        return "MEDIUM"
    else:
        return "LOW"


print("\nLoading test set for final bucketed evaluation...")
X_test = np.load(f"{SEQ_DIR}/X_test.npy")
y_test = np.load(f"{SEQ_DIR}/y_test.npy")
mask_test = np.load(f"{SEQ_DIR}/mask_test.npy")

test_prob = model.predict(X_test, batch_size=64, verbose=1)
real_mask_test = mask_test == 1
y_true_test = y_test[real_mask_test]
y_prob_test = test_prob[real_mask_test]

buckets = np.array([bucket(p) for p in y_prob_test])

print("\n================ BUCKETED TEST RESULTS ================")
for b in ["HIGH", "MEDIUM", "LOW"]:
    mask_b = buckets == b
    n = mask_b.sum()
    sepsis_rate = y_true_test[mask_b].mean() if n > 0 else 0
    print(f"{b:7s}: {n:>7,} hourly records | "
          f"{100*sepsis_rate:.2f}% were actually sepsis-positive")

pid_test = np.load(f"{SEQ_DIR}/pid_test.npy", allow_pickle=True)
patient_true = np.array([y_test[i][mask_test[i] == 1].max() for i in range(len(pid_test))])
patient_max_prob = np.array([test_prob[i][mask_test[i] == 1].max() for i in range(len(pid_test))])
patient_bucket = np.array([bucket(p) for p in patient_max_prob])

print("\n=== Patient-level: highest risk bucket EVER reached during stay ===")
for b in ["HIGH", "MEDIUM", "LOW"]:
    mask_b = patient_bucket == b
    n = mask_b.sum()
    if n > 0:
        sepsis_among = patient_true[mask_b].sum()
        print(f"{b:7s}: {n:>6,} patients | {int(sepsis_among)} actually had sepsis "
              f"({100*sepsis_among/n:.1f}% precision at this level)")
    else:
        print(f"{b:7s}: 0 patients")

sepsis_caught_high_or_med = ((patient_true == 1) &
                              ((patient_bucket == "HIGH") | (patient_bucket == "MEDIUM"))).sum()
total_sepsis = (patient_true == 1).sum()
print(f"\nOf {total_sepsis} true sepsis patients, {sepsis_caught_high_or_med} "
      f"({100*sepsis_caught_high_or_med/total_sepsis:.1f}%) reached at least MEDIUM risk.")

print("\nStep 10 complete. Next: Step 11 (Streamlit Dashboard).")
