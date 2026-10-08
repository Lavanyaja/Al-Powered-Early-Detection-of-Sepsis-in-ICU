"""
Step 8: Evaluate Model on Test Set
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from lstm_model import elementwise_bce

SEQ_DIR = "processed_data/sequences"
MODEL_DIR = "models"
FIG_DIR = "figures"

print("Loading test data...")
X_test = np.load(f"{SEQ_DIR}/X_test.npy")
y_test = np.load(f"{SEQ_DIR}/y_test.npy")
mask_test = np.load(f"{SEQ_DIR}/mask_test.npy")
pid_test = np.load(f"{SEQ_DIR}/pid_test.npy", allow_pickle=True)

print("Loading trained model...")
model = keras.models.load_model(
    f"{MODEL_DIR}/best_model.keras",
    custom_objects={"elementwise_bce": elementwise_bce},
    compile=False,
)

print("Generating predictions on test set...")
y_pred_prob = model.predict(X_test, batch_size=64, verbose=1)

real_mask = mask_test == 1
y_true_flat = y_test[real_mask]
y_prob_flat = y_pred_prob[real_mask]
y_pred_flat = (y_prob_flat >= 0.5).astype(int)

print(f"\nEvaluating on {len(y_true_flat):,} real hourly records "
      f"from {len(pid_test):,} test patients")

roc_auc = roc_auc_score(y_true_flat, y_prob_flat)
pr_auc = average_precision_score(y_true_flat, y_prob_flat)

print("\n================ TEST SET RESULTS ================")
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

print("\nClassification report (threshold = 0.5):")
print(classification_report(y_true_flat, y_pred_flat,
                             target_names=["No Sepsis", "Sepsis"], digits=3))

cm = confusion_matrix(y_true_flat, y_pred_flat)
print("Confusion matrix:")
print(cm)

patient_true = np.array([y_test[i][mask_test[i] == 1].max() for i in range(len(pid_test))])
patient_max_prob = np.array([y_pred_prob[i][mask_test[i] == 1].max() for i in range(len(pid_test))])
patient_pred = (patient_max_prob >= 0.5).astype(int)

sepsis_patients = patient_true == 1
caught = (patient_pred[sepsis_patients] == 1).sum()
total_sepsis_patients = sepsis_patients.sum()
print(f"\n=== Patient-level: of {total_sepsis_patients} sepsis patients in test set, "
      f"model flagged {caught} at least once ({100*caught/total_sepsis_patients:.1f}%) ===")

false_alarms = ((patient_true == 0) & (patient_pred == 1)).sum()
total_non_sepsis = (patient_true == 0).sum()
print(f"=== False alarms: {false_alarms}/{total_non_sepsis} non-sepsis patients "
      f"flagged at least once ({100*false_alarms/total_non_sepsis:.1f}%) ===")

fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
prec, rec, _ = precision_recall_curve(y_true_flat, y_prob_flat)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(rec, prec, label=f"AP = {pr_auc:.3f}")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_title("Confusion Matrix")
axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["No Sepsis", "Sepsis"])
axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["No Sepsis", "Sepsis"])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                      color="white" if cm[i, j] > cm.max()/2 else "black")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_evaluation.png", dpi=120)
plt.close()

print(f"\nEvaluation plots saved to: {FIG_DIR}/08_evaluation.png")
print("Step 8 complete. Next: Step 9 (Explainability - SHAP-based reasons).")
