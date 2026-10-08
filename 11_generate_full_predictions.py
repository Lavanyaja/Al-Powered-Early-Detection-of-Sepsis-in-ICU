"""
Step 11a: Generate Predictions for the FULL Population (all 40,336 patients)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from lstm_model import elementwise_bce

SEQ_LEN = 48
PAD_VALUE = -100.0
SEQ_DIR = "processed_data/sequences"
OUT_DIR = "processed_data/full_population"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading feature dataset (all patients)...")
df = pd.read_parquet("processed_data/features.parquet")
df = df.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True)

with open(f"{SEQ_DIR}/feature_cols.json") as f:
    feature_cols = json.load(f)
scaler = joblib.load(f"{SEQ_DIR}/scaler.pkl")

print("Scaling full dataset with the training-fitted scaler...")
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.transform(df[feature_cols])

print("Building 48-hour sequences for ALL patients (this takes a few minutes)...")
X_list, mask_list, pid_list = [], [], []
grouped = df_scaled.groupby("PatientID")

for pid, g in grouped:
    g = g.sort_values("ICULOS")
    feats = g[feature_cols].values
    n = len(feats)

    if n >= SEQ_LEN:
        feats_seq = feats[-SEQ_LEN:]
        m = np.ones(SEQ_LEN, dtype=np.float32)
    else:
        pad_len = SEQ_LEN - n
        pad_feats = np.full((pad_len, feats.shape[1]), PAD_VALUE, dtype=np.float32)
        feats_seq = np.vstack([pad_feats, feats])
        m = np.concatenate([np.zeros(pad_len), np.ones(n)]).astype(np.float32)

    X_list.append(feats_seq)
    mask_list.append(m)
    pid_list.append(pid)

X_all = np.array(X_list, dtype=np.float32)
mask_all = np.array(mask_list, dtype=np.float32)
pid_all = np.array(pid_list)
print(f"X_all shape: {X_all.shape}")

np.save(f"{OUT_DIR}/X_all.npy", X_all)
np.save(f"{OUT_DIR}/mask_all.npy", mask_all)
np.save(f"{OUT_DIR}/pid_all.npy", pid_all)

print("\nLoading trained model...")
model = keras.models.load_model(
    f"{MODEL_DIR}/best_model.keras", compile=False,
    custom_objects={"elementwise_bce": elementwise_bce},
)

print("Running predictions on all patients...")
all_probs = model.predict(X_all, batch_size=128, verbose=1)

with open(f"{MODEL_DIR}/risk_thresholds.json") as f:
    thresholds = json.load(f)


def bucket_of(prob):
    if prob >= thresholds["high"]:
        return "High"
    elif prob >= thresholds["medium"]:
        return "Medium"
    else:
        return "Low"


print("Building patient summary table...")
meta = df.groupby("PatientID").agg(
    Age=("Age", "first"),
    Gender=("Gender", "first"),
    Source=("Source", "first"),
    SepsisEver=("SepsisLabel", "max"),
).reset_index()
meta = meta.set_index("PatientID")

records = []
for i, pid in enumerate(pid_all):
    real_idx = np.where(mask_all[i] == 1)[0]
    current_prob = float(all_probs[i][real_idx[-1]])
    peak_prob = float(all_probs[i][real_idx].max())
    icu_los = len(real_idx)
    row = meta.loc[pid]
    records.append({
        "PatientID": pid,
        "Age": int(row["Age"]) if pd.notna(row["Age"]) else None,
        "Gender": "Male" if row["Gender"] == 1 else "Female",
        "Source": row["Source"],
        "ICU_LOS": icu_los,
        "CurrentRisk": current_prob,
        "PeakRisk": peak_prob,
        "Bucket": bucket_of(current_prob),
        "ActualOutcome": "Sepsis" if row["SepsisEver"] == 1 else "No Sepsis",
    })

summary_df = pd.DataFrame(records)
summary_df.to_csv(f"{OUT_DIR}/patient_summary.csv", index=False)

print(f"\n================ SUMMARY ================")
print(f"Total patients: {len(summary_df)}")
print(summary_df["Bucket"].value_counts())
print(f"\nSaved to: {OUT_DIR}/patient_summary.csv")
print(f"Sequences saved to: {OUT_DIR}/X_all.npy, mask_all.npy, pid_all.npy")
print("\nStep 11a complete. Next: run the Flask backend (backend.py).")
