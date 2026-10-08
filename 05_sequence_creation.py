"""
Step 5: Sequence Creation for LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

SEQ_LEN = 48
PAD_VALUE = -100.0

OUT_DIR = "processed_data/sequences"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading feature dataset...")
df = pd.read_parquet("processed_data/features.parquet")
df = df.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True)

exclude_cols = ["PatientID", "Source", "SepsisLabel", "ICULOS_row"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Using {len(feature_cols)} features for the model.")

patient_labels = df.groupby("PatientID")["SepsisLabel"].max()
patient_ids = patient_labels.index.to_numpy().astype(str)
labels = patient_labels.to_numpy().astype(int)

train_ids, temp_ids, train_y, temp_y = train_test_split(
    patient_ids, labels, test_size=0.30, stratify=labels, random_state=42
)
val_ids, test_ids, val_y, test_y = train_test_split(
    temp_ids, temp_y, test_size=0.50, stratify=temp_y, random_state=42
)

print(f"\nPatient split:")
print(f"  Train: {len(train_ids)} ({train_y.mean()*100:.2f}% sepsis)")
print(f"  Val:   {len(val_ids)} ({val_y.mean()*100:.2f}% sepsis)")
print(f"  Test:  {len(test_ids)} ({test_y.mean()*100:.2f}% sepsis)")

print("\nFitting scaler on training data only...")
train_mask_rows = df["PatientID"].isin(train_ids)
scaler = StandardScaler()
scaler.fit(df.loc[train_mask_rows, feature_cols])

df_scaled = df.copy()
df_scaled[feature_cols] = scaler.transform(df[feature_cols])

joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")
with open(f"{OUT_DIR}/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

def build_sequences(patient_id_list):
    X_list, y_list, mask_list, pid_list = [], [], [], []

    grouped = df_scaled[df_scaled["PatientID"].isin(patient_id_list)].groupby("PatientID")

    for pid, g in grouped:
        g = g.sort_values("ICULOS")
        feats = g[feature_cols].values
        labels_seq = g["SepsisLabel"].values
        n = len(feats)

        if n >= SEQ_LEN:
            feats_seq = feats[-SEQ_LEN:]
            lab_seq = labels_seq[-SEQ_LEN:]
            m = np.ones(SEQ_LEN, dtype=np.float32)
        else:
            pad_len = SEQ_LEN - n
            pad_feats = np.full((pad_len, feats.shape[1]), PAD_VALUE, dtype=np.float32)
            feats_seq = np.vstack([pad_feats, feats])
            lab_seq = np.concatenate([np.zeros(pad_len), labels_seq])
            m = np.concatenate([np.zeros(pad_len), np.ones(n)]).astype(np.float32)

        X_list.append(feats_seq)
        y_list.append(lab_seq)
        mask_list.append(m)
        pid_list.append(pid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    mask = np.array(mask_list, dtype=np.float32)
    pids = np.array(pid_list)
    return X, y, mask, pids

print("\nBuilding TRAIN sequences...")
X_train, y_train, mask_train, pid_train = build_sequences(train_ids)
print(f"  X_train shape: {X_train.shape}")

print("Building VAL sequences...")
X_val, y_val, mask_val, pid_val = build_sequences(val_ids)
print(f"  X_val shape: {X_val.shape}")

print("Building TEST sequences...")
X_test, y_test, mask_test, pid_test = build_sequences(test_ids)
print(f"  X_test shape: {X_test.shape}")

np.save(f"{OUT_DIR}/X_train.npy", X_train)
np.save(f"{OUT_DIR}/y_train.npy", y_train)
np.save(f"{OUT_DIR}/mask_train.npy", mask_train)
np.save(f"{OUT_DIR}/pid_train.npy", pid_train)

np.save(f"{OUT_DIR}/X_val.npy", X_val)
np.save(f"{OUT_DIR}/y_val.npy", y_val)
np.save(f"{OUT_DIR}/mask_val.npy", mask_val)
np.save(f"{OUT_DIR}/pid_val.npy", pid_val)

np.save(f"{OUT_DIR}/X_test.npy", X_test)
np.save(f"{OUT_DIR}/y_test.npy", y_test)
np.save(f"{OUT_DIR}/mask_test.npy", mask_test)
np.save(f"{OUT_DIR}/pid_test.npy", pid_test)

print(f"\nAll sequence data saved to: {OUT_DIR}/")
print("Step 5 complete. Next: Step 6 (Build LSTM Model).")
