"""
Step 4: Feature Engineering
"""

import pandas as pd
import numpy as np

print("Loading cleaned + raw datasets...")
df_clean = pd.read_parquet("processed_data/cleaned.parquet")
df_raw = pd.read_parquet("processed_data/merged_raw.parquet")

df_clean = df_clean.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True)
df_raw = df_raw.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True)

key_labs_for_flags = ["WBC", "Lactate", "Platelets", "Creatinine",
                       "BUN", "Bilirubin_total", "Glucose", "Potassium"]

print("Adding 'measured recently' flags...")
for col in key_labs_for_flags:
    df_clean[f"{col}_measured"] = df_raw[col].notna().astype(int)

trend_vitals = ["HR", "MAP", "Resp", "O2Sat", "Temp", "SBP"]

print("Computing hour-to-hour deltas...")
for col in trend_vitals:
    df_clean[f"{col}_delta"] = df_clean.groupby("PatientID")[col].diff().fillna(0)

print("Computing rolling 6-hour stats (this takes a moment)...")
for col in trend_vitals:
    grp = df_clean.groupby("PatientID")[col]
    df_clean[f"{col}_roll_mean6"] = (
        grp.rolling(window=6, min_periods=1).mean().reset_index(drop=True)
    )
    df_clean[f"{col}_roll_std6"] = (
        grp.rolling(window=6, min_periods=1).std().fillna(0).reset_index(drop=True)
    )

print("Computing clinical ratio features...")
df_clean["Shock_Index"] = df_clean["HR"] / df_clean["SBP"].replace(0, np.nan)
df_clean["Shock_Index"] = df_clean["Shock_Index"].fillna(df_clean["Shock_Index"].median())

df_clean["BUN_Creatinine_ratio"] = df_clean["BUN"] / df_clean["Creatinine"].replace(0, np.nan)
df_clean["BUN_Creatinine_ratio"] = df_clean["BUN_Creatinine_ratio"].fillna(
    df_clean["BUN_Creatinine_ratio"].median()
)

print(f"\nFinal feature set shape: {df_clean.shape}")
print(f"Total columns now: {df_clean.shape[1]}")

new_cols = [c for c in df_clean.columns if any(
    c.endswith(s) for s in ["_measured", "_delta", "_roll_mean6", "_roll_std6"]
)] + ["Shock_Index", "BUN_Creatinine_ratio"]
print(f"\nNew engineered columns ({len(new_cols)}):")
print(new_cols)

df_clean.to_parquet("processed_data/features.parquet", index=False)
print("\nSaved to: processed_data/features.parquet")
print("Step 4 complete. Next: Step 5 (Sequence Creation for LSTM).")
