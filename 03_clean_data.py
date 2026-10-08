"""
Step 3: Data Cleaning
"""

import pandas as pd
import numpy as np

print("Loading merged dataset...")
df = pd.read_parquet("processed_data/merged_raw.parquet")
print(f"Loaded {len(df):,} rows, {df['PatientID'].nunique():,} patients")

df = df.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True)

vital_cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
lab_cols = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
            "Alkalinephos", "Calcium", "Chloride", "Creatinine",
            "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
            "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
            "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]

measurement_cols = vital_cols + lab_cols

print(f"\nMissing % BEFORE cleaning (sample of worst columns):")
print((df[measurement_cols].isna().mean() * 100).sort_values(ascending=False).head(10).round(2))

print("\nForward/backward filling within each patient (this takes a moment)...")
df[measurement_cols] = (
    df.groupby("PatientID")[measurement_cols]
      .apply(lambda g: g.ffill().bfill())
      .reset_index(drop=True)
)

print("Filling remaining gaps with global medians...")
medians = df[measurement_cols].median()
df[measurement_cols] = df[measurement_cols].fillna(medians)

df["Unit1"] = df["Unit1"].fillna(0)
df["Unit2"] = df["Unit2"].fillna(0)
df["HospAdmTime"] = df["HospAdmTime"].fillna(df["HospAdmTime"].median())

clip_ranges = {
    "HR": (20, 250), "O2Sat": (50, 100), "Temp": (25, 43),
    "SBP": (40, 280), "MAP": (20, 200), "DBP": (10, 180),
    "Resp": (2, 70), "Glucose": (20, 1000), "WBC": (0, 100),
}
for col, (lo, hi) in clip_ranges.items():
    if col in df.columns:
        df[col] = df[col].clip(lo, hi)

print(f"\nMissing % AFTER cleaning (should all be 0):")
print(df[measurement_cols].isna().sum().sum(), "total remaining missing values")

df.to_parquet("processed_data/cleaned.parquet", index=False)
print("\nSaved cleaned dataset to: processed_data/cleaned.parquet")
print("Step 3 complete. Next: Step 4 (Feature Engineering).")
