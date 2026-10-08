"""
Step 2: Exploratory Data Analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")

print("Loading merged dataset...")
df = pd.read_parquet("processed_data/merged_raw.parquet")
print(f"Loaded {len(df):,} rows, {df['PatientID'].nunique():,} patients\n")

# 1. Class imbalance
patient_labels = df.groupby("PatientID")["SepsisLabel"].max()
print("=== Class Balance ===")
print(f"Patients WITH sepsis   : {patient_labels.sum():,} "
      f"({100*patient_labels.mean():.2f}%)")
print(f"Patients WITHOUT sepsis: {(patient_labels==0).sum():,} "
      f"({100*(1-patient_labels.mean()):.2f}%)")

fig, ax = plt.subplots(figsize=(5, 4))
patient_labels.value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"], ax=ax)
ax.set_xticklabels(["No Sepsis", "Sepsis"], rotation=0)
ax.set_title("Patient-level Class Balance")
ax.set_ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_balance.png", dpi=120)
plt.close()

# 2. ICU stay length distribution
stay_length = df.groupby("PatientID")["ICULOS_row"].max() + 1
print(f"\n=== ICU Stay Length (hours) ===")
print(stay_length.describe())

fig, ax = plt.subplots(figsize=(6, 4))
stay_length.clip(upper=200).hist(bins=50, ax=ax, color="#4C72B0")
ax.set_title("ICU Stay Length Distribution (clipped at 200h)")
ax.set_xlabel("Hours")
ax.set_ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_stay_length.png", dpi=120)
plt.close()

# 3. Vital sign patterns: sepsis vs non-sepsis
key_vitals = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Glucose"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(key_vitals):
    data_pos = df.loc[df.SepsisLabel == 1, col].dropna()
    data_neg = df.loc[df.SepsisLabel == 0, col].dropna()
    axes[i].hist(data_neg, bins=40, alpha=0.5, label="No Sepsis", density=True, color="#4C72B0")
    axes[i].hist(data_pos, bins=40, alpha=0.5, label="Sepsis", density=True, color="#C44E52")
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_vitals_sepsis_vs_not.png", dpi=120)
plt.close()

# 4. Missingness
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 10))
missing_pct.plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Missing Value % per Column")
ax.set_xlabel("% Missing")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_missingness.png", dpi=120)
plt.close()

# 5. Correlation
corr = df[key_vitals + ["SepsisLabel"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation: Key Vitals vs SepsisLabel")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_correlation.png", dpi=120)
plt.close()

# 6. Age by sepsis
patient_meta = df.groupby("PatientID").agg(
    Age=("Age", "first"),
    Gender=("Gender", "first"),
    Sepsis=("SepsisLabel", "max")
)
print("\n=== Age stats by sepsis outcome ===")
print(patient_meta.groupby("Sepsis")["Age"].describe())

fig, ax = plt.subplots(figsize=(6, 4))
patient_meta.boxplot(column="Age", by="Sepsis", ax=ax)
ax.set_title("Age by Sepsis Outcome")
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_age_by_sepsis.png", dpi=120)
plt.close()

print(f"\nAll plots saved to '{FIG_DIR}/' folder.")
print("Step 2 complete. Next: Step 3 (Data Cleaning).")
