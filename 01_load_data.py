"""
Step 1: Load & Merge Full Dataset (training_setA + training_setB)
"""

import os
import glob
import pandas as pd
from tqdm import tqdm

DATA_DIR = "/Users/veena/Sepsis_Project 2/challenge-2019/training"

OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_folder(folder_path, source_label):
    files = sorted(glob.glob(os.path.join(folder_path, "*.psv")))
    print(f"Found {len(files)} files in {folder_path}")

    if len(files) == 0:
        print(f"  WARNING: no .psv files found in {folder_path}.")
        return []

    frames = []
    bad_files = []

    for f in tqdm(files, desc=f"Loading {source_label}"):
        try:
            patient_id = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f, sep="|")
            df["PatientID"] = patient_id
            df["Source"] = source_label
            df["ICULOS_row"] = range(len(df))
            frames.append(df)
        except Exception as e:
            bad_files.append((f, str(e)))

    if bad_files:
        print(f"  {len(bad_files)} files failed to load in {source_label}:")
        for f, err in bad_files[:5]:
            print(f"    {f}: {err}")

    return frames


def main():
    folder_A = os.path.join(DATA_DIR, "training_setA")
    folder_B = os.path.join(DATA_DIR, "training_setB")

    frames_A = load_folder(folder_A, "A")
    frames_B = load_folder(folder_B, "B")

    all_frames = frames_A + frames_B

    if not all_frames:
        print("\nNo data loaded. Check DATA_DIR.")
        return

    print("\nMerging all patients into one DataFrame...")
    df = pd.concat(all_frames, ignore_index=True)

    n_patients = df["PatientID"].nunique()
    n_rows = len(df)
    n_sepsis_patients = df.groupby("PatientID")["SepsisLabel"].max().sum()

    print("\n================ DATASET SUMMARY ================")
    print(f"Total patients        : {n_patients}")
    print(f"  - from setA         : {df[df.Source=='A']['PatientID'].nunique()}")
    print(f"  - from setB         : {df[df.Source=='B']['PatientID'].nunique()}")
    print(f"Total hourly records  : {n_rows}")
    print(f"Total columns         : {df.shape[1]}")
    print(f"Patients who develop sepsis: {n_sepsis_patients} "
          f"({100*n_sepsis_patients/n_patients:.2f}%)")
    print(f"Sepsis-positive hourly rows: {df['SepsisLabel'].sum()} "
          f"({100*df['SepsisLabel'].mean():.3f}% of all rows)")

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing value % per column (top 15 worst):")
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    print(missing_pct.head(15).round(2))

    out_path = os.path.join(OUTPUT_DIR, "merged_raw.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved merged dataset to: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")
    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
