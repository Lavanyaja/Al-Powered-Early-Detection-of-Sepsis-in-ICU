"""
Step 11b: Flask Backend for the Sepsis Risk Dashboard
Run with:  python backend.py
Then open: http://localhost:8000
"""

import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from explainability import SepsisExplainer, FEATURE_INFO

OUT_DIR = "processed_data/full_population"
MODEL_DIR = "models"

app = Flask(__name__)

print("Loading patient summary table...")
summary_df = pd.read_csv(f"{OUT_DIR}/patient_summary.csv")

print("Loading sequence data...")
X_all = np.load(f"{OUT_DIR}/X_all.npy")
mask_all = np.load(f"{OUT_DIR}/mask_all.npy")
pid_all = np.load(f"{OUT_DIR}/pid_all.npy", allow_pickle=True)
pid_to_idx = {pid: i for i, pid in enumerate(pid_all)}

with open(f"{MODEL_DIR}/risk_thresholds.json") as f:
    THRESHOLDS = json.load(f)


def bucket_of(prob):
    if prob >= THRESHOLDS["high"]:
        return "High"
    elif prob >= THRESHOLDS["medium"]:
        return "Medium"
    else:
        return "Low"


print("Loading model + building explainer (one-time)...")
explainer = SepsisExplainer()

DOCTORS = ["Dr. Sharma", "Dr. Iyer", "Dr. Nair", "Dr. Reddy", "Dr. Menon", "Dr. Rao"]

REASONS_BY_LEVEL = {"High": 6, "Medium": 4, "Low": 2}
TIER_DISPLAY_CAP = {"Low": "moderate", "Medium": "moderate", "High": "severe"}
TIER_DISPLAY_FLOOR = {"Low": "mild", "Medium": "mild", "High": "moderate"}
_TIER_ORDER = {"mild": 0, "moderate": 1, "severe": 2}


def patient_name(pid):
    try:
        num = int(str(pid).lstrip("p").lstrip("0") or "0")
    except ValueError:
        num = 0
    return f"ICU Patient #{num}"


def assigned_guide(pid):
    return DOCTORS[hash(pid) % len(DOCTORS)]


@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/summary")
def api_summary():
    total = len(summary_df)
    high = int((summary_df["Bucket"] == "High").sum())
    medium = int((summary_df["Bucket"] == "Medium").sum())
    low = int((summary_df["Bucket"] == "Low").sum())
    avg_risk = round(summary_df["CurrentRisk"].mean() * 100, 1)

    dataset_counts = summary_df.groupby("Source")["PatientID"].nunique()
    datasets = []
    if "A" in dataset_counts:
        datasets.append({"name": "Training Set A", "patients": int(dataset_counts["A"])})
    if "B" in dataset_counts:
        datasets.append({"name": "Training Set B", "patients": int(dataset_counts["B"])})

    return jsonify({
        "total_patients": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "avg_risk_score": avg_risk,
        "datasets": datasets,
    })


@app.route("/api/patients")
def api_patients():
    search = request.args.get("search", "").strip().lower()
    risk = request.args.get("risk", "all").lower()

    df = summary_df.copy()

    if risk != "all":
        df = df[df["Bucket"].str.lower() == risk]

    if search:
        names = df["PatientID"].apply(patient_name).str.lower()
        mask = df["PatientID"].str.lower().str.contains(search) | names.str.contains(search)
        df = df[mask]

    df = df.sort_values("CurrentRisk", ascending=False)

    results = []
    for _, row in df.iterrows():
        results.append({
            "patient_id": row["PatientID"],
            "patient_name": patient_name(row["PatientID"]),
            "Age": int(row["Age"]) if row["Age"] == row["Age"] else None,
            "Gender": str(row["Gender"]),
            "ICU_LOS": int(row["ICU_LOS"]),
            "risk_score": round(row["CurrentRisk"] * 100, 1),
            "risk_level": row["Bucket"],
            "assigned_guide": assigned_guide(row["PatientID"]),
        })
    return jsonify(results)


@app.route("/api/patient/<pid>")
def api_patient_detail(pid):
    if pid not in pid_to_idx:
        return jsonify({"error": "Patient not found"}), 404

    idx = pid_to_idx[pid]
    row = summary_df[summary_df["PatientID"] == pid].iloc[0]

    risk_score, reasons, raw_last_hour = explainer.explain_structured(
        X_all[idx], mask_all[idx], top_k=6
    )

    bucket = row["Bucket"]
    cap_n = REASONS_BY_LEVEL.get(bucket, 5)
    reasons = reasons[:cap_n]

    display_cap = TIER_DISPLAY_CAP.get(bucket, "severe")
    display_floor = TIER_DISPLAY_FLOOR.get(bucket, "mild")
    for r in reasons:
        if _TIER_ORDER[r["tier_css"]] > _TIER_ORDER[display_cap]:
            r["tier_css"] = display_cap
            r["tier"] = display_cap
        elif _TIER_ORDER[r["tier_css"]] < _TIER_ORDER[display_floor]:
            r["tier_css"] = display_floor
            r["tier"] = display_floor
    if reasons and bucket == "High":
        reasons[0]["tier_css"] = "severe"
        reasons[0]["tier"] = "severe"

    key_vitals = {}
    for feat in ["Resp", "SBP", "Temp", "HR", "Lactate"]:
        if feat in raw_last_hour:
            label, unit, lo, hi = FEATURE_INFO.get(feat, (feat, "", None, None))
            val = round(float(raw_last_hour[feat]), 1)
            is_abnormal = (lo is not None) and (val < lo or val > hi)
            key_vitals[feat] = {
                "label": label, "value": val, "unit": unit, "is_abnormal": bool(is_abnormal)
            }

    predicted_label = 1 if row["Bucket"] in ("High", "Medium") else 0

    return jsonify({
        "patient_id": pid,
        "patient_name": patient_name(pid),
        "risk_level": row["Bucket"],
        "predicted_label": predicted_label,
        "risk_score": round(risk_score * 100, 1),
        "Age": int(row["Age"]) if row["Age"] == row["Age"] else None,
        "Gender": str(row["Gender"]),
        "assigned_guide": assigned_guide(pid),
        "key_vitals": key_vitals,
        "reasons": reasons,
    })


@app.route("/api/patient/<pid>/timeline")
def api_patient_timeline(pid):
    if pid not in pid_to_idx:
        return jsonify({"error": "Patient not found"}), 404

    idx = pid_to_idx[pid]
    try:
        risk_per_hour, raw_vitals = explainer.get_timeline(X_all[idx], mask_all[idx])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 200

    n = len(risk_per_hour)
    vitals_out = {}
    for feat in ["HR", "SBP", "Temp", "Resp", "Lactate"]:
        if feat in raw_vitals:
            vitals_out[feat] = [round(float(v), 1) for v in raw_vitals[feat]]

    return jsonify({
        "patient_id": pid,
        "patient_name": patient_name(pid),
        "hours": list(range(1, n + 1)),
        "risk_scores": [round(float(p) * 100, 1) for p in risk_per_hour],
        "buckets": [bucket_of(float(p)) for p in risk_per_hour],
        "vitals": vitals_out,
    })


@app.route("/api/notify/<pid>", methods=["POST"])
def api_notify(pid):
    guide = assigned_guide(pid)
    return jsonify({"message": f"{guide} has been notified about patient {pid}."})


if __name__ == "__main__":
    print(f"\nDashboard ready for {len(summary_df):,} patients.")
    print("Open http://localhost:8000 in your browser.\n")
    app.run(debug=True, port=8000)
