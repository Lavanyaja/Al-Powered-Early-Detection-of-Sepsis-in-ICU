"""
explainability.py
Generates human-readable + structured "reasons" behind a patient's
risk score using Gradient x Input attribution.

Reasons now ONLY include vitals/labs genuinely outside their
clinical normal range - ranked by how strongly each abnormal
vital is driving the risk score.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import json
from lstm_model import elementwise_bce

SEQ_DIR = "processed_data/sequences"
MODEL_DIR = "models"

FEATURE_INFO = {
    "HR": ("Heart Rate", "bpm", 60, 100),
    "O2Sat": ("Oxygen Saturation", "%", 95, 100),
    "Temp": ("Temperature", "°C", 36.1, 37.8),
    "SBP": ("Systolic BP", "mmHg", 90, 130),
    "MAP": ("Mean Arterial Pressure", "mmHg", 65, 100),
    "DBP": ("Diastolic BP", "mmHg", 60, 90),
    "Resp": ("Respiratory Rate", "breaths/min", 12, 20),
    "EtCO2": ("End-Tidal CO2", "mmHg", 35, 45),
    "BaseExcess": ("Base Excess", "mmol/L", -2, 2),
    "HCO3": ("Bicarbonate (HCO3)", "mmol/L", 22, 28),
    "FiO2": ("Fraction of Inspired O2", "", 0.21, 0.4),
    "pH": ("Blood pH", "", 7.35, 7.45),
    "PaCO2": ("Arterial CO2 (PaCO2)", "mmHg", 35, 45),
    "SaO2": ("Arterial O2 Saturation", "%", 95, 100),
    "AST": ("AST (liver enzyme)", "U/L", 8, 40),
    "BUN": ("Blood Urea Nitrogen", "mg/dL", 7, 20),
    "Alkalinephos": ("Alkaline Phosphatase", "U/L", 44, 147),
    "Calcium": ("Calcium", "mg/dL", 8.5, 10.5),
    "Chloride": ("Chloride", "mmol/L", 96, 106),
    "Creatinine": ("Creatinine", "mg/dL", 0.6, 1.3),
    "Bilirubin_direct": ("Direct Bilirubin", "mg/dL", 0, 0.3),
    "Glucose": ("Glucose", "mg/dL", 70, 140),
    "Lactate": ("Lactate", "mmol/L", 0.5, 2.0),
    "Magnesium": ("Magnesium", "mg/dL", 1.7, 2.2),
    "Phosphate": ("Phosphate", "mg/dL", 2.5, 4.5),
    "Potassium": ("Potassium", "mmol/L", 3.5, 5.0),
    "Bilirubin_total": ("Total Bilirubin", "mg/dL", 0.1, 1.2),
    "TroponinI": ("Troponin I", "ng/mL", 0, 0.04),
    "Hct": ("Hematocrit", "%", 36, 50),
    "Hgb": ("Hemoglobin", "g/dL", 12, 17.5),
    "PTT": ("Partial Thromboplastin Time", "sec", 25, 35),
    "WBC": ("White Blood Cell Count", "x10^9/L", 4, 11),
    "Fibrinogen": ("Fibrinogen", "mg/dL", 200, 400),
    "Platelets": ("Platelet Count", "x10^9/L", 150, 450),
    "Shock_Index": ("Shock Index (HR/SBP)", "", 0.5, 0.7),
    "BUN_Creatinine_ratio": ("BUN/Creatinine Ratio", "", 10, 20),
}


def _is_reason_candidate(feature_name):
    return feature_name in FEATURE_INFO


def _is_abnormal(feature_name, raw_value):
    if feature_name not in FEATURE_INFO:
        return False
    _, _, lo, hi = FEATURE_INFO[feature_name]
    if lo is None:
        return False
    return raw_value < lo or raw_value > hi


def _severity_tier(feature_name, raw_value):
    label, unit, lo, hi = FEATURE_INFO[feature_name]
    range_str = f"{lo}-{hi}{(' ' + unit) if unit else ''}"
    span = max(hi - lo, 1e-6)

    if raw_value > hi:
        pct_over = (raw_value - hi) / span
        tier = "severe" if pct_over > 0.5 else ("moderate" if pct_over > 0.15 else "mild")
        return tier, "Above Normal", range_str
    else:
        pct_under = (lo - raw_value) / span
        tier = "severe" if pct_under > 0.5 else ("moderate" if pct_under > 0.15 else "mild")
        return tier, "Below Normal", range_str


def _friendly_reason(feature_name, raw_value, direction, tier, status):
    label, unit, lo, hi = FEATURE_INFO[feature_name]
    unit_str = f" {unit}" if unit else ""
    risk_word = "increasing" if direction > 0 else "still elevated, though currently"
    qualifier = "elevated" if status == "Above Normal" else "low"
    severity_word = {"severe": "markedly", "moderate": "notably", "mild": "slightly"}[tier]
    return (f"{label} is {severity_word} {qualifier} at {raw_value:.1f}{unit_str} "
            f"(normal: {lo}-{hi}{unit_str}) — {risk_word} risk")


class SepsisExplainer:
    def __init__(self):
        self.model = keras.models.load_model(
            f"{MODEL_DIR}/best_model.keras", compile=False,
            custom_objects={"elementwise_bce": elementwise_bce},
        )
        with open(f"{SEQ_DIR}/feature_cols.json") as f:
            self.feature_cols = json.load(f)
        self.scaler = joblib.load(f"{SEQ_DIR}/scaler.pkl")

        last_step_output = self.model.output[:, -1]
        self.submodel = keras.Model(inputs=self.model.input, outputs=last_step_output)

    def predict_risk(self, patient_seq):
        x = np.expand_dims(patient_seq, axis=0).astype(np.float32)
        return float(self.submodel(x, training=False).numpy()[0])

    def predict_trajectory(self, patient_seq):
        x = np.expand_dims(patient_seq, axis=0).astype(np.float32)
        return self.model(x, training=False).numpy()[0]

    def get_timeline(self, patient_seq, patient_mask):
        traj = self.predict_trajectory(patient_seq)
        real_idx = np.where(patient_mask == 1)[0]
        risk_per_hour = traj[real_idx]
        raw_all_hours = self.scaler.inverse_transform(patient_seq[real_idx])
        raw_vitals = {f: raw_all_hours[:, i] for i, f in enumerate(self.feature_cols)}
        return risk_per_hour, raw_vitals

    def _get_candidate_pool(self, patient_seq, patient_mask, recent_hours=6, pool_size=25):
        x = tf.convert_to_tensor(
            np.expand_dims(patient_seq, axis=0).astype(np.float32)
        )
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = self.submodel(x, training=False)
        grads = tape.gradient(output, x).numpy()[0]

        attribution = grads * patient_seq

        real_idx = np.where(patient_mask == 1)[0]
        recent_idx = real_idx[-recent_hours:] if len(real_idx) >= recent_hours else real_idx

        signed_attr = attribution[recent_idx].sum(axis=0)
        abs_attr = np.abs(signed_attr)

        candidate_indices = np.array([
            i for i, f in enumerate(self.feature_cols) if _is_reason_candidate(f)
        ])
        candidate_abs = abs_attr[candidate_indices]
        order = np.argsort(candidate_abs)[::-1][:pool_size]
        pool_idx = candidate_indices[order]

        last_real_hour = real_idx[-1]
        raw_last_hour = self.scaler.inverse_transform(
            patient_seq[last_real_hour].reshape(1, -1)
        )[0]
        raw_last_hour_dict = dict(zip(self.feature_cols, raw_last_hour))

        pool = [(self.feature_cols[i], raw_last_hour[i], signed_attr[i]) for i in pool_idx]
        risk_score = self.predict_risk(patient_seq)
        return risk_score, pool, raw_last_hour_dict

    def _abnormal_only(self, pool, top_k):
        abnormal = [(f, v, d) for f, v, d in pool if _is_abnormal(f, v)]
        return abnormal[:top_k]

    def explain(self, patient_seq, patient_mask, top_k=5, recent_hours=6):
        risk_score, pool, _ = self._get_candidate_pool(patient_seq, patient_mask, recent_hours)
        abnormal = self._abnormal_only(pool, top_k)
        reasons = []
        for fname, raw_val, direction in abnormal:
            tier, status, _ = _severity_tier(fname, raw_val)
            reasons.append(_friendly_reason(fname, raw_val, direction, tier, status))
        return risk_score, reasons

    def explain_structured(self, patient_seq, patient_mask, top_k=5, recent_hours=6):
        risk_score, pool, raw_last_hour_dict = self._get_candidate_pool(
            patient_seq, patient_mask, recent_hours
        )
        abnormal = self._abnormal_only(pool, top_k)

        reasons = []
        for fname, raw_val, direction in abnormal:
            label, unit, lo, hi = FEATURE_INFO[fname]
            tier, status, normal_range = _severity_tier(fname, raw_val)
            reasons.append({
                "feature": label,
                "value": round(float(raw_val), 2),
                "unit": unit,
                "note": None,
                "tier": tier,
                "tier_css": tier,
                "status": status,
                "normal_range": normal_range,
            })
        return risk_score, reasons, raw_last_hour_dict
