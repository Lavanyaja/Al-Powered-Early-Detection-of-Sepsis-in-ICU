"""
Step 11: Sepsis Risk Dashboard (Streamlit)
Run with:  streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tensorflow import keras
from lstm_model import elementwise_bce
from explainability import SepsisExplainer

SEQ_DIR = "processed_data/sequences"
MODEL_DIR = "models"

st.set_page_config(page_title="Sepsis Risk Dashboard", layout="wide")

BUCKET_COLORS = {"HIGH": "#C44E52", "MEDIUM": "#DD8452", "LOW": "#55A868"}


@st.cache_resource
def load_model_and_thresholds():
    model = keras.models.load_model(
        f"{MODEL_DIR}/best_model.keras", compile=False,
        custom_objects={"elementwise_bce": elementwise_bce},
    )
    with open(f"{MODEL_DIR}/risk_thresholds.json") as f:
        thresholds = json.load(f)
    return model, thresholds


@st.cache_resource
def load_explainer():
    return SepsisExplainer()


@st.cache_data
def load_test_sequences():
    X_test = np.load(f"{SEQ_DIR}/X_test.npy")
    y_test = np.load(f"{SEQ_DIR}/y_test.npy")
    mask_test = np.load(f"{SEQ_DIR}/mask_test.npy")
    pid_test = np.load(f"{SEQ_DIR}/pid_test.npy", allow_pickle=True)
    return X_test, y_test, mask_test, pid_test


@st.cache_data
def load_raw_features():
    df = pd.read_parquet("processed_data/features.parquet")
    return df


@st.cache_data
def compute_all_predictions(_model, X_test):
    return _model.predict(X_test, batch_size=64, verbose=0)


def bucket_of(prob, thresholds):
    if prob >= thresholds["high"]:
        return "HIGH"
    elif prob >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


model, thresholds = load_model_and_thresholds()
explainer = load_explainer()
X_test, y_test, mask_test, pid_test = load_test_sequences()
raw_df = load_raw_features()
all_probs = compute_all_predictions(model, X_test)

records = []
for i in range(len(pid_test)):
    real_idx = np.where(mask_test[i] == 1)[0]
    current_prob = float(all_probs[i][real_idx[-1]])
    max_prob = float(all_probs[i][real_idx].max())
    actual = "Sepsis" if y_test[i][real_idx].max() == 1 else "No Sepsis"
    records.append({
        "PatientID": pid_test[i],
        "Current Risk": current_prob,
        "Peak Risk (during stay)": max_prob,
        "Bucket": bucket_of(current_prob, thresholds),
        "Actual Outcome": actual,
        "ICU Hours": len(real_idx),
    })
summary_df = pd.DataFrame(records).sort_values("Current Risk", ascending=False).reset_index(drop=True)

st.title("🏥 Sepsis Early Detection Dashboard")
st.caption("LSTM-based hourly sepsis risk prediction with explainable AI reasons — "
           f"trained on {len(pid_test):,} held-out test patients shown here "
           "(model trained on 40,336 patients total).")

tab1, tab2 = st.tabs(["📊 Population Overview", "🔍 Patient Detail"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(summary_df))
    col2.metric("High Risk", int((summary_df["Bucket"] == "HIGH").sum()))
    col3.metric("Medium Risk", int((summary_df["Bucket"] == "MEDIUM").sum()))
    col4.metric("Low Risk", int((summary_df["Bucket"] == "LOW").sum()))

    st.markdown("---")

    bucket_filter = st.multiselect(
        "Filter by risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
    )
    search_id = st.text_input("Search Patient ID (e.g. p000123)", "")

    filtered = summary_df[summary_df["Bucket"].isin(bucket_filter)]
    if search_id:
        filtered = filtered[filtered["PatientID"].str.contains(search_id, case=False)]

    def color_bucket(val):
        return f"background-color: {BUCKET_COLORS.get(val, '')}; color: white; font-weight: bold;"

    st.dataframe(
        filtered.style.applymap(color_bucket, subset=["Bucket"])
                .format({"Current Risk": "{:.3f}", "Peak Risk (during stay)": "{:.3f}"}),
        use_container_width=True,
        height=500,
    )

with tab2:
    default_patient = summary_df.iloc[0]["PatientID"]
    selected_pid = st.selectbox(
        "Select a patient", summary_df["PatientID"].tolist(),
        index=summary_df["PatientID"].tolist().index(default_patient),
    )

    row_idx = np.where(pid_test == selected_pid)[0][0]
    real_idx = np.where(mask_test[row_idx] == 1)[0]
    patient_probs = all_probs[row_idx][real_idx]
    patient_true = y_test[row_idx][real_idx]

    current_risk = float(patient_probs[-1])
    current_bucket = bucket_of(current_risk, thresholds)

    st.subheader(f"Patient {selected_pid}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Risk Score", f"{current_risk:.3f}")
    c2.markdown(
        f"<div style='background-color:{BUCKET_COLORS[current_bucket]}; "
        f"padding:10px; border-radius:8px; text-align:center; color:white; "
        f"font-weight:bold; font-size:20px;'>{current_bucket} RISK</div>",
        unsafe_allow_html=True,
    )
    c3.metric("ICU Hours Recorded", len(real_idx))

    st.markdown("### Risk Score Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=patient_probs, mode="lines+markers", name="Risk Score",
                              line=dict(color="#4C72B0", width=2)))
    fig.add_hline(y=thresholds["high"], line_dash="dash", line_color="#C44E52",
                  annotation_text="High threshold")
    fig.add_hline(y=thresholds["medium"], line_dash="dash", line_color="#DD8452",
                  annotation_text="Medium threshold")
    sepsis_hours = np.where(patient_true == 1)[0]
    if len(sepsis_hours) > 0:
        fig.add_trace(go.Scatter(x=sepsis_hours, y=patient_probs[sepsis_hours],
                                  mode="markers", name="Actual Sepsis Hour",
                                  marker=dict(color="black", size=10, symbol="x")))
    fig.update_layout(xaxis_title="ICU Hour", yaxis_title="Predicted Risk",
                       yaxis_range=[0, 1], height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Why this risk score? (Top contributing factors)")
    with st.spinner("Computing explanation..."):
        _, reasons = explainer.explain(X_test[row_idx], mask_test[row_idx], top_k=5)
    for r in reasons:
        icon = "🔴" if "increasing" in r else "🟢"
        st.markdown(f"{icon} {r}")

    st.markdown("### Key Vitals Over Stay")
    patient_raw = raw_df[raw_df["PatientID"] == selected_pid].sort_values("ICULOS")
    vitals_to_show = ["HR", "O2Sat", "Temp", "MAP", "Resp", "WBC"]
    fig2 = go.Figure()
    for v in vitals_to_show:
        if v in patient_raw.columns:
            fig2.add_trace(go.Scatter(x=patient_raw["ICULOS"], y=patient_raw[v],
                                       mode="lines", name=v))
    fig2.update_layout(xaxis_title="ICU Hour", yaxis_title="Value", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Actual outcome for this patient: "
               f"{'Sepsis diagnosed during stay' if patient_true.max() == 1 else 'No sepsis'}")
