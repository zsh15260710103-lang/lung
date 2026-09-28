import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ===================== 1. Load deployment package =====================
@st.cache_resource
def load_assets():
    with open("deploy_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    try:
        with open("deploy_calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
    except FileNotFoundError:
        calibrator = None
    with open("deploy_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, calibrator, config

model, calibrator, config = load_assets()
threshold = config.get("threshold", 0.5)
use_calibrated = config.get("use_calibrated", False) and (calibrator is not None)

# ===================== 2. Variable definitions (label -> code) =====================
# *** Order must match the training order ***
FEATURES = ['TNM', 'CEA', 'KI67', 'Differentiation', 'Size']

TNM_OPTIONS = {
    "Stage Ia":  0,
    "Stage Ib":  1,
    "Stage IIa": 2,
    "Stage IIb": 3,
    "Stage IIIa": 4,
}
KI67_OPTIONS = {
    "KI-67 < 20%": 0,
    "KI-67 >= 20% and < 40%": 1,
    "KI-67 >= 40%": 2,
}
DIFF_OPTIONS = {
    "Well differentiated":       0,
    "Moderately differentiated": 1,
    "Poorly differentiated":     2,
}
SIZE_OPTIONS = {
    "Tumor size < 2 cm": 0,
    "2 cm <= tumor size < 4 cm": 1,
    "Tumor size >= 4 cm": 2,
}

# ===================== 3. Page layout =====================
st.set_page_config(
    page_title="Recurrence Risk Prediction in Lung Adenocarcinoma",
    page_icon="🫁",
    layout="centered",
)
st.title("Postoperative Recurrence Risk Prediction in Lung Adenocarcinoma")
st.caption(
    "Enter the patient's clinicopathological information to estimate the "
    "individualized risk of postoperative recurrence in lung adenocarcinoma."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tumor characteristics")
    tnm_label = st.selectbox("TNM stage", list(TNM_OPTIONS.keys()), index=0)
    size_label = st.selectbox("Tumor size", list(SIZE_OPTIONS.keys()), index=0)

with col2:
    st.subheader("Pathology and biomarkers")
    ki67_label = st.selectbox("KI-67 proliferation index", list(KI67_OPTIONS.keys()), index=0)
    diff_label = st.selectbox("Degree of differentiation", list(DIFF_OPTIONS.keys()), index=0)

st.divider()
cea_value = st.number_input(
    "CEA (ng/mL)",
    min_value=0.0,
    max_value=1000.0,          # ★ 加上限，防误输
    value=2.0,
    step=0.1,
    help="Carcinoembryonic antigen; reference value is generally < 5 ng/mL.",
)

# ===================== 4. Prediction =====================
if st.button("Predict recurrence risk", use_container_width=True):
    X_new = pd.DataFrame([{
        "TNM": TNM_OPTIONS[tnm_label],
        "CEA": cea_value,
        "KI67": KI67_OPTIONS[ki67_label],
        "Differentiation": DIFF_OPTIONS[diff_label],
        "Size": SIZE_OPTIONS[size_label],
    }], columns=FEATURES)

    prob_raw = model.predict_proba(X_new)[:, 1][0]
    prob = calibrator.predict_proba(X_new)[:, 1][0] if use_calibrated else prob_raw

    is_high = prob >= threshold

    st.divider()
    st.subheader("Prediction result")

    c1, c2 = st.columns(2)
    with c1:
        # ★ 校准后概率即真实风险，可直接称为“复发风险”
        st.metric("Predicted recurrence risk", f"{prob:.1%}")
    with c2:
        st.metric("Risk category", "High risk" if is_high else "Low risk")

    st.progress(float(prob))

    if is_high:
        st.error(
            f"High risk of postoperative recurrence (predicted risk {prob:.1%}). "
            f"Intensified surveillance and intervention may be warranted."
        )
    else:
        st.success(
            f"Low risk of postoperative recurrence (predicted risk {prob:.1%}). "
            f"Routine follow-up is suggested."
        )

    with st.expander("View inputs"):
        st.write({
            "TNM stage": tnm_label,
            "Tumor size": size_label,
            "KI-67": ki67_label,
            "Differentiation": diff_label,
            "CEA (ng/mL)": cea_value,
        })

    st.caption(
        f"Decision threshold = {threshold}; predicted probabilities were "
        f"{'calibrated (Platt scaling)' if use_calibrated else 'not calibrated'}. "
        f"This tool was developed for patients with lung adenocarcinoma undergoing "
        f"curative-intent surgery, and is intended for clinical decision support only; "
        f"it does not constitute a diagnostic conclusion."
    )



