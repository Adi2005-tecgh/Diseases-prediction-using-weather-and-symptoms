# app.py
# Weather & Disease Prediction Streamlit app
# Minimal edits included to allow loading joblib files saved from another sklearn version.
# Keep this file next to a `models/` folder containing:
#   - weather_disease_model.joblib
#   - feature_names.joblib
#   - label_encoder.joblib
# and an optional `outputs/` folder with figures/ and tables/ for downloads.

import os
import importlib
import glob
import zipfile

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Weather & Disease Prediction", layout="wide", page_icon="🧑‍⚕️")

# --------------------------
# Aesthetic CSS - beige theme
# --------------------------
st.markdown("""
<style>
:root{
  --bg-beige: #f5f0e6;
  --card-white: #ffffff;
  --muted: #6b6b6b;
  --accent1: #b998ff;  /* soft purple */
  --accent2: #ffb199;  /* soft peach */
  --radius: 22px;
  --shadow: 0 10px 30px rgba(15,23,42,0.06);
}

/* page background */
.stApp {
  background: var(--bg-beige);
  color: #222;
}

/* Title */
.main-title {
  text-align: center;
  font-size: 34px !important;
  font-weight: 800 !important;
  margin-top: 10px;
  color: #1f2937;
}
.sub {
  text-align: center;
  color: var(--muted);
  margin-bottom: 24px;
}

/* soft card */
.soft-card {
  background: var(--card-white);
  padding: 20px;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid rgba(0,0,0,0.03);
  margin-bottom: 20px;
}

/* pastel blocks for visual echo of reference */
.pastel-1 { background: linear-gradient(135deg, rgba(185,152,255,0.12), rgba(255,177,153,0.07)); padding:14px; border-radius:18px; }
.pastel-2 { background: linear-gradient(135deg, rgba(255,177,153,0.10), rgba(255,226,241,0.06)); padding:14px; border-radius:18px; }

/* section title */
.section-title { font-size:18px; font-weight:700; color:#111827; margin-bottom:8px; }

/* Inputs label weight */
.stNumberInput label, .stMultiSelect label, .stRadio label {
  font-weight:600 !important;
  color:#333 !important;
}

/* Button styling */
.stButton > button {
  background: linear-gradient(135deg, var(--accent1), #8ea2ff);
  color: white;
  border-radius: 12px;
  padding: 10px 22px;
  font-size: 16px;
  border: none;
  box-shadow: 0 12px 30px rgba(137, 96, 255, 0.18);
}
.stButton > button:hover {
  transform: translateY(-2px);
}

/* Chart container */
.chart-box { background: var(--card-white); padding: 12px; border-radius: 14px; box-shadow: var(--shadow); }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.markdown('<div class="main-title">🧑‍⚕️ Diseases prediction using weather and symptoms</div>', unsafe_allow_html=True)



# --------------------------
# Load model + features + encoder (minimal, compatible approach)
# --------------------------
MODEL_PATH = "models/weather_disease_model.joblib"
FEATURE_PATH = "models/feature_names.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

# Temporary compatibility shim:
# Some sklearn versions changed internal helper names. Create a small placeholder
# so joblib.load can unpickle objects saved with a different sklearn internal layout.
try:
    _ct_mod = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compatibility placeholder for older/newer sklearn internal name."""
            pass
        setattr(_ct_mod, "_RemainderColsList", _RemainderColsList)
except Exception:
    # If the module path is different or not present, ignore — matching sklearn is the robust fix.
    pass

# attempt to load; show error and stop if missing
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    st.error(
        "Failed to load model or resources: "
        f"{e}\n\nIf this is an sklearn version mismatch, you should install the same scikit-learn "
        "version used to train the model (e.g. `pip install scikit-learn==X.Y.Z`)."
    )
    st.stop()

# if the loaded object is a Pipeline, pipeline will have named_steps
IS_PIPELINE = hasattr(model, "named_steps")

# --------------------------
# Symptom list (unchanged)
# --------------------------
SYMPTOM_LIST = [
    "nausea", "joint_pain", "abdominal_pain", "high_fever", "chills", "fatigue",
    "runny_nose", "pain_behind_the_eyes", "dizziness", "headache", "chest_pain",
    "vomiting", "cough", "shivering", "asthma_history", "high_cholesterol",
    "diabetes", "obesity", "hiv_aids", "nasal_polyps", "asthma",
    "high_blood_pressure", "severe_headache", "weakness", "trouble_seeing",
    "fever", "body_aches", "sore_throat", "sneezing", "diarrhea",
    "rapid_breathing", "rapid_heart_rate", "swollen_glands", "rashes",
    "sinus_headache", "facial_pain"
]

# --------------------------
# Layout: inputs left, predict right
# --------------------------
left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌦️ Weather & Demographics</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input("Temperature (°C)", value=30.0, format="%.1f", key="temp")
    with c2:
        humidity = st.number_input("Humidity (%)", value=50.0, format="%.1f", key="hum")
    with c3:
        wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, format="%.1f", key="wind")

    c4, c5 = st.columns(2)
    with c4:
        age = st.number_input("Age", min_value=0, max_value=120, value=25, key="age")
    with c5:
        gender = st.radio("Gender", ["Male", "Female"], index=0, key="gender")
        gender_encoded = 0 if gender == "Male" else 1

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤒 Symptoms</div>', unsafe_allow_html=True)
    selected_symptoms = st.multiselect("Select symptoms (start typing):", SYMPTOM_LIST, default=[])
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)

    # Build feature vector exactly as before
    symptom_sum = len(selected_symptoms)

    user_features = {
        "Temperature (C)": temperature,
        "Humidity": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Age": age,
        "Gender": gender_encoded,
        "symptom_sum": symptom_sum,
        # 'temp_x_fever' intentionally left out if not present in feature_names; handled below
    }

    for symptom in SYMPTOM_LIST:
        user_features[symptom] = 1 if symptom in selected_symptoms else 0

    # Ensure all feature_names are present in user_features (missing → fill 0)
    X_input = pd.DataFrame([[user_features.get(f, 0) for f in feature_names]], columns=feature_names)

    # Small dev-time check: warn if feature_names contain unexpected columns
    missing = [c for c in feature_names if c not in X_input.columns]
    if missing:
        st.warning(f"Warning: some expected features are missing from input: {missing[:10]}")

    # Require at least one symptom to predict
    if st.button("🔮 Predict Disease"):
        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom before predicting. The model requires symptom inputs.")
        else:
            try:
                # Use pipeline if available (it includes preprocessing)
                if IS_PIPELINE:
                    pred_idx = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
                else:
                    # model is a plain estimator — assume X_input is already in the expected form
                    pred_idx = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None

                # Inverse transform predicted index → disease name
                prediction = label_encoder.inverse_transform([pred_idx])[0]
                classes = label_encoder.inverse_transform(np.arange(len(proba))) if proba is not None else []
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.success(f"Predicted Disease: **{prediction}**")

            # Top-5 probability chart (matplotlib)
            if proba is not None:
                top_idx = np.argsort(proba)[::-1][:5]
                top_diseases = [classes[i] for i in top_idx]
                top_probs = [proba[i] for i in top_idx]

                st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(top_diseases[::-1], np.array(top_probs[::-1]))
                ax.set_xlabel("Probability")
                ax.set_xlim(0, 1)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
                ax.set_title("Top 5 Most Likely Diseases")
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Model does not provide probabilities (no predict_proba). Showing predicted label only.")

    st.markdown('</div>', unsafe_allow_html=True)

# small footer spacing
st.write("")
