import streamlit as st
import joblib
import numpy as np
import time, os, csv

# --- Utility functions ---
def now_ms():
    return time.perf_counter() * 1000

def log_latency(times):
    os.makedirs("logs", exist_ok=True)
    path = "logs/latency_log.csv"
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=times.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(times)

# --- Load trained model ---
model = joblib.load("best_stack_model.pkl")

# --- Prediction function with timing ---
def predict(input_data):
    t0 = now_ms()
    X = np.array([input_data])
    t1 = now_ms()
    prediction = model.predict(X)
    t2 = now_ms()
    probability = model.predict_proba(X)
    t3 = now_ms()

    times = {
        "preprocess_ms": round(t1 - t0, 2),
        "inference_ms": round(t2 - t1, 2),
        "probability_ms": round(t3 - t2, 2),
        "total_ms": round(t3 - t0, 2)
    }
    return prediction[0], probability[0][1], times

# --- Streamlit UI ---
st.set_page_config(page_title="RenalAI: CKD Predictor", layout="centered")

# --- Custom CSS for centering ---
st.markdown(
    """
    <style>
    .block-container {
        text-align: center;
    }
    h1, h2, h3, h4, h5, h6, p {
        text-align: center;
    }
    .stButton>button {
        display: block;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("RenalAI: Chronic Kidney Disease Risk Prediction")

st.markdown("""
RenalAI is an **AI-powered clinical decision support tool** that predicts the risk of  
**Chronic Kidney Disease (CKD)** using patient baseline health parameters.
""")

col1, col2 = st.columns(2)

with col1:
    eGFR = st.number_input("eGFR (Baseline)", min_value=5.0, max_value=150.0, step=0.1, value=90.0, format="%.1f")
    creatinine = st.number_input("Creatinine (Baseline, mg/dL)", min_value=0.1, max_value=15.0, step=0.1, value=1.2, format="%.1f")
    diabetes = st.selectbox("History of Diabetes", ("Yes", "No"))
    cholesterol = st.number_input("Cholesterol (Baseline, mg/dL)", min_value=100, max_value=400, step=1, value=200)
    age = st.number_input("Age (years)", min_value=18, max_value=100, step=1, value=50)

with col2:
    dm_meds = st.selectbox("On Diabetes Medications (DMmeds)", ("Yes", "No"))
    aceiarb = st.selectbox("On ACEI/ARB Medication", ("Yes", "No"))
    sbp = st.number_input("Systolic BP (Baseline, mmHg)", min_value=80, max_value=200, step=1, value=120)
    dbp = st.number_input("Diastolic BP (Baseline, mmHg)", min_value=50, max_value=120, step=1, value=80)
    bmi = st.number_input("BMI (Baseline, kg/m²)", min_value=10.0, max_value=50.0, step=0.1, value=25.0, format="%.1f")

# --- Encode categorical variables ---
diabetes_val = 1 if diabetes == "Yes" else 0
dm_meds_val = 1 if dm_meds == "Yes" else 0
aceiarb_val = 1 if aceiarb == "Yes" else 0

# --- Collect input data in correct order ---
input_data = [
    eGFR,
    creatinine,
    diabetes_val,
    cholesterol,
    age,
    dm_meds_val,
    aceiarb_val,
    sbp,
    dbp,
    bmi
]

# --- Prediction Button ---
if st.button("🔍 Predict"):
    prediction, probability, times = predict(input_data)
    log_latency(times)

    # Show Prediction
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error("⚠️ RenalAI suggests you may have **Chronic Kidney Disease (CKD)**. Please consult a healthcare provider.")
        st.write(f"Model Confidence: **{probability * 100:.2f}%** chance of CKD.")
    else:
        st.success("✅ RenalAI suggests you are **not at risk of CKD** based on the current data.")
        st.write(f"Model Confidence: **{(1 - probability) * 100:.2f}%** chance of being healthy.")

    # Probabilities
    st.subheader("📊 Prediction Probabilities")
    st.write(f"**Not CKD:** {(1 - probability) * 100:.2f}%")
    st.write(f"**CKD:** {probability * 100:.2f}%")