import streamlit as st
import numpy as np
import joblib
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# LOAD MODEL & SCALER (SAFE)
# ==========================================
MODEL_PATH = "student_gpa_model.pkl"
SCALER_PATH = "gpa_scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or Scaler files not found. Please run the training script first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==========================================
# APP TITLE & DESCRIPTION
# ==========================================
st.title("🎓 Student GPA Predictor")
st.write("""
This app predicts a student's **GPA** based on demographics, academic habits,
and parental involvement.  
Fill in the details below and click **Predict Student GPA**.
""")

# ==========================================
# INPUT FORM
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Education")
    age = st.number_input("Age", min_value=15, max_value=18, value=16)

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Male" if x == 0 else "Female"
    )

    ethnicity = st.selectbox(
        "Ethnicity",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x]
    )

    parental_edu = st.selectbox(
        "Parental Education",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: [
            "None",
            "High School",
            "Some College",
            "Bachelor's",
            "Higher"
        ][x]
    )

    study_time = st.slider("Weekly Study Time (hours)", 0, 20, 10)
    absences = st.slider("Absences (days missed)", 0, 30, 5)

with col2:
    st.subheader("Activities & Support")

    tutoring = st.radio(
        "Tutoring",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    parental_support = st.selectbox(
        "Parental Support",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: [
            "None",
            "Low",
            "Moderate",
            "High",
            "Very High"
        ][x]
    )

    extracurriculars = st.radio(
        "Extracurricular Activities",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    sports = st.radio(
        "Sports",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    music = st.radio(
        "Music",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    volunteering = st.radio(
        "Volunteering",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

# ==========================================
# FEATURE VECTOR (MUST MATCH TRAINING ORDER)
# ==========================================
input_features = np.array([[
    age,
    gender,
    ethnicity,
    parental_edu,
    study_time,
    absences,
    tutoring,
    parental_support,
    extracurriculars,
    sports,
    music,
    volunteering
]])

# ==========================================
# PREDICTION
# ==========================================
if st.button("🎯 Predict Student GPA"):
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.success(f"### 📊 Predicted GPA: **{prediction:.2f}**")

    # GPA Progress Bar (0–4 scale)
    st.progress(min(max(prediction / 4.0, 0.0), 1.0))

    if prediction >= 3.5:
        st.balloons()
        st.info("🌟 Excellent! This student is projected to be in the top academic tier.")
    elif prediction < 2.0:
        st.warning("⚠️ This student may benefit from additional academic support.")
    else:
        st.info("📘 This student is performing at an average to good academic level.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("📚 Machine Learning GPA Predictor | Random Forest Regression")
