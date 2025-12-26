import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
# Ensure these files are in the same folder as App.py
try:
    model = joblib.load('student_gpa_model.pkl')
    scaler = joblib.load('gpa_scaler.pkl')
except:
    st.error("Error: Model or Scaler files not found. Please run the training script first.")

st.set_page_config(page_title="Student GPA Predictor", page_icon="🎓")

st.title("🎓 Student GPA Predictor")
st.write("""
This app predicts a student's **GPA** based on their demographics, habits, and parental involvement.
Fill in the details below and click **Predict**.
""")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Education")
    age = st.number_input("Age", min_value=15, max_value=18, value=16)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], 
                             format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x])
    parental_edu = st.selectbox("Parental Education", options=[0, 1, 2, 3, 4],
                                format_func=lambda x: ["None", "High School", "Some College", "Bachelor's", "Higher"][x])
    study_time = st.slider("Weekly Study Time (hours)", 0, 20, 10)
    absences = st.slider("Absences (days missed)", 0, 30, 5)

with col2:
    st.subheader("Activities & Support")
    tutoring = st.radio("Tutoring", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    parental_support = st.selectbox("Parental Support", options=[0, 1, 2, 3, 4],
                                    format_func=lambda x: ["None", "Low", "Moderate", "High", "Very High"][x])
    extracurriculars = st.radio("Extracurriculars", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    sports = st.radio("Sports", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    music = st.radio("Music", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    volunteering = st.radio("Volunteering", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Prepare input for prediction
# Order must match exactly: Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, 
# Tutoring, ParentalSupport, Extracurriculars, Sports, Music, Volunteering
input_features = np.array([[age, gender, ethnicity, parental_edu, study_time, absences, 
                            tutoring, parental_support, extracurriculars, sports, music, volunteering]])

if st.button("Predict Student GPA"):
    # 1. Scale inputs using the loaded scaler
    input_scaled = scaler.transform(input_features)
    
    # 2. Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # 3. Show Result
    st.markdown("---")
    st.success(f"### Predicted GPA: {prediction:.2f}")
    
    # Visualization
    st.progress(min(max(prediction/4.0, 0.0), 1.0))
    if prediction >= 3.5:
        st.balloons()
        st.info("Excellent! This student is projected to be in the top tier.")
    elif prediction < 2.0:
        st.warning("Attention: This student may need additional academic support.")

