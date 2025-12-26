import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
try:
    model = joblib.load('best_gpa_model.pkl')
    scaler = joblib.load('gpa_scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found. Please ensure 'best_gpa_model.pkl' and 'gpa_scaler.pkl' are in the same folder as this script.")
    st.stop()

st.title("🎓 Student GPA Predictor")
st.write("Enter the student details below to predict their GPA.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=20, value=17)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x==0 else "Female")
    ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], help="0: Caucasian, 1: African American, 2: Asian, 3: Other")
    parent_edu = st.selectbox("Parental Education", options=[0, 1, 2, 3, 4], help="0: None, 1: High School, 2: Some College, 3: Bachelor's, 4: Higher")
    study_time = st.slider("Weekly Study Time (hours)", 0, 20, 10)
    absences = st.number_input("Absences", min_value=0, max_value=30, value=5)

with col2:
    tutoring = st.radio("Tutoring", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    parent_support = st.selectbox("Parental Support", options=[0, 1, 2, 3, 4], help="0: None to 4: Very High")
    extracurricular = st.radio("Extracurricular Activities", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    sports = st.radio("Plays Sports", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    music = st.radio("Music Activities", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    volunteering = st.radio("Volunteering", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

# Predict button
if st.button("Predict GPA"):
    # Prepare input data
    input_data = pd.DataFrame([[
        age, gender, ethnicity, parent_edu, study_time, 
        absences, tutoring, parent_support, extracurricular, 
        sports, music, volunteering
    ]], columns=['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 
                 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
                 'Sports', 'Music', 'Volunteering'])
    
    # Scale and predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    st.success(f"### Predicted GPA: {prediction[0]:.2f}")
    
    # Visual indicator
    if prediction[0] >= 3.0:
        st.balloons()
