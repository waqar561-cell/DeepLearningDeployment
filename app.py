import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the saved model and scaler
model = load_model('heart_disease_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Heart Disease Prediction App")
st.write("Enter the patient's data below to predict the likelihood of heart disease.")

# Create input fields for the 13 features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of peak exercise ST segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0=null, 1=fixed, 2=normal, 3=reversible)", [0, 1, 2, 3])

# Prediction Logic
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0
    
    # Display results
    if prediction == 1:
        st.error(f"Prediction: Heart Disease Detected (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"Prediction: No Heart Disease Detected (Probability: {1-prediction_prob:.2f})")
