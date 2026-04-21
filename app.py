import streamlit as st
import numpy as np
import joblib

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("💰 Salary Prediction App")

# -------------------------
# INPUT FIELDS
# -------------------------
experience = st.number_input("Experience", 0, 50)
education = st.number_input("Education Level (1-4)", 1, 4)
age = st.number_input("Age", 18, 100)
certifications = st.number_input("Certifications", 0, 20)
projects = st.number_input("Projects", 0, 50)

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Salary"):

    input_data = np.array([[
        experience,
        education,
        age,
        certifications,
        projects
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ₹ {round(prediction, 2)}")

    # Feature display
    st.bar_chart({
        "experience": experience,
        "education": education,
        "age": age,
        "certifications": certifications,
        "projects": projects
    })
