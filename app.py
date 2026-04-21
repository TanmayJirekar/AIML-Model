import streamlit as st
import numpy as np
import joblib
import pandas as pd
from datetime import datetime

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")

st.set_page_config(page_title="Salary Predictor", layout="centered")

# -------------------------
# SESSION STORAGE (HISTORY)
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# TABS
# -------------------------
tab1, tab2 = st.tabs(["🔮 Predict Salary", "📊 History"])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:

    st.title("💰 Salary Prediction System")

    experience = st.number_input("Experience", 0, 50)
    education = st.number_input("Education Level (1-4)", 1, 4)
    age = st.number_input("Age", 18, 100)
    certifications = st.number_input("Certifications", 0, 20)
    projects = st.number_input("Projects", 0, 50)

    if st.button("Predict Salary"):

        input_data = np.array([[
            experience,
            education,
            age,
            certifications,
            projects
        ]])

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Salary: ₹ {round(prediction, 2)}")

        # -------------------------
        # SAVE TO HISTORY
        # -------------------------
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experience": experience,
            "education": education,
            "age": age,
            "certifications": certifications,
            "projects": projects,
            "predicted_salary": round(prediction, 2)
        })

        # -------------------------
        # VISUALIZATION
        # -------------------------
        st.bar_chart({
            "experience": experience,
            "education": education,
            "age": age,
            "certifications": certifications,
            "projects": projects
        })

# =========================================================
# TAB 2 - HISTORY
# =========================================================
with tab2:

    st.title("📊 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        # Optional: download CSV
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

        # Simple trend graph
        st.line_chart(df["predicted_salary"])
