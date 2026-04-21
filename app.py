import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Salary Predictor", layout="centered")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "predicted" not in st.session_state:
    st.session_state.predicted = False

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Salary",
    "📊 History",
    "📈 Analytics Dashboard"
])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:

    st.title("💰 Salary Prediction System")

    experience = st.number_input("Experience", 0, 50, value=1)
    education = st.number_input("Education Level (1-4)", 1, 4, value=1)
    age = st.number_input("Age", 18, 100, value=25)
    certifications = st.number_input("Certifications", 0, 20, value=0)
    projects = st.number_input("Projects", 0, 50, value=1)

    if st.button("Predict Salary"):

        # -------------------------
        # MODEL INPUT (ORDER FIXED)
        # -------------------------
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
        # SAVE HISTORY
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

        st.session_state.predicted = True

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

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

        st.line_chart(df["predicted_salary"])

# =========================================================
# TAB 3 - ANALYTICS DASHBOARD
# =========================================================
with tab3:

    st.title("📈 Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("⚠ Make at least one prediction to view analytics.")
    else:

        df = pd.DataFrame(st.session_state.history)

        # -------------------------
        # KPI METRICS
        # -------------------------
        st.subheader("📌 Key Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", len(df))
        col2.metric("Avg Salary", f"₹ {df['predicted_salary'].mean():,.0f}")
        col3.metric("Max Salary", f"₹ {df['predicted_salary'].max():,.0f}")

        st.divider()

        # -------------------------
        # SALARY TREND
        # -------------------------
        st.subheader("📊 Salary Trend")
        st.line_chart(df["predicted_salary"])

        st.divider()

        # -------------------------
        # DISTRIBUTION
        # -------------------------
        st.subheader("📉 Salary Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["predicted_salary"], bins=10)
        ax.set_title("Salary Distribution")
        ax.set_xlabel("Salary")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        st.divider()

        # -------------------------
        # FEATURE INSIGHTS
        # -------------------------
        st.subheader("🧠 Feature Insights")

        feature_cols = ["experience", "education", "age", "certifications", "projects"]

        avg_features = df[feature_cols].mean()

        st.bar_chart(avg_features)

        st.caption("Average values used in predictions")
