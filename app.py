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

    # -------------------------
    # ORIGINAL INPUTS
    # -------------------------
    experience = st.number_input("Experience", 0, 50, value=1)
    education = st.number_input("Education Level (1-4)", 1, 4, value=1)
    age = st.number_input("Age", 18, 100, value=25)
    certifications = st.number_input("Certifications", 0, 20, value=0)
    projects = st.number_input("Projects", 0, 50, value=1)

    # =====================================================
    # 🔥 WHAT-IF ANALYSIS (NEW FEATURE)
    # =====================================================
    st.subheader("🧠 What-If Analysis (Interactive AI)")

    w_exp = st.slider("Change Experience", 0, 50, experience)
    w_edu = st.slider("Change Education", 1, 4, education)
    w_age = st.slider("Change Age", 18, 100, age)
    w_cert = st.slider("Change Certifications", 0, 20, certifications)
    w_proj = st.slider("Change Projects", 0, 50, projects)

    what_if_input = np.array([[
        w_exp,
        w_edu,
        w_age,
        w_cert,
        w_proj
    ]])

    what_if_salary = model.predict(what_if_input)[0]

    st.info(f"📊 What-If Predicted Salary: ₹ {round(what_if_salary, 2)}")

    # =====================================================
    # 🔥 SKILL IMPACT ANALYZER (NEW FEATURE)
    # =====================================================
    st.subheader("📊 Skill Impact Analyzer")

    base_input = np.array([[experience, education, age, certifications, projects]])
    base_salary = model.predict(base_input)[0]

    impact = {}

    feature_names = ["Experience", "Education", "Age", "Certifications", "Projects"]

    for i, name in enumerate(feature_names):

        temp = base_input.copy()
        temp[0][i] += 1

        new_salary = model.predict(temp)[0]
        impact[name] = new_salary - base_salary

    st.bar_chart(impact)

    # =====================================================
    # MAIN PREDICTION BUTTON
    # =====================================================
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
        # TREND
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

        st.bar_chart(df[feature_cols].mean())

        st.caption("Average values used in predictions")
