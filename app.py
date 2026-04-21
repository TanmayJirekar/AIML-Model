import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Salary Predictor Pro", page_icon="💰", layout="wide")

# -------------------------
# LOAD MODEL & FEATURES
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, feature_names = load_model()

# -------------------------
# SESSION STATE INIT
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    [data-theme="dark"] .stMetric { background-color: #262730; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("💰 AI Salary Prediction Platform")
st.markdown("Predict, Analyze, and Simulate salaries using advanced Machine Learning.")

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Single Prediction",
    "🧠 What-If Analysis",
    "📂 Batch Predictions",
    "📊 Analytics Dashboard"
])

# =========================================================
# TAB 1 - SINGLE PREDICTION
# =========================================================
with tab1:
    st.header("Predict Salary for a Single Candidate")
    st.markdown("Fill out the details below to estimate the market salary.")
    
    col1, col2 = st.columns(2)
    with col1:
        experience = st.number_input("Years of Experience", 0, 50, value=2, step=1)
        education = st.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: {1:"High School", 2:"Bachelors", 3:"Masters", 4:"PhD"}[x])
        age = st.number_input("Age", 18, 100, value=25)
    with col2:
        certifications = st.number_input("Number of Certifications", 0, 20, value=1)
        projects = st.number_input("Number of Projects Completed", 0, 50, value=3)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Salary", type="primary", use_container_width=True):
        input_data = np.array([[experience, education, age, certifications, projects]])
        prediction = model.predict(input_data)[0]
        
        st.success("Prediction generated successfully!")
        st.metric(label="Predicted Annual Salary", value=f"₹ {prediction:,.2f}")
        
        # Save to history
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experience": experience,
            "education_level": education,
            "age": age,
            "certifications": certifications,
            "projects": projects,
            "predicted_salary": round(prediction, 2)
        })

# =========================================================
# TAB 2 - WHAT-IF ANALYSIS (SEPARATED & DIFFERENT)
# =========================================================
with tab2:
    st.header("🧠 What-If Scenario Analysis")
    st.markdown("Set a base profile and adjust sliders to simulate and visualize the **real-time impact** on the salary.")
    
    col_base, col_sim = st.columns([1, 2])
    
    with col_base:
        st.subheader("1. Base Profile")
        with st.container():
            base_exp = st.number_input("Base Exp", 0, 50, value=2, key="b_exp")
            base_edu = st.selectbox("Base Edu", [1,2,3,4], key="b_edu", format_func=lambda x: {1:"High School", 2:"Bachelors", 3:"Masters", 4:"PhD"}[x])
            base_age = st.number_input("Base Age", 18, 100, value=25, key="b_age")
            base_cert = st.number_input("Base Certs", 0, 20, value=1, key="b_cert")
            base_proj = st.number_input("Base Projects", 0, 50, value=3, key="b_proj")
            
            base_input = np.array([[base_exp, base_edu, base_age, base_cert, base_proj]])
            base_salary = model.predict(base_input)[0]
            st.metric(label="Base Salary", value=f"₹ {base_salary:,.2f}")
        
    with col_sim:
        st.subheader("2. Simulate Changes")
        w_exp = st.slider("Simulate Experience", 0, 50, base_exp, key="w_exp")
        w_edu = st.slider("Simulate Education", 1, 4, base_edu, key="w_edu")
        w_age = st.slider("Simulate Age", 18, 100, base_age, key="w_age")
        w_cert = st.slider("Simulate Certifications", 0, 20, base_cert, key="w_cert")
        w_proj = st.slider("Simulate Projects", 0, 100, base_proj, key="w_proj")
        
        sim_input = np.array([[w_exp, w_edu, w_age, w_cert, w_proj]])
        sim_salary = model.predict(sim_input)[0]
        diff = sim_salary - base_salary
        
        st.metric(label="Simulated Salary", value=f"₹ {sim_salary:,.2f}", delta=f"₹ {diff:,.2f}")
        
        # We want to show the independent isolated impacts as a bar chart
        st.markdown("### Independent Impact of Each Change")
        diff_exp = model.predict([[w_exp, base_edu, base_age, base_cert, base_proj]])[0] - base_salary
        diff_edu = model.predict([[base_exp, w_edu, base_age, base_cert, base_proj]])[0] - base_salary
        diff_age = model.predict([[base_exp, base_edu, w_age, base_cert, base_proj]])[0] - base_salary
        diff_cert = model.predict([[base_exp, base_edu, base_age, w_cert, base_proj]])[0] - base_salary
        diff_proj = model.predict([[base_exp, base_edu, base_age, base_cert, w_proj]])[0] - base_salary
        
        impacts = {
            "Experience": diff_exp,
            "Education": diff_edu,
            "Age": diff_age,
            "Certifications": diff_cert,
            "Projects": diff_proj
        }
        
        fig_impact = px.bar(
            x=list(impacts.keys()), 
            y=list(impacts.values()),
            labels={'x': 'Feature Modified', 'y': 'Salary Impact (₹)'},
            title="What if I ONLY changed this individual feature?",
            color=list(impacts.values()),
            color_continuous_scale=px.colors.diverging.RdYlGn
        )
        st.plotly_chart(fig_impact, use_container_width=True)


# =========================================================
# TAB 3 - BATCH PREDICTIONS
# =========================================================
with tab3:
    st.header("📂 Batch Salary Prediction")
    st.markdown("Upload a CSV file containing multiple profiles to predict their salaries simultaneously. Ensure the columns match the exact names: `experience`, `education_level`, `age`, `certifications`, `projects`.")
    
    # Download sample format
    sample_df = pd.DataFrame({
        "experience": [2, 5, 8],
        "education_level": [2, 3, 4],
        "age": [23, 28, 35],
        "certifications": [0, 2, 5],
        "projects": [5, 10, 20]
    })
    
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample CSV Format", data=sample_csv, file_name="sample_batch.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Upload Candidates CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(batch_df.head(5))
            
            # Ensure columns match
            missing_cols = [col for col in feature_names if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
            else:
                if st.button("Predict Batch Salaries", type="primary"):
                    with st.spinner("Generating predictions..."):
                        preds = model.predict(batch_df[feature_names])
                        batch_df["Predicted_Salary"] = np.round(preds, 2)
                    st.success(f"Successfully generated predictions for {len(batch_df)} profiles.")
                    st.dataframe(batch_df)
                    
                    # Download results
                    res_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results CSV", data=res_csv, file_name="predicted_salaries.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# =========================================================
# TAB 4 - ANALYTICS & HISTORY
# =========================================================
with tab4:
    st.header("📊 Performance Analytics & History")
    
    if len(st.session_state.history) == 0:
        st.info("Make at least one single prediction to start gathering history.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        
        # -------------------------
        # KPI METRICS
        # -------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df_hist))
        col2.metric("Avg Predicted Salary", f"₹ {df_hist['predicted_salary'].mean():,.0f}")
        col3.metric("Max Predicted Salary", f"₹ {df_hist['predicted_salary'].max():,.0f}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Salary Trend")
            fig_trend = px.line(df_hist, x="time", y="predicted_salary", markers=True, title="Predictions Over Time")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("Salary Distribution")
            fig_hist = px.histogram(df_hist, x="predicted_salary", nbins=10, title="Distribution of Predicted Salaries", marginal="box", color_discrete_sequence=['indianred'])
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Correlation Insights")
        avg_features = df_hist[["experience", "certifications", "projects"]].mean().reset_index()
        avg_features.columns = ["Feature", "Average Value"]
        fig_bar = px.bar(avg_features, x="Feature", y="Average Value", color="Feature", title="Average Input Profile (from history)")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Prediction History Logs")
        st.dataframe(df_hist, use_container_width=True)
        
        csv_hist = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full History", csv_hist, "prediction_history.csv", "text/csv")
