import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from database import save_prediction, get_all_predictions
from ml_models import get_prediction, get_feature_importances, get_model_metrics, predict_future_growth, get_ai_advice, load_models
from utils import parse_resume, generate_pdf_report

st.set_page_config(page_title="AI Career Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric, .stDataFrame, .stExpander, div[data-testid="stForm"] {
        background: rgba(30, 30, 30, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

if not load_models():
    st.error("⚠️ Models not found. Please run `train.py` first.")
    st.stop()

st.sidebar.title("🧠 AI Career System")
page = st.sidebar.radio("Navigation Logs", [
    "🏠 Dashboard & Predictor",
    "📈 Career Growth & Advisor",
    "🧬 Model Comparison & XAI",
    "📂 Resume Auto-Parser",
    "📊 History & Reports"
])

def create_base_form(col1, col2, prefix=""):
    with col1:
        exp = st.number_input("Years of Exp", 0, 50, value=3, key=f"{prefix}exp")
        edu = st.selectbox("Education Level", [1,2,3,4], format_func=lambda x:{1:"High School", 2:"Bachelors", 3:"Masters", 4:"PhD"}[x], key=f"{prefix}edu")
        age = st.number_input("Age", 18, 100, value=25, key=f"{prefix}age")
    with col2:
        cert = st.number_input("Certifications", 0, 20, value=2, key=f"{prefix}cert")
        proj = st.number_input("Projects", 0, 100, value=5, key=f"{prefix}proj")
    return {"experience": exp, "education_level": edu, "age": age, "certifications": cert, "projects": proj}

if page == "🏠 Dashboard & Predictor":
    st.title("💸 AI Salary Prediction Dashboard")
    st.markdown("Industry-grade predictor utilizing advanced ML algorithms.")
    
    col1, col2 = st.columns(2)
    profile = create_base_form(col1, col2)
    model_choice = st.selectbox("Select Model Engine for standard predictions:", ["Random Forest", "XGBoost", "Linear Regression"])
    
    if st.button("Predict Optimal Salary", type="primary"):
        salary = get_prediction(profile, model_name=model_choice)
        save_prediction(profile['experience'], profile['education_level'], profile['age'], profile['certifications'], profile['projects'], salary, model_choice)
        st.success("Successfully processed!")
        st.metric("Predicted Annual Salary", f"₹ {salary:,.2f}", delta=f"{profile['experience']} Yrs Exp")
        
        # Industry Benchmarking
        st.subheader("Industry Benchmarking")
        avg = 1000000
        top10 = 2500000
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = salary,
            title = {'text': "Market Position"},
            gauge = {
                'axis': {'range': [0, 3000000]},
                'steps': [
                    {'range': [0, avg], 'color': "lightgray"},
                    {'range': [avg, top10], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': salary}}))
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Career Growth & Advisor":
    st.title("📈 Career Growth Trajectory & AI Advisor")
    st.info("Simulate future salary progression organically over the coming decade.")
    col1, col2 = st.columns(2)
    profile = create_base_form(col1, col2, "cg_")
    
    if st.button("Analyze Career Path"):
        df_growth = predict_future_growth(profile, "XGBoost")
        
        st.subheader("Projected Salary Curve (Next 10 Years)")
        fig = px.line(df_growth, x="Year", y="Salary", markers=True, title="Compounding Salary Potential")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🤖 AI Chatbot Advisor")
        advice = get_ai_advice(profile)
        st.info(advice)

elif page == "🧬 Model Comparison & XAI":
    st.title("🧬 Explainable AI & Model Dash")
    
    st.subheader("Model Performance Metrics")
    metrics = get_model_metrics()
    m_df = pd.DataFrame(metrics).T
    st.dataframe(m_df.style.highlight_min(subset=["MAE"], color='lightgreen').highlight_max(subset=["R2"], color='lightgreen'))
    
    st.subheader("XAI: Feature Importances")
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Linear Regression"])
    fi = get_feature_importances(model_choice)
    fig = px.bar(x=list(fi.keys()), y=list(fi.values()), labels={'x':'Feature', 'y':'Importance Weight'}, title=f"What drives {model_choice} decisions?")
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚖️ What-If Simulator":
    st.title("⚖️ Advanced What-If Scenario Comparison")
    
    col1, col2 = st.columns(2)
    st.markdown("### Scenario A (Baseline)")
    prof_a = create_base_form(col1, col2, "wa_")
    sal_a = get_prediction(prof_a, "Random Forest")
    
    st.markdown("---")
    st.markdown("### Scenario B (Target)")
    c1, c2 = st.columns(2)
    prof_b = create_base_form(c1, c2, "wb_")
    sal_b = get_prediction(prof_b, "Random Forest")
    
    diff = sal_b - sal_a
    st.metric("Target Salary Difference", f"₹ {diff:,.2f}", delta=f"₹ {diff:,.2f}" if diff != 0 else None)
    
    fig = go.Figure(data=[
        go.Bar(name='Scenario A', x=list(prof_a.keys()), y=list(prof_a.values())),
        go.Bar(name='Scenario B', x=list(prof_b.keys()), y=list(prof_b.values()))
    ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

elif page == "📂 Resume Auto-Parser":
    st.title("📂 PDF Resume NLP Parser")
    
    uploaded_file = st.file_uploader("Upload your resume context (PDF)", type=["pdf"])
    if uploaded_file:
        parsed_data = parse_resume(uploaded_file)
        if parsed_data:
            st.success("Extracted features using NLP!")
            st.json(parsed_data)
            
            parsed_data['age'] = 25 # Default logic since resumes rarely have age
            sal = get_prediction(parsed_data, "XGBoost")
            st.metric("Auto-Predicted Salary", f"₹ {sal:,.2f}")
        else:
            st.error("Failed to parse.")

elif page == "📊 History & Reports":
    st.title("📊 History & PDF Reports")
    
    df = get_all_predictions()
    if not df.empty:
        st.dataframe(df)
        
        st.subheader("Generate AI Career Report")
        if st.button("Generate Downloadable PDF"):
            # Mock profile passing from latest run
            last_prof = df.iloc[-1].to_dict()
            profile = {
                "experience": last_prof["experience"],
                "education_level": last_prof["education_level"],
                "age": last_prof["age"],
                "certifications": last_prof["certifications"],
                "projects": last_prof["projects"]
            }
            growth = predict_future_growth(profile, "Random Forest")
            advice = get_ai_advice(profile)
            
            pdf_bytes = generate_pdf_report(profile, growth, advice)
            st.download_button(label="📥 Download AI Report", data=pdf_bytes, file_name="AI_Career_Report.pdf", mime="application/pdf")
    else:
        st.info("No prediction history found in SQLite.")
