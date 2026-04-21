import joblib
import pandas as pd
import numpy as np

# Lazy load models
_models = {}

def load_models():
    if not _models:
        try:
            _models['Random Forest'] = joblib.load("rf_model.pkl")
            _models['Linear Regression'] = joblib.load("lr_model.pkl")
            _models['XGBoost'] = joblib.load("xgb_model.pkl")
            _models['features'] = joblib.load("features.pkl")
            _models['metrics'] = joblib.load("model_metrics.pkl")
        except FileNotFoundError:
            return False
    return True

def get_prediction(input_data_dict, model_name="Random Forest"):
    """
    input_data_dict: dict containing experience, education_level, age, certifications, projects
    Returns predicted salary float
    """
    if not load_models():
        return 0.0
        
    df = pd.DataFrame([input_data_dict])
    # ensure columns ordering
    df = df[_models['features']]
    model = _models.get(model_name, _models['Random Forest'])
    
    pred = model.predict(df)[0]
    return float(pred)

def get_feature_importances(model_name="Random Forest"):
    if not load_models():
        return {}
    
    model = _models.get(model_name, _models['Random Forest'])
    features = _models['features']
    
    if model_name in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_
    elif model_name == "Linear Regression":
        # use absolute coefficients
        importances = np.abs(model.coef_)
        # normalize
        importances = importances / np.sum(importances)
    else:
        return {}

    return dict(zip(features, importances))

def get_model_metrics():
    if not load_models():
        return {}
    return _models.get('metrics', {})

def predict_future_growth(current_profile, model_name="Random Forest"):
    """
    Simulates organic growth over 1, 3, 5, 10 years.
    Assuming +1 exp per year, +some certs/projects naturally over time.
    """
    if not load_models():
        return pd.DataFrame()
        
    years_ahead = [1, 3, 5, 10]
    results = []
    
    for y in years_ahead:
        sim = current_profile.copy()
        sim['experience'] += y
        sim['age'] += y
        # simulate gaining 1 cert every 3 years
        sim['certifications'] += int(y / 3)
        # simulate gaining 2 projects every year
        sim['projects'] += (y * 2)
        
        salary = get_prediction(sim, model_name)
        results.append({
            "Year": f"+{y} Year{'s' if y>1 else ''}",
            "Salary": salary
        })
        
    return pd.DataFrame(results)

def get_ai_advice(profile):
    """
    Rules-based AI advice generator comparing current profile to max achievable
    """
    advice = []
    
    if profile['experience'] < 5:
        advice.append("💡 **Experience Focus**: You are in the early stages of your career. Focus on gathering foundational experience. The model highly rewards years of experience.")
        
    if profile['education_level'] < 3:
        advice.append("🎓 **Education Boost**: Upgrading your education to a Master's degree will unlock a significant tier jump in your potential salary.")
        
    if profile['certifications'] < 3:
        advice.append("📜 **Certifications**: You have low certifications. Earning highly-recognized industry certs (AWS, Azure) will provide strong synergy multipliers to your existing experience.")
        
    if profile['projects'] < 5:
        advice.append("🛠 **Projects Portfolio**: Try building and deploying more practical projects. High project counts demonstrate applied knowledge.")
        
    if not advice:
        advice.append("🌟 **Excellent Profile**: You are on a highly optimized career track! Keep mastering niche skills and taking leadership roles!")
        
    return "\n\n".join(advice)
