import joblib
import pandas as pd
import numpy as np
import os
import sys

_models = {}

def load_models(force_retrain=False):
    global _models
    
    # If forced to retrain or files are missing entirely
    missing_files = not all(os.path.exists(f) for f in ["rf_model.pkl", "lr_model.pkl", "xgb_model.pkl", "features.pkl", "model_metrics.pkl"])
    
    if force_retrain or missing_files:
        from train import train_and_save_models
        train_and_save_models()

    if not _models or force_retrain:
        try:
            _models['Random Forest'] = joblib.load("rf_model.pkl")
            _models['Linear Regression'] = joblib.load("lr_model.pkl")
            _models['XGBoost'] = joblib.load("xgb_model.pkl")
            _models['features'] = joblib.load("features.pkl")
            _models['metrics'] = joblib.load("model_metrics.pkl")
            
            # Fire a dummy prediction to strictly catch Scikit-Learn validation mismatches (e.g. monotonic_cst error)
            dummy_df = pd.DataFrame([[1, 1, 25, 1, 1]], columns=_models['features'])
            _models['Random Forest'].predict(dummy_df)
            
        except Exception as e:
            print(f"Pickle compatibility error detected: {e}. Retraining native models...", file=sys.stderr)
            # If the dummy prediction crashes because local pickles were from an old scikit-learn
            # We strictly retrain using the cloud server's exact scikit-learn version
            from train import train_and_save_models
            train_and_save_models()
            
            # Reload
            _models['Random Forest'] = joblib.load("rf_model.pkl")
            _models['Linear Regression'] = joblib.load("lr_model.pkl")
            _models['XGBoost'] = joblib.load("xgb_model.pkl")
            _models['features'] = joblib.load("features.pkl")
            _models['metrics'] = joblib.load("model_metrics.pkl")
            
    return True

def get_prediction(input_data_dict, model_name="Random Forest"):
    load_models()
    df = pd.DataFrame([input_data_dict])
    df = df[_models['features']]
    model = _models.get(model_name, _models['Random Forest'])
    pred = model.predict(df)[0]
    return float(pred)

def get_feature_importances(model_name="Random Forest"):
    load_models()
    model = _models.get(model_name, _models['Random Forest'])
    features = _models['features']
    
    if model_name in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_
    elif model_name == "Linear Regression":
        importances = np.abs(model.coef_)
        importances = importances / np.sum(importances)
    else:
        return {}

    return dict(zip(features, importances))

def get_model_metrics():
    load_models()
    return _models.get('metrics', {})

def predict_future_growth(current_profile, model_name="Random Forest"):
    load_models()
    years_ahead = [1, 3, 5, 10]
    results = []
    for y in years_ahead:
        sim = current_profile.copy()
        sim['experience'] += y
        sim['age'] += y
        sim['certifications'] += int(y / 3)
        sim['projects'] += (y * 2)
        
        salary = get_prediction(sim, model_name)
        results.append({
            "Year": f"+{y} Year{'s' if y>1 else ''}",
            "Salary": salary
        })
    return pd.DataFrame(results)

def get_ai_advice(profile):
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
