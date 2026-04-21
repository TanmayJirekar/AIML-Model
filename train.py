import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save_models():
    print("Initiating native model training for current environment...")
    np.random.seed(42)

    data_size = 4000
    df = pd.DataFrame({
        "experience": np.random.randint(0, 20, data_size),
        "education_level": np.random.randint(1, 5, data_size),
        "age": np.random.randint(21, 60, data_size),
        "certifications": np.random.randint(0, 10, data_size),
        "projects": np.random.randint(1, 30, data_size)
    })

    df["salary"] = (
        df["experience"] * 70000 +
        df["education_level"] * 50000 +
        df["certifications"] * 15000 +
        df["projects"] * 10000 +
        (df["experience"] * df["certifications"] * 2000) + 
        np.random.randint(0, 30000, data_size)
    )

    feature_columns = ["experience", "education_level", "age", "certifications", "projects"]
    X = df[feature_columns]
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    metrics = {}
    for name, m in [("Random Forest", rf_model), ("Linear Regression", lr_model), ("XGBoost", xgb_model)]:
        preds = m.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics[name] = {"MAE": mae, "R2": r2}

    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(lr_model, "lr_model.pkl")
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(feature_columns, "features.pkl")
    joblib.dump(metrics, "model_metrics.pkl")
    print("All models natively trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
