import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1. CREATE SAMPLE DATASET
# -----------------------------
np.random.seed(42)

data_size = 1000

df = pd.DataFrame({
    "experience": np.random.randint(0, 15, data_size),
    "education_level": np.random.randint(1, 5, data_size),  # 1=school ... 4=PhD
    "age": np.random.randint(21, 60, data_size),
    "certifications": np.random.randint(0, 10, data_size),
    "projects": np.random.randint(1, 20, data_size)
})

# -----------------------------
# 2. CREATE SALARY TARGET
# -----------------------------
df["salary"] = (
    df["experience"] * 60000 +
    df["education_level"] * 40000 +
    df["certifications"] * 12000 +
    df["projects"] * 8000 +
    np.random.randint(0, 30000, data_size)
)

# -----------------------------
# 3. FEATURES & TARGET
# -----------------------------
feature_columns = [
    "experience",
    "education_level",
    "age",
    "certifications",
    "projects"
]

X = df[feature_columns]
y = df["salary"]

# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. MODEL TRAINING
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 7. SAVE MODEL + FEATURES
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(feature_columns, "features.pkl")

print("Model saved as model.pkl")
print("Features saved as features.pkl")
