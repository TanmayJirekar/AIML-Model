from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# -----------------------------
# LOAD MODEL + FEATURES
# -----------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICTION API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # -------------------------
    # INPUTS (must match training order)
    # -------------------------
    experience = int(data["experience"])
    education_level = int(data["education_level"])
    age = int(data["age"])
    certifications = int(data["certifications"])
    projects = int(data["projects"])

    # -------------------------
    # MODEL INPUT
    # -------------------------
    input_data = np.array([[
        experience,
        education_level,
        age,
        certifications,
        projects
    ]])

    prediction = model.predict(input_data)[0]

    # -------------------------
    # GRAPH GENERATION
    # -------------------------
    labels = features
    values = [experience, education_level, age, certifications, projects]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.title("User Input Features")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    graph_base64 = base64.b64encode(img.getvalue()).decode()

    # -------------------------
    # RESPONSE
    # -------------------------
    return jsonify({
        "predicted_salary": round(prediction, 2),
        "graph": graph_base64
    })


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
