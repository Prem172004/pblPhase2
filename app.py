from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///instance/bloodmark.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ✅ Load ML Model for Blood Group Prediction (Added Feature)
MODEL_PATH = r"C:\Users\abdhe\OneDrive\Documents\GitHub\BloodMark\template\model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ✅ Existing User Model (Unchanged)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

# ✅ Existing Routes (Unchanged)
@app.route("/")
def land():
    return render_template("land.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/donation")
def donation():
    return render_template("donation.html")

# ✅ Machine Learning Prediction Route (Added Feature)
@app.route("/predict", methods=['POST'])
def predict():
    if 'fingerprint' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        img = load_img(request.files['fingerprint'], target_size=(64, 64))  # Resize image
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)  # Normalize

        prediction = model.predict(img_array)
        blood_group_index = int(np.argmax(prediction))  # Assuming classification output

        blood_groups = ["A", "B", "AB", "O", "-A", "-B", "-AB", "-O"]

        return jsonify({"blood_group": blood_groups[blood_group_index]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Error Handling (Unchanged)
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# ✅ Run Application (Unchanged)
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=8000)