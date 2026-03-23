from pathlib import Path
from threading import Lock
from datetime import datetime
import csv
import hashlib
import secrets
import json

import joblib
import numpy as np
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(32)

MODEL_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

USERS_CSV = DATA_DIR / "users.csv"
PATIENT_DATA_CSV = DATA_DIR / "patient_data.csv"

MODEL_LOCK = Lock()
MODEL = None
SCALER = None
TARGET_ENCODER = None
FEATURE_COLUMNS = None
RISK_LEVEL_MAP = None


def init_csv_files():
    """Initialize CSV files if they don't exist."""
    if not USERS_CSV.exists():
        with open(USERS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'name', 'password_hash', 'created_at'])
    
    if not PATIENT_DATA_CSV.exists():
        with open(PATIENT_DATA_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'user_id', 'user_name', 'gender', 'age', 'height', 'weight', 
                           'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'CH2O', 
                           'SCC', 'PAL', 'MTRANS', 'predicted_risk_level', 'confidence', 'created_at', 'result_data'])


def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_next_user_id():
    """Get the next available user ID."""
    if not USERS_CSV.exists():
        return 1
    with open(USERS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        rows = list(reader)
        return len(rows) + 1 if rows else 1


def get_next_patient_id():
    """Get the next available patient ID."""
    if not PATIENT_DATA_CSV.exists():
        return 1
    with open(PATIENT_DATA_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        rows = list(reader)
        return len(rows) + 1 if rows else 1


def user_exists(name):
    """Check if a username already exists."""
    if not USERS_CSV.exists():
        return False
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] == name:
                return True
    return False


def get_user_by_name(name):
    """Get user by name."""
    if not USERS_CSV.exists():
        return None
    with open(USERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] == name:
                return row
    return None


def load_artifacts() -> None:
    global MODEL, SCALER, TARGET_ENCODER, FEATURE_COLUMNS, RISK_LEVEL_MAP

    if MODEL is not None:
        return

    with MODEL_LOCK:
        if MODEL is None:
            MODEL = load_model(MODEL_DIR / "ann_obesity_model.keras")
            SCALER = joblib.load(MODEL_DIR / "scaler.pkl")
            TARGET_ENCODER = joblib.load(MODEL_DIR / "target_encoder.pkl")
            FEATURE_COLUMNS = joblib.load(MODEL_DIR / "feature_columns.pkl")
            RISK_LEVEL_MAP = joblib.load(MODEL_DIR / "risk_level_map.pkl")


def parse_payload(payload: dict) -> np.ndarray:
    missing_fields = [key for key in FEATURE_COLUMNS if key not in payload]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    values = []
    for feature in FEATURE_COLUMNS:
        raw_value = payload[feature]
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for field '{feature}': {raw_value}") from exc

    return np.array(values, dtype=np.float32).reshape(1, -1)


@app.get("/")
def home():
    return jsonify(
        {
            "service": "ObesiCare API",
            "status": "running",
            "endpoint": "/predict",
        }
    )


@app.post("/auth/register")
def register():
    """Register a new user with name and password."""
    try:
        init_csv_files()
        
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be valid JSON."}), 400
        
        name = payload.get('name', '').strip()
        password = payload.get('password', '').strip()
        
        if not name or len(name) < 2:
            return jsonify({"error": "Name must be at least 2 characters."}), 400
        
        if not password or len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."}), 400
        
        if user_exists(name):
            return jsonify({"error": "This username already exists."}), 409
        
        user_id = get_next_user_id()
        password_hash = hash_password(password)
        created_at = datetime.now().isoformat()
        
        with open(USERS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, name, password_hash, created_at])
        
        return jsonify({
            "success": True,
            "message": "User registered successfully.",
            "user_id": user_id,
            "name": name
        }), 201
    
    except Exception as exc:
        return jsonify({"error": f"Registration failed: {exc}"}), 500


@app.post("/auth/login")
def login():
    """Login user with name and password."""
    try:
        init_csv_files()
        
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be valid JSON."}), 400
        
        name = payload.get('name', '').strip()
        password = payload.get('password', '').strip()
        
        if not name or not password:
            return jsonify({"error": "Username and password are required."}), 400
        
        user = get_user_by_name(name)
        if not user:
            return jsonify({"error": "Invalid username or password."}), 401
        
        password_hash = hash_password(password)
        if user['password_hash'] != password_hash:
            return jsonify({"error": "Invalid username or password."}), 401
        
        session['user_id'] = user['user_id']
        session['user_name'] = user['name']
        
        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user_id": user['user_id'],
            "name": user['name']
        }), 200
    
    except Exception as exc:
        return jsonify({"error": f"Login failed: {exc}"}), 500


@app.post("/auth/logout")
def logout():
    """Logout current user."""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."}), 200


@app.post("/predict")
def predict():
    try:
        init_csv_files()
        load_artifacts()

        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        # Extract user info
        user_id = payload.get('user_id')
        user_name = payload.get('user_name')
        
        if not user_id or not user_name:
            return jsonify({"error": "User ID and name are required for prediction."}), 400

        # Extract health features
        input_data = parse_payload(payload)
        input_scaled = SCALER.transform(input_data)

        probabilities = MODEL.predict(input_scaled, verbose=0)[0]
        predicted_class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_index])

        predicted_risk_level = RISK_LEVEL_MAP.get(
            predicted_class_index,
            str(TARGET_ENCODER.inverse_transform([predicted_class_index])[0]),
        )

        # Store patient data for future reference and personalization
        patient_id = get_next_patient_id()
        created_at = datetime.now().isoformat()
        
        patient_record = [
            patient_id,
            user_id,
            user_name,
            payload.get('Gender', ''),
            payload.get('Age', ''),
            payload.get('Height', ''),
            payload.get('Weight', ''),
            payload.get('family_history_with_overweight', ''),
            payload.get('FAVC', ''),
            payload.get('FCVC', ''),
            payload.get('NCP', ''),
            payload.get('CAEC', ''),
            payload.get('CH2O', ''),
            payload.get('SCC', ''),
            payload.get('PAL', ''),
            payload.get('MTRANS', ''),
            predicted_risk_level,
            confidence,
            created_at,
            json.dumps({
                "predicted_class_index": predicted_class_index,
                "confidence": confidence,
                "probabilities": probabilities.tolist()
            })
        ]
        
        with open(PATIENT_DATA_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(patient_record)

        return jsonify(
            {
                "predicted_class_index": predicted_class_index,
                "predicted_risk_level": predicted_risk_level,
                "confidence": round(confidence, 4),
                "patient_id": patient_id,
                "message": "Prediction saved to patient record."
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"Model artifacts not found: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500



@app.get("/patient-history/<int:user_id>")
def get_patient_history(user_id):
    """Retrieve all health records for a patient (for personalization)."""
    try:
        init_csv_files()
        
        records = []
        with open(PATIENT_DATA_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['user_id']) == user_id:
                    records.append(row)
        
        if not records:
            return jsonify({"error": "No records found for this user."}), 404
        
        return jsonify({
            "user_id": user_id,
            "total_records": len(records),
            "records": records
        }), 200
    
    except Exception as exc:
        return jsonify({"error": f"Error retrieving history: {exc}"}), 500


if __name__ == "__main__":
    init_csv_files()
    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
