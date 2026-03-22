from pathlib import Path
from threading import Lock

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_LOCK = Lock()
MODEL = None
SCALER = None
TARGET_ENCODER = None
FEATURE_COLUMNS = None
RISK_LEVEL_MAP = None


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


@app.post("/predict")
def predict():
    try:
        load_artifacts()

        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        input_data = parse_payload(payload)
        input_scaled = SCALER.transform(input_data)

        probabilities = MODEL.predict(input_scaled, verbose=0)[0]
        predicted_class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_index])

        predicted_risk_level = RISK_LEVEL_MAP.get(
            predicted_class_index,
            str(TARGET_ENCODER.inverse_transform([predicted_class_index])[0]),
        )

        return jsonify(
            {
                "predicted_class_index": predicted_class_index,
                "predicted_risk_level": predicted_risk_level,
                "confidence": round(confidence, 4),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"Model artifacts not found: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
