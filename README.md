# A Data-Centric Deep Learning Based Mobile Application for Predicting Early Warnings and Symptoms of Adolescent Obesity

This project is a full-stack AI-powered healthcare web application that predicts obesity risk levels using an ANN model trained with TensorFlow/Keras and provides lifestyle recommendations through a mobile-style web interface.

## Project Structure

obesity-risk-prediction-system/
- backend/
  - app.py
  - requirements.txt
  - Procfile
  - models/
    - ann_obesity_model.keras
    - scaler.pkl
    - target_encoder.pkl
    - feature_columns.pkl
    - risk_level_map.pkl
  - training/
    - train_model.py
  - test_api.py
- frontend/
  - login.html
  - form.html
  - result.html
  - style.css
  - script.js
- dataset/
  - obesity_dataset.csv
- documentation/
  - system_architecture.png
  - model_workflow.png
  - block_diagram.png
- obesity_dataset_preprocessed.csv
- README.md

## System Workflow

Login -> Health Questionnaire -> Flask API /predict -> StandardScaler preprocessing -> ANN model inference -> Risk level output + recommendations

## Machine Learning Layer

- Dataset source: dataset/obesity_dataset.csv
- Preprocessing:
  - Feature validation
  - Numeric casting
  - StandardScaler normalization
- Model:
  - Dense(128, relu) + Dropout(0.30)
  - Dense(64, relu) + Dropout(0.20)
  - Dense(7, softmax)
- Artifacts saved with joblib and Keras:
  - ann_obesity_model.keras
  - scaler.pkl
  - target_encoder.pkl
  - feature_columns.pkl
  - risk_level_map.pkl

## Backend API Layer

- Framework: Flask + Flask-CORS
- Lazy loading enabled for model/scaler/encoder artifacts
- Endpoint:
  - POST /predict
- Request body example:

{
  "Gender": 1,
  "Age": 23,
  "Height": 1.8,
  "Weight": 92,
  "family_history_with_overweight": 1,
  "FAVC": 1,
  "FCVC": 2,
  "NCP": 3,
  "CAEC": 2,
  "SMOKE": 0,
  "CH2O": 2,
  "SCC": 0,
  "FAF": 1,
  "TUE": 1,
  "CALC": 1,
  "MTRANS": 3
}

- Response example:

{
  "predicted_class_index": 4,
  "predicted_risk_level": "Obesity Type I",
  "confidence": 0.91
}

## Frontend (Mobile-style)

- login.html: User profile collection (name, mobile, email, optional API URL)
- form.html: Health and lifestyle questionnaire
- result.html: Predicted risk, confidence, personalized recommendations
- localStorage used to retain:
  - userProfile
  - apiBaseUrl
  - lastQuestionnaire
  - predictionResult

## Run Locally

1. Open a terminal in backend/
2. Install dependencies:
   pip install -r requirements.txt
3. Train and save model artifacts:
   python training/train_model.py
4. Start Flask API:
   python app.py
5. Open frontend/login.html in a browser (or serve frontend with any static server)

## API Test Script

From backend/ run:
python test_api.py

## Cloud Deployment (Render)

1. Create a new Web Service from this repository.
2. Set Root Directory to backend.
3. Build command:
   pip install -r requirements.txt
4. Start command:
   gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
5. Upload model artifacts to backend/models (or run training during build pipeline and persist artifacts if your deploy process supports it).
6. Copy Render HTTPS URL and set it in frontend login page as API Base URL.

## Recommendation Logic

Recommendations are shown on the result page and adapt to risk category, including:
- Increasing physical activity
- Reducing high-calorie foods
- Increasing vegetable intake
- Maintaining a balanced diet
- Seeking healthcare professionals for higher risk levels
