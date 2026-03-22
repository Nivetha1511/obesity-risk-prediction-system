import requests


API_URL = "http://127.0.0.1:5000/predict"

payload = {
    "Gender": 1,
    "Age": 23,
    "Height": 1.80,
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
    "MTRANS": 3,
}

response = requests.post(API_URL, json=payload, timeout=30)
print("Status:", response.status_code)
print("Response:", response.json())
