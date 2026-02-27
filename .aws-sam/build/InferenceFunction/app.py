import json
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = joblib.load(os.path.join(BASE_DIR, "hgb_model.pkl"))
COLUMNS = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))

def validate_payload(payload: dict) -> None:
    required = ["area_type", "location", "total_sqft", "bath", "balcony", "BHK"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    sqft = float(payload["total_sqft"])
    bath = float(payload["bath"])
    balcony = float(payload["balcony"])
    bhk = float(payload["BHK"])

    if sqft <= 0:
        raise ValueError("total_sqft must be > 0")
    if bhk <= 0:
        raise ValueError("BHK must be > 0")
    if bath <= 0:
        raise ValueError("bath must be > 0")
    if balcony < 0:
        raise ValueError("balcony must be >= 0")

    if not isinstance(payload["area_type"], str) or not payload["area_type"].strip():
        raise ValueError("area_type must be a non-empty string")
    if not isinstance(payload["location"], str) or not payload["location"].strip():
        raise ValueError("location must be a non-empty string")

def predict_price(payload: dict) -> float:
    input_df = pd.DataFrame([{
        "area_type": payload["area_type"],
        "location": payload["location"],
        "total_sqft": float(payload["total_sqft"]),
        "bath": float(payload["bath"]),
        "balcony": float(payload["balcony"]),
        "BHK": float(payload["BHK"]),
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=COLUMNS, fill_value=0)

    return float(MODEL.predict(input_encoded)[0])

def lambda_handler(event, context):
    try:
        body = event.get("body", "{}")
        payload = json.loads(body) if isinstance(body, str) else body

        validate_payload(payload)
        pred = predict_price(payload)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": True, "predicted_price_lakhs": pred})
        }

    except ValueError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": False, "error": str(e)})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": False, "error": f"Internal error: {str(e)}"})
        }