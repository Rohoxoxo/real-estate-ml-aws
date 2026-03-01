import json
import joblib
import pandas as pd
import os
import logging
import uuid
import boto3

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# S3 config
BUCKET = "real-estate-model-artifacts-rc"
MODEL_S3_KEY = "models/hgb_model.pkl"
COLUMNS_S3_KEY = "models/model_columns.pkl"
LOCATION_ENCODING_S3_KEY = "models/location_encoding.pkl"

MODEL_LOCAL = "/tmp/hgb_model.pkl"
COLUMNS_LOCAL = "/tmp/model_columns.pkl"
LOCATION_ENCODING_LOCAL = "/tmp/location_encoding.pkl"

# Download model artifacts from S3 on startup
s3_client = boto3.client("s3", region_name="us-east-1")

if not os.path.exists(MODEL_LOCAL):
    logger.info("Downloading model from S3...")
    s3_client.download_file(BUCKET, MODEL_S3_KEY, MODEL_LOCAL)
    logger.info("Model downloaded")

if not os.path.exists(COLUMNS_LOCAL):
    logger.info("Downloading columns from S3...")
    s3_client.download_file(BUCKET, COLUMNS_S3_KEY, COLUMNS_LOCAL)
    logger.info("Columns downloaded")

if not os.path.exists(LOCATION_ENCODING_LOCAL):
    logger.info("Downloading location encoding from S3...")
    s3_client.download_file(BUCKET, LOCATION_ENCODING_S3_KEY, LOCATION_ENCODING_LOCAL)
    logger.info("Location encoding downloaded")

MODEL = joblib.load(MODEL_LOCAL)
COLUMNS = joblib.load(COLUMNS_LOCAL)
LOCATION_ENCODING = joblib.load(LOCATION_ENCODING_LOCAL)
logger.info(f"Model loaded. Column count: {len(COLUMNS)}")


def validate_payload(payload: dict) -> None:
    required = ["area_type", "location", "total_sqft", "bath", "balcony", "BHK"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    sqft = float(payload["total_sqft"])
    bath = float(payload["bath"])
    balcony = float(payload["balcony"])
    bhk = float(payload["BHK"])

    if not (300 <= sqft <= 10000):
        raise ValueError("Total area must be between 300 and 10,000 sq.ft")
    if not (1 <= bhk <= 10):
        raise ValueError("BHK must be between 1 and 10")
    if not (1 <= bath <= 10):
        raise ValueError("Bathrooms must be between 1 and 10")
    if not (0 <= balcony <= 3):
        raise ValueError("Balconies must be between 0 and 3")
    if bath > bhk + 2:
        raise ValueError(f"Bathrooms ({int(bath)}) cannot exceed BHK ({int(bhk)}) + 2")
    if sqft / bhk < 300:
        raise ValueError(f"Minimum 300 sq.ft per BHK required")
    if not isinstance(payload["area_type"], str) or not payload["area_type"].strip():
        raise ValueError("area_type must be a non-empty string")
    if not isinstance(payload["location"], str) or not payload["location"].strip():
        raise ValueError("location must be a non-empty string")


def predict_price(payload: dict) -> float:
    location = payload["location"]
    area_type = payload["area_type"]

    # Use target encoding for location
    location_encoded = LOCATION_ENCODING.get(location, LOCATION_ENCODING.get("other", 70.0))

    input_df = pd.DataFrame([{
        "total_sqft": float(payload["total_sqft"]),
        "bath": float(payload["bath"]),
        "balcony": float(payload["balcony"]),
        "BHK": float(payload["BHK"]),
        "area_type": area_type,
    }])

    input_encoded = pd.get_dummies(input_df, columns=["area_type"], drop_first=True)
    input_encoded = input_encoded.fillna(0).astype(float)
    input_encoded["location_encoded"] = location_encoded
    input_encoded = input_encoded.reindex(columns=COLUMNS, fill_value=0)

    return float(MODEL.predict(input_encoded)[0])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"ok": False, "error": "No JSON body"}), 400

        validate_payload(payload)
        pred = predict_price(payload)

        return jsonify({
            "ok": True,
            "predicted_price_lakhs": round(pred, 2),
            "request_id": str(uuid.uuid4())
        })

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)