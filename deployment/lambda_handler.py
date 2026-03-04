import json, boto3, joblib, numpy as np, pandas as pd, os, uuid
from datetime import datetime, timezone
from decimal import Decimal

MODEL_PATH = "/tmp/hgb_model.pkl"
COLUMNS_PATH = "/tmp/model_columns.pkl"
LOCATION_ENCODING_PATH = "/tmp/location_encoding.pkl"
BUCKET = "real-estate-model-artifacts-rc"
TABLE_NAME = "real-estate-predictions"

def download_artifacts():
    if not os.path.exists(MODEL_PATH):
        s3 = boto3.client("s3")
        s3.download_file(BUCKET, "models/hgb_model.pkl", MODEL_PATH)
        s3.download_file(BUCKET, "models/model_columns.pkl", COLUMNS_PATH)
        s3.download_file(BUCKET, "models/location_encoding.pkl", LOCATION_ENCODING_PATH)

download_artifacts()
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)
location_encoding = joblib.load(LOCATION_ENCODING_PATH)

# DynamoDB resource — initialized once at cold start
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))

        input_df = pd.DataFrame([{
            "total_sqft": body["total_sqft"],
            "bath": body["bath"],
            "balcony": body["balcony"],
            "BHK": body["BHK"],
            "area_type": body["area_type"],
        }])

        # One-hot encode area_type only (matching training)
        input_encoded = pd.get_dummies(input_df, columns=["area_type"], drop_first=True)

        # Target encode location (matching training)
        location = body["location"]
        input_encoded["location_encoded"] = location_encoding.get(
            location, np.median(list(location_encoding.values()))
        )

        # Align columns to training
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_aligned)[0]
        request_id = str(uuid.uuid4())
        predicted_price = round(float(prediction), 2)

        # ── Log prediction to DynamoDB ────────────────────────────────────
        table.put_item(Item={
            "request_id":            request_id,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "area_type":             body["area_type"],
            "location":              location,
            "total_sqft":            Decimal(str(body["total_sqft"])),
            "bath":                  Decimal(str(body["bath"])),
            "balcony":               Decimal(str(body["balcony"])),
            "BHK":                   Decimal(str(body["BHK"])),
            "predicted_price_lakhs": Decimal(str(predicted_price)),
        })
        # ─────────────────────────────────────────────────────────────────

        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"},
            "body": json.dumps({
                "ok": True,
                "predicted_price_lakhs": predicted_price,
                "request_id": request_id
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"ok": False, "error": str(e)})
        }