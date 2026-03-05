import json, boto3, joblib, numpy as np, pandas as pd, os, uuid
from datetime import datetime, timezone
from decimal import Decimal

MODEL_PATH = "/tmp/hgb_model.pkl"
COLUMNS_PATH = "/tmp/model_columns.pkl"
LOCATION_ENCODING_PATH = "/tmp/location_encoding.pkl"
BUCKET = "real-estate-model-artifacts-rc"
TABLE_NAME = "real-estate-predictions"

# ── Validation constants (based on cleaned training data) ────────────────────
VALID_AREA_TYPES = ["Super built-up  Area", "Plot  Area", "Built-up  Area", "Carpet  Area"]
SQFT_MIN, SQFT_MAX = 300, 36000
BATH_MIN, BATH_MAX = 1, 16
BALCONY_MIN, BALCONY_MAX = 0, 3
BHK_MIN, BHK_MAX = 1, 16
# ─────────────────────────────────────────────────────────────────────────────

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

def validate_input(body):
    """Validate all input fields. Returns (is_valid, error_message)."""
    errors = []

    # Check required fields exist
    required = ["total_sqft", "bath", "balcony", "BHK", "area_type", "location"]
    for field in required:
        if field not in body:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors

    # Validate total_sqft
    try:
        sqft = float(body["total_sqft"])
        if sqft < SQFT_MIN or sqft > SQFT_MAX:
            errors.append(f"total_sqft must be between {SQFT_MIN} and {SQFT_MAX} sqft. Got: {sqft}")
    except (ValueError, TypeError):
        errors.append("total_sqft must be a number")

    # Validate bath
    try:
        bath = float(body["bath"])
        if bath < BATH_MIN or bath > BATH_MAX:
            errors.append(f"bath must be between {BATH_MIN} and {BATH_MAX}. Got: {bath}")
    except (ValueError, TypeError):
        errors.append("bath must be a number")

    # Validate balcony
    try:
        balcony = float(body["balcony"])
        if balcony < BALCONY_MIN or balcony > BALCONY_MAX:
            errors.append(f"balcony must be between {BALCONY_MIN} and {BALCONY_MAX}. Got: {balcony}")
    except (ValueError, TypeError):
        errors.append("balcony must be a number")

    # Validate BHK
    try:
        bhk = float(body["BHK"])
        if bhk < BHK_MIN or bhk > BHK_MAX:
            errors.append(f"BHK must be between {BHK_MIN} and {BHK_MAX}. Got: {bhk}")
    except (ValueError, TypeError):
        errors.append("BHK must be a number")

    # Validate area_type
    if body.get("area_type") not in VALID_AREA_TYPES:
        errors.append(f"area_type must be one of: {VALID_AREA_TYPES}. Got: '{body.get('area_type')}'")

    # Validate location exists in training data
    if body.get("location") not in location_encoding:
        errors.append(f"Unknown location: '{body.get('location')}'. Please select a valid Bengaluru location.")

    # Logical check — bath shouldn't exceed BHK by more than 2
    try:
        if float(body["bath"]) > float(body["BHK"]) + 2:
            errors.append(f"Number of bathrooms ({body['bath']}) seems too high for {body['BHK']} BHK")
    except (ValueError, TypeError):
        pass

    if errors:
        return False, errors

    return True, []

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))

        # ── Validate input ────────────────────────────────────────────────
        is_valid, errors = validate_input(body)
        if not is_valid:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"},
                "body": json.dumps({
                    "ok": False,
                    "error": "Invalid input",
                    "details": errors
                })
            }
        # ─────────────────────────────────────────────────────────────────

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