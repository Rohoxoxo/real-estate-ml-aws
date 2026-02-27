import json
from app import lambda_handler

event = {
    "body": json.dumps({
        "area_type": "Super built-up  Area",
        "location": "Whitefield",
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1,
        "BHK": 2
    })
}

print(lambda_handler(event, None))

event_invalid = {
    "body": json.dumps({
        "area_type": "Super built-up  Area",
        "location": "Whitefield",
        "total_sqft": -500,
        "bath": 2,
        "balcony": 1,
        "BHK": 2
    })
}

print(lambda_handler(event_invalid, None))

