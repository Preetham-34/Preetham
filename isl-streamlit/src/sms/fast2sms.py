import requests
from typing import Tuple

FAST2SMS_URL = "https://www.fast2sms.com/dev/bulkV2"

def send_sms_fast2sms(api_key: str, numbers: str, message: str) -> Tuple[bool, str]:
    if not numbers.strip():
        return False, "No recipients configured."
    headers = {"authorization": api_key}
    payload = {
        "route": "v3",
        "sender_id": "TXTIND",
        "message": message,
        "language": "english",
        "flash": 0,
        "numbers": numbers,
    }
    try:
        r = requests.post(FAST2SMS_URL, headers=headers, data=payload, timeout=10)
        ok = r.status_code == 200 and '"return":true' in r.text.lower()
        return ok, r.text
    except Exception as e:
        return False, str(e)
