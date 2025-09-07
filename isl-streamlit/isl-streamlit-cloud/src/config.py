import os
from dotenv import load_dotenv

load_dotenv()

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY", "REPLACE_ME")
SMS_DEFAULT_RECIPIENTS = os.getenv("SMS_DEFAULT_RECIPIENTS", "").strip()
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("1", "true", "yes", "y")
EMERGENCY_THRESHOLD = float(os.getenv("EMERGENCY_THRESHOLD", "0.75"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "20"))
