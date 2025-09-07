import os
from dotenv import load_dotenv

load_dotenv()

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY", "REPLACE_ME")
SMS_DEFAULT_RECIPIENTS = os.getenv("SMS_DEFAULT_RECIPIENTS", "").strip()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("1", "true", "yes", "y")
EMERGENCY_THRESHOLD = float(os.getenv("EMERGENCY_THRESHOLD", "0.75"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "20"))

# ICE servers
STUN_URLS = ["stun:stun.l.google.com:19302"]

TURN_URLS = [u.strip() for u in os.getenv("TURN_URLS", "").split(",") if u.strip()]
TURN_USERNAME = os.getenv("TURN_USERNAME", "").strip()
TURN_CREDENTIAL = os.getenv("TURN_CREDENTIAL", "").strip()

def build_rtc_configuration():
    ice_servers = [{"urls": STUN_URLS}]
    if TURN_URLS and TURN_USERNAME and TURN_CREDENTIAL:
        ice_servers.append({
            "urls": TURN_URLS,
            "username": TURN_USERNAME,
            "credential": TURN_CREDENTIAL,
        })
    return {"iceServers": ice_servers}
