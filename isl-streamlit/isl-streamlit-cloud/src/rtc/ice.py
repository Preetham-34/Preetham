import os
from typing import Dict, List

def _get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

def _get_secret_list(key: str) -> List[str]:
    raw = _get_secret(key, "")
    return [s.strip() for s in raw.split(",") if s.strip()]

def _try_twilio_ice_servers():
    sid = _get_secret("TWILIO_ACCOUNT_SID", "")
    token = _get_secret("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        return None
    try:
        from twilio.rest import Client
        client = Client(sid, token)
        t = client.tokens.create()
        servers = []
        for s in t.ice_servers:
            urls = s.get("urls") or s.get("url")
            if not urls:
                continue
            item = {"urls": urls}
            if s.get("username"):
                item["username"] = s["username"]
            if s.get("credential"):
                item["credential"] = s["credential"]
            servers.append(item)
        if servers:
            return {"iceServers": servers}
    except Exception:
        return None
    return None

def build_rtc_configuration() -> Dict:
    cfg = _try_twilio_ice_servers()
    if cfg:
        return cfg
    turn_urls = _get_secret_list("TURN_URLS")
    turn_user = _get_secret("TURN_USERNAME", "")
    turn_cred = _get_secret("TURN_CREDENTIAL", "")
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if turn_urls and turn_user and turn_cred:
        ice_servers.append({"urls": turn_urls, "username": turn_user, "credential": turn_cred})
    return {"iceServers": ice_servers}
