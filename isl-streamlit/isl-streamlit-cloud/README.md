# ISL → Text (Streamlit) — Streamlit Cloud Ready

A Streamlit-based app for **Indian Sign Language → text** with **emergency detection** and **SMS**.
This version is tuned for **Streamlit Community Cloud**: it reads **TURN/Twilio** settings from **Secrets**.

## Quick Start (Cloud)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create an app from the repo.
3. In the app's **Settings → Secrets**, add one of:
   - **Twilio (recommended)**:
     ```
     TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     TWILIO_AUTH_TOKEN = "your_auth_token"
     ```
   - **Static TURN** (your own TURN provider):
     ```
     TURN_URLS = "turns:your.turn.host:5349?transport=tcp,turn:your.turn.host:3478?transport=tcp"
     TURN_USERNAME = "user"
     TURN_CREDENTIAL = "pass"
     ```
4. (Optional) Add:
   ```
   FAST2SMS_API_KEY = "YOUR_FAST2SMS_KEY"
   SMS_DEFAULT_RECIPIENTS = "9999999999,8888888888"
   ```
5. Deploy. The app will fetch ICE servers from Twilio automatically or use your TURN settings.

## Local Dev
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional
streamlit run app.py
```

## Notes
- On remote/Cloud, WebRTC usually **requires a TURN relay**. This app supports **Twilio Network Traversal** or **static TURN** via secrets.
- Replace the mocked model stubs in `src/models/*` with your real models when ready.
