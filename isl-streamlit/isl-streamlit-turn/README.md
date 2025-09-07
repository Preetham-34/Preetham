# ISL → Text (Streamlit) with Emergency Alerts

A Streamlit-based web app for **Indian Sign Language (ISL) → text** with **emergency gesture detection** and **SMS alerts** (Fast2SMS).  
This repository includes a working scaffold with **mock models** so you can run it immediately; swap in your real models (MediaPipe, CNN+BiLSTM, T5, YOLO) later without changing the UI.

## Features
- Real-time webcam capture via WebRTC (`streamlit-webrtc`).
- Gesture recognition and sentence formation (mocked by default).
- Emergency detection path with SMS alerts (Fast2SMS).
- Throttling and basic logging of alerts.
- Dockerfile for easy deployment.

## Quick Start (Local)
1. Python 3.10+ recommended.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. (Optional) Copy `.env.example` to `.env` and set variables:
   ```bash
   cp .env.example .env
   # Edit FAST2SMS_API_KEY, SMS_DEFAULT_RECIPIENTS, etc.
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

> The app runs in **mock mode** by default (no heavy ML libs required). To enable real models later, set `MOCK_MODE=false` in `.env` and implement model loaders in `src/models/*` (see `TODO` markers).

## Environment Variables
Create `.env` or set environment variables directly:
- `FAST2SMS_API_KEY` (required for SMS sending; otherwise the app stays in dry-run for SMS)
- `SMS_DEFAULT_RECIPIENTS` (comma-separated phone numbers, optional)
- `MOCK_MODE` (default `true`; set `false` to use real models when wired)
- `EMERGENCY_THRESHOLD` (default `0.75`)
- `ALERT_COOLDOWN_SEC` (default `20`)

## Project Structure
```
.
├── app.py
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml
├── src/
│   ├── config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── fps.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db.py
│   ├── sms/
│   │   ├── __init__.py
│   │   └── fast2sms.py
│   └── models/
│       ├── __init__.py
│       ├── factory.py
│       ├── landmarks.py
│       ├── gesture.py
│       ├── sentence.py
│       └── emergency.py
├── models/           # Place your .tflite / .onnx files here
│   └── .gitkeep
└── data/
    └── .gitkeep
```

## Docker
Build and run:
```bash
docker build -t isl-streamlit .
docker run -it --rm -p 8501:8501 --env-file .env isl-streamlit
```
Navigate to http://localhost:8501

## Deployment Notes
- **HTTPS** is required for WebRTC in production. Place Streamlit behind a reverse proxy (e.g., Nginx) with TLS, or use a platform that provides HTTPS.
- For scale or heavy models, consider splitting inference into a separate service (FastAPI) and let Streamlit call it via HTTP. Keep this UI unchanged.

## Swapping in Real Models
- **Landmarks (MediaPipe)**: Implement in `src/models/landmarks.py`.
- **Gesture model (CNN+BiLSTM)**: `src/models/gesture.py`.
- **Sentence generation (T5)**: `src/models/sentence.py`.
- **Emergency detection (YOLO)**: `src/models/emergency.py`.
- Use TF-Lite/ONNX and quantization to achieve real-time on CPU.

## License
MIT


## WebRTC (STUN/TURN) Setup
If you see a yellow banner "Connection is taking longer than expected", configure a TURN server (STUN alone may not traverse NAT/firewalls).

### Quick start with coturn (docker-compose)
```
cd ops/turn
docker compose up -d
```
Then set in `.env` (match `turnserver.conf` creds):
```
TURN_URLS=turn:YOUR_SERVER_IP:3478?transport=udp,turn:YOUR_SERVER_IP:3478?transport=tcp
TURN_USERNAME=turnuser
TURN_CREDENTIAL=turnpass
```
> For stricter networks, enable TLS on coturn (port 5349) and use `turns:` URLs.

### Enable in the app
The app reads these variables and builds the ICE config automatically:
- `TURN_URLS` (comma-separated)
- `TURN_USERNAME`
- `TURN_CREDENTIAL`

Also ensure your deployment is served over **HTTPS** to allow camera access on remote origins.
