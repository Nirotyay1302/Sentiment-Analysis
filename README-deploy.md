Deployment guide

Option A — Streamlit Community Cloud (fastest):
- Remove heavy system deps (OCR won't work unless added via a custom build)
- Push repo to GitHub and create a new Streamlit app pointing to the repository.

Option B — Docker (recommended if you need OCR):
- Build the image:
  docker build -t sentiment-app:latest .
- Run locally:
  docker run -p 8501:8501 -v $(pwd)/model.joblib:/app/model.joblib sentiment-app:latest

Option C — VPS or cloud VM:
- Install Python and Tesseract on the machine.
- Clone repo, setup venv, install requirements, and run the Streamlit server.

Notes:
- Make sure `model.joblib` is included in your deployed image or accessible to the runtime.
- For security, don't commit `model.joblib` to a public repo unless it's safe to share.
