# Sentiment Analyzer

This repository contains a Streamlit app for sentiment analysis of messy CSVs and social media content, with OCR support for screenshots.

Quick start (local):

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Install Tesseract (required for OCR). On Windows, download from:
https://github.com/UB-Mannheim/tesseract/wiki

3. Place your trained model file `model.joblib` in the repository root.

4. Run the app:

```powershell
.\.venv\Scripts\python -m streamlit run d:\sentiment\app.py
```

Deployment options

- Streamlit Community Cloud: easiest for pure-Streamlit apps (no system-level Tesseract). If you need OCR, use a Docker-based deploy or host with Tesseract installed.
- Docker: the repo contains a sample Dockerfile you can use to build an image with Tesseract installed.

See `Dockerfile` and `README-deploy.md` for more details.
