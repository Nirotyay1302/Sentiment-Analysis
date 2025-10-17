FROM python:3.11-slim

# Minimal system deps (no Tesseract or image libs required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false"]
