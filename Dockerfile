FROM python:3.11-slim

# Install system dependencies for Tesseract and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    poppler-utils \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false"]
