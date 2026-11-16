FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/patterns data/labeled models reports logs mlruns

ENV PORT=10000 \
    TELEGRAM_BOT_TOKEN="9088596436:AAHOp0uhVtA4snr9ARmmncS_ljwnIkxyq68" \
    TELEGRAM_CHAT_ID="6124128574" \
    OCTA_REAL_ACCOUNT="125726194"

EXPOSE 10000

CMD ["python", "render_main.py"]
