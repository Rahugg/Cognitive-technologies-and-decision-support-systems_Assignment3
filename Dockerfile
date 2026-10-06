FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN mkdir -p data/raw data/processed models

EXPOSE 8501 8888

FROM base AS app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM base AS notebook
COPY requirements-notebook.txt .
RUN pip install -r requirements-notebook.txt
