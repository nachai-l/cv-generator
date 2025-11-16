FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (extend if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps from requirements.txt (generated via uv)
RUN pip install --upgrade pip && pip install -r requirements.txt

ENV PORT=8080
ENV APP_ENV=prod

# FastAPI app is in api.py → app = FastAPI(...)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
