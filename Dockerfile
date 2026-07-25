# syntax=docker/dockerfile:1
# BioIntel Agent image — used by both the API and the Streamlit UI services.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# System deps: build tools for a few wheels, curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast resolver/installer).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY biointel ./biointel
# Install the project + runtime deps into the system environment.
RUN uv pip install --system --no-cache .

# Copy the rest of the source (configs, alembic, scripts).
COPY . .

EXPOSE 8000 8501

# Default command is overridden by docker-compose for each service.
CMD ["uvicorn", "biointel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
