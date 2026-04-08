FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    MAX_CONCURRENT_ENVS=4

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml /app/
COPY src /app/src
COPY server /app/server

RUN pip install --upgrade pip && pip install .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT}"]
