FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create the non-root user and the data directory BEFORE switching user,
# so a fresh named volume mounted at /data inherits this ownership.
RUN useradd --create-home --uid 1000 simuser \
    && mkdir -p /data \
    && chown simuser:simuser /data

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY tests ./tests

USER simuser

# Placeholder command. Replaced when the CLI exists.
CMD ["python", "-c", "import simengine; print(simengine.__version__)"]