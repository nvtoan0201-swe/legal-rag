# syntax=docker/dockerfile:1
# CPU-only image: smaller and portable (no CUDA runtime needed in the container).
# Host installs via uv sync keep GPU torch; only the container runs CPU inference.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so source edits do not invalidate the expensive layer.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --no-dev --no-install-project \
        --no-install-package torch --no-install-package torchvision \
        --no-install-package triton \
    && uv pip install --python /app/.venv/bin/python --torch-backend=cpu torch torchvision

COPY . .
RUN uv sync --no-dev \
        --no-install-package torch --no-install-package torchvision \
        --no-install-package triton

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
