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
# uv.lock resolves torch with its full CUDA stack on Linux (cuda-toolkit,
# nvidia-* wheels — several GB). The container excludes every CUDA package and
# installs the pinned CPU-only wheels from the PyTorch CPU index instead.
# No second sync after COPY: the project is not a package (no build-system),
# so all dependencies come from this sync, and an exact re-sync would
# uninstall the CPU torch installed below.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package triton \
        --no-install-package cuda-bindings \
        --no-install-package cuda-pathfinder \
        --no-install-package cuda-toolkit \
        --no-install-package nvidia-cublas \
        --no-install-package nvidia-cuda-cupti \
        --no-install-package nvidia-cuda-nvrtc \
        --no-install-package nvidia-cuda-runtime \
        --no-install-package nvidia-cudnn-cu13 \
        --no-install-package nvidia-cufft \
        --no-install-package nvidia-cufile \
        --no-install-package nvidia-curand \
        --no-install-package nvidia-cusolver \
        --no-install-package nvidia-cusparse \
        --no-install-package nvidia-cusparselt-cu13 \
        --no-install-package nvidia-nccl-cu13 \
        --no-install-package nvidia-nvjitlink \
        --no-install-package nvidia-nvshmem-cu13 \
        --no-install-package nvidia-nvtx \
    && uv pip install --python /app/.venv/bin/python --torch-backend=cpu \
        torch==2.14.1 torchvision==0.29.1

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
