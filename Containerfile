# Containerfile for mlx-omni-server
# Note: MLX is optimized for Apple Silicon (macOS + Metal). Running inside
# a Linux container will typically fall back to CPU-only and may not leverage
# Metal acceleration. Build for linux/arm64 on Apple Silicon hosts.

# ---------- Base image ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models MLX_OMNI_LOG_LEVEL=info MLX_OMNI_PORT=10240

# Install OS dependencies commonly needed by scientific Python stacks
# (kept minimal to keep image slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only project metadata first for better layer caching
COPY pyproject.toml ./
COPY README.md LICENSE ./

# Copy source
COPY src ./src

# Install uv (fast Python package manager) and use it to install the project
# See: https://docs.astral.sh/uv/
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project and its runtime dependencies only using uv
# - Build via hatchling as declared in pyproject
RUN uv pip install --system --no-cache .

# Create cache/log dirs and make them writable
RUN mkdir -p /models /logs && chmod -R 777 /models /logs

EXPOSE 10240

# Healthcheck: query a lightweight endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD \
    curl -fsS http://127.0.0.1:${MLX_OMNI_PORT}/health || exit 1

# Default command runs the server
# Use env MLX_OMNI_PORT to change the port
CMD ["sh", "-c", "mlx-omni-server --host 0.0.0.0 --port ${MLX_OMNI_PORT}"]
