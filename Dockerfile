# ============================================================
# Dockerfile — RAG Service
# ============================================================
#
# INTERVIEW QUESTIONS:
#   Q: What is a multi-stage Docker build?
#   A: Use one image to BUILD the app (with compilers, dev tools),
#      then copy only the final artifacts into a smaller RUNTIME image.
#      Result: much smaller production image. Here: builder ~1GB → final ~200MB.
#
#   Q: Why run as non-root in Docker?
#   A: Security principle of least privilege. If the container is
#      compromised, the attacker gets a non-root shell, not root access
#      to the host. Required by many K8s security policies (PodSecurityStandards).
#
#   Q: What is .dockerignore?
#   A: Like .gitignore — prevents files (node_modules, .git, __pycache__,
#      test files, secrets) from being sent to the Docker build context.
#      Speeds up builds and prevents accidental secret inclusion.
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
# Copy requirements before source code — Docker only re-runs this layer
# if requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production Image ─────────────────────────────────
FROM python:3.11-slim AS production

# Metadata
LABEL org.opencontainers.image.title="RAG Service"
LABEL org.opencontainers.image.description="Production RAG API — LLMOps Tutorial"
LABEL org.opencontainers.image.source="https://github.com/AyorindeTayo/llm-llmops-tutorial"

# Create non-root user (security best practice)
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/

# Create directories the app needs
RUN mkdir -p /app/cache /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check built into the image
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8080    # Application
EXPOSE 8000    # Prometheus metrics

# Default command — can be overridden in K8s Deployment
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
