# =============================================================================
# Network Traffic Anomaly & Threat Classification
# Docker Image
# =============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY ["project 1/requirements.txt", "requirements-main.txt"]
COPY ["project 1/dashboard/requirements.txt", "requirements-dashboard.txt"]
COPY ["project 1/siem_integration/requirements.txt", "requirements-siem.txt"]

# Create combined requirements and install dependencies
RUN cat requirements-main.txt requirements-dashboard.txt requirements-siem.txt | \
    grep -v "^#" | grep -v "^$" | sort -u > requirements-combined.txt && \
    pip install --no-cache-dir -r requirements-combined.txt && \
    pip install --no-cache-dir gunicorn && \
    rm -f requirements-*.txt

# Copy the entire project
COPY ["project 1/", "/app/"]

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Expose the Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Default command - run the dashboard
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "--chdir", "/app/dashboard", "app:app"]

