# Use Debian slim for better compatibility with scientific Python packages
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for scientific Python packages
# These ensure packages like matplotlib and xgboost can run properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train_pipeline.py .
COPY main.py .

# Create necessary directories that will be used at runtime
RUN mkdir -p artifacts models charts content

# Copy optional data - these directories should exist in your repo
# If you don't need to bundle data/models in the image, they will be empty
COPY artifacts/ ./artifacts/
COPY models/ ./models/
COPY content/ ./content/
COPY tess_clean_no_na.csv ./tess_clean_no_na.csv

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV MATPLOTLIB_BACKEND=Agg

# Expose port (Zeabur will auto-detect this)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)"

# Run the application with production settings
# Run uvicorn directly (not with python -m) to ensure proper pickle deserialization
CMD ["uvicorn", "train_pipeline:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
