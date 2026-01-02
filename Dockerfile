# YOLOv8 Object Detection - Production Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .

# Create directories for data and outputs
RUN mkdir -p data/images outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run image detection
# Users can override with docker run commands
CMD ["python", "-m", "src.main", "--config", "config.yaml"]
