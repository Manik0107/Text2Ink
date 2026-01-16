# Use official slim Python image
FROM python:3.11

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by Pillow and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies first (cacheable layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Ensure output folder exists
RUN mkdir -p /app/output_images

# Use a non-root user for safety
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser /app
USER appuser

# Default command: run the main script. You can override with `docker run <image> python main.py`.
CMD ["python", "main.py"]
