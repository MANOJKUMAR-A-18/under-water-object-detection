# Underwater Detection Gradio App - Dockerfile
# Use a slim Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable logs straight to stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps (opencv needs some)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_professional.txt ./
RUN pip install --no-cache-dir -r requirements_professional.txt

# Copy source
COPY . ./

# Gradio listens on 0.0.0.0, use PORT provided by platform (default 7860)
ENV SERVER_NAME=0.0.0.0 \
    PORT=7860 \
    GRADIO_SHARE=false

EXPOSE 7860

# Run the app
CMD ["python", "app_simple_professional.py"]
