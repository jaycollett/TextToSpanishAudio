FROM python:3.10-slim

# Install system dependencies (e.g., build tools and libsndfile for audio processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache Coqui TTS Spanish male model, auto-accept license
RUN echo y | python -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits', gpu=False)"

# Copy the application code
COPY . .

# Help with cuda vram use
ENV Pexport=PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

# Expose the port your Flask app listens on
EXPOSE 5055

# Start the app (using the built-in server; for production consider using gunicorn)
CMD ["python", "app.py"]
