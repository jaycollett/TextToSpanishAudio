FROM python:3.9-slim

# Install system dependencies (e.g., build tools and libsndfile for audio processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Help with cuda vram use
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Expose the port your Flask app listens on
EXPOSE 5055

# Start the app (using the built-in server; for production consider using gunicorn)
CMD ["python", "app.py"]
