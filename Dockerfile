# ──────────────────────────────────────────────
# Transcribe v2 — Free Transcriber Web UI (optimized)
# Build:  docker build -t transcribe-v2 .
# Run:    docker run -p 7860:7860 -v whisper-models:/root/.cache/huggingface transcribe-v2
# ──────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies:
#   ffmpeg/ffprobe  — audio decode + duration probe
#   libsndfile1     — soundfile backend
#   gcc             — build wheels for librosa/numpy if no wheel available
ARG EXTRA_SYS_DEPS=""
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        gcc \
        $EXTRA_SYS_DEPS \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY app.py .
COPY transcribe_diarize.py .

# Expose the Gradio port
EXPOSE 7860

# Runtime knobs (override with -e on docker run)
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    TRANSCRIBER_MAX_CONCURRENT=1 \
    TRANSCRIBER_MAX_MODELS=2 \
    TRANSCRIBER_MAX_AUDIO_MINUTES=180 \
    TRANSCRIBER_FFMPEG_TIMEOUT=600 \
    TRANSCRIBER_LOG_LEVEL=INFO

# Whisper model cache — mount a volume here to persist downloaded models
# e.g. docker run -v whisper-models:/root/.cache/huggingface -p 7860:7860 transcribe-v2

# Healthcheck: Gradio serves "/" with 200 once ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/',timeout=3).status==200 else 1)"

CMD ["python", "app.py"]
