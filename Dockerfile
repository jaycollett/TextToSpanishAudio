# syntax=docker/dockerfile:1.7
#
# Two-stage build. Kept on Debian (glibc) rather than Alpine: coqui-tts pulls
# torch/torchaudio, which publish no musllinux wheels at all, so musl here
# would mean compiling PyTorch from source. Raised python:3.10-slim ->
# python:3.12-slim-trixie instead, which is the highest release coqui-tts
# 0.26.0 accepts (its metadata is `>=3.10,<3.13`).
#
# The builder carries build-essential and the pip cache; the runtime stage
# receives the finished virtualenv, the pre-warmed TTS model and the app. No
# compilers ship.

# ---- Stage 1: build the virtualenv and pre-warm the TTS model ----
FROM python:3.12-slim-trixie AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# Build-time system dependencies (compilers, and libsndfile for the audio
# packages that probe for it during install).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

# Copy and install Python dependencies.
#
# The transformers constraint is a build unblocker, not a hardening change.
# coqui-tts 0.26.0 declares `transformers>=4.47.0` with no upper bound, and
# newer transformers releases dropped two APIs that coqui-tts 0.26.0 imports
# at module scope: `pytorch_utils.isin_mps_friendly` (gone in 5.x) and the
# top-level `BeamSearchScorer` (gone in the later 4.5x line). Any from-scratch
# build of this image has therefore failed for some time - confirmed against
# the ORIGINAL python:3.10 base, so this predates the base bump and is not
# caused by it.
#
# 4.47-4.49 is the window coqui-tts 0.26.0 was written against; coqui-tts
# 0.27.0 is the release that moved to `transformers>=4.52.1`.
#
# The proper fix is to pin transformers in requirements.txt, or to upgrade
# coqui-tts to 0.27.x. It is pinned here instead to keep this CVE pass out of
# the project's declared dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 'transformers>=4.47,<4.50'

# Download and cache Coqui TTS Spanish male model, auto-accept license.
# TTS_HOME pins the cache to a known path so the runtime stage can copy it.
ENV TTS_HOME=/opt/tts
RUN mkdir -p /opt/tts \
    && echo y | python -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits', gpu=False)"

# ---- Stage 2: runtime ----
FROM python:3.12-slim-trixie

# Runtime system dependencies only: libsndfile for `soundfile`, ffmpeg for
# pydub's transcoding. No build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Help with cuda vram use
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

# NUMBA_CACHE_DIR / MPLBACKEND matter once the process is not root. librosa is
# numba-jitted, and numba's default cache location is beside the module source
# inside /opt/venv, which a non-root user cannot write - it raises "cannot
# cache function '__o_fold': no locator available" at import time. Pointing the
# cache at the runtime user's home fixes it. Agg keeps matplotlib from looking
# for a display.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TTS_HOME=/opt/tts \
    HOME=/home/appuser \
    NUMBA_CACHE_DIR=/home/appuser/.cache/numba \
    XDG_CACHE_HOME=/home/appuser/.cache \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tts /opt/tts

# Copy the application code
COPY . .

# Non-root runtime. Numeric UID/GID so K8S runAsNonRoot can verify without a
# passwd lookup.
# /data holds the SQLite job database and the generated audio (app.py creates
# both at import time). It is pre-created and owned here, because a non-root
# process cannot mkdir at the filesystem root.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/appuser \
        --shell /usr/sbin/nologin appuser \
    && mkdir -p /home/appuser/.cache/numba /data/audiofiles \
    && chown -R appuser:appuser /app /opt/tts /home/appuser /data

USER 1000:1000

# Job database and generated audio. Declared after the chown above so a new
# volume starts out owned by appuser. Mount a named volume or host path here;
# otherwise Docker creates an anonymous volume that a recreated container
# does not reuse.
VOLUME /data

# Expose the port your Flask app listens on
EXPOSE 5055

# Start the app (using the built-in server; for production consider using gunicorn)
CMD ["python", "app.py"]
