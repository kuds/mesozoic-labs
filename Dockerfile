FROM python:3.11-slim

# MuJoCo runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libosmesa6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Headless rendering for MuJoCo
ENV MUJOCO_GL=osmesa

WORKDIR /app

# Install Python dependencies first (for Docker layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[train]"

# Copy project source code
COPY environments/ environments/
COPY configs/ configs/

# Install the package in editable mode
COPY README.md .
RUN pip install --no-cache-dir -e ".[train]"

ENTRYPOINT ["python"]
