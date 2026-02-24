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

# Install Python dependencies first (for Docker layer caching).
# Copy only pyproject.toml + README (needed by metadata) and install the
# package non-editable.  Because environments/ is empty the package itself
# is a no-op, but all *dependencies* get cached in this layer.
COPY pyproject.toml README.md ./
RUN mkdir -p environments && \
    pip install --no-cache-dir ".[train,viz]"

# Copy project source code
COPY environments/ environments/
COPY configs/ configs/

# Re-install in editable mode (deps already cached, so this is fast)
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python"]
