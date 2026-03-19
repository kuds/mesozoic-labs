"""Simulation-wide default constants for Mesozoic Labs environments.

Centralises magic numbers that were previously scattered across species
environment files and training scripts.  Per-species constants should
remain in their respective modules; only values shared across two or
more species belong here.
"""

# ---------------------------------------------------------------------------
# Sensor layout (matches MJCF sensor definition order across all species)
# ---------------------------------------------------------------------------
SENSOR_GYRO_START: int = 0
SENSOR_ACCEL_START: int = 3
SENSOR_QUAT_START: int = 6

# ---------------------------------------------------------------------------
# VecNormalize defaults (used by train_base.py)
# ---------------------------------------------------------------------------
DEFAULT_NORM_OBS: bool = True
DEFAULT_NORM_REWARD: bool = True
DEFAULT_CLIP_OBS: float = 10.0
DEFAULT_CLIP_REWARD: float = 50.0

# ---------------------------------------------------------------------------
# Physics defaults
# ---------------------------------------------------------------------------
DEFAULT_FRAME_SKIP: int = 5
TAIL_ANGULAR_VEL_MAX: float = 10.0  # rad/s — normalisation ceiling
