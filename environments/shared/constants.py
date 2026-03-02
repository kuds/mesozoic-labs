"""Simulation-wide default constants.

Centralises values that were previously hardcoded across multiple
species modules. Import from here instead of duplicating magic numbers.
"""

# ── Sensor layout (matches MJCF sensor order across all species) ─────────
SENSOR_GYRO_START = 0
SENSOR_ACCEL_START = 3
SENSOR_QUAT_START = 6

# ── VecNormalize defaults (used by train_base.py) ────────────────────────
DEFAULT_NORM_OBS = True
DEFAULT_NORM_REWARD = True
DEFAULT_CLIP_OBS = 10.0
DEFAULT_CLIP_REWARD = 50.0

# ── Physics defaults ─────────────────────────────────────────────────────
DEFAULT_FRAME_SKIP = 5
TAIL_ANGULAR_VEL_MAX = 10.0  # rad/s — normalisation cap for tail instability

# ── Species list (for parametrised tests) ────────────────────────────────
ALL_SPECIES = ("velociraptor", "trex", "brachiosaurus")
