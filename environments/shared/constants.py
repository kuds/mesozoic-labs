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
TAIL_ANGULAR_VEL_MAX: float = 10.0  # rad/s — normalisation ceiling

# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
# First seed of the publication_evaluation seed role; the registered panel
# family is 3042-3081 — see provenance.evaluation_protocols.
PUBLICATION_SEED_START: int = 3042

# Environment steps between in-training evaluations (EvalCallback): the SB3
# notebook's train_stage, the command line's --eval-freq default and
# train_base's.  100,000 since 2026-11-28 (the maintainer's decision; 50,000
# before): at 40 episodes every 50,000 steps the in-training evaluation was
# about 45% of an 11M T. rex stance's wall clock on a Colab L4, and it
# certifies nothing (every gate is judged after training on its own panel).
# The evaluation-counted collapse backstop keys were halved with it
# (collapse_min_evals 20 -> 10, collapse_patience 10 -> 5) so they keep their
# step spans (1M and 500k).  It enters no digest; stage_config.json records it
# as run.eval_freq.
DEFAULT_EVAL_FREQ: int = 100_000
