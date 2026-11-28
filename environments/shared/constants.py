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
# train_base's.  100,000 since 2026-11-28 (decision D-D79 (4); 50,000
# before): at 40 episodes every 50,000 steps the in-training evaluation was
# about 45% of an 11M T. rex stance's wall clock on a Colab L4.  In the
# notebook it judges no gate (every gate is judged after training on its own
# panel), though it ranks the checkpoints the handoff is chosen from; the
# command line's curriculum also advances the stance gates that can pass
# in training (reward_and_length/v1, stance_quality/v1) on it.  The collapse
# backstop's wait, collapse_min_evals, was halved with it (20 -> 10 on the
# eight configs that declare it, 6 declared on the T. rex recovery) to keep
# its step span; its patience, a count of consecutive low evaluations
# calibrated as a count, was not.  It enters no digest; stage_config.json
# records it as run.eval_freq.
DEFAULT_EVAL_FREQ: int = 100_000
