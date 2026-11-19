"""Test-session setup for the analytics tests: MuJoCo's GL off, and the analytics folder, rl/ and the repository root
on sys.path (``import gait_core``, ``import gait_metrics``, ``import environments...``) from any working directory."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("MPLBACKEND", "Agg")

ANALYTICS = Path(__file__).resolve().parents[1]
RL_DIR = ANALYTICS.parent
REPO_ROOT = RL_DIR.parents[2]
for p in (str(ANALYTICS), str(RL_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
