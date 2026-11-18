"""Test-session defaults for the Triceratops RL tests.

MuJoCo's OpenGL is disabled before anything imports MuJoCo: no test renders, and a GL backend loaded next to torch
can crash a stable-baselines3 policy build (seen in the T. rex form study's tests). The rl directory and the repository
root go on sys.path so ``import trike_env`` and ``import environments...`` work from any working directory.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

RL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_DIR.parents[2]
for p in (str(RL_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
