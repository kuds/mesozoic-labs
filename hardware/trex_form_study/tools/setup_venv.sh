#!/usr/bin/env bash
# Create the form study's Python environment: a venv holding the pinned packages of tools/requirements-study.txt, then
# an import check. Run it from anywhere:
#
#   bash hardware/trex_form_study/tools/setup_venv.sh [VENV_DIR]     # default VENV_DIR: hardware/trex_form_study/.venv
#   TORCH=cpu bash hardware/trex_form_study/tools/setup_venv.sh      # CPU-only torch (about 1 GB instead of 6 GB)
#   PYTHON=/usr/local/bin/python3.13 bash ...                        # choose the interpreter (default: python3)
#
# The default VENV_DIR is git-ignored (.venv/) and, unlike a cloud session's scratchpad, sits on the checkout's own
# disk. The records since Stage 1 were made on Python 3.13.18; another 3.13 works for development, but say so in any
# record you make with it. Then, from hardware/trex_form_study:
#
#   export PY=$PWD/.venv/bin/python
#   MUJOCO_GL=disable PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests
#
# MUJOCO_GL=disable matters: with libOSMesa loaded, torch segfaults while building a stable-baselines3 policy
# (tests/conftest.py; LESSONS_LEARNED.md). Render only in a separate process with MUJOCO_GL=osmesa.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY=$(dirname "$HERE")
VENV=${1:-$STUDY/.venv}
PYTHON=${PYTHON:-python3}
TORCH=${TORCH:-pypi}

"$PYTHON" - <<'EOF'
import sys
v = sys.version_info
if (v.major, v.minor) != (3, 13):
    sys.exit(f"Python {sys.version.split()[0]}: the study's packages are pinned for Python 3.13 (records: 3.13.18)")
if sys.version.split()[0] != "3.13.18":
    print(f"note: Python {sys.version.split()[0]}, not the records' 3.13.18", file=sys.stderr)
EOF

if [ -e "$VENV/bin/python" ]; then
  echo "using the existing venv at $VENV"
else
  "$PYTHON" -m venv "$VENV"
fi
PIP=("$VENV/bin/python" -m pip install --no-cache-dir --disable-pip-version-check)
if [ "$TORCH" = cpu ]; then
  # 2.15.0+cpu satisfies requirements-study.txt's torch==2.15.0 (PEP 440 ignores the local label).
  "${PIP[@]}" torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
fi
"${PIP[@]}" -r "$HERE/requirements-study.txt"

cd "$STUDY"
MUJOCO_GL=disable PYTHONPATH=. "$VENV/bin/python" - <<'EOF'
import sys

import gymnasium
import matplotlib
import mujoco
import numpy
import PIL
import scipy
import stable_baselines3
import torch

import designlab  # noqa: F401  (the study's package; sets MUJOCO_GL only when it is unset)
from designlab.stage1 import BODIES

print("python", sys.version.split()[0], "| mujoco", mujoco.__version__, "| numpy", numpy.__version__, "| scipy",
      scipy.__version__, "| gymnasium", gymnasium.__version__, "| sb3", stable_baselines3.__version__, "| torch",
      torch.__version__, "| matplotlib", matplotlib.__version__, "| pillow", PIL.__version__)
print("Stage 1 registry:", ", ".join(sorted(BODIES)))
EOF
echo "venv ready: $VENV"
echo "next: PYTHONPATH=. $VENV/bin/python tools/restore_inputs.py check   (from $STUDY)"
