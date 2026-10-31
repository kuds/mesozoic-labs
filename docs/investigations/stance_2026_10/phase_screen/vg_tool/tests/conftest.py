"""vg_tool tests: import vg_sigma from the folder above (read only; -B / PYTHONDONTWRITEBYTECODE: no bytecode)."""
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# VG_SIGMA_DIR: only the mutation check (val/mutants.py) points the tests at a mutated copy of vg_sigma.py
TOOL = Path(os.environ.get("VG_SIGMA_DIR") or HERE.parent)
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))
