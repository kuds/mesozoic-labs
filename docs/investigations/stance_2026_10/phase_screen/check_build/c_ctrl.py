"""PS build checker: is PS's control BS2's control?  Runs PS's OWN run_bs.py (tools/run_bs.py, unmodified file, loaded
as a module) in its no-optimizer-step mode (--no-update: train() a no-op) with BS2's validation seed 9201, so that its
per-rollout evidence and evaluation ledger can be compared with BS2's own V2 run of control_9201
(BS2/validate/v2final/control_9201, read only).  Two in-memory patches, both paths only: run_bs.bs_root() points at
PS/check_build/ctrl_root (so the run's folder and TMPDIR are inside the checker's folder), and the validation-seed tuple
gets 9201 (BS2's).  Nothing else is changed.
    PY -I -B c_ctrl.py UNTIL
"""
import importlib.util
import sys
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
ROOT = PS / "check_build" / "ctrl_root"
spec = importlib.util.spec_from_file_location("run_bs", str(PS / "tools" / "run_bs.py"))
R = importlib.util.module_from_spec(spec)
sys.modules["run_bs"] = R
spec.loader.exec_module(R)
R.bs_root = lambda: ROOT
R.C.VALIDATION_SEEDS = (*R.C.VALIDATION_SEEDS, 9201)
(ROOT / "validate").mkdir(parents=True, exist_ok=True)
until = int(sys.argv[1])
sys.exit(R.main(["--arm", "control", "--seed", "9201", "--no-update", "--until", str(until),
                 "--out", str(ROOT / "validate")]))
