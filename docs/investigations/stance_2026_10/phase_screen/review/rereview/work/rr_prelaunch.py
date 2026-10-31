"""Re-reviewer: score_bs.score on the REAL <PS> root, read only (score() writes nothing; no run exists yet)."""
import importlib.util
import sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("score_bs_pre", "/home/user/work/phase_screen/tools/score_bs.py")
SB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SB)
res, lines, code = SB.score("/home/user/work/phase_screen")
print("exit", code, "status", res.get("status"), "decided", res.get("n_decided", res.get("decided")))
print("provisional", res.get("provisional"))
