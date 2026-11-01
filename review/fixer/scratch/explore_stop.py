import sys, shutil
from pathlib import Path
sys.path.insert(0, "/home/user/work/pc_study/tools/tests")
import synth_pc as S
C = S.C
root = Path("/home/user/work/pc_study/review/fixer/scratch/stoproot")
shutil.rmtree(root, ignore_errors=True)
rnd = S.Round(root / "r")
E = C.GRID_STEPS[50]
print("E", E, "FINAL", C.FINAL)
for s in (60, 80):
    S.standard_run(rnd, s, ledger=S.ledger_with_cmd(0.05))
S.standard_run(rnd, 42, ledger=S.ledger_with_cmd(0.05, E=E), E=E, hop_watch=True)
out, st = rnd.score()
print(out["incomplete"])
for x in out["per_seed"]:
    print(x["seed"], x["verdict"], (x.get("classes") or {}).get("cls"), x.get("finished"), (x.get("td") or {}).get("td"))
rd = out["reading"]
print(rd and (rd["line"], rd["one_test"]["k"], rd["one_test"]["pc_infinite"]))
shutil.rmtree(root, ignore_errors=True)
