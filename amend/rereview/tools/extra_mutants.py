"""Re-reviewer's extra mutants (not in mutate_pc.py): each changes one amended rule in a copy of ST/tools/{pc,tests};
the suite tests/test_score_pc.py is run on the copy.  SURVIVED = the reviewed tests do not pin that rule.

    PY -I -B extra_mutants.py WORK LOG
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

T = Path("/home/user/work/pc_study/tools")
PY = sys.executable
MUT = [
    ("X1 an off run's SNAPSHOT counted as an opening", "score_pc.py",
     'if n["kind"] in OPENING_KINDS and C.arm_of(n["run"]) == "pc"),',
     'if n["kind"] in OPENING_KINDS and C.arm_of(n["run"]) in ("pc", "off")),'),
    ("X2 an off launch in the opening's minute counted blind", "score_pc.py",
     'lo["time"] < first["time"])', 'lo["time"] <= first["time"])'),
    ("X3 the stock-critic why not read", "complete_r10round.py",
     ' and chk.get("why") == STOCK_CRITIC_WHY)', ')'),
    ("X4 an unfilled off_arm falls back to the phase pins", "complete_r10round.py",
     '        if not isinstance(blk, dict):\n            return None\n',
     '        if not isinstance(blk, dict):\n            return C.pin(pins, key)\n'),
    ("X5 the launch-check CLI ignores an unfilled switch pin", "complete_r10round.py",
     '("hyperparameters_sha256", "task_sha256", "plant_identity", "critic_phase_switch")',
     '("hyperparameters_sha256", "task_sha256", "plant_identity")'),
    ("X6 an unfinished COMPLETE run enters at its TD", "score_pc.py",
     'if (x.get("verdict") == "COMPLETE" and x.get("finished") and x.get("td")',
     'if (x.get("verdict") == "COMPLETE" and x.get("td")'),
    ("X7 -infinity not counted in the rank", "pc_common.py",
     '        n_inf = sum(1 for v in xs if v == math.inf) - sum(1 for v in xs if v == -math.inf)',
     '        n_inf = sum(1 for v in xs if v == math.inf)'),
    ("X8 replication block from the phase runs", "score_pc.py",
     'out["replication"] = P.replication(per_arm["off"], comparators)',
     'out["replication"] = P.replication(per_arm["pc"], comparators)'),
    ("X9 off_arm unfilled is not PROVISIONAL", "r10round_common.py",
     '            "critic_phase_switch", "critic_phase_spec",                                   # PC\n            "off_arm")',
     '            "critic_phase_switch", "critic_phase_spec")'),
    ("X10 the not-blind rule reads the first run's LAUNCH, not the seed's run", "score_pc.py",
     '        lo = CP.launch_note(notes, do["run"]) if do and do["run"] else None',
     '        lo = CP.launch_note(notes, do["first"]) if do and do["run"] else None'),
    ("X11 off TD presence: a not-blind off run keeps its TD", "score_pc.py",
     '        off_td = [(-math.inf if x["run"] in nb else test_value(x, -math.inf)) for x in per_arm["off"]]',
     '        off_td = [test_value(x, -math.inf) for x in per_arm["off"]]'),
]


def main():
    work, log = Path(sys.argv[1]), Path(sys.argv[2])
    if work.exists():
        print("REFUSED: work exists")
        return 2
    out = []
    for i, (name, fname, old, new) in enumerate(MUT, 1):
        d = work / f"x{i}"
        shutil.copytree(T / "pc", d / "pc")
        shutil.copytree(T / "tests", d / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        f = d / "pc" / fname
        src = f.read_text()
        n = src.count(old)
        if n != 1:
            out.append(f"{name}: NOT APPLIED ({n})")
            shutil.rmtree(d)
            continue
        f.write_text(src.replace(old, new))
        t0 = time.time()
        p = subprocess.run(["nice", "-n", "10", PY, "-I", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            str(d / "tests" / "test_score_pc.py"), f"--basetemp={d / 'tmp'}"],
                           capture_output=True, text=True, cwd=str(d))
        failed = [ln for ln in p.stdout.splitlines() if ln.startswith("FAILED")][:1]
        out.append(f"{name}: {'KILLED' if p.returncode else 'SURVIVED'} (exit {p.returncode}, {time.time() - t0:.0f} s) "
                   f"{failed[0] if failed else ''}")
        print(out[-1], flush=True)
        shutil.rmtree(d)
    log.write_text("\n".join(out) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
