"""Reviewer's own mutants of score_pc_dev.py, run against builder A's test suite (copy in review/rt/tests): which
guards does the suite leave untested?

    cd review/rt && PY -I -B own_mutants.py <outdir>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path("/home/user/work/pc_study/amend2/tools/score_pc_dev.py")
TESTS = HERE / "tests"

MUTANTS = [
    ("R1 OFF RELAUNCH lines ignored when finding the last OFF launch",
     'n["kind"] in ("LAUNCH", "RELAUNCH") and C.arm_of(n["run"]) == "off"]',
     'n["kind"] == "LAUNCH" and C.arm_of(n["run"]) == "off"]'),
    ("R2 no JSON difference is ever UNEXPECTED",
     '        return "C1\'s listing of a dropped line"\n    return "UNEXPECTED"',
     '        return "C1\'s listing of a dropped line"\n    return "blindness"'),
    ("R3 no output-line difference is ever UNEXPECTED",
     '    return "UNEXPECTED (not a blindness line)"', '    return "blindness: the seed\'s OFF run blind or not"'),
    ("R4 containment always reported unchanged",
     "changed = {d: sorted(set(before[d].items()) ^ set(after[d].items())) for d in before}",
     "changed = {d: [] for d in before}"),
    ("R5 the load-closure check skipped",
     '        if f"{m}.py" not in got:\n            raise Refused',
     '        if False:\n            raise Refused'),
    ("R6 the decision line's sha256 never checked",
     "ok = sha256_bytes(body.encode(\"utf-8\")) == DECISION_LINE_SHA256", "ok = True"),
    ("R7 the deviation copy's self-check removed",
     "        check_deviation_copy(C, asread, dev, drop_lines, notes)\n", ""),
    ("R8 PHASE SNAPSHOT of any arm dropped (arm test removed)",
     'n["kind"] == "SNAPSHOT" and C.arm_of(n["run"]) == "pc"', 'n["kind"] == "SNAPSHOT"'),
]


def main(argv):
    root = Path(argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    src = TOOL.read_text(encoding="utf-8")
    surv = 0
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, (name, src.count(old))
        d = root / name.split()[0]
        d.mkdir(parents=True, exist_ok=True)
        (d / "tmp").mkdir(exist_ok=True)
        dev = d / "score_pc_dev.py"
        dev.write_text(src.replace(old, new), encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(d / "tmp"), "LANG": "C.UTF-8", "AMEND2_DEV_TOOL": str(dev)}
        p = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            f"--basetemp={d / 'pt'}", str(TESTS / "test_score_pc_dev.py")],
                           capture_output=True, text=True, cwd=str(TESTS), env=env)
        (d / "pytest.log").write_text(p.stdout + p.stderr)
        killed = p.returncode != 0
        surv += not killed
        failed = [x for x in p.stdout.splitlines() if x.startswith("FAILED ")]
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} (pytest exit {p.returncode}; "
              f"{failed[0][7:].split(' - ')[0] if failed else p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''})",
              flush=True)
    print(f"{len(MUTANTS)} reviewer mutants, {surv} survived builder A's suite")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main(sys.argv))
