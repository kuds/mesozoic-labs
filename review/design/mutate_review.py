"""Design-lens reviewer's own mutants of the PC reading/test/habit mapping (none of them in ST/tools/mutate_pc.py's 22).
Each mutant changes one rule in a copy of ST/tools/{pc,tests} under --work; it is then run against (a) the adapter's
suite tests/test_score_pc.py and (b) this reviewer's independent comparison (indep_score.py's checks, pointed at the
mutated pc_common).  Copies deleted after.  Report only.

    PY -I -B mutate_review.py --work DIR --log FILE
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ST = Path("/home/user/work/pc_study/tools")
HERE = Path(__file__).resolve().parent
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"

MUTANTS = [
    ("R1 line 4 at g >= 2 (ROUND's G+ >= 2 rule kept)", "pc_common.py", "    elif g == 3 and hs == 0:\n",
     "    elif g >= 2 and hs == 0:\n"),
    ("R2 PRESENCE rails not counted in r", "pc_common.py", "    r_total = r + len(presence_rails)\n",
     "    r_total = r\n"),
    ("R3 UNREAD not counted in u", "pc_common.py", "        elif c[\"cls\"] == \"UNREAD\":\n            u += 1\n",
     "        elif c[\"cls\"] == \"UNREAD\":\n            pass\n"),
    ("R4 line 3 (PARKED) before line 2 (INCOMPLETE)", "pc_common.py",
     "    elif u >= 1:\n        line, head = 2, LINE_HEAD[2].format(read=3 - u, u=u)\n    elif p >= 1:\n"
     "        line, head = 3, LINE_HEAD[3].format(p=p)\n",
     "    elif p >= 1:\n        line, head = 3, LINE_HEAD[3].format(p=p)\n    elif u >= 1:\n"
     "        line, head = 2, LINE_HEAD[2].format(read=3 - u, u=u)\n"),
    ("R5 DRIVE ONLY counted as SATURATING in hs", "pc_common.py", "        hs += h == \"SATURATING\"\n",
     "        hs += h in (\"SATURATING\", \"DRIVE ONLY\")\n"),
    ("R6 line 5 prints g, not 3 - g", "pc_common.py", "f\"{3 - g} GATE FAIL\" if 3 - g else None",
     "f\"{g} GATE FAIL\" if 3 - g else None"),
    ("R7 PRESENCE basis before ledger basis", "pc_common.py",
     "        if \"ledger\" in basis:\n            b = BASIS_LEDGER\n        elif presence_rails:\n            b = BASIS_LOST\n",
     "        if presence_rails:\n            b = BASIS_LOST\n        elif \"ledger\" in basis:\n            b = BASIS_LEDGER\n"),
    ("R8 T needs 1 episode", "pc_common.py",
     "t_at = [lab for lab, v in m1.items() if (v[\"over_bar_full_plus\"] + v[\"over_bar_full_tie\"]) >= 2",
     "t_at = [lab for lab, v in m1.items() if (v[\"over_bar_full_plus\"] + v[\"over_bar_full_tie\"]) >= 1"),
    ("R9 D at one J checkpoint", "pc_common.py", "            if len(hits) >= 2:\n", "            if len(hits) >= 1:\n"),
    ("R10 L without consecutiveness", "pc_common.py", "        own = [s for s in flagged if s - C.GRID in flagged]\n",
     "        own = list(flagged)\n"),
    ("R11 T read at R1/final only", "pc_common.py",
     "        t_at = [lab for lab, v in m1.items() if",
     "        t_at = [lab for lab, v in m1.items() if lab in (str(r1), \"final\") and"),
    ("R12 STOPPED COMPLETE run enters the test at its TD", "score_pc.py",
     "x.get(\"verdict\") == \"COMPLETE\" and x.get(\"finished\") and x.get(\"td\")",
     "x.get(\"verdict\") == \"COMPLETE\" and x.get(\"td\")"),
    ("R13 TD skips the first evaluation", "pc_common.py",
     "    used = [(s, abs(e[\"cmd_l\"])) for s, e in per.items() if e[\"cmd_l\"] is not None]\n",
     "    used = [(s, abs(e[\"cmd_l\"])) for s, e in per.items() if e[\"cmd_l\"] is not None and s > 131072]\n"),
    ("R14 one test on the largest PC value only", "pc_common.py",
     "    obs = math.fsum(vals[:3]) if all(math.isfinite(v) for v in vals[:3]) else math.inf\n",
     "    obs = math.fsum(vals[:3]) if any(math.isfinite(v) for v in vals[:3]) else math.inf\n"),
    ("R15 u counts only LOST (not never-launched/NOT FETCHED)", "pc_common.py",
     "        if d.get(\"verdict\") != \"COMPLETE\":\n            u += 1\n",
     "        if d.get(\"verdict\") != \"COMPLETE\":\n            u += d.get(\"verdict\") == \"LOST\"\n"),
    ("R16 line 4 ignores g", "pc_common.py", "    elif g == 3 and hs == 0:\n", "    elif hs == 0:\n"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    if work.exists():
        print(f"REFUSED: {work} exists")
        return 2
    log = []
    k_suite = k_mine = k_any = 0
    for i, (name, fname, old, new) in enumerate(MUTANTS, 1):
        d = work / f"r{i}"
        shutil.copytree(ST / "pc", d / "pc", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ST / "tests", d / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        f = d / "pc" / fname
        src = f.read_text()
        n = src.count(old)
        if n != 1:
            log.append(f"{name}: NOT APPLIED (old text occurs {n} times)")
            shutil.rmtree(d)
            continue
        f.write_text(src.replace(old, new))
        t0 = time.time()
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
        p = subprocess.run(["nice", "-n", "10", PY, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            str(d / "tests" / "test_score_pc.py"), f"--basetemp={d / 'tmp'}"],
                           capture_output=True, text=True, cwd=str(d), env=env)
        suite_killed = p.returncode != 0
        failed = [ln for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
        # the reviewer's independent comparison, against the mutated pc_common (score_pc mutants: reading-level only)
        q = subprocess.run(["nice", "-n", "10", PY, "-I", "-B", str(HERE / "indep_score.py"), "--pc", str(d / "pc"),
                            "--quick"], capture_output=True, text=True, cwd=str(d), env=env)
        mine_killed = q.returncode != 0
        tot = [ln for ln in q.stdout.splitlines() if ln.startswith("TOTAL")]
        k_suite += suite_killed
        k_mine += mine_killed
        k_any += suite_killed or mine_killed
        log.append(f"{name} ({fname}): suite {'KILLED' if suite_killed else 'SURVIVED'} "
                   f"({len(failed)} failed{': ' + failed[0].split('::')[-1] if failed else ''}); reviewer's comparison "
                   f"{'KILLED' if mine_killed else 'SURVIVED'} ({tot[0] if tot else q.stderr.strip()[-200:]}); "
                   f"{time.time() - t0:.1f} s")
        shutil.rmtree(d)
    log.append(f"suite killed {k_suite} of {len(MUTANTS)}; reviewer's comparison killed {k_mine}; either {k_any}")
    Path(a.log).write_text("\n".join(log) + "\n")
    print("\n".join(log))
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
