"""Fixer's mutation check of score_bs.py v1.1 against tools/tests/test_score_bs.py.  Each mutant: a throw-away copy of
<BS>/tools under <BS>/review/fixer/mutants/<id>, one exact-once string replacement in score_bs.py, pytest on the copy's
test file, the copy removed.  A mutant is KILLED when pytest fails.

    cd /tmp && PY -I -B <BS>/review/fixer/fx_mutants.py <BS>
"""
import shutil
import subprocess
import sys
from pathlib import Path

BS = Path(sys.argv[1]).resolve()
WORK = BS / "review" / "fixer" / "mutants"
PY = sys.executable
M = [
    ("R6 markers tool unchecked", 'if mk is not None and mk.get("tool_sha256") != read_pin:', "if False:"),
    ("R2 addition dropped", "        additions.append(ADDITION_AS_COUNTED)", "        pass"),
    ("R2 addition old wording", 'ADDITION_AS_COUNTED = "(the as-counted p, report only, is <= 0.05; the filled p decides)"',
     'ADDITION_AS_COUNTED = "(REDUCED holds only without the missing continuations)"'),
    ("R2 line-3 text old wording", '(not excluded: a reduction of about half the rise or smaller (power about 0.3-0.9 "',
     '(not tested: a reduction smaller than about half the rise (power about 0.3-0.9 "'),
    ("R2 SD never computed", "if len(counted[arm]) >= 2 else None)", "if False else None)"),
    ("R2 SD not printed", "f\"{fnum(dec['sds'][arm], '.5f')} (report only)\")", "f\"n/a (report only)\")"),
    ("ties not counted", "count = sum(1 for s in sums.values() if s >= obs)",
     "count = sum(1 for s in sums.values() if s > obs)"),
    ("fill reversed", "INF_CONTROL, INF_BLIND = (-1, Fraction(0)), (1, Fraction(0))",
     "INF_CONTROL, INF_BLIND = (1, Fraction(0)), (-1, Fraction(0))"),
    ("alpha 0.1", "ALPHA = Fraction(1, 20)", "ALPHA = Fraction(1, 10)"),
    ("rise strict", "informative = rise is not None and rise >= RISE_MIN",
     "informative = rise is not None and rise > RISE_MIN"),
    ("decide on as-counted", "    elif filled[0] <= ALPHA:\n",
     "    elif as_counted is not None and as_counted[0] <= ALPHA:\n"),
    ("line order", "    if not informative:\n        line = 1\n    elif filled[0] <= ALPHA:\n        line = 2\n",
     "    if filled[0] <= ALPHA:\n        line = 2\n    elif not informative:\n        line = 1\n"),
    ("primary rows shifted", "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))",
     "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K + 1, 1, -1))"),
    ("STOPPED counted", '    if c["cls"] != COUNTED:\n        return None, c.get("cause") or c["cls"]',
     '    if c["cls"] not in (COUNTED, STOPPED):\n        return None, c.get("cause") or c["cls"]'),
    ("blind slope unchecked", "                if eff not in (None, 0.0):", "                if False:"),
    ("median -> mean", "med = {arm: (statistics.median(counted[arm])", "med = {arm: (statistics.mean(counted[arm])"),
    ("incomplete reads markers", "    if undecided or unread:\n", "    if undecided:\n"),
    ("baseline wrong", "BASELINE_S = -0.03157372563146055", "BASELINE_S = -0.03324194939341396"),
    ("probe sha link dropped", "len(src) != 2 or probe.get(\"sha256\") != src[1]:", "len(src) != 2:"),
    ("own-sha pin unchecked", "        if own not in text:", "        if False:"),
]
WORK.mkdir(parents=True, exist_ok=True)
rows = []
for i, (label, old, new) in enumerate(M, 1):
    d = WORK / f"m{i:02d}"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(BS / "tools", d / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    p = d / "tools" / "score_bs.py"
    src = p.read_text()
    n = src.count(old)
    if n != 1:
        rows.append(f"M{i:02d} {label}: NOT APPLIED (pattern found {n} times)")
        shutil.rmtree(d)
        continue
    p.write_text(src.replace(old, new))
    r = subprocess.run([PY, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={d / 'pt'}",
                        str(d / "tools" / "tests" / "test_score_bs.py")], capture_output=True, text=True, cwd="/tmp",
                       env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    rows.append(f"M{i:02d} {label}: {'KILLED' if r.returncode != 0 else 'SURVIVED'} ({last})")
    shutil.rmtree(d)
shutil.rmtree(WORK)
print("\n".join(rows))
print(f"{sum('KILLED' in r for r in rows)} of {len(rows)} killed")
