"""VP0 / P2: the fixer's suites (PYTHONPATH = the repo) against ROUND's known 25 ids (known25.txt 62e01b73...).
usage: PY -I -B suite_compare.py"""
import re
from pathlib import Path
F = Path("/home/user/work/pc_study/review/fixer/suites")
K = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/runbranch/fixer/logs/known25.txt")
known = sorted(x.strip() for x in K.read_text().splitlines() if x.strip())
failed, totals = set(), {}
for name in ("shared", "trex"):
    log = (F / name / "pytest.log").read_text(errors="replace")
    for m in re.finditer(r"^(FAILED|ERROR) (\S+::\S+)", log, re.M):
        failed.add(m.group(2))
    t = [ln for ln in log.splitlines() if re.match(r"^=+ .*(passed|failed).* in [0-9.]+s", ln)]
    totals[name] = t[-1] if t else "(no summary line)"
failed = sorted(failed)
print("totals:", totals)
print(f"failed ids: {len(failed)}; known25: {len(known)}")
print("failed and not known:", [x for x in failed if x not in known])
print("known and not failed:", [x for x in known if x not in failed])
extra3 = [x for x in failed if "test_refusals_come_before_the_run" in x]
print("the 3 digest_snapshot subprocess tests among the failures:", extra3)
print("RESULT", "EXACTLY THE KNOWN 25" if failed == known else "NOT EXACTLY THE KNOWN 25")
(F / "failed_ids.txt").write_text("\n".join(failed) + "\n")
