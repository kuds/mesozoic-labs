"""Fixer: write ST/review/FIXES.txt from FIXES_body.txt, checks.txt and verdicts.txt (fills @SUITES@ from
suites/compare.txt and @EVIDENCE@ from the evidence files' sha256)."""
import hashlib
from pathlib import Path
F = Path("/home/user/work/pc_study/review/fixer")
cmp = (F / "suites/compare.txt").read_text()
import ast, re
_d = ast.literal_eval(next(l for l in cmp.splitlines() if l.startswith("totals:"))[len("totals: "):])
_c = lambda v: re.sub(r"^=+ | in [0-9.]+s.*$", "", v).replace(", 3 warnings", "")
tot = f"totals: shared {_c(_d['shared'])}; T. rex {_c(_d['trex'])}"
res = next(l for l in cmp.splitlines() if l.startswith("RESULT"))
nk = next(l for l in cmp.splitlines() if l.startswith("failed and not known"))
suites = f"{tot.replace('totals: ', '')}; {res.replace('RESULT ', '')} ({nk}); fixer/suites/compare.txt, failed_ids.txt"
ev = ["suites/compare.txt", "suites/failed_ids.txt", "suites/shared/pytest.log", "suites/trex/pytest.log",
      "rb/a5_compare.txt", "rb/a6_compare.txt", "rb/evalrows_compare.txt", "equiv/compare_base_vs_head_off.txt",
      "vp2d1/vp2_part_a.out", "vp2d1/vp2_record.json", "vp2d1/d1.out", "lint/mypy_nosb3_fixed.txt",
      "/home/user/work/pc_study/tools/dry/vp3.out"]
evid = "\n" + "\n".join(f"    {hashlib.sha256((F / p).read_bytes()).hexdigest()}  {p}" for p in ev)
checks = (F / "checks.txt").read_text().replace("@SUITES@", suites).replace("@EVIDENCE@", evid)
body = (F / "FIXES_body.txt").read_text().replace("@CHECKS@\n", checks).replace("@VERDICTS@\n", (F / "verdicts.txt").read_text())
assert "@" not in body.replace("@anthropic", ""), "unfilled placeholder"
(Path("/home/user/work/pc_study/review/FIXES.txt")).write_text(body)
print(hashlib.sha256(body.encode()).hexdigest())
