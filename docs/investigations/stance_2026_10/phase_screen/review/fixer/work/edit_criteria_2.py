"""Fixer: criteria.txt edits, batch 2 (wording touch-ups of batch 1)."""
from pathlib import Path

p = Path("/home/user/work/phase_screen/criteria.txt")
s = p.read_text()
R = []
R.append(("""  the reader's <PS>/tmp/read_<name>/; reads/<name>; logs/<name>.attempt<k>.log; the TensorBoard events file under
  <PS>/tmp/<name>/tb/, fixer P-B1) except: run_meta.json's fields status,""",
"""  the reader's <PS>/tmp/read_<name>/; reads/<name>; logs/<name>.attempt<k>.log; its TensorBoard events file under
  <PS>/tmp/<name>/tb/, fixer P-B1) except: run_meta.json's fields status,"""))
R.append(("""- The critic's error by episode step (#655 Open Question 1, lines 953-975; the PR-655 review's""",
"""- The critic's error by episode step (#655 section 9, Open Question 1, lines 966-971; the PR-655 review's"""))
for old, new in R:
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)
p.write_text(s)
print("applied", len(R))
