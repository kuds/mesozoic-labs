"""Fixer: criteria.txt edit, batch 4 (the disk sentence: two measurements, the shortfall left to step 4(d))."""
from pathlib import Path

p = Path("/home/user/work/phase_screen/criteria.txt")
s = p.read_text()
old = """  [measured, df -h /, WORK/ADAPT_NOTES.txt]: BELOW the launch rule; 1,597,640,704 B at 15:30 UTC, below even the
  per-start floor of 1.5 GiB [measured, df -B1 /, <PS>/review/fixer/notes.txt] (the fixer's re-validation under
  <PS>/validate/fixer holds 112.6 MB, removable after the re-review; other lanes' folders grew meanwhile): about
  1.2 GB must be freed for the 2.6 GiB rule (process review P-m1)."""
new = """  [measured, df -h /, WORK/ADAPT_NOTES.txt]: BELOW the launch rule; 1,597,640,704 B at 15:30 UTC (below even the
  per-start floor of 1.5 GiB) and 2,086,477,824 B at 15:33 UTC, as other lanes' folders came and went [measured,
  df -B1 /, <PS>/review/fixer/notes.txt]; the fixer's re-validation under <PS>/validate/fixer holds 112.6 MB,
  removable after the re-review. The shortfall to the 2.6 GiB rule (2,791,728,742 B) is measured again at step 4(d)
  and freed first (process review P-m1)."""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("applied 1")
