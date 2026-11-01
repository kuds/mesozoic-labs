"""D2(b)/P1 (VP0's evaluation-row clause): the evaluation callback's eval_episodes.csv rows at 131,072 for the same
actor with the switch ON (rb_on.py, s42, 17 rollouts, no update) and OFF (rollbuf.py, s42, 33 rollouts, no update):
both actors are the seed's initial actor (no optimizer step; A5 shows the ON and OFF rollouts equal), so every row
must be equal; the header must be equal too.  Columns whose name says wall-clock time would be set aside: the header
has none (time_to_toe_min_s is a simulated-time quantity and is compared).  usage: PY -I -B evalrow_compare.py RBDIR"""
import csv, hashlib, sys
from pathlib import Path
d = Path(sys.argv[1])
off_f, on_f = d / "scr_42_33/eval_episodes.csv", d / "scr_on_42_17/eval_episodes.csv"
h_off = next(csv.reader(open(off_f, newline=""))); h_on = next(csv.reader(open(on_f, newline="")))
off = [r for r in csv.DictReader(open(off_f, newline="")) if r["timesteps"] == "131072"]
on = [r for r in csv.DictReader(open(on_f, newline="")) if r["timesteps"] == "131072"]
print(f"header equal: {h_off == h_on} ({len(h_off)} columns)")
print(f"rows at 131,072: OFF {len(off)}, ON {len(on)}")
diff = [(i, c, a[c], b[c]) for i, (a, b) in enumerate(zip(off, on)) for c in h_off if a[c] != b[c]]
print(f"differing cells: {len(diff)}", diff[:10])
lines_off = [l for l in open(off_f, newline="").read().splitlines() if l.split(",")[1:2] == ["131072"]]
lines_on = [l for l in open(on_f, newline="").read().splitlines() if l.split(",")[1:2] == ["131072"]]
print(f"raw 131,072 lines byte-equal: {lines_off == lines_on} (sha256 OFF {hashlib.sha256(chr(10).join(lines_off).encode()).hexdigest()[:16]}, ON {hashlib.sha256(chr(10).join(lines_on).encode()).hexdigest()[:16]})")
ok = h_off == h_on and len(off) == len(on) == 10 and not diff
print("RESULT", "EVAL ROWS EQUAL ON/OFF (same actor)" if ok else "EVAL ROWS DIFFER")
