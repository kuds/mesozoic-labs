"""Triage of WORK/seeds_grep.out (grep -w hits of the 15 candidate numbers): keep a hit only when the matched line's
CONTENT (not its path) names a seed (the word 'seed' in the content) or a run name ending _<number>.  Prints the kept
hits and a per-file count of the dropped ones.  Read only.  Usage: PY -I -B seeds_triage.py WORK/seeds_grep.out"""
import re
import sys
from collections import Counter

NUMS = "2701|2702|2703|2704|3701|2711|2712|2713|2714|3711|2721|2722|2723|2724|3721"
tok = re.compile(rf"(?<![0-9A-Za-z_.])({NUMS})(?![0-9A-Za-z_]|\.[0-9])")
run = re.compile(rf"_({NUMS})(?![0-9A-Za-z])")
kept, dropped = [], Counter()
for line in open(sys.argv[1], errors="replace"):
    parts = line.rstrip("\n").split(":", 2)
    if len(parts) < 3:
        continue
    path, ln, content = parts
    if not tok.search(content) and not run.search(content):
        dropped[path] += 1
        continue
    if re.search(r"seed", content, re.I) or run.search(content):
        kept.append((path, ln, content.strip()[:200]))
    else:
        dropped[path] += 1
print(f"kept {len(kept)} hit(s) whose content names a seed or a run _<number>; dropped {sum(dropped.values())}")
for p, ln, c in kept:
    print(f"  {p}:{ln}: {c}")
