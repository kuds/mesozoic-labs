"""LB design input: are the proposed continuation seeds new?  Read only; prints; writes nothing.

Each continuation seed s goes to the model and training env 0 (s), envs 1-3 (s + 1 ... s + 3) and the evaluation env
(s + 1000) (LB/tools/bs_common.derived_seeds, read here).  The used set is built from the studies' own files where they
exist (the ECS's ecs_common.SEEDS and derived_seeds, BS's and BS2's SCREEN.json, LB's bs_common.VALIDATION_SEEDS and the
reader's seed constants) and from the seed ranges the notes and the run log name (listed below with their sources).
Then a text search over S1 and S2 (text-like files under 2 MiB, this folder excluded) for every derived number of the
candidates, as a whole number (not part of a longer number or a decimal), printed for triage.
Usage: PY -I -B seeds_check.py
"""
import importlib.util
import json
import os
import re
from pathlib import Path

CW = Path(__file__).resolve().parent
S2 = CW.parents[2]
S1 = S2.parent.parent / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad"
LB = S2 / "r10round/long_blind"


def short(p):
    return str(p).replace(str(S1), "S1").replace(str(S2), "S2")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lbc = load("lb_common_ro", LB / "tools/bs_common.py")
ecs = load("ecs_common_ro", S2 / "r10round/ecs/tools/ecs_common.py")
derived = lbc.derived_seeds
print(f"LB derived_seeds(2701) = {derived(2701)} (bs_common.N_ENVS {lbc.N_ENVS}, EVAL_SEED_OFFSET {lbc.EVAL_SEED_OFFSET})")
used = {}


def add(name, seeds):
    for s in seeds:
        used.setdefault(int(s), set()).add(name)


for (start, arm), seeds in ecs.SEEDS.items():
    for s in seeds:
        add(f"ECS {start}_{arm}_{s} (derived)", ecs.derived_seeds(s))
for nm, rel in (("BS", "r10round/bias_screen/SCREEN.json"), ("BS2", "r10round/bias_screen2/SCREEN.json")):
    for arm, s in json.loads((S2 / rel).read_text())["queue"]:
        add(f"{nm} {arm}_{s} (derived)", derived(s))
for nm, base in (("BS validation", (9101, 9111, 9121, 9131, 9141)), ("BS2 validation", (9201, 9211, 9221, 9231, 9241)),
                 ("LB validation (bs_common.VALIDATION_SEEDS)", lbc.VALIDATION_SEEDS)):
    for s in base:
        add(f"{nm} {s} (derived)", derived(s))
named = [
    ("M1 traces / M3 certificate / M2 panel / K3 panel / clamp / warning margin, 3042-3081 (round criteria lines "
     "119-128)", range(3042, 3082)),
    ("reader OWN 5001-5010 (read_bs.OWN_SEEDS; early warning)", range(5001, 5011)),
    ("reader on-policy 9301-9302 (read_bs.ONPOLICY_SEEDS); BS2 criteria's 9301-9305", range(9301, 9306)),
    ("early-bias probe 9101-9133", range(9101, 9134)),
    ("verify_early_bias 7001-7216, 7501-7502", list(range(7001, 7217)) + [7501, 7502]),
    ("termination probe 7001-7128, 6001-6002 (its NOTES/run lines)", list(range(7001, 7129)) + [6001, 6002]),
    ("phase probe 7700 (+ envs 7701-7703), 12345 (its NOTES/run lines)", [7700, 7701, 7702, 7703, 12345]),
    ("training runs' seeds (r9/r10: 42, 52, 60, 70, 80)", (42, 52, 60, 70, 80)),
]
for nm, seeds in named:
    add(nm, seeds)
print(f"used numbers: {len(used)}")

CANDIDATES = {"PROPOSED 2701 / 2711 / 2721": (2701, 2711, 2721), "alternative 2801 / 2811 / 2821": (2801, 2811, 2821)}
for label, seeds in CANDIDATES.items():
    print(f"\n{label}:")
    for s in seeds:
        d = derived(s)
        hits = {x: sorted(used[x]) for x in d if x in used}
        print(f"  {s}: derived {d} -> collisions with the named sets {hits if hits else 'none'}; validation seed "
              f"{s in lbc.VALIDATION_SEEDS}")

EXT = {".py", ".txt", ".json", ".md", ".sh", ".toml", ".log", ".out", ".jsonl", ".tsv", ".csv", ".js", ".yaml", ".yml",
       ".cfg", ".ini", ".err"}
nums = sorted({x for seeds in CANDIDATES.values() for s in seeds for x in derived(s)})
pat = re.compile(r"(?<![\d.])(" + "|".join(str(n) for n in nums) + r")(?![\d]|\.\d)")
hits, scanned, skipped = {}, 0, 0
for root in (S1, S2):
    for dp, dn, fn in os.walk(root):
        if str(dp).startswith(str(CW)):
            dn[:] = []
            continue
        dn[:] = [d for d in dn if d not in (".git", "venv", "__pycache__", "node_modules", "caches")]
        for f in fn:
            p = Path(dp) / f
            if p.suffix not in EXT:
                continue
            try:
                if p.stat().st_size > 2 * 1024 * 1024 or p.is_symlink():
                    skipped += 1
                    continue
                txt = p.read_text(errors="replace")
            except OSError:
                skipped += 1
                continue
            scanned += 1
            for ln, line in enumerate(txt.splitlines(), 1):
                for m in pat.finditer(line):
                    hits.setdefault(int(m.group(1)), []).append((short(p), ln, line.strip()[:150]))
print(f"\ntext search: {scanned} files scanned, {skipped} skipped (> 2 MiB, symlink or unreadable)")
for n in nums:
    h = hits.get(n, [])
    print(f"  {n}: {len(h)} hit(s)")
    for x in h[:6]:
        print(f"      {x[0]}:{x[1]}: {x[2]}")
