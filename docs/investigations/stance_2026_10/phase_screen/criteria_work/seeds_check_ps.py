"""PS design input: are the proposed continuation seeds new?  Read only; prints; writes nothing but stdout.

Adapted from LB's seeds_check.py (S2/r10round/long_blind/criteria_work/seeds_check.py, read only).  Each continuation
seed s goes to the model and training env 0 (s), envs 1-3 (s + 1 ... s + 3) and the evaluation env (s + 1000)
(BS2/tools/bs_common.derived_seeds, read here; PS reuses BS2's harness).  The used set is built from the studies' own
files where they exist (the ECS's ecs_common.SEEDS and derived_seeds, BS's and BS2's SCREEN.json, LB's draft
SCREEN_v1.json, the validation seeds of BS, BS2, LB and PS (PS/tools/bs_common.VALIDATION_SEEDS if the builder has
written it; else the brief's 9401-9441), the reader's seed constants) and from the seed ranges the notes and the run log
name (listed below with their sources).  Then a text search over S1, S2 and /home/user/work (text-like files under
2 MiB, this folder excluded) for every derived number of the candidates as a whole number (not part of a longer number
or a decimal), printed for triage.
Usage: cd /tmp && nice -n 10 PY -I -B seeds_check_ps.py > seeds_check_ps.out 2> seeds_check_ps.err
"""
import importlib.util
import json
import os
import re
from pathlib import Path

CW = Path(__file__).resolve().parent
PS = CW.parent
WORK = PS.parent
import sys as _sys
# S2 is the venv owner: PY = S2/venv/bin/python (run with PY -I -B); S1 is its sibling session scratchpad.
S2 = Path(_sys.executable).parents[2]
S1 = S2.parent.parent / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad"
assert (S2 / "r10round/bias_screen2").is_dir() and (S1 / "foot/stance6").is_dir(), "run with S2/venv/bin/python"


def clean(s):
    """Printed text without the scratchpads' temp-root prefix (short names instead)."""
    return s.replace(str(S2), "S2").replace(str(S1), "S1").replace(str(S2.parent.parent) + "/", "<TMPROOT>/")
LB = S2 / "r10round/long_blind"
BS2 = S2 / "r10round/bias_screen2"


def short(p):
    s = str(p)
    for k, v in ((str(S1), "S1"), (str(S2), "S2"), (str(WORK), "W")):
        s = s.replace(k, v)
    return s


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs2c = load("bs2_common_ro", BS2 / "tools/bs_common.py")
lbc = load("lb_common_ro", LB / "tools/bs_common.py")
ecs = load("ecs_common_ro", S2 / "r10round/ecs/tools/ecs_common.py")
derived = bs2c.derived_seeds
print(f"BS2 derived_seeds(2601) = {derived(2601)} (bs_common.N_ENVS {bs2c.N_ENVS}, EVAL_SEED_OFFSET "
      f"{bs2c.EVAL_SEED_OFFSET}); LB's derived_seeds equal: {lbc.derived_seeds(2601) == derived(2601)}")
ps_common = PS / "tools/bs_common.py"
ps_val = None
if ps_common.is_file():
    try:
        psc = load("ps_common_ro", ps_common)
        ps_val = tuple(getattr(psc, "VALIDATION_SEEDS", ()) or ())
        print(f"PS/tools/bs_common.py present: VALIDATION_SEEDS {ps_val}; derived_seeds equal: "
              f"{psc.derived_seeds(2601) == derived(2601)}")
    except Exception as exc:  # noqa: BLE001 - the builder's file may be mid-edit; fall back to the brief
        print(f"PS/tools/bs_common.py present but not importable now ({type(exc).__name__}: {exc}); using the brief's")
if not ps_val:
    ps_val = (9401, 9411, 9421, 9431, 9441)
    print(f"PS validation seeds taken from the build brief (9401-9441): {ps_val}")
used = {}


def add(name, seeds):
    for s in seeds:
        used.setdefault(int(s), set()).add(name)


for (start, arm), seeds in ecs.SEEDS.items():
    for s in seeds:
        add(f"ECS {start}_{arm}_{s} (derived)", ecs.derived_seeds(s))
for nm, path in (("BS", S2 / "r10round/bias_screen/SCREEN.json"), ("BS2", BS2 / "SCREEN.json"),
                 ("LB draft", LB / "criteria_work/SCREEN_v1.json")):
    for arm, s in json.loads(path.read_text())["queue"]:
        add(f"{nm} {arm}_{s} (derived)", derived(s))
for nm, base in (("BS validation", (9101, 9111, 9121, 9131, 9141)), ("BS2 validation", (9201, 9211, 9221, 9231, 9241)),
                 ("LB validation (bs_common.VALIDATION_SEEDS)", lbc.VALIDATION_SEEDS),
                 ("PS validation", ps_val), ("LB's named alternative 2801 / 2811 / 2821 (reserved)", (2801, 2811, 2821))):
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
    ("verify_why FR1-FR3 8810, 8820, 8830 (+ envs + 1..3), smoke 9990 (its NOTES)",
     [8810 + i for i in range(4)] + [8820 + i for i in range(4)] + [8830 + i for i in range(4)] + [9990]),
    ("training runs' seeds (r9/r10: 42, 52, 60, 70, 80)", (42, 52, 60, 70, 80)),
]
for nm, seeds in named:
    add(nm, seeds)
print(f"used numbers: {len(used)}")

CANDIDATES = {"PROPOSED 2601 / 2611 / 2621 / 2631": (2601, 2611, 2621, 2631),
              "alternative 2901 / 2911 / 2921 / 2931": (2901, 2911, 2921, 2931)}
for label, seeds in CANDIDATES.items():
    print(f"\n{label}:")
    for s in seeds:
        d = derived(s)
        hits = {x: sorted(used[x]) for x in d if x in used}
        print(f"  {s}: derived {d} -> collisions with the named sets {hits if hits else 'none'}; a validation seed "
              f"{s in ps_val or s in lbc.VALIDATION_SEEDS}")

EXT = {".py", ".txt", ".json", ".md", ".sh", ".toml", ".log", ".out", ".jsonl", ".tsv", ".csv", ".js", ".yaml", ".yml",
       ".cfg", ".ini", ".err"}
nums = sorted({x for seeds in CANDIDATES.values() for s in seeds for x in derived(s)})
pat = re.compile(r"(?<![\d.])(" + "|".join(str(n) for n in nums) + r")(?![\d]|\.\d)")
hits, scanned, skipped = {}, 0, 0
for root in (S1, S2, WORK):
    for dp, dn, fn in os.walk(root):
        if str(dp).startswith(str(CW)):
            dn[:] = []
            continue
        dn[:] = [d for d in dn if d not in (".git", "venv", "__pycache__", "node_modules", "caches", "site-packages")]
        for f in fn:
            p = Path(dp) / f
            if p.suffix not in EXT:
                continue
            try:
                if p.is_symlink() or p.stat().st_size > 2 * 1024 * 1024:
                    skipped += 1
                    continue
                txt = p.read_text(errors="replace")
            except OSError:
                skipped += 1
                continue
            scanned += 1
            for ln, line in enumerate(txt.splitlines(), 1):
                for m in pat.finditer(line):
                    hits.setdefault(int(m.group(1)), []).append((short(p), ln, clean(line.strip())[:160]))
print(f"\ntext search: {scanned} files scanned, {skipped} skipped (> 2 MiB, symlink or unreadable); site-packages, "
      "venv, .git, caches excluded")
SEEDLIKE = re.compile(r"seed|SEED|_\d{4}\b|\bs\d{4}\b|arm|queue|control_|critic_|blind_|phase_")
for n in nums:
    h = hits.get(n, [])
    k = [x for x in h if SEEDLIKE.search(x[2])]
    print(f"  {n}: {len(h)} hit(s), {len(k)} on a line naming a seed, an arm or a run")
    for x in k[:8]:
        print(f"      {x[0]}:{x[1]}: {x[2]}")
print("\nEND")
