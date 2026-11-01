"""Amendment: do the switch-off RUN_IDs collide with anything?  Read-only.  Checks (1) RUNLOG (every line), (2) the
stance6 round's ppo folder listing (S1/foot/stance6/fetch/folders.json: every recorded folder name and every
_ppo_listing(s) folder list), (3) the round's names (the phase RUN_IDs, their _b, the stance6 RUN_IDs) for equality and
for prefix relations either way (the tools read "<RUN_ID>" and "<RUN_ID>_<suffix>" as the run's: of_round, base_seed),
(4) the tools' RUNLOG token pattern on the phase names (no phase name is read as an off name or back).
    PY -I -B names_check_off.py
"""
import json
import re
from pathlib import Path

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
RUNLOG = S1 / "foot/stance4/live/run_log.txt"
FOLDERS = S1 / "foot/stance6/fetch/folders.json"
SEEDS = (60, 42, 80)
OFF = [f"stance7_r10LTSBoff_s{s}" for s in SEEDS]
OFF_ALL = OFF + [x + "_b" for x in OFF]
PC = [f"stance7_r10LTSBpc_s{s}" for s in SEEDS]
PC_ALL = PC + [x + "_b" for x in PC]
S6 = [f"stance6_r10LTSB_s{s}" for s in SEEDS]
lines = RUNLOG.read_text(encoding="utf-8", errors="replace").splitlines()
print(f"RUNLOG {RUNLOG}: {len(lines)} lines")
for pat in ("stance7_r10LTSBoff", "r10LTSBoff", "LTSBoff", "switch-off s"):
    hits = [i for i, ln in enumerate(lines, 1) if pat in ln]
    print(f"  lines containing {pat!r}: {len(hits)} {hits}")
NOTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})? UTC\s+STANCE7\s")
for i in [i for i, ln in enumerate(lines, 1) if "LTSBoff" in ln]:
    print(f"    line {i}: STANCE7 note form: {bool(NOTE_RE.match(lines[i - 1]))}; {lines[i - 1][:150]}")
d = json.loads(FOLDERS.read_text(encoding="utf-8"))
names = set(k for k in d if not k.startswith("_") and k != "ppo_folder_id")
for key in ("_ppo_listing",):
    names |= set((d.get(key) or {}).get("folders") or [])
for x in d.get("_ppo_listings") or []:
    names |= set(x.get("folders") or [])
last = d.get("_ppo_listing") or {}
print(f"folders.json {FOLDERS}: {len(names)} distinct folder names; last listing utc {last.get('utc')} complete {last.get('complete')}")
for n in OFF_ALL:
    eq = n in names
    pre = sorted(m for m in names if m.startswith(n) or n.startswith(m))
    print(f"  {n}: equal to a listed folder {eq}; prefix relation with listed folders {pre}")
print("  listed folders containing 'LTSBoff' or 'stance7':", sorted(m for m in names if "LTSBoff" in m or "stance7" in m))
others = PC_ALL + S6
for n in OFF_ALL:
    rel = [m for m in others if m == n or m.startswith(n + "_") or n.startswith(m + "_") or m.startswith(n) or n.startswith(m)]
    print(f"  {n} vs the round's names (phase RUN_IDs and _b, stance6 RUN_IDs): equal/prefix {rel}")
tok = re.compile(r"stance7_r10LTSB(?:pc|off)_s\d+\w*")
print("  token pattern stance7_r10LTSB(?:pc|off)_s\\d+\\w* reads:", {n: tok.fullmatch(n) is not None for n in OFF_ALL + PC_ALL})
print("  arm by prefix (pc / off):", {n: ("pc" if n.startswith("stance7_r10LTSBpc_s") else "off" if n.startswith("stance7_r10LTSBoff_s") else None) for n in OFF_ALL + PC_ALL})
labels = [f"r10 LTSB switch-off s{s}" for s in SEEDS]
print("  labels", labels, "in RUNLOG:", [sum(1 for ln in lines if lab in ln) for lab in labels])
