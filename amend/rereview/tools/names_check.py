"""Re-reviewer's own collision check of the six RUN_IDs (and their _b) and labels: RUNLOG, the stance6 Drive listing
(folders.json), every name pair for prefix relations, and the tools' token pattern / arm_of."""
import json
import sys
from pathlib import Path

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
RL = S1 / "foot/stance4/live/run_log.txt"
FJ = S1 / "foot/stance6/fetch/folders.json"
sys.path.insert(0, "/home/user/work/pc_study/tools/pc")
import r10round_common as C  # noqa: E402

phase = [f"stance7_r10LTSBpc_s{s}" for s in (60, 42, 80)]
off = [f"stance7_r10LTSBoff_s{s}" for s in (60, 42, 80)]
s6 = [f"stance6_r10LTSB_s{s}" for s in (60, 42, 80)]
mine = phase + off + [x + "_b" for x in phase + off]
labels = [f"r10 LTSB switch-off s{s}" for s in (60, 42, 80)] + [f"r10 LTSB phase-critic s{s}" for s in (60, 42, 80)]
lines = RL.read_text(encoding="utf-8", errors="replace").splitlines()
print(f"RUNLOG {len(lines)} lines")
for tok in ("LTSBoff", "stance7_", "switch-off s", "r10LTSBoff"):
    hits = [i for i, ln in enumerate(lines, 1) if tok in ln]
    print(f"  {tok!r}: lines {hits}")
for lab in labels:
    hits = [i for i, ln in enumerate(lines, 1) if lab in ln]
    print(f"  label {lab!r}: lines {hits}")
fj = json.loads(FJ.read_text())


def names(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "name" and isinstance(v, str):
                yield v
            else:
                yield from names(v)
    elif isinstance(o, list):
        for v in o:
            yield from names(v)


folder_names = sorted(set(fj["_ppo_listing"]["folders"]) | {f for x in fj.get("_ppo_listings", [])
                                                            for f in x.get("folders", [])} | set(names(fj)))
print("last complete listing:", fj["_ppo_listing"]["utc"], fj["_ppo_listing"]["complete"])
print(f"folders.json: {len(folder_names)} names")
rel = [(a, b) for a in mine for b in folder_names if a == b or a.startswith(b) or b.startswith(a)]
print("  equal or prefix-related to a listed folder:", rel)
print("  any listed name with 'stance7' or 'LTSBoff':", [n for n in folder_names if "stance7" in n or "LTSBoff" in n])
allnames = phase + off + s6
pairs = [(a, b) for a in allnames for b in allnames if a != b and (a.startswith(b) or b.startswith(a))]
print("prefix relation among the 9 round/stance6 names:", pairs)
print("token pattern reads:", {n: bool(C.RUN_TOKEN_RE.fullmatch(n)) for n in mine})
print("arm_of:", {n: C.arm_of(n) for n in mine + s6 + ["stance7_r10LTSBoff_s6", "stance7_r10LTSBpc_s600"]})
print("RUN_IDS:", C.RUN_IDS)
