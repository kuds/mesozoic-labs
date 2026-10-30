"""Dry run of score_bs.py: a synthetic screen root in which EVERY queue entry (4 control, 4 critic_blind_tail3) carries
the PARENT's own markers (reads/parent_s42/markers.json, read_bs.py --reference over the same window).  Changed per
entry: only the identity fields (label, run, kind, arm, seed, stop_reason) and, for the critic_blind_tail3 entries, the
loop rows' critic form ("blind+raw") and effective critic slope (0.0), which read_bs.py computes that way for a blind
run by construction.  The probe rows (the primary's input), the ledger and every other marker are the parent's, byte for
byte in value.  check.json is synthetic (a passing CHECK naming the parent probe's sha256); run_meta.json is a stand-in
(the V2 final run_meta of the same arm, 31 no-update rollouts).  Nothing here is a continuation.
    PY -I -B dryrun_build.py <BS> <dry root>"""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

BS, DRY = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(BS / "tools"))
import bs_common as C  # noqa: E402

assert str(DRY.resolve()).startswith(str(BS.resolve()) + "/"), "the dry root must be under <BS>"
if DRY.exists():
    shutil.rmtree(DRY)
DRY.mkdir(parents=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


shutil.copyfile(BS / "SCREEN.json", DRY / "SCREEN.json")
parent = json.loads((BS / "reads" / "parent_s42" / "markers.json").read_text())
d, problems = C.load_screen(str(DRY / "SCREEN.json"))
assert not problems
read_sha = sha(BS / "tools" / "read_bs.py")
for arm, seed in C.queue_of(d):
    name = C.run_name(arm, seed)
    rd = DRY / "runs" / name
    rd.mkdir(parents=True)
    meta = rd / "run_meta.json"
    shutil.copyfile(BS / "validate" / "v2final" / f"{arm}_9101" / "run_meta.json", meta)
    check = {"schema": "bs-check/1", "version": "dry run (synthetic)", "tool_sha256": read_sha, "run": str(rd),
             "name": name, "mode": "continuation", "passed": True, "failures": [], "items": {},
             "sources": {"run_meta.json": [str(meta), sha(meta)],
                         "policy_probe.npz": [parent["probe"]["file"], parent["probe"]["sha256"]]},
             "arm": arm, "seed": seed, "S": C.S, "E": C.E, "T": C.E, "end": C.E, "stop_reason": "end"}
    (rd / "check.json").write_text(json.dumps(check, indent=1) + "\n")
    m = copy.deepcopy(parent)
    m.update(label=name, run=str(rd), kind="continuation", arm=arm, seed=seed, stop_reason="end")
    if arm == "critic_blind_tail3":
        for row in m["loop"]:
            row["critic_form"] = "blind+raw"
            for states in row["critic_slope_obs6"].values():
                states["effective"] = dict(states["raw"], dV_dq3_per_deg=0.0, dV_dq3_per_deg_sd=0.0)
    out = DRY / "reads" / name
    out.mkdir(parents=True)
    (out / "markers.json").write_text(json.dumps(m, indent=1) + "\n")
    print(name, "probe sha", m["probe"]["sha256"][:16], "rows", m["probe"]["n_rows"])
