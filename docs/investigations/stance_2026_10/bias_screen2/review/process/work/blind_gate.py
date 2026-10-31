"""Process review of BS2: read_bs.full_read's outcome-blind gate on a synthetic COPY root (argv[1]; never BS2).
One continuation folder with a passing, current check.json and a GARBAGE policy_probe.npz: while the other 7 queue
entries are undecided the read must refuse with the outcome-blind message (never reaching the probe); once all 8 have
passing check.json files it must get past the gate (and then fail on the garbage probe).  Writes only under argv[1]."""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
assert ROOT.name == "copyroot" and "review/process/work" in str(ROOT)
spec = importlib.util.spec_from_file_location("bs_common", str(ROOT / "tools" / "bs_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
sys.modules["bs_common"] = C
C.BS = str(ROOT)
C.READS = str(ROOT / "reads")
spec = importlib.util.spec_from_file_location("read_bs", str(ROOT / "tools" / "read_bs.py"))
RB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RB)
assert RB.C is C


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fake_run(name, arm, seed):
    run = ROOT / "runs" / name
    (run / "01_stance").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"arm": arm, "seed": seed, "T": C.E, "stop_reason": "end"}))
    (run / C.EVIDENCE).write_text("{}\n")
    (run / RB.PROBE_REL).write_bytes(b"GARBAGE, not an npz: an outcome file must never be reached before the gate\n")
    check = {"passed": True, "mode": "continuation", "sources": {
        "run_meta.json": [str(run / "run_meta.json"), sha(run / "run_meta.json")],
        "policy_probe.npz": [str(run / RB.PROBE_REL), sha(run / RB.PROBE_REL)],
        C.EVIDENCE: [str(run / C.EVIDENCE), sha(run / C.EVIDENCE)]}}
    (run / C.CHECK_JSON).write_text(json.dumps(check))
    return run


shutil.rmtree(ROOT / "runs", ignore_errors=True)
queue = C.queue_of(json.loads((ROOT / "SCREEN.json").read_text()))
first = fake_run(C.run_name(*queue[0]), *queue[0])
for label in ("1 of 8 decided", "8 of 8 decided"):
    try:
        RB.full_read(first)
        print(f"{label}: full_read returned (unexpected)")
    except RB.InputError as exc:
        print(f"{label}: InputError: {str(exc)[:300]}")
    if label.startswith("1"):
        for arm, seed in queue[1:]:
            fake_run(C.run_name(arm, seed), arm, seed)
print("reads/ written:", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "reads").rglob("*")) if (ROOT / "reads").exists()
      else "none")
shutil.rmtree(ROOT / "runs")
shutil.rmtree(ROOT / "reads", ignore_errors=True)
