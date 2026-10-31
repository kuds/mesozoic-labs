"""Process review of BS2: do the pin checks catch a changed byte?  Runs ONLY on a COPY root (argv[1]) holding copies
of BS2's criteria.txt, SCREEN.json and the six pinned tools, with a FAKE APPROVED written here (never BS2/APPROVED).

For each case the copy is restored from its pristine bytes, one change is made, and then:
  - run_bs.preflight(ns) of the COPY's run_bs.py with bs_common.BS pointed at the copy root (preflight reads only and
    creates nothing; run() is never called, so nothing can train);
  - score_bs.score(copy root) of the COPY's score_bs.py (no runs exist: the screen is "incomplete", so only the pins
    and check.json are looked at; the PROVISIONAL items are printed).
Reads the real F2 inputs, M1/M5 pins and BS/code (the seal) read only.  Writes nothing outside the copy root except
its stdout.  Usage:  cd /tmp && nice -n 10 PY -I -B pin_check.py <copy root>
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
assert ROOT.name == "copyroot" and "review/process/work" in str(ROOT), ROOT
TOOLS = ROOT / "tools"
PINNED = ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py", "score_bs.py")
FILES = ["criteria.txt", "SCREEN.json", *(f"tools/{t}" for t in PINNED)]
PRISTINE = {f: (ROOT / f).read_bytes() for f in FILES}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write_approved(tool_keys=PINNED, code_digest=None):
    appr = {"criteria_sha256": sha(PRISTINE["criteria.txt"]), "screen_sha256": sha(PRISTINE["SCREEN.json"]),
            "code_tree_digest": code_digest or "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1",
            "tool_sha256": {t: sha(PRISTINE[f"tools/{t}"]) for t in tool_keys},
            "approved_by": "FAKE: process-review copy root only; not an approval", "utc": "n/a"}
    (ROOT / "APPROVED").write_text(json.dumps(appr) + "\n")


def restore():
    for f, b in PRISTINE.items():
        (ROOT / f).write_bytes(b)


def flip(rel, old, new):
    """Replace the first occurrence of old by new (same length: one changed byte)."""
    b = (ROOT / rel).read_bytes()
    assert len(old) == len(new) and old in b, (rel, old)
    (ROOT / rel).write_bytes(b.replace(old, new, 1))
    return f"{rel}: first {old!r} -> {new!r}"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = load("rb_copy", TOOLS / "run_bs.py")
RB.C.BS = str(ROOT)                       # the copy's run_bs now treats the copy root as <BS>
assert str(RB.bs_root()) == str(ROOT) and str(RB.TOOLS) == str(TOOLS)
for k in ("bs_common", "read_bs", "bs_hooks"):
    sys.modules.pop(k, None)
SB = load("score_bs_copy", TOOLS / "score_bs.py")
assert str(SB.TOOLS) == str(TOOLS)


def preflight(arm, seed):
    ns = argparse.Namespace(arm=arm, seed=seed, save_loaded_only=False, no_update=False, until=None, out=None,
                            preflight_only=True)
    return RB.preflight(ns)                # never RB.run()


def scorer():
    try:
        result, _lines, code = SB.score(ROOT)
        return {"exit": code, "status": result.get("status"), "provisional": result.get("provisional")}
    except SB.InputError as exc:
        return {"exit": 2, "input_error": str(exc)}


CASES = [
    ("M0 nothing changed (APPROVED pins the copy's 8 files)", None),
    ("M1 criteria.txt one byte", lambda: flip("criteria.txt", b"CRITERIA v1", b"CRITERIA v2")),
    ("M2a SCREEN.json one byte in _note", lambda: flip("SCREEN.json", b"BS2's queue", b"BS3's queue")),
    ("M2b SCREEN.json one byte in the queue (seed 2531 -> 2532)",
     lambda: flip("SCREEN.json", b'["critic_blind_tail3", 2531]', b'["critic_blind_tail3", 2532]')),
    ("M3a tools/run_bs.py one byte", lambda: flip("tools/run_bs.py", b"one continuation", b"One continuation")),
    ("M3b tools/read_bs.py one byte", lambda: flip("tools/read_bs.py", b"the reader", b"the Reader")),
    ("M3c tools/run_waves_bs.sh one byte", lambda: flip("tools/run_waves_bs.sh", b"# BS2:", b"# BS3:")),
    ("M3d tools/bs_common.py one byte", lambda: flip("tools/bs_common.py", b"pins and constants", b"Pins and constants")),
    ("M3e tools/bs_hooks.py one byte", lambda: flip("tools/bs_hooks.py", b"the", b"thE")),
    ("M3f tools/score_bs.py one byte", lambda: flip("tools/score_bs.py", b"the scorer", b"the Scorer")),
    ("M4 APPROVED code_tree_digest one hex digit", "code"),
    ("M5 APPROVED without score_bs.py in tool_sha256", "noscore"),
]

out = {}
for title, mutate in CASES:
    restore()
    write_approved()
    what = "none"
    if mutate == "code":
        write_approved(code_digest="431fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1")
        what = "APPROVED code_tree_digest 331fe8fc... -> 431fe8fc..."
    elif mutate == "noscore":
        write_approved(tool_keys=PINNED[:5])
        what = "APPROVED tool_sha256 without score_bs.py"
    elif mutate is not None:
        what = mutate()
    rec = {"change": what, "preflight_control_2501": preflight("control", 2501),
           "preflight_critic_blind_tail3_2531": preflight("critic_blind_tail3", 2531), "scorer": scorer()}
    if title.startswith("M0"):
        rec["preflight_all_8"] = {f"{a}_{s}": preflight(a, s) for a, s in SB.QUEUE}
    out[title] = rec
    pf = rec["preflight_control_2501"]
    print(f"== {title}\n   change: {what}\n   preflight control_2501: {'PASS' if not pf else 'REFUSED'} {pf}\n"
          f"   preflight critic_blind_tail3_2531: {rec['preflight_critic_blind_tail3_2531']}\n"
          f"   scorer: {rec['scorer']}", flush=True)
    if "preflight_all_8" in rec:
        print(f"   preflight all 8 queue entries: { {k: (v or 'PASS') for k, v in rec['preflight_all_8'].items()} }")
restore()
(ROOT / "APPROVED").unlink()
assert all((ROOT / f).read_bytes() == b for f, b in PRISTINE.items())
print("restored the copy's 8 files to their pristine bytes; fake APPROVED removed")
json.dump(out, open(ROOT.parent / "pin_check.json", "w"), indent=1)
