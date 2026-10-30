"""Fixer: does run_bs.py's preflight, UNMODIFIED, accept an APPROVED of the form criteria.txt's HOW THE COORDINATOR PINS
gives (criteria_sha256, screen_sha256, code_tree_digest, tool_sha256 with the five tools AND "score_bs.py",
approved_by, utc)?  And does score_bs.py accept the same APPROVED (its six-tool and PINS checks)?

A temporary root <BS>/review/fixer/preflight_root holds byte copies of <BS>/criteria.txt and <BS>/SCREEN.json and that
APPROVED.  run_bs.bs_root is pointed at it in memory; run_bs.run is replaced by a function that raises, so no
continuation can start whatever happens; run_bs.main is called with --preflight-only for every SCREEN.json entry
(expected: exit 0, "preflight": "pass").  Everything else preflight checks is the real thing: the five tools at <BS>/tools,
the start pair and records at FMAN sha256, the code copy's seal (M1's check_copy on <BS>/code), the stack, the M5 pins,
free disk.  Negative controls: a wrong five-tool pin, and a criteria_sha256 that is not the file's: refused.
The temporary root is removed after; <BS>/APPROVED is never written.

    cd /tmp && PY -I -B <BS>/review/fixer/fx_preflight.py <BS>
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

BS = Path(sys.argv[1]).resolve()
TMP = BS / "review" / "fixer" / "preflight_root"
assert not (BS / "APPROVED").exists(), "the real <BS>/APPROVED exists: this check is for before the pin"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BS / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rb = load("run_bs")
C = rb.C


def never_run(ns):
    raise SystemExit("fx_preflight: run() reached -- refusing to start a continuation")


rb.run = never_run
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
shutil.copyfile(BS / "criteria.txt", TMP / "criteria.txt")
shutil.copyfile(BS / "SCREEN.json", TMP / "SCREEN.json")
tools = {t: sha(BS / "tools" / t) for t in ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py",
                                             "score_bs.py")}
approved = {"criteria_sha256": sha(BS / "criteria.txt"), "screen_sha256": sha(BS / "SCREEN.json"),
            "code_tree_digest": C.TREE_DIGEST, "tool_sha256": tools,
            "approved_by": "fixer stand-in (temporary root; not an approval)", "utc": "stand-in"}
(TMP / "APPROVED").write_text(json.dumps(approved, indent=1) + "\n")
print("APPROVED (temporary root) =", json.dumps(approved, indent=1))
rb.bs_root = lambda: TMP

screen = json.loads((BS / "SCREEN.json").read_text())
ok = True
for arm, seed in screen["queue"]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rb.main(["--arm", arm, "--seed", str(seed), "--preflight-only"])
    out = buf.getvalue()
    verdict = json.loads(out[out.index("{"):])
    print(f"preflight-only {arm} {seed}: exit {rc}; {verdict['preflight']}; problems {verdict['problems']}")
    ok &= rc == 0 and verdict["preflight"] == "pass" and verdict["problems"] == []
print("approval_problems() with the form above:", rb.approval_problems())

# negative controls (the check is live)
bad = json.loads(json.dumps(approved))
bad["tool_sha256"]["bs_hooks.py"] = "0" * 64
(TMP / "APPROVED").write_text(json.dumps(bad, indent=1) + "\n")
neg1 = rb.preflight(rb.parse(["--arm", "control", "--seed", "2101", "--preflight-only"]))
print("negative control, wrong bs_hooks.py pin:", neg1)
bad = json.loads(json.dumps(approved))
bad["criteria_sha256"] = "1" * 64
(TMP / "APPROVED").write_text(json.dumps(bad, indent=1) + "\n")
neg2 = rb.preflight(rb.parse(["--arm", "critic_blind_tail3", "--seed", "2131", "--preflight-only"]))
print("negative control, criteria_sha256 not the file's:", neg2)
ok &= any("bs_hooks.py" in x for x in neg1) and any("criteria_sha256" in x for x in neg2)

# the scorer on the same root (no run folders: incomplete); its PROVISIONAL list must hold only the code copy's
# absence from the temporary root, and the real <BS>/code must measure clean
(TMP / "APPROVED").write_text(json.dumps(approved, indent=1) + "\n")
SB = load("score_bs")
classes = [{"name": C.run_name(a, s), "arm": a, "seed": s, "cls": SB.UNDECIDED} for a, s in SB.QUEUE]
prov = SB.provisional_items(TMP, classes, {})
print("score_bs PROVISIONAL items on the temporary root:", prov)
seal = SB.code_seal_items(BS / "code")
print("score_bs code seal on the real <BS>/code:", seal)
ok &= prov == [f"code copy {TMP / 'code'} missing: not measured"] and seal == []
shutil.rmtree(TMP)
print("RESULT:", "PASS" if ok else "FAIL", "(nothing started; the temporary root removed; <BS>/APPROVED not written)")
sys.exit(0 if ok else 1)
