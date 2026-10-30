"""Fixer re-run (copy of review/rv_endtoend.py, TMP moved to review/fixer/e2e_tmp) of the reviewer's end-to-end runs of score_bs.py on roots derived from the writer's dry-run root (the parent's real markers;
only the 8 primary rows' rollout_mu14_mean are replaced), and an audit of every file the scorer opens while the screen
is incomplete (outcome-blindness).  Everything is written under <BS>/review/tmp.

    cd /tmp && PY -I -B <BS>/review/rv_endtoend.py
"""
import importlib.util
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

BS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen")
DRY = BS / "scratch" / "criteria_work" / "dryrun_root"
TMP = BS / "review" / "fixer" / "e2e_tmp"
spec = importlib.util.spec_from_file_location("score_bs", BS / "tools" / "score_bs.py")
SB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SB)


def fresh(name):
    root = TMP / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(DRY, root, ignore=shutil.ignore_patterns("score"))
    return root


def set_primary(root, name, value, nan_row=None):
    p = root / "reads" / name / "markers.json"
    m = json.loads(p.read_text())
    st = m["probe"]["steps"]
    for k, s in enumerate(SB.PRIMARY_STEPS):
        i = st.index(s)
        m["probe"]["rollout_mu14_mean"][i] = float("nan") if nan_row == k else value
    p.write_text(json.dumps(m, indent=1) + "\n")


def scenario(label, values, lost=(), nan=None):
    root = fresh(label)
    for (arm, seed), v in zip(SB.QUEUE, values, strict=True):
        set_primary(root, f"{arm}_{seed}", v, nan_row=3 if nan == f"{arm}_{seed}" else None)
    if lost:
        (root / "lost.json").write_text(json.dumps({"schema": SB.C.LOST_SCHEMA, "lost": {
            n: {"cause": "crashes", "detail": "reviewer synthetic"} for n in lost}}) + "\n")
    rc = SB.main(["--root", str(root)])
    res = json.loads((root / "score" / "score.json").read_text())
    print(f"== {label}: exit {rc}; line {res['line']} {res['line_key']}; p filled {res['p_filled']['count']}/70; "
          f"as counted {res['p_as_counted'] and (res['p_as_counted']['count'], res['p_as_counted']['total'])}; "
          f"control median {res['control_median']:+.5f} rise {res['control_rise']:+.5f}; additions {res['additions']}")
    return res


TMP.mkdir(parents=True, exist_ok=True)
sys.stdout = open(TMP / "rv_endtoend_scorer_stdout.txt", "w")  # the scorer prints its full text; keep it in a file
r = []
# queue order: control 2101, blind 2101, control 2111, blind 2111, ...
r.append(("A_boundary", scenario("A_boundary", [0.3125, 0.048828125, 0.1171875, 0.185546875, 0.2734375, 0.087890625,
                                                0.224609375, 0.146484375])))
r.append(("A_prime", scenario("A_prime", [0.3125, 0.048828125, 0.107421875, 0.185546875, 0.2734375, 0.1171875,
                                          0.224609375, 0.146484375])))
r.append(("blind_lost", scenario("blind_lost", [0.30, 0.01, 0.29, 0.02, 0.28, 0.03, 0.27, 0.04],
                                 lost=("critic_blind_tail3_2131",))))
r.append(("nan_primary", scenario("nan_primary", [0.30, 0.01, 0.29, 0.02, 0.28, 0.03, 0.27, 0.04],
                                  nan="control_2111")))
r.append(("flat", scenario("flat", [0.05, 0.01, 0.06, 0.02, 0.04, 0.03, 0.05, 0.04])))
sys.stdout.close()
sys.stdout = sys.__stdout__
for label, res in r:
    print(f"{label:<12} line {res['line']} {res['line_key']:<22} p filled {res['p_filled']['count']}/70  "
          f"as counted {res['p_as_counted'] and (res['p_as_counted']['count'], res['p_as_counted']['total'])}  "
          f"rise {res['control_rise']:+.5f}  additions {res['additions']}  status {res['status']}")
assert [x[1]["line"] for x in r] == [2, 3, 3, 3, 1]
assert r[0][1]["p_filled"]["count"] == 3 and r[1][1]["p_filled"]["count"] == 5
assert r[2][1]["p_filled"]["count"] >= 36 and r[3][1]["p_filled"]["count"] >= 36

# OUTCOME-BLINDNESS AUDIT: one entry undecided (its check.json removed); record every path the scorer opens.
root = fresh("blind_audit")
(root / "runs" / "critic_blind_tail3_2131" / "check.json").unlink()
opened = []


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        opened.append(str(args[0]))


sys.addaudithook(hook)
res, lines, code = SB.score(root)
sys.stdout.flush()
touched = sorted({p for p in opened if str(root) in p})
bad = [p for p in touched if "/reads/" in p or p.endswith("run_meta.json") or "/01_stance/" in p]
print("AUDIT incomplete: status", res["status"], "exit", code, "; files opened under the root:")
for p in touched:
    print("   ", p.replace(str(root), "<root>"))
print("AUDIT markers / run_meta / run records opened while incomplete:", bad or "none")
assert not bad and res["status"] == "incomplete"
txt = "\n".join(lines)
assert "THE READING" not in txt and "+0." not in txt.split("STATUS")[1]
print("END-TO-END AND AUDIT AGREE WITH THE CRITERIA")
