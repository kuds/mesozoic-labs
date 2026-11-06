"""Fix lane: one-place mutants of the fix stage's new code, each run against the tests that should catch it, on a
scratch copy of the tools (the real files are only read). A mutant is killed when its tests fail. Nothing trains.

usage: python -I -B mutate_fix.py TOOLS SCRATCH OUT.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS, SCRATCH, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PY = sys.executable
FILES = ["pilot4_judge.py", "pilot4_straight.py", "pilot4_score.py", "test_pilot4_score.py", "nb_pilot_runner4.py",
         "iso.py", "gait_probe_pilot.py", "fixtures/gait_cases.json", "tests/test_pilot4_judge.py",
         "tests/test_pilot4_straight.py", "tests/test_nb_pilot_runner4.py"]
if SCRATCH.exists():
    raise SystemExit(f"{SCRATCH} exists")
for f in FILES:
    (SCRATCH / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TOOLS / f, SCRATCH / f)
INI = SCRATCH / "pytest.ini"
INI.write_text("[pytest]\n")
ENV = dict(os.environ, PILOT4_TREE=str(TOOLS / "tree"), STRAIGHT_TREE=str(TOOLS / "tree"),
           PILOT3_SCORE=str(TOOLS.parents[1] / "pilot3" / "tools" / "pilot3_score.py"),
           PILOT3_JUDGING=str(TOOLS.parents[1] / "pilot3" / "judging"),
           PILOT4_STRAIGHT_VAL=str(TOOLS / "out" / "fix" / "val_final"),
           PILOT4_JUDGE_E2E=str(TOOLS / "out" / "judge" / "e2e" / "OUT"),
           PILOT3_RUNS=str(TOOLS.parents[1] / "pilot3" / "runs" / "final"),
           PILOT3_JUDGE=str(TOOLS.parents[1] / "pilot3" / "tools" / "pilot3_judge.py"),
           PILOT4_SETUP_RECORDS=str(TOOLS.parents[1] / "pilot4" / "build" / "check" / "judge" / "fake_runs"),
           STRAIGHT_P3_RUNS=str(TOOLS.parents[1] / "pilot3" / "runs" / "final"),
           STRAIGHT_P3_CELLS=str(TOOLS.parents[1] / "pilot3" / "judging" / "OUT"))
ENV.pop("PYTHONPATH", None)


def pytest(test: str, k: str) -> list[str]:
    return [PY, "-I", "-B", "-m", "pytest", "-p", "no:cacheprovider", "-c", str(INI), "--rootdir", str(SCRATCH),
            "-q", "-x", "-k", k, str(SCRATCH / test)]


SCORE = [PY, "-I", "-B", str(SCRATCH / "test_pilot4_score.py")]
JT, ST, RT = "tests/test_pilot4_judge.py", "tests/test_pilot4_straight.py", "tests/test_nb_pilot_runner4.py"
MUTANTS = [
    ("F01 judge: no anchor check at the pilot budget", "pilot4_judge.py",
     "        problems += anchor_problems(hp_here, task_here, committed_table)\n", "",
     pytest(JT, "anchor or moved or text_anchors")),
    ("F02 judge: the recipe anchor not compared", "pilot4_judge.py",
     "    if hp_here != HYPERPARAMETERS_SHA256:", "    if False:", pytest(JT, "anchor or moved")),
    ("F03 judge: the table anchor compared loosely (==)", "pilot4_judge.py",
     "    if not same_json(committed_table, COMMAND_TABLE):", "    if committed_table != COMMAND_TABLE:",
     pytest(JT, "anchor or moved")),
    ("F04 judge: the drift record ignored", "pilot4_judge.py",
     "    why_drift = drift_problem(kwargs_drift)\n", "    why_drift = None\n", pytest(JT, "drift")),
    ("F05 judge: a tuple not read as a list in the drift", "pilot4_judge.py",
     "        a_cmp = list(a) if isinstance(a, tuple) else a\n", "        a_cmp = a\n", pytest(JT, "drift")),
    ("F06 judge: the drift over the record's keys only", "pilot4_judge.py",
     "    for k in sorted(set(rec) | set(env_kwargs)):", "    for k in sorted(set(rec)):", pytest(JT, "drift")),
    ("F07 judge: no default --expect-head", "pilot4_judge.py",
     "    ap.add_argument(\"--expect-head\", default=PILOT_HEAD,", "    ap.add_argument(\"--expect-head\", default=None,",
     pytest(JT, "default_expect_head")),
    ("F08 judge: a null warm_start_log_std accepted", "pilot4_judge.py",
     "        if isinstance(block, dict) and \"warm_start_log_std\" in block:",
     "        if isinstance(block, dict) and block.get(\"warm_start_log_std\") is not None:",
     pytest(JT, "binding_record")),
    ("F09 judge: prey within 0.5 m accepted", "pilot4_judge.py",
     "            or not all(is_number(v) and float(v) == w for v, w in zip(prey, PREY_LATERAL_RANGE))):",
     "            or not all(is_number(v) and abs(float(v) - w) <= 0.5 for v, w in zip(prey, PREY_LATERAL_RANGE))):",
     pytest(JT, "binding_record")),
    ("S01 straight: the drift record not read", "pilot4_straight.py",
     "    drift = meta.get(\"env_kwargs_drift_vs_recorded\")\n", "    drift = None\n", pytest(ST, "drift")),
    ("S02 straight: r read at m_900", "pilot4_straight.py",
     "READ_K = 950  ", "READ_K = 900  ", pytest(ST, "m950 or reads_m950")),
    ("S03 straight: the guard without Tensor.backward", "pilot4_straight.py",
     "    torch.Tensor.backward = blocked(\"torch.Tensor.backward\")\n", "", pytest(ST, "training_guard")),
    ("S04 straight: the crab note's sign dropped", "pilot4_straight.py",
     "median crab {crab:+.1f} deg", "median crab {crab:.1f} deg", pytest(ST, "crab_note")),
    ("C01 score: the tool sha256 not checked", "pilot4_score.py",
     "    if rec.get(\"tool_sha256\") != STRAIGHT_TOOL_SHA256:", "    if False:", SCORE),
    ("C02 score: the constants not checked", "pilot4_score.py",
     "    if not isinstance(const, dict) or const != STRAIGHT_CONSTANTS:", "    if False:", SCORE),
    ("C03 score: verdicts not recomputed", "pilot4_score.py",
     "             if k not in bad and redo.get(k) != verdicts[k]}", "             if False}", SCORE),
    ("C04 score: (a)'s bar exclusive", "pilot4_score.py",
     "    return x is not None and math.isfinite(x) and x <= bar", "    return x is not None and math.isfinite(x) and x < bar",
     SCORE),
    ("C05 score: (c) read below 30 counting", "pilot4_score.py",
     "n_counting < k[\"C_MIN_COUNTING\"]:", "n_counting < 29:", SCORE),
    ("C06 score: inf read as a pass", "pilot4_score.py",
     "    return x is not None and math.isfinite(x) and x <= bar", "    return x is not None and x <= bar or x == math.inf",
     SCORE),
    ("C07 score: a drifted cell scored", "pilot4_score.py",
     "    if extra:\n        out[\"problems\"] += extra\n        out[\"outcome\"] = None\n", "", SCORE),
    ("C08 score: the pinned sha256 is not the tool's", "pilot4_score.py",
     "STRAIGHT_TOOL_SHA256 = \"sha256:9", "STRAIGHT_TOOL_SHA256 = \"sha256:8", SCORE),
    ("C09 score: a pending video with the look counted (Q23)", "pilot4_score.py",
     "        (counted if (video == \"agree\" and parts is not None and look == \"straight\") else could).append(s)",
     "        (counted if (video != \"disagree\" and parts is not None and look == \"straight\") else could).append(s)",
     SCORE),
    ("R01 runner: the QUICK_TEST refusal removed", "nb_pilot_runner4.py",
     "        if quick_test is not True and not a.stop_at_learn and not a.plan_only:",
     "        if False:", pytest(RT, "quick_test")),
    ("R02 runner: a truthy QUICK_TEST accepted", "nb_pilot_runner4.py",
     "        if quick_test is not True and not a.stop_at_learn and not a.plan_only:",
     "        if not quick_test and not a.stop_at_learn and not a.plan_only:", pytest(RT, "quick_test")),
    ("R03 runner: the refusal also blocks --stop-at-learn", "nb_pilot_runner4.py",
     "        if quick_test is not True and not a.stop_at_learn and not a.plan_only:",
     "        if quick_test is not True and not a.plan_only:", pytest(RT, "quick_test")),
]
results = []
for name, f, old, new, cmd in MUTANTS:
    path = SCRATCH / f
    orig = path.read_text()
    if orig.count(old) != 1:
        results.append({"mutant": name, "applied": False, "why": f"found {orig.count(old)} times"})
        continue
    path.write_text(orig.replace(old, new))
    try:
        r = subprocess.run(cmd, cwd=SCRATCH, env=ENV, capture_output=True, text=True, timeout=900)
    finally:
        path.write_text(orig)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    results.append({"mutant": name, "applied": True, "exit": r.returncode, "killed": r.returncode != 0, "tail": tail})
    print(name, "KILLED" if r.returncode != 0 else "SURVIVED", flush=True)
# the unmutated copy must pass the same selections first-hand
base = []
for cmd in {tuple(m[4]) for m in MUTANTS}:
    r = subprocess.run(list(cmd), cwd=SCRATCH, env=ENV, capture_output=True, text=True, timeout=900)
    base.append({"cmd": " ".join(cmd[-3:]), "exit": r.returncode, "tail": (r.stdout + r.stderr).strip().splitlines()[-1:]})
summary = {"mutants": results, "unmutated": base,
           "killed": sum(1 for x in results if x.get("killed")), "applied": sum(1 for x in results if x.get("applied")),
           "unmutated_all_pass": all(b["exit"] == 0 for b in base)}
OUT.write_text(json.dumps(summary, indent=1) + "\n")
print(f"killed {summary['killed']} of {summary['applied']} applied ({len(results)} listed); unmutated pass: "
      f"{summary['unmutated_all_pass']}")
shutil.rmtree(SCRATCH)
