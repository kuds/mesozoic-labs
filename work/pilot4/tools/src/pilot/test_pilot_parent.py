"""Cases for pilot_parent.py (the screen's parent rule) and pilot_judge.py's pure checks (resume point, run end).

Plain Python; run: python3 -I -B test_pilot_parent.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_judge as pj  # noqa: E402  (module level is stdlib only; the repository is imported inside main())
import pilot_parent as pp  # noqa: E402

cases = []


def check(name, got, want):
    cases.append((name, got == want, got, want))


R9 = [{"run": "r9 arm L s42 6M", "clean": 29, "joint_noise_falls": 1, "own_noise_falls": 0, "noise_falls": 1},
      {"run": "r9 arm L s52 6M", "clean": 38, "joint_noise_falls": 1, "own_noise_falls": 0, "noise_falls": 1},
      {"run": "r9 arm L s60 6M", "clean": 40, "joint_noise_falls": 6, "own_noise_falls": 0, "noise_falls": 6}]


def r10(c42, f42, c52, f52, c60, f60, paths=None):
    out = []
    for s, c, f in ((42, c42, f42), (52, c52, f52), (60, c60, f60)):
        out.append({"run": f"r10 arm L s{s} 6M", "clean": c, "joint_noise_falls": f, "own_noise_falls": 0,
                    "noise_falls": f, "path": (paths or {}).get(s)})
    return out


def score(overall, cands, failed=None):
    return {"overall": overall, "failed": failed if failed is not None else ([] if overall == "PASS" else ["K2"]),
            "missing": [], "invalid": [], "best_stance_policy": {"candidates": cands}}


# K1-K4 all hold: the best r10 checkpoint, even when r9 s60 ranks above every r10 checkpoint
res = pp.choose(score("PASS", r10(38, 2, 39, 3, 37, 0) + R9))
check("PASS: r10 only, even below r9 s60's 40/40", res["parent"], "r10 arm L s52 6M")
check("PASS: path B", res["path"], "B")
check("PASS: MANUAL_LOAD_PATH names the seed", res["manual_load_path"].endswith("r10_armL_s52/01_stance/models/stage1_6291456_steps"),
      True)
res = pp.choose(score("PASS", r10(40, 3, 40, 1, 39, 0) + R9))
check("PASS: clean tie inside r10 broken by noise falls", res["parent"], "r10 arm L s52 6M")
res = pp.choose(score("PASS", r10(40, 2, 40, 2, 39, 0) + R9))
check("PASS: full tie inside r10 -> no parent", res["parent"], None)
check("PASS: full tie names the tied", res["tied"], ["r10 arm L s42 6M", "r10 arm L s52 6M"])
# Any criterion fails: best of six
res = pp.choose(score("FAIL", r10(39, 0, 38, 0, 37, 0) + R9))
check("FAIL: r9 s60 40/40 beats r10 39/40", res["parent"], "r9 arm L s60 6M")
check("FAIL: path A", res["path"], "A")
check("FAIL: path A sha", res["parent_sha256"], pp.R9_S60_SHA256)
res = pp.choose(score("FAIL", r10(40, 5, 38, 0, 37, 0) + R9))
check("FAIL: r10 40/40 with 5 falls beats r9 s60 (6)", res["parent"], "r10 arm L s42 6M")
res = pp.choose(score("FAIL", r10(40, 6, 38, 0, 37, 0) + R9))
check("FAIL: 40/40 and 6 falls each -> tie, no parent", (res["parent"], res["tied"]),
      (None, ["r10 arm L s42 6M", "r9 arm L s60 6M"]))
check("FAIL: tie is a maintainer decision", "maintainer" in res.get("decision_needed", ""), True)
# PASS but with a missing judge is not "all four hold"
res = pp.choose(dict(score("PASS", r10(39, 0, 38, 0, 37, 0) + R9), missing=["x"]))
check("PASS with a missing judge falls back to six", res["parent"], "r9 arm L s60 6M")
# A noise-fall count that is unmeasured (score_r10.py writes noise_falls None) cannot win or lose a tie-break
def unmeasured(cands, run):
    return [dict(c, joint_noise_falls=None, own_noise_falls=None, noise_falls=None) if c["run"] == run else c
            for c in cands]


res = pp.choose(score("FAIL", unmeasured(r10(40, 0, 30, 0, 30, 0), "r10 arm L s42 6M") + R9))
check("FAIL: r10 40/40 unmeasured vs r9 s60 40/40 -> no parent (the round-2 verifier's case)",
      (res["parent"], res["tied"]), (None, ["r10 arm L s42 6M", "r9 arm L s60 6M"]))
check("FAIL: the unmeasured tie says why", "unmeasured for r10 arm L s42 6M" in res["ranking_why"], True)
check("FAIL: the unmeasured tie is a maintainer decision and names why",
      "maintainer" in res["decision_needed"] and "unmeasured" in res["decision_needed"], True)
check("FAIL: the unmeasured candidate is listed", res["unmeasured_noise_falls"], ["r10 arm L s42 6M"])
res = pp.choose(score("PASS", unmeasured(r10(39, 0, 39, 2, 37, 0), "r10 arm L s42 6M") + R9))
check("PASS: r10 39 unmeasured vs r10 39 measured -> no parent (even though the measured one has falls)",
      (res["parent"], res["tied"]), (None, ["r10 arm L s42 6M", "r10 arm L s52 6M"]))
res = pp.choose(score("PASS", unmeasured(unmeasured(r10(39, 0, 39, 0, 37, 0), "r10 arm L s42 6M"), "r10 arm L s52 6M")
                      + R9))
check("PASS: two unmeasured tied on clean -> no parent", res["parent"], None)
res = pp.choose(score("PASS", unmeasured(r10(40, 0, 39, 1, 37, 0), "r10 arm L s42 6M") + R9))
check("PASS: unmeasured but alone on most clean wins on clean (no tie-break needed)",
      (res["parent"], res["ranking_why"]), ("r10 arm L s42 6M", "most clean (40/40)"))
res = pp.choose(score("PASS", unmeasured(r10(38, 0, 39, 1, 39, 2), "r10 arm L s42 6M") + R9))
check("PASS: an unmeasured candidate below the top clean does not block the tie-break",
      (res["parent"], res["unmeasured_noise_falls"]), ("r10 arm L s52 6M", ["r10 arm L s42 6M"]))
with tempfile.TemporaryDirectory() as tmp:
    sj = Path(tmp) / "score.json"
    sj.write_text(json.dumps(score("FAIL", unmeasured(r10(40, 0, 30, 0, 30, 0), "r10 arm L s42 6M") + R9)))
    check("main exits 1 with no parent on an unmeasured tie", pp.main([str(sj), "--json", str(Path(tmp) / "o.json")]), 1)
    check("main's JSON names no parent", json.loads((Path(tmp) / "o.json").read_text())["parent"], None)

# The r10 parent's sha256 comes from its judge file
with tempfile.TemporaryDirectory() as tmp:
    jp = Path(tmp) / "judge_6291456.json"
    jp.write_text(json.dumps({"model_sha256": "sha256:" + "ab" * 32}))
    res = pp.choose(score("PASS", r10(40, 0, 39, 0, 38, 0, paths={42: str(jp)}) + R9))
    check("PASS: parent sha from the r10 judge file", res["parent_sha256"], "sha256:" + "ab" * 32)
    sj = Path(tmp) / "score.json"
    sj.write_text(json.dumps(dict(score("PASS", r10(40, 0, 39, 0, 38, 0)), validation_synthetic=True)))
    check("refuses a validation-synthetic score", pp.main([str(sj)]), 2)

# pilot_judge: the resume rule (ramp 2,000,000, warm-up 300,000 in the pilot config)
RAMP, WARM = 2_000_000, 300_000
check("not resumed", pj.resume_check({}, RAMP, WARM), (None, None))
step, why = pj.resume_check({"resume_load_path": "/x/03_locomotion/models/stage2_524288_steps.zip"}, RAMP, WARM)
check("resume at 524,288 is off-schedule", (step, why is not None and why.startswith("OFF-SCHEDULE")), (524288, True))
step, why = pj.resume_check({"resume_load_path": "/x/models/stage2_1572864_steps.zip"}, RAMP, WARM)
check("resume at 1,572,864 is off-schedule", why is not None, True)
check("resume at 2,097,152 is on schedule",
      pj.resume_check({"resume_load_path": "/x/models/stage2_2097152_steps.zip"}, RAMP, WARM), (2097152, None))
check("resume at 3,670,016 is on schedule",
      pj.resume_check({"resume_load_path": "/x/models/stage2_3670016_steps"}, RAMP, WARM), (3670016, None))
step, why = pj.resume_check({"resume_load_path": "/x/models/stage2_final.zip"}, RAMP, WARM)
check("resume from a non-periodic checkpoint is refused", (step, why is not None), (None, True))
check("sha normalisation", pj.norm_sha("SHA256:" + "AB" * 32), "ab" * 32)
# pilot_judge: where a judgement goes (an off-schedule report never shares a path with the seed's restart)
OUTD = Path("/x/OUT")
check("on-schedule cell", pj.cell_path(OUTD, 42, 4194304, None, Path("/r/RUN_s42_b")), OUTD / "s42" / "4194304")
check("off-schedule cell goes under off_schedule/<run>",
      pj.cell_path(OUTD, 42, 4194304, "OFF-SCHEDULE: ...", Path("/r/RUN_s42")),
      OUTD / "off_schedule" / "RUN_s42" / "s42" / "4194304")
check("off-schedule report and the restart's judgement never share a path",
      pj.cell_path(OUTD, 42, 2097152, "OFF-SCHEDULE: ...", Path("/r/RUN_s42"))
      != pj.cell_path(OUTD, 42, 2097152, None, Path("/r/RUN_s42_b")), True)
check("sha normalisation refuses a prefix", pj.norm_sha("70f3a444"), None)

# pilot_judge: the run-end record
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp) / "03_locomotion"
    models = stage / "models"
    models.mkdir(parents=True)
    for s in (524288, 1048576, 3670016):
        (models / f"stage2_{s}_steps.zip").write_text("x")
    with zipfile.ZipFile(models / "stage2_final.zip", "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": 3801088}))
    e = pj.run_end(models, 4194304, stage)
    check("run_end newest periodic", e["newest_periodic_steps"], 3670016)
    check("run_end stopped early from the final model", (e["final_num_timesteps"], e["stopped_early"]), (3801088, True))
    (models / "stage2_final.zip").unlink()
    (stage / "stage_summary.txt").write_text("Stage: 2 (Locomotion)\nTimesteps:      4,194,304\n")
    e = pj.run_end(models, 4194304, stage)
    check("run_end complete from stage_summary.txt", (e["final_num_timesteps"], e["stopped_early"]), (4194304, False))
    (stage / "stage_summary.txt").unlink()
    e = pj.run_end(models, 4194304, stage)
    check("run_end unknown without a final record", e["stopped_early"], None)

failed = [c for c in cases if not c[1]]
for name, ok, got, want in cases:
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
print(f"{len(cases) - len(failed)}/{len(cases)} passed")
sys.exit(1 if failed else 0)
