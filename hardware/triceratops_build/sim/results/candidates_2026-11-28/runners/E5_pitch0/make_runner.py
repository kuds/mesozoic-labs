"""Generate run_e5_pitch0.py from E5's run_e5.py by literal text replacements only (E5_PITCH0_PLAN.md item 10).
Each replacement must match the source the stated number of times, or nothing is written."""
import hashlib
from pathlib import Path

SRC = Path("/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E5/code/run_e5.py")
DST = Path(__file__).resolve().parent / "run_e5_pitch0.py"
EXPECT_SRC_SHA = "ad49597ecbb74be271ee555d1d4e23881723edd7c549d12c71533f3f52de13ef"

text = SRC.read_text()
assert hashlib.sha256(text.encode()).hexdigest() == EXPECT_SRC_SHA, "run_e5.py changed"

NEW_DOC = '''"""E5 pitch-0 branch (sim/results/candidates_2026-11-28/E5/pitch0/E5_PITCH0_PLAN.md), runner. SIMULATION only.
Generated from E5's run_e5.py (sha256 ad49597e...) by make_runner.py's literal replacements; everything else is E5's
runner unchanged, including its readings R1-R6. Changed after a result: written after E5's pitched held-out results
and review finding E5-R1 (prereg section 3 item 9's pitch-0 alternative, run as a contingency; the lead's ruling is
open).
Differences from run_e5.py:
  - stage 1 is plan.json's 33 points with body_pitch_deg set to 0.0 (ids H<size>-p0-s1-NN); every episode refuses a
    nonzero pitch;
  - stage 3 (only if no stage-2 setting meets the rule) drops e5_neighbours' two body-pitch neighbours (ids
    H<size>-p0-s3-NN);
  - held-out: the chosen setting on 301-310, 401-410, 501-510 and 601-610; the frozen controller (supporting) on
    501-510 and 601-610 only; SUCCEEDS only if all four sets pass;
  - files: scratch candidates/E5_pitch0/, results sim/results/candidates_2026-11-28/E5/pitch0/ (e5p0_*), the plan's
    sha256 checked before tune and heldout and recorded in every result file; tuning rows carry seeds and episode
    wall time; per-seed tables carry the 2-8 s speed and hind knee (context, review E5-R2).
Usage: python run_e5_pitch0.py {tune,heldout,summarise} --size {0.60,0.50} [--workers 2]
"""
'''

REPL = [
    # (old, new, expected count)
    ('RUNNER = Path(__file__).resolve()\nSCR = C.CAND_SCRATCH / "E5"\nEPI = SCR / "episodes"\nMODELS = SCR / "models"\n'
     'OUT = C.RESULTS / "E5"\n',
     'RUNNER = Path(__file__).resolve()\nSCR = C.CAND_SCRATCH / "E5_pitch0"\nEPI = SCR / "episodes"\n'
     'MODELS = C.CAND_SCRATCH / "E5" / "models"     # E5\'s 50 cm model, reused (sha256 checked before every use)\n'
     'OUT = C.RESULTS / "E5" / "pitch0"\nPLAN = OUT / "E5_PITCH0_PLAN.md"\n'
     'HELD = {**C.E5_HELDOUT, "501-510": tuple(range(501, 511)), "601-610": tuple(range(601, 611))}\n'
     'FROZEN_SETS = ("501-510", "601-610")\nP0 = "p0"\n', 1),
    ('                 "heldout": set(C.E5_HELDOUT["301-310"]) | set(C.E5_HELDOUT["401-410"]),\n'
     '                 "heldout_frozen": set(C.E5_HELDOUT["301-310"]) | set(C.E5_HELDOUT["401-410"])}\n',
     '                 "heldout": {s for k in HELD for s in HELD[k]},\n'
     '                 "heldout_frozen": {s for k in FROZEN_SETS for s in HELD[k]}}\n\n\n'
     'def _check_plan():\n'
     '    rec = SCR / "plan_sha256.txt"\n'
     '    if not PLAN.exists() or not rec.exists() or rec.read_text().split()[0] != sha(PLAN):\n'
     '        raise RuntimeError("the pitch-0 plan is missing or changed since its sha256 was recorded")\n\n\n'
     'def _extra(size_key):\n'
     '    ex = {"pitch0_plan": PLAN}\n'
     '    if size_key == "0.50":\n'
     '        ex["model_50cm"] = model_path(size_key)\n'
     '    return ex\n', 1),
    ('    pitch = float(setting["body_pitch_deg"])\n    lg = LazyLogger()\n',
     '    pitch = float(setting["body_pitch_deg"])\n'
     '    if pitch != 0.0:\n'
     '        raise RuntimeError(f"pitch-0 branch: setting {sid} has body pitch {pitch}")\n'
     '    lg = LazyLogger()\n', 1),
    ('        raise RuntimeError("stage-1 settings differ from plan.json\'s")\n    return mine\n',
     '        raise RuntimeError("stage-1 settings differ from plan.json\'s")\n'
     '    out = []\n'
     '    for i, st in enumerate(mine):          # the same points, body pitch set to 0 (prereg item 9)\n'
     '        assert st["id"] == f"H{SIZES[size_key]:.2f}-s1-{i:02d}"\n'
     '        out.append({**st, "body_pitch_deg": 0.0, "id": f"H{SIZES[size_key]:.2f}-{P0}-s1-{i:02d}",\n'
     '                    "source_point": st["id"], "source_body_pitch_deg": st["body_pitch_deg"]})\n'
     '    return out\n', 1),
    ('def tune(size_key, workers):\n    size = SIZES[size_key]\n',
     'def tune(size_key, workers):\n    _check_plan()\n    size = SIZES[size_key]\n', 1),
    ('        s3 = C.e5_neighbours(settings_of[best_id], size, 0)\n',
     '        s3 = [st for st in C.e5_neighbours(settings_of[best_id], size, 0) if st["body_pitch_deg"] == 0.0]\n'
     '        for i, st in enumerate(s3):\n'
     '            st["id"] = f"H{size:.2f}-{P0}-s3-{i:02d}"\n', 1),
    ('def heldout(size_key, workers):\n', 'def heldout(size_key, workers):\n    _check_plan()\n', 1),
    ('    seeds = [s for k in ("301-310", "401-410") for s in C.E5_HELDOUT[k]]\n'
     '    jobs = [(size_key, "heldout", st, c, sd) for c in CONDS for sd in seeds]\n'
     '    pool_map(jobs, workers)\n'
     '    jobs = [(size_key, "heldout_frozen", frozen, c, sd) for c in CONDS for sd in seeds]\n',
     '    seeds = [s for k in HELD for s in HELD[k]]\n'
     '    jobs = [(size_key, "heldout", st, c, sd) for c in CONDS for sd in seeds]\n'
     '    pool_map(jobs, workers)\n'
     '    fseeds = [s for k in FROZEN_SETS for s in HELD[k]]\n'
     '    jobs = [(size_key, "heldout_frozen", frozen, c, sd) for c in CONDS for sd in fseeds]\n', 1),
    ('            "slip_max": max(sv) if sv else None, "wall_s": e["wall_s"]}\n',
     '            "slip_max": max(sv) if sv else None, "wall_s": e["wall_s"],\n'
     '            "speed_2_8": r.get("speed_2_8_mps"), "hind_knee_2_8": r.get("hind_knee_rms_over_rating_2_8")}\n', 1),
    ('                                                                "slip_max")}}\n',
     '                                                                "slip_max", "speed_2_8", "hind_knee_2_8")}}\n', 1),
    ('    rows = [json.loads(ln) for ln in tuning_log_path(size_key).read_text().splitlines()]\n',
     '    rows = [json.loads(ln) for ln in tuning_log_path(size_key).read_text().splitlines()]\n'
     '    for r in rows:                      # review E5-R4: each row\'s seeds and episode wall time\n'
     '        sd = list(C.E5_SCREEN_SEEDS) if r["stage"] == "s1" else list(C.E5_TUNING_SEEDS)\n'
     '        cs = C.E5_SCREEN_CONDITIONS if r["stage"] == "s1" else CONDS\n'
     '        r["seeds"] = sd\n'
     '        r["conditions"] = list(cs)\n'
     '        r["episode_wall_s_sum"] = round(sum(load_ep(size_key, r["stage"], r["id"], c, s)["wall_s"]\n'
     '                                            for c in cs for s in sd), 1)\n', 1),
    ('        per = {}\n        for name, seeds in C.E5_HELDOUT.items():\n',
     '        per = {}\n'
     '        sets = HELD if stage == "heldout" else {k: HELD[k] for k in FROZEN_SETS}\n'
     '        for name, seeds in sets.items():\n', 1),
    ('                       "SUCCEEDS_both_sets" if stage == "heldout" else "would_meet_rule_both_sets_supporting":\n'
     '                           all(v["rule"]["SUCCEEDS"] for v in per.values())}\n',
     '                       "SUCCEEDS_all_four_sets" if stage == "heldout" else "would_meet_rule_both_sets_supporting":\n'
     '                           all(v["rule"]["SUCCEEDS"] for v in per.values())}\n'
     '        if stage == "heldout":\n'
     '            held[stage]["reading_lead_sets_301_310_401_410"] = all(per[k]["rule"]["SUCCEEDS"]\n'
     '                                                                   for k in ("301-310", "401-410"))\n'
     '            held[stage]["reading_rerun_sets_501_510_601_610"] = all(per[k]["rule"]["SUCCEEDS"]\n'
     '                                                                    for k in FROZEN_SETS)\n', 1),
    ('"<= 1.0 of 6.75 N m in every episode of every condition); SUCCEEDS if both sets pass",',
     '"<= 1.0 of 6.75 N m in every episode of every condition); the pitch-0 branch SUCCEEDS only if all four "\n'
     '                       "sets (301-310, 401-410, 501-510, 601-610) pass (E5_PITCH0_PLAN.md item 7)",', 1),
    ('"heldout_seeds": C.E5_HELDOUT,', '"heldout_seeds": HELD, "heldout_frozen_seeds": list(FROZEN_SETS),', 1),
    ('C.provenance(RUNNER, extra=({"model_50cm": model_path(size_key)}\n'
     '                                                        if size_key == "0.50" else None))',
     'C.provenance(RUNNER, extra=_extra(size_key))', 1),
    ('C.provenance(RUNNER, extra=({"model_50cm": model_path(size_key)}\n'
     '                                                     if size_key == "0.50" else None))',
     'C.provenance(RUNNER, extra=_extra(size_key))', 1),
    ('C.provenance(RUNNER, extra=({"model_50cm": model_path(size_key)}\n'
     '                                                         if size_key == "0.50" else None))',
     'C.provenance(RUNNER, extra=_extra(size_key))', 1),
    ('e5_choice_', 'e5p0_choice_', 6),
    ('e5_tuning_', 'e5p0_tuning_', 2),
    ('e5_heldout_', 'e5p0_heldout_', 1),
    ('"experiment": "E5",', '"experiment": "E5 pitch-0 branch (changed after a result)",', 3),
]

body = text[text.index("from __future__"):]
for old, new, n in REPL:
    got = body.count(old)
    assert got == n, (got, n, old[:80])
    body = body.replace(old, new)
out = NEW_DOC + body
DST.write_text(out)
print("wrote", DST, hashlib.sha256(out.encode()).hexdigest())
