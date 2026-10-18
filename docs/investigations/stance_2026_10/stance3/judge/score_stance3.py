"""Score the three T. rex stance3 screen runs against their fixed criteria (foot/stance3/criteria.txt, sha256 1ce910dc...).

    python3 score_stance3.py JUDGED_ROOT [--runs-root RUNS_ROOT] [--json OUT.json]
    python3 score_stance3.py --inspect JUDGE.json [...]          (one file's counts and validity, nothing scored)

JUDGED_ROOT/<run id>/judge_<slot>.json are the files judge_stance3.py wrote (full mode), one per checkpoint, for
  Run 1 stance3_armL_s60  (arm L,  branch claude/screen-trex-r9-arml-12m, seed 60)
  Run 2 stance3_armLT_s60 (arm LT, branch claude/screen-trex-r9-armlt,    seed 60)
  Run 3 stance3_armLT_s52 (arm LT, branch claude/screen-trex-r9-armlt,    seed 52)
with <slot> one of 4194304 5242880 6291456 8388608 10485760 final (the judged checkpoints J), 3145728 (report only),
robust_best, best and the S3 checkpoint's step (the selection comparison).  RUNS_ROOT/<run id>/[01_stance/] holds each
run's eval_episodes.csv and train_episodes.csv (default RUNS_ROOT = JUDGED_ROOT).  Plain python (no repository import).

The criteria, per run (criteria.txt):
  K1 gate held    v2 PASS at 6,291,456 and at final: the repository verdict PASS and, recomputed from the 40 rows, clean
                  >= 37/40 with the exact Clopper-Pearson LCB >= 0.80, full horizon >= 38/40, hop-or-fall <= 1 (a
                  disagreement fails the checkpoint and is listed); across J at most one checkpoint below 37/40 clean
                  and none below 33/40.
  K2 fore-aft     at every checkpoint in J:
                  K2a  <= 3 of 40 episodes with the pad's max_sole_cop_fore_aft > 0.80, every episode counted, a
                       non-full-horizon (or unmeasured) one counting as over;
                  K2b  <= 3 of 40 episodes with either foot's signed whole-foot CoP <label>_foot_cop_fore_aft outside
                       [-0.929, +0.571] span half-lengths, a non-full-horizon (or unmeasured) one counting as outside;
                  K2c  the panel median of the both-feet whole-foot CoP x_c (over the full-horizon episodes; mm ahead of
                       the MTP III joints) within [-60, +30] mm of the statue's median on the same seeds (the same
                       file's statue panel, likewise over its full-horizon episodes: 39 of 40, seed 3081 falls), at
                       each of the six checkpoints, and its range over the six <= 30 mm.  (criteria.txt's "equals the
                       COM projection to within 0.5 mm" holds for the medians, not per statue episode: up to 1.74 mm.)
  K3 robustness   at 6,291,456 and at final: joint-space noise falls <= 2 of 20 and own-sigma falls <= 1 of 20.
  K4 roll         at 6,291,456 and at final: <= 3 of 40 full-horizon episodes with max_sole_tilt_deg > 2.0 or
                  max_sole_corner_lift_m > 0.004.
  HOLDS = K1-K4 all hold; FORE-AFT HOLDS = K1 and K2 hold.  Bars are inclusive as written ("at most", "within"); the
  K2c millimetre comparisons are made on values rounded to 1e-6 mm.
A missing judge file, or one that fails the validity checks (smoke, relaxed bindings -- --allow-commit-mismatch among
them --, another mode, arm, seed, slot or clone HEAD, a run whose stage_config.json records another git_commit, n_envs
other than 4, another hyperparameters_sha256 or another seed, a checkpoint whose num_timesteps is not its slot's
(12,001,280 for final, the step itself for a periodic slot), panel seeds / bars / noise configuration other than the
screen's, no K2 fields), fails every criterion that needs it and is listed; a run with a missing or invalid J file is
INCOMPLETE and the decision line PROVISIONAL.
Then the recipe verdicts and the decision-table line(s) of criteria.txt, the best final checkpoint of the three, and,
report only: K5 (any 4 consecutive evaluations after 3,145,728 whose full-horizon mean com_fore_aft_mean_m lies outside
[statue - 0.060, statue + 0.030] m; the statue's reading in that frame comes from the judge files' statue panel), the
late-fall fraction per 262,144-step bin from train_episodes.csv (200 < length < 1000; flagged above 5%), and the S0-S3
selection comparison (S3: the latest periodic checkpoint at or after 3,145,728 whose 8 evaluations up to and including
its step were all 10/10 full horizon with no full-horizon episode over 0.80 on the pad or the whole-foot statistic).

--validation (validation of this script only) accepts relaxed-binding files and marks the output VALIDATION ONLY.
--ledgers RUN_ID=DIR (repeatable) reads that run's eval_episodes.csv / train_episodes.csv from DIR[/01_stance] instead
of RUNS_ROOT/RUN_ID: for a run restarted under "<RUN_ID>_b" (colab_steps.txt step 9), whose judge files still go to
JUDGED_ROOT/<RUN_ID>/ (the judge's --out names them).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys

SCRATCH = "scratchpad"
CRITERIA_FILE = SCRATCH + "/foot/stance3/criteria.txt"
CRITERIA_SHA256 = "1ce910dcd6e2ff7b41a4f592b9f34b9a6c9cb398c523e88f0ebd19af754f18fa"
FROZEN_PARAMS = SCRATCH + "/foot/symmetry/design/frozen_params.json"
FROZEN_PARAMS_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
JUDGE_SCHEMA = "stance3-judge/v1"
R9_PHYSICS_SHA256 = "sha256:3d68aec73fbc4642681ed601bedaf3524459d693d6a7afce036bb14bac872d67"
HEADS = {"L": "7e6b557db999900bbc95df63ca651b1f2f4e2df1", "LT": "1af2d4d767aebc89fa2cf73fc59a11004048f797"}
#: criteria.txt "n_envs 4", "12,000,000 steps (final 12,001,280)", "the committed PPO schedules": the stance
#: hyperparameters_sha256.PPO of configs/digest_snapshot.generated.txt, the same on both heads.
N_ENVS = 4
FINAL_TIMESTEPS = 12001280
HP_SHA256 = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"
RUNS = (("stance3_armL_s60", "L", 60, "Run 1"), ("stance3_armLT_s60", "LT", 60, "Run 2"), ("stance3_armLT_s52", "LT", 52, "Run 3"))
ARM_ENV = {"L": {"leg_home_pose_weight": 0.5, "leg_home_pose_tolerance": 0.10},
           "LT": {"leg_home_pose_weight": 1.0, "leg_home_pose_tolerance": 0.05}}
ARM_ENV_COMMON = {"lateral_lean_weight": 0.25, "lateral_lean_tolerance_m": 0.015, "leg_home_pose_reference": "settled"}
J = (("4.2M", 4194304), ("5.2M", 5242880), ("6.3M", 6291456), ("8.4M", 8388608), ("10.5M", 10485760), ("final", "final"))
REPORT_SLOTS = (("3.1M", 3145728),)
NOISE_SLOTS = ("6.3M", "final")
PANEL_SEEDS = list(range(3042, 3082))
NOISE_SEEDS = list(range(3042, 3062))
FEET = ("r", "l")
PANEL_N, NOISE_N = 40, 20
# criteria.txt, fixed before any training
K1_MIN_CLEAN, K1_MIN_LCB, K1_MIN_FULL, K1_MAX_HOP = 37, 0.80, 38, 1
K1_DURABLE_BELOW, K1_DURABLE_MAX_BELOW, K1_FLOOR = 37, 1, 33
K2A_BAR, K2_MAX = 0.80, 3
K2B_LO, K2B_HI = -0.929, 0.571
K2C_LO_MM, K2C_HI_MM, K2C_RANGE_MM = -60.0, 30.0, 30.0
K3_JOINT_MAX, K3_OWN_MAX = 2, 1
K4_TILT, K4_LIFT, K4_MAX = 2.0, 0.004, 3
FOOT_REPORT_BAR = 0.80
K5_AFTER, K5_LO_M, K5_HI_M, K5_RUN = 3145728, -0.060, 0.030, 4
LATE_BIN, LATE_FLAG = 262144, 0.05
S3_FROM, S3_CADENCE, S3_EVALS, S3_EPISODES = 3145728, 524288, 8, 10
#: The r9 stance stage's bars the panel must have been scored at (the criteria's numbers among them).
R9_BARS = {"min_eval_episodes": 40, "min_clean_stance_lcb": 0.8, "min_full_horizon_fraction": 0.95,
           "max_hop_or_fall_episodes": 1, "max_sole_cop_fore_aft": 0.8, "max_sole_tilt_deg": 2.0,
           "max_sole_corner_lift_m": 0.004, "settle_steps": 200}
#: stance_gate_v2.HOP_OR_FALL_KEYS plus the early end ("horizon").
HOP_KEYS = {"min_all_feet_support", "max_touchdown_rate", "max_window_displacement_m", "max_actuator_saturation_fraction",
            "max_window_airborne_substeps", "max_window_peak_floor_force_bw", "horizon"}
#: K1 failure modes the decision table's line 4 names: saturation, hop, nosedive, head contact.
NEW_MODE_RAILS = {"max_actuator_saturation_fraction", "min_all_feet_support", "max_touchdown_rate",
                  "max_window_airborne_substeps", "max_window_peak_floor_force_bw"}
NEW_MODE_TERMINATIONS = {"nosedive", "head_contact"}


# ------------------------------------------------------------------------------------------------ arithmetic
def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(0, k + 1)), 1.0)


def binomial_lcb(successes: int, trials: int, alpha: float = 0.05) -> float:
    """recovery_gate.binomial_lcb, verbatim arithmetic: the exact one-sided Clopper-Pearson lower bound."""
    if successes == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if 1.0 - _binom_cdf(successes - 1, trials, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def reason_key(r: str) -> str:
    return r.split(":", 1)[0].strip()


def over(v, bar) -> bool:
    return v is not None and v > bar


def num(v):
    """A finite float, or None (a CSV cell, a JSON null, NaN)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def mm(v_m):
    return None if v_m is None else round(1000.0 * v_m, 6)


def median(values):
    v = [x for x in (num(x) for x in values) if x is not None]
    return statistics.median(v) if v else None


def fmean(values):
    v = [x for x in (num(x) for x in values) if x is not None]
    return sum(v) / len(v) if v else None


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def k2b_outside(e: dict) -> bool:
    if not e.get("reached_horizon"):
        return True
    w = e.get("foot_cop_fore_aft") or {}
    for f in FEET:
        v = num(w.get(f))
        if v is None or v < K2B_LO or v > K2B_HI:
            return True
    return False


def k2a_over(e: dict) -> bool:
    v = num(e.get("max_sole_cop_fore_aft"))
    return (not e.get("reached_horizon")) or v is None or v > K2A_BAR


# ------------------------------------------------------------------------------------------------ one judge file
def panel_counts(d: dict) -> dict:
    """Every number the criteria read from one judge file, recomputed from its per-episode rows."""
    p = d["panel"]
    eps = p["episodes_detail"]
    st = p.get("statue_episodes_detail") or []
    n = len(eps)
    full = [bool(e["reached_horizon"]) for e in eps]
    clean = [bool(e["clean"]) for e in eps]
    hop = [(not f) or any(reason_key(r) in HOP_KEYS for r in (e.get("reasons") or [])) for e, f in zip(eps, full, strict=True)]
    k = sum(clean)
    lcb = binomial_lcb(k, n) if n else 0.0
    roll = [f and (over(num(e.get("max_sole_tilt_deg")), K4_TILT) or over(num(e.get("max_sole_corner_lift_m")), K4_LIFT))
            for e, f in zip(eps, full, strict=True)]
    xc = median(e.get("xc_both_feet_m") for e in eps if e["reached_horizon"])
    st_xc = median(e.get("xc_both_feet_m") for e in st if e.get("reached_horizon"))
    com = median(e.get("com_mtp3_head_m") for e in eps if e["reached_horizon"])
    st_com = median(e.get("com_mtp3_head_m") for e in st if e.get("reached_horizon"))
    rails: dict = {}
    for e in eps:
        for r in e.get("reasons") or []:
            rails[reason_key(r)] = rails.get(reason_key(r), 0) + 1
    out = {
        "n": n, "n_clean": k, "lcb": lcb, "lcb_recorded": p.get("clean_lcb"), "full": sum(full), "falls": n - sum(full),
        "hop_or_fall": sum(hop), "hop_or_fall_recorded": p.get("hop_or_fall_episodes"), "verdict": p.get("verdict"),
        "roll_rail": sum(roll),
        "k2a": sum(1 for e in eps if k2a_over(e)), "k2a_seeds": [e["seed"] for e in eps if k2a_over(e)],
        "k2b": sum(1 for e in eps if k2b_outside(e)), "k2b_seeds": [e["seed"] for e in eps if k2b_outside(e)],
        "k2a_recorded": (p.get("summary") or {}).get("k2a_count"), "k2b_recorded": (p.get("summary") or {}).get("k2b_count"),
        "xc_median_mm": mm(xc), "statue_xc_median_mm": mm(st_xc),
        "xc_offset_mm": None if xc is None or st_xc is None else round(1000.0 * xc - 1000.0 * st_xc, 6),
        "com_mtp3_median_mm": mm(com), "statue_com_mtp3_median_mm": mm(st_com),
        "com_offset_mm": None if com is None or st_com is None else round(1000.0 * com - 1000.0 * st_com, 6),
        "statue_n": len(st), "statue_full": sum(1 for e in st if e.get("reached_horizon")),
        "statue_fall_seeds": [e["seed"] for e in st if not e.get("reached_horizon")],
        "statue_com_footsite_mean_m": fmean(e.get("com_fore_aft_footsite_m") for e in st if e.get("reached_horizon")),
        "foot_cop_report_fh": sum(1 for e, f in zip(eps, full, strict=True)
                                  if f and over(num(e.get("max_foot_cop_fore_aft")), FOOT_REPORT_BAR)),
        "mean_reward": p.get("mean_reward"), "statue_mean_reward": p.get("statue_mean_reward"),
        "terminations": p.get("terminations"), "failing_rails": rails,
        "noise_joint_falls": None, "noise_own_falls": None,
    }
    comp = {"n_eq_40": n == PANEL_N, "clean_ge_37": k >= K1_MIN_CLEAN, "lcb_ge_0.80": lcb >= K1_MIN_LCB,
            "full_ge_38": sum(full) >= K1_MIN_FULL, "hop_or_fall_le_1": sum(hop) <= K1_MAX_HOP}
    out["k1_components"] = comp
    out["k1_components_pass"] = all(comp.values())
    out["k1_repo_pass"] = p.get("verdict") == "PASS"
    out["k1_pass"] = out["k1_components_pass"] and out["k1_repo_pass"]
    out["k1_agree"] = out["k1_components_pass"] == out["k1_repo_pass"]
    for key, name in (("noise_matched_control_joint", "noise_joint_falls"), ("noise_matched_own", "noise_own_falls")):
        nz = d.get(key)
        if nz:
            rows = nz.get("episodes_detail") or []
            out[name] = sum(1 for r in rows if not r["reached_horizon"])
            out[name + "_recorded"] = nz.get("falls")
            out[name.replace("falls", "fall_seeds")] = [r["seed"] for r in rows if not r["reached_horizon"]]
    return out


def validity_issues(d: dict, *, arm: str, seed: int, steps, label: str, expect_head: str, frozen: dict,
                    validation: bool, needs_noise: bool) -> tuple[list[str], list[str]]:
    """(why the file cannot be scored in its slot, the validation relaxations it was accepted with)."""
    bad, relaxed_ok = [], []
    if d.get("schema") != JUDGE_SCHEMA:
        bad.append(f"schema {d.get('schema')!r}")
    if d.get("mode") != "full":
        bad.append(f"a {d.get('mode')!r} judge (not the full panel)")
    if d.get("smoke"):
        bad.append("a smoke judge (shortened panels)")
    if d.get("synthetic"):
        bad.append("a synthetic copy")
    if d.get("arm") != arm:
        bad.append(f"judged as arm {d.get('arm')!r}, the run is arm {arm}")
    stem = "stage1_final" if steps == "final" else f"{steps}_model" if steps in ("best", "robust_best") else f"stage1_{steps}_steps"
    if d.get("seed") != seed or d.get("steps") != steps or d.get("checkpoint") != stem:
        bad.append(f"slot s{seed}/{steps} holds seed {d.get('seed')} steps {d.get('steps')} ({d.get('checkpoint')})")
    git = d.get("git") or {}
    if git.get("head") != expect_head or git.get("status"):
        bad.append(f"judged on clone HEAD {git.get('head')} (dirty={bool(git.get('status'))}), not {expect_head}")
    checks = d.get("checks") or {}
    relaxed = checks.get("relaxed") or {}
    soft = []
    for key in ("no_symmetry_check", "allow_missing_records", "allow_seed_mismatch", "allow_commit_mismatch"):
        if relaxed.get(key) is not False:
            soft.append(f"judged with {key}")
    if relaxed.get("task_binding") != "require":
        soft.append(f"judged with task_binding {relaxed.get('task_binding')!r}")
    if not (checks.get("symmetry") or {}).get("mirror_map_sha256_match"):
        soft.append("checkpoint mesozoic_symmetry mirror_map_sha256 does not match the clone")
    # The run's own stage_config.json, as the judge recorded it: the commit it trained on, n_envs, the recipe digest and
    # the seed (criteria.txt: all binding checks on, commit included; n_envs 4; the committed PPO schedules).
    sc = checks.get("stage_config") or {}
    if sc.get("git_commit_match") is not True:
        soft.append(f"the run's stage_config.json git_commit {sc.get('git_commit')!r} is not the judged clone HEAD")
    if sc.get("run_n_envs") != N_ENVS:
        soft.append(f"the run's stage_config.json n_envs {sc.get('run_n_envs')!r} is not {N_ENVS}")
    if sc.get("run_hyperparameters_sha256") != HP_SHA256:
        soft.append(f"the run's stage_config.json hyperparameters_sha256 {sc.get('run_hyperparameters_sha256')!r} is not "
                    f"the committed {HP_SHA256}")
    if sc.get("run_seed") != seed:
        soft.append(f"the run's stage_config.json seed {sc.get('run_seed')!r} is not the slot's {seed}")
    if validation:
        relaxed_ok += soft
    else:
        bad += soft
    if checks.get("hard_failures"):
        bad.append(f"binding failures {checks.get('hard_failures')}")
    # criteria.txt: final is stage1_final at 12,001,280 steps (an early stop's stage1_final is not "final"); a periodic
    # slot's checkpoint holds exactly its step.  best / robust_best hold whatever step they were selected at.
    want_ts = FINAL_TIMESTEPS if steps == "final" else steps if isinstance(steps, int) and not isinstance(steps, bool) else None
    got_ts = checks.get("checkpoint_num_timesteps")
    if want_ts is not None and got_ts != want_ts:
        bad.append(f"the checkpoint holds num_timesteps {got_ts!r}; slot {steps} needs {want_ts:,}")
    if not (checks.get("plant_physics_sha256") or {}).get("match"):
        bad.append("run plant_identity.json physics_sha256 does not match the clone")
    plant = d.get("plant_identity") or {}
    if plant.get("physics_sha256") != R9_PHYSICS_SHA256 or plant.get("physics_revision") != 9:
        bad.append(f"plant {plant.get('physics_revision')} {plant.get('physics_sha256')} is not r9")
    if not (checks.get("seed") or {}).get("match"):
        bad.append(f"checkpoint seed {(checks.get('seed') or {}).get('checkpoint')} != slot seed {seed}")
    env = d.get("arm_env_kwargs") or {}
    want = {**ARM_ENV_COMMON, **ARM_ENV[arm]}
    if any(env.get(k) != v for k, v in want.items()):
        bad.append(f"the clone's stance [env] {({k: env.get(k) for k in want})} is not arm {arm}'s")
    p = d.get("panel") or {}
    eps = p.get("episodes_detail") or []
    if [e.get("seed") for e in eps] != PANEL_SEEDS:
        bad.append("panel episodes are not seeds 3042-3081 in order")
    if p.get("scored_gate_kind") != "stance_quality/v2":
        bad.append(f"panel scored as {p.get('scored_gate_kind')!r}")
    th = p.get("thresholds") or {}
    wrong = {k: th.get(k) for k, v in R9_BARS.items() if th.get(k) != v}
    if wrong:
        bad.append(f"panel bars differ from the r9 stance stage's: {wrong}")
    if "max_foot_cop_fore_aft" in th:
        bad.append("panel scored with the whole-foot rail (not the r9 stance stage)")
    st = p.get("statue_episodes_detail") or []
    if (p.get("statue_on_same_seeds") or {}).get("n") != PANEL_N or [e.get("seed") for e in st] != PANEL_SEEDS:
        bad.append("no 40-episode statue panel on the same seeds")
    for name, rows in (("panel", eps), ("statue", st)):
        if rows and not all(isinstance(e.get("foot_cop_fore_aft"), dict) and all(f in e["foot_cop_fore_aft"] for f in FEET)
                            and "xc_both_feet_m" in e for e in rows):
            bad.append(f"{name} rows without the K2b / K2c fields")
    if needs_noise:
        h4 = frozen["criteria"]["H4"]
        cj, ow = d.get("noise_matched_control_joint") or {}, d.get("noise_matched_own") or {}
        ref = h4["control_sigma"][str(seed)]
        checks6 = {
            "control joint present": bool(cj),
            "control joint space": cj.get("space") == "joint",
            "control joint n": cj.get("n") == NOISE_N and [r.get("seed") for r in cj.get("episodes_detail") or []] == NOISE_SEEDS,
            "control joint spawns": cj.get("seeds") == h4["spawn_range"],
            "control joint sigma (frozen, this seed)": cj.get("sigma_source") == ref["source"] and cj.get("sigma_vector") == ref["vector"],
            "own present": bool(ow),
            "own sigma_source": ow.get("sigma_source") == "own" and ow.get("space") == "policy",
            "own n": ow.get("n") == NOISE_N and [r.get("seed") for r in ow.get("episodes_detail") or []] == NOISE_SEEDS,
            "own spawns": ow.get("seeds") == h4["spawn_range"],
        }
        bad += [f"H4 noise configuration: {k}" for k, ok in checks6.items() if not ok]
    return bad, relaxed_ok


# ------------------------------------------------------------------------------------------------ one run
def load_cell(root: str, run: str, arm: str, seed: int, steps, label: str, *, expect_head: str, frozen: dict,
              validation: bool, needs_noise: bool, notes: list) -> tuple[dict | None, str | None, dict | None]:
    """(counts or None, 'missing' / 'invalid' / None, the invalid entry)."""
    path = os.path.join(root, run, f"judge_{steps}.json")
    if not os.path.exists(path):
        return None, "missing", {"path": path}
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, "invalid", {"path": path, "issues": [f"unreadable: {exc}"]}
    bad, soft = validity_issues(d, arm=arm, seed=seed, steps=steps, label=label, expect_head=expect_head, frozen=frozen,
                                validation=validation, needs_noise=needs_noise)
    if bad:
        return None, "invalid", {"path": path, "issues": bad}
    c = panel_counts(d)
    c.update(path=path, model_sha256=d.get("model_sha256"), task_sha256=(d.get("panel") or {}).get("task_sha256"),
             validation_relaxations=soft)
    if soft:
        notes.append(f"{path}: accepted for VALIDATION ONLY with {soft}")
    if not c["k1_agree"]:
        notes.append(f"{path}: repository verdict {c['verdict']} disagrees with the criteria components {c['k1_components']} (counts as not passing)")
    if c["lcb_recorded"] is not None and abs(c["lcb_recorded"] - c["lcb"]) > 1e-12:
        notes.append(f"{path}: recorded LCB {c['lcb_recorded']} != recomputed {c['lcb']}")
    for name in ("hop_or_fall", "k2a", "k2b"):
        rec = c.get(name + "_recorded")
        if rec is not None and rec != c[name]:
            notes.append(f"{path}: recorded {name} {rec} != recomputed {c[name]}")
    for name in ("noise_joint_falls", "noise_own_falls"):
        if c.get(name + "_recorded") is not None and c[name + "_recorded"] != c[name]:
            notes.append(f"{path}: recorded {name} {c[name + '_recorded']} != recomputed {c[name]}")
    # (A run's stage_config.json git_commit other than the clone HEAD is a relaxation in validity_issues: refused, or
    # listed among the VALIDATION ONLY relaxations above.)
    return c, None, None


def run_criteria(cells: dict) -> dict:
    """K1-K4, HOLDS and FORE-AFT for one run; a None cell (missing or invalid) fails every criterion that needs it."""
    labels = [lab for lab, _ in J]
    missing = [lab for lab in labels if cells.get(lab) is None]
    # K1
    k1_at = {lab: (None if cells.get(lab) is None else cells[lab]["k1_pass"]) for lab in NOISE_SLOTS}
    clean = {lab: (None if cells.get(lab) is None else cells[lab]["n_clean"]) for lab in labels}
    below37 = [lab for lab in labels if clean[lab] is not None and clean[lab] < K1_DURABLE_BELOW]
    below33 = [lab for lab in labels if clean[lab] is not None and clean[lab] < K1_FLOOR]
    durable = not missing and len(below37) <= K1_DURABLE_MAX_BELOW and not below33
    K1 = {"pass": bool(k1_at["6.3M"]) and bool(k1_at["final"]) and durable, "pass_at": k1_at, "durable": durable,
          "clean": clean, "below_37": below37, "below_33": below33, "missing": missing,
          "detail": {lab: None if cells.get(lab) is None else
                     {"repo_verdict": cells[lab]["verdict"], "clean": cells[lab]["n_clean"], "lcb": cells[lab]["lcb"],
                      "full": cells[lab]["full"], "hop_or_fall": cells[lab]["hop_or_fall"],
                      "components": cells[lab]["k1_components"]} for lab in NOISE_SLOTS}}
    # K2
    k2a = {lab: (None if cells.get(lab) is None else cells[lab]["k2a"]) for lab in labels}
    k2b = {lab: (None if cells.get(lab) is None else cells[lab]["k2b"]) for lab in labels}
    med = {lab: (None if cells.get(lab) is None else cells[lab]["xc_median_mm"]) for lab in labels}
    off = {lab: (None if cells.get(lab) is None else cells[lab]["xc_offset_mm"]) for lab in labels}
    k2a_ok = all(v is not None and v <= K2_MAX for v in k2a.values())
    k2b_ok = all(v is not None and v <= K2_MAX for v in k2b.values())
    window_ok = all(v is not None and K2C_LO_MM <= v <= K2C_HI_MM for v in off.values())
    present = [v for v in med.values() if v is not None]
    rng = round(max(present) - min(present), 6) if len(present) == len(labels) else None
    range_ok = rng is not None and rng <= K2C_RANGE_MM
    K2 = {"pass": k2a_ok and k2b_ok and window_ok and range_ok,
          "K2a": {"pass": k2a_ok, "counts": k2a, "over": [lab for lab, v in k2a.items() if v is None or v > K2_MAX]},
          "K2b": {"pass": k2b_ok, "counts": k2b, "over": [lab for lab, v in k2b.items() if v is None or v > K2_MAX],
                  "seeds": {lab: (None if cells.get(lab) is None else cells[lab]["k2b_seeds"]) for lab in labels}},
          "K2c": {"pass": window_ok and range_ok, "window_pass": window_ok, "range_pass": range_ok,
                  "median_mm": med, "offset_from_statue_mm": off, "range_mm": rng,
                  "statue_median_mm": {lab: (None if cells.get(lab) is None else cells[lab]["statue_xc_median_mm"]) for lab in labels}}}
    # K3
    jf = {lab: (None if cells.get(lab) is None else cells[lab]["noise_joint_falls"]) for lab in NOISE_SLOTS}
    of = {lab: (None if cells.get(lab) is None else cells[lab]["noise_own_falls"]) for lab in NOISE_SLOTS}
    K3 = {"pass": all(v is not None and v <= K3_JOINT_MAX for v in jf.values()) and all(v is not None and v <= K3_OWN_MAX for v in of.values()),
          "joint_falls": jf, "own_falls": of}
    # K4
    rr = {lab: (None if cells.get(lab) is None else cells[lab]["roll_rail"]) for lab in NOISE_SLOTS}
    K4 = {"pass": all(v is not None and v <= K4_MAX for v in rr.values()), "roll_rail": rr}
    out = {"K1": K1, "K2": K2, "K3": K3, "K4": K4, "complete": not missing}
    out["failed"] = [k for k in ("K1", "K2", "K3", "K4") if not out[k]["pass"]]
    out["HOLDS"] = not out["failed"]
    out["FORE_AFT_HOLDS"] = K1["pass"] and K2["pass"]
    statues = sorted({cells[lab]["statue_xc_median_mm"] for lab in labels if cells.get(lab) is not None
                      and cells[lab]["statue_xc_median_mm"] is not None})
    out["statue_median_mm_spread"] = round(statues[-1] - statues[0], 6) if statues else None
    return out


def k1_failure_modes(cells: dict) -> dict:
    """The K1-failing checkpoints' rails and terminations, and whether they are the decision table's 'new' modes."""
    modes, new = {}, False
    for lab, _ in J:
        c = cells.get(lab)
        if c is None or (c["n_clean"] >= K1_MIN_CLEAN and (lab not in NOISE_SLOTS or c["k1_pass"])):
            continue
        rails = {k: v for k, v in c["failing_rails"].items()}
        terms = {k: v for k, v in (c.get("terminations") or {}).items() if k != "truncated"}
        modes[lab] = {"rails": rails, "terminations": terms}
        if (set(rails) & NEW_MODE_RAILS) or (set(terms) & NEW_MODE_TERMINATIONS):
            new = True
    return {"by_checkpoint": modes, "new_modes": new}


# ------------------------------------------------------------------------------------------------ ledgers (report only)
def _find(runs_root: str, run: str, name: str, ledger_dir: str | None = None) -> str | None:
    base = ledger_dir if ledger_dir else os.path.join(runs_root, run)
    for p in (os.path.join(base, "01_stance", name), os.path.join(base, name)):
        if os.path.exists(p):
            return p
    return None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def evaluations(rows: list[dict]) -> list[dict]:
    """One entry per evaluation of eval_episodes.csv, in step order."""
    by: dict = {}
    for r in rows:
        ts = int(float(r["timesteps"]))
        by.setdefault(ts, []).append(r)
    out = []
    for ts in sorted(by):
        rs = by[ts]
        full = [r for r in rs if str(r.get("full_horizon")).strip() in ("1", "True", "true", "1.0")]
        out.append({"timesteps": ts, "n": len(rs), "n_full": len(full),
                    "com_fh_mean_m": fmean(r.get("com_fore_aft_mean_m") for r in full),
                    "pad_over_fh": sum(1 for r in full if over(num(r.get("max_sole_cop_fore_aft")), K2A_BAR)),
                    "foot_over_fh": sum(1 for r in full if over(num(r.get("max_foot_cop_fore_aft")), FOOT_REPORT_BAR))})
    return out


def k5_readout(evals: list[dict], statue_m) -> dict:
    """K5 (report only): runs of >= 4 consecutive evaluations at or after 3,145,728 steps whose full-horizon mean
    com_fore_aft_mean_m lies outside [statue - 0.060, statue + 0.030] m (an evaluation with no full-horizon episode
    counts as outside)."""
    if statue_m is None:
        return {"computed": False, "why": "no statue reading in the foot-site frame (no judge file)"}
    lo, hi = statue_m + K5_LO_M, statue_m + K5_HI_M
    seq = [e for e in evals if e["timesteps"] >= K5_AFTER]
    flagged, run = [], []
    for e in seq + [None]:
        out = e is not None and (e["com_fh_mean_m"] is None or not lo <= e["com_fh_mean_m"] <= hi)
        if out:
            run.append(e)
            continue
        if len(run) >= K5_RUN:
            flagged.append({"from": run[0]["timesteps"], "to": run[-1]["timesteps"], "evaluations": len(run),
                            "com_fh_mean_m": [r["com_fh_mean_m"] for r in run]})
        run = []
    return {"computed": True, "statue_m": statue_m, "window_m": [lo, hi], "evaluations_considered": len(seq),
            "flag": bool(flagged), "runs": flagged}


def late_falls(rows: list[dict]) -> dict:
    """Per 262,144-step bin (k*262144, (k+1)*262144]: training episodes with 200 < length < 1000, over all episodes."""
    bins: dict = {}
    for r in rows:
        ts, length = num(r.get("timesteps")), num(r.get("length"))
        if ts is None or length is None:
            continue
        k = int((int(ts) - 1) // LATE_BIN)
        b = bins.setdefault(k, [0, 0])
        b[1] += 1
        b[0] += int(200 < length < 1000)
    table = [{"bin_end": (k + 1) * LATE_BIN, "late": b[0], "episodes": b[1], "fraction": b[0] / b[1] if b[1] else None}
             for k, b in sorted(bins.items())]
    flagged = [t for t in table if t["fraction"] is not None and t["fraction"] > LATE_FLAG]
    peak = max(table, key=lambda t: t["fraction"] or 0.0) if table else None
    return {"bins": table, "flagged": flagged, "peak": peak}


def s3_step(evals: list[dict]) -> dict:
    """The latest periodic checkpoint at or after 3,145,728 whose 8 evaluations up to and including its step were all
    10/10 full horizon with no full-horizon episode over 0.80 on the pad or the whole-foot statistic."""
    if not evals:
        return {"step": None, "why": "no evaluations"}
    last = max(e["timesteps"] for e in evals)
    best, considered = None, 0
    n = S3_FROM
    while n <= last:
        prior = [e for e in evals if e["timesteps"] <= n][-S3_EVALS:]
        considered += 1
        if len(prior) == S3_EVALS and all(e["n"] == S3_EPISODES and e["n_full"] == S3_EPISODES and e["pad_over_fh"] == 0
                                          and e["foot_over_fh"] == 0 for e in prior):
            best = n
        n += S3_CADENCE
    return {"step": best, "checkpoints_considered": considered,
            "why": None if best is not None else "no periodic checkpoint met the rule"}


# ------------------------------------------------------------------------------------------------ verdicts
def decision(per_run: dict) -> dict:
    """criteria.txt 'RECIPE VERDICTS AND WHAT FOLLOWS', every line that applies (in the table's order)."""
    r1, r2, r3 = (per_run[r[0]]["criteria"] for r in RUNS)
    lines = []
    if r2["HOLDS"] and r3["HOLDS"]:
        lines.append("1. L+T HOLDS on both Run 2 and Run 3: propose committing L+T (arm L's switches plus the two leg keys, "
                     "collapse reference 4460.5), then certify it on 2 seeds under the committed config.")
    elif r2["FORE_AFT_HOLDS"] and r3["FORE_AFT_HOLDS"]:
        lines.append("2. L+T FORE-AFT HOLDS on both but fails K3 or K4: the drift is fixed; the next round adds the leg "
                     "exploration-noise floor to L+T.")
    if not r2["K2"]["pass"] or not r3["K2"]["pass"]:
        if r1["HOLDS"]:
            lines.append("3a. L+T fails K2 on a seed and Run 1 HOLDS: arm L (config-only) stays the candidate; replicate it "
                         "on 2 more seeds before committing.")
        elif not r1["K2"]["pass"]:
            lines.append("3b. L+T fails K2 on a seed and Run 1 also fails K2: a joint-space price does not hold the stance; "
                         "the next round tests a fore-aft term that needs code or the v6b ankle spring.")
        else:
            lines.append(f"3. L+T fails K2 on a seed; Run 1 holds K2 but fails {r1['failed']}: neither 3a nor 3b applies "
                         "as written (maintainer's call).")
    for name, (run, *_rest) in zip(("Run 2", "Run 3"), RUNS[1:], strict=True):
        c = per_run[run]["criteria"]
        if not c["K1"]["pass"] and c["K2"]["pass"]:
            fm = per_run[run]["k1_failure_modes"]
            if fm["new_modes"]:
                lines.append(f"4. {name} fails K1 through new failure modes while K2 holds ({fm['by_checkpoint']}): the dose "
                             "is too strong; next: weight 0.75 at tolerance 0.05.")
            else:
                lines.append(f"4? {name} fails K1 while K2 holds, but without saturation, hop, nosedive or head contact "
                             f"({fm['by_checkpoint']}): line 4 does not apply as written (maintainer's call).")
    if r2["HOLDS"] != r3["HOLDS"]:
        lines.append("5. Exactly one L+T seed HOLDS: the recipe is not separated from seed luck; replicate before any commit.")
    if not lines:
        lines.append("No line of the decision table applies as written (maintainer's call).")
    complete = all(per_run[r[0]]["criteria"]["complete"] for r in RUNS)
    return {"lines": lines, "provisional": not complete}


def best_final(per_run: dict) -> dict:
    cands = []
    for run, _arm, _seed, name in RUNS:
        c = per_run[run]["cells"].get("final")
        if c is None:
            continue
        j, o = c["noise_joint_falls"], c["noise_own_falls"]
        cands.append({"run": f"{name} {run}", "clean": c["n_clean"], "joint_noise_falls": j, "own_noise_falls": o,
                      "noise_falls": None if j is None or o is None else j + o, "path": c["path"]})
    if not cands:
        return {"best": None, "tied": [], "candidates": []}
    key = lambda x: (-x["clean"], x["noise_falls"] if x["noise_falls"] is not None else 10 ** 9)  # noqa: E731
    cands.sort(key=key)
    top = [c for c in cands if key(c) == key(cands[0])]
    return {"best": top[0]["run"] if len(top) == 1 else None, "tied": [c["run"] for c in top] if len(top) > 1 else [],
            "rule": "most clean panel episodes at final; tie -> fewer noise falls (joint-space + own-sigma)", "candidates": cands}


# ------------------------------------------------------------------------------------------------ scoring
def score(root: str, *, runs_root: str | None = None, frozen_path: str = FROZEN_PARAMS, heads: dict | None = None,
          validation: bool = False, ledger_dirs: dict | None = None) -> dict:
    fsha = sha256_file(frozen_path)
    if fsha != FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{frozen_path} sha256 {fsha} != {FROZEN_PARAMS_SHA256}")
    with open(frozen_path) as fh:
        frozen = json.load(fh)
    heads = dict(HEADS, **(heads or {}))
    runs_root = runs_root or root
    ledger_dirs = {k: os.path.abspath(v) for k, v in (ledger_dirs or {}).items()}
    unknown = sorted(set(ledger_dirs) - {r[0] for r in RUNS})
    if unknown:
        raise SystemExit(f"--ledgers names no screen run: {unknown} (the runs: {[r[0] for r in RUNS]})")
    crit = {"path": CRITERIA_FILE, "sha256": sha256_file(CRITERIA_FILE) if os.path.exists(CRITERIA_FILE) else None}
    crit["matches"] = crit["sha256"] == CRITERIA_SHA256
    per_run, notes = {}, []
    if not crit["matches"]:
        notes.append(f"criteria file {CRITERIA_FILE} sha256 {crit['sha256']} != the fixed {CRITERIA_SHA256}")
    for run, arm, seed, name in RUNS:
        cells, missing, invalid = {}, [], []
        slots = list(J) + list(REPORT_SLOTS) + [("robust_best", "robust_best"), ("best", "best")]
        for lab, steps in slots:
            c, why, entry = load_cell(root, run, arm, seed, steps, lab, expect_head=heads[arm], frozen=frozen,
                                      validation=validation, needs_noise=lab in NOISE_SLOTS, notes=notes)
            cells[lab] = c
            if why == "missing" and (lab, steps) in J:
                missing.append(entry["path"])
            elif why == "invalid":
                invalid.append(entry)
        crits = run_criteria(cells)
        # report only: ledgers, K5, late falls, S3
        ldir = ledger_dirs.get(run)
        if ldir:
            notes.append(f"{run}: ledgers read from {ldir} (--ledgers; a run restarted under another RUN_ID)")
        ev_path = _find(runs_root, run, "eval_episodes.csv", ldir)
        tr_path = _find(runs_root, run, "train_episodes.csv", ldir)
        evals = evaluations(read_csv(ev_path)) if ev_path else []
        statue_site = next((cells[lab]["statue_com_footsite_mean_m"] for lab, _ in J + REPORT_SLOTS
                            if cells.get(lab) is not None and cells[lab]["statue_com_footsite_mean_m"] is not None), None)
        s3 = s3_step(evals)
        if s3["step"] is not None:
            c, why, entry = load_cell(root, run, arm, seed, s3["step"], "S3", expect_head=heads[arm], frozen=frozen,
                                      validation=validation, needs_noise=False, notes=notes)
            cells["S3"] = c
            if why == "invalid":
                invalid.append(entry)
        per_run[run] = {
            "arm": arm, "seed": seed, "name": name, "head": heads[arm], "cells": cells, "missing": missing, "invalid": invalid,
            "criteria": crits, "k1_failure_modes": k1_failure_modes(cells),
            "ledgers": {"eval_episodes": ev_path, "train_episodes": tr_path},
            "K5": k5_readout(evals, statue_site) if ev_path else {"computed": False, "why": "no eval_episodes.csv"},
            "late_falls": late_falls(read_csv(tr_path)) if tr_path else None,
            "S3": s3 if ev_path else {"step": None, "why": "no eval_episodes.csv"},
        }
        if crits["statue_median_mm_spread"] and crits["statue_median_mm_spread"] > 0.01:
            notes.append(f"{run}: the statue's x_c median differs across its judge files by {crits['statue_median_mm_spread']} mm")
    return {"schema": "stance3-score/v1", "criteria": crit, "judged_root": os.path.abspath(root),
            "runs_root": os.path.abspath(runs_root), "ledger_dirs": ledger_dirs, "validation": validation, "heads": heads,
            "frozen_params_sha256": fsha,
            "runs": per_run, "decision": decision(per_run), "best_final": best_final(per_run), "notes": notes}


# ------------------------------------------------------------------------------------------------ printing
def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render(res: dict) -> str:
    L = []
    L.append("T. rex stance3 screen: score against foot/stance3/criteria.txt "
             f"(sha256 {res['criteria']['sha256']}; {'matches the fixed text' if res['criteria']['matches'] else 'DOES NOT MATCH'})")
    if res["validation"]:
        L.append("*** VALIDATION ONLY: relaxed-binding judge files accepted (--validation); not a screen verdict ***")
    L.append(f"judge files: {res['judged_root']}   ledgers: {res['runs_root']}")
    L.append("K2c medians are over full-horizon episodes only, the policy's and the statue's on the same seeds (the "
             "statue stands 39 of 40; seed 3081 falls).")
    for run, arm, seed, name in RUNS:
        r = res["runs"][run]
        c, cells = r["criteria"], r["cells"]
        L.append("")
        L.append(f"== {name} {run} (arm {arm}, seed {seed}, clone HEAD {r['head'][:8]})"
                 + ("" if c["complete"] else "   INCOMPLETE (a J judge file is missing or invalid)"))
        L.append("   ckpt    verdict clean  LCB   full hop | K2a K2b  x_c med  statue  offset mm | roll | noise joint/own")
        for lab, _steps in list(J) + list(REPORT_SLOTS):
            x = cells.get(lab)
            tag = "" if (lab, _steps) in J else "  (report only)"
            if x is None:
                L.append(f"   {lab:<7} missing/invalid{tag}")
                continue
            L.append(f"   {lab:<7} {x['verdict']:>7} {x['n_clean']:>2}/40 {x['lcb']:.3f} {x['full']:>4} {x['hop_or_fall']:>3} |"
                     f" {x['k2a']:>3} {x['k2b']:>3} {fmt(x['xc_median_mm']):>8} {fmt(x['statue_xc_median_mm']):>7} {fmt(x['xc_offset_mm']):>8}  |"
                     f" {x['roll_rail']:>4} | {fmt(x['noise_joint_falls'])}/{fmt(x['noise_own_falls'])}{tag}")
        k1, k2 = c["K1"], c["K2"]
        L.append(f"   K1 {'PASS' if k1['pass'] else 'FAIL'}: v2 PASS at 6.3M {fmt(k1['pass_at']['6.3M'])}, at final "
                 f"{fmt(k1['pass_at']['final'])}; below 37/40 {k1['below_37'] or 'none'} (<= 1 allowed), below 33/40 "
                 f"{k1['below_33'] or 'none'}" + (f"; missing {k1['missing']}" if k1["missing"] else ""))
        L.append(f"   K2 {'PASS' if k2['pass'] else 'FAIL'}: K2a {'pass' if k2['K2a']['pass'] else 'FAIL ' + str(k2['K2a']['over'])}; "
                 f"K2b {'pass' if k2['K2b']['pass'] else 'FAIL ' + str(k2['K2b']['over'])}; "
                 f"K2c window {'pass' if k2['K2c']['window_pass'] else 'FAIL'} [-60, +30] mm, range "
                 f"{fmt(k2['K2c']['range_mm'])} mm {'pass' if k2['K2c']['range_pass'] else 'FAIL'} (<= 30)")
        L.append(f"   K3 {'PASS' if c['K3']['pass'] else 'FAIL'}: joint {c['K3']['joint_falls']} (<= 2/20), own "
                 f"{c['K3']['own_falls']} (<= 1/20)")
        L.append(f"   K4 {'PASS' if c['K4']['pass'] else 'FAIL'}: roll rail {c['K4']['roll_rail']} (<= 3/40)")
        L.append(f"   -> {'HOLDS' if c['HOLDS'] else 'does not hold (failed ' + ', '.join(c['failed']) + ')'}; "
                 f"FORE-AFT {'HOLDS' if c['FORE_AFT_HOLDS'] else 'does not hold'}")
        if r["missing"]:
            L.append("   missing J judge files (fail every criterion that needs them): " + "; ".join(r["missing"]))
        for x in r["invalid"]:
            L.append(f"   invalid: {x['path']}: {'; '.join(x['issues'])}")
        k5 = r["K5"]
        L.append("   report only:")
        if k5.get("computed"):
            L.append(f"     K5 trajectory: statue {k5['statue_m']:+.4f} m, window [{k5['window_m'][0]:+.4f}, {k5['window_m'][1]:+.4f}] m, "
                     f"{k5['evaluations_considered']} evaluations from 3,145,728: "
                     + ("FLAG " + "; ".join(f"{x['from']:,}-{x['to']:,} ({x['evaluations']})" for x in k5["runs"]) if k5["flag"] else "no flag"))
        else:
            L.append(f"     K5 trajectory: not computed ({k5.get('why')})")
        lf = r["late_falls"]
        if lf:
            pk = lf["peak"]
            peak = None if not pk or pk["fraction"] is None else 100.0 * pk["fraction"]
            L.append(f"     late falls (200 < length < 1000) per 262,144 steps: peak {fmt(peak)}% at {pk['bin_end']:,}"
                     if pk else "     late falls: no rows")
            if lf["flagged"]:
                L.append("       above 5%: " + ", ".join(f"{t['bin_end']:,} {100 * t['fraction']:.1f}%" for t in lf["flagged"]))
        else:
            L.append("     late falls: no train_episodes.csv")
        s3 = r["S3"]
        sel = []
        for tag, lab in (("S0 robust_best", "robust_best"), ("S1 best", "best"), ("S2 final", "final"),
                         (f"S3 {s3['step']:,}" if s3.get("step") else "S3 -", "S3")):
            x = cells.get(lab)
            sel.append(f"{tag}: " + ("not judged/invalid" if x is None else
                                     f"clean {x['n_clean']}/40, own-noise falls {fmt(x['noise_own_falls'])}, K2a {x['k2a']}, K2b {x['k2b']}"))
        L.append("     selection: " + " | ".join(sel) + ("" if s3.get("step") else f"  (S3: {s3.get('why')})"))
    L.append("")
    d = res["decision"]
    L.append("RECIPE VERDICTS: " + ", ".join(
        f"{name} {'HOLDS' if res['runs'][run]['criteria']['HOLDS'] else ('FORE-AFT HOLDS' if res['runs'][run]['criteria']['FORE_AFT_HOLDS'] else 'fails')}"
        for run, _a, _s, name in RUNS))
    L.append("DECISION TABLE" + (" (PROVISIONAL: judge files missing or invalid)" if d["provisional"] else "") + ":")
    for line in d["lines"]:
        L.append(f"   {line}")
    b = res["best_final"]
    L.append("Best final checkpoint (stance parent for walking pilot 2): "
             + (b["best"] if b["best"] else ("TIE: " + ", ".join(b["tied"]) if b.get("tied") else "none available")))
    for x in b["candidates"]:
        L.append(f"   {x['run']:<28} clean {x['clean']:>2}/40  joint + own noise falls {fmt(x['noise_falls'])}")
    if res["notes"]:
        L.append("")
        L.append("Notes:")
        for n in res["notes"]:
            L.append(f"   {n}")
    return "\n".join(L)


def inspect(paths: list[str]) -> int:
    """One file's counts as the scorer reads them, and the issues that would make it invalid in its own slot."""
    with open(FROZEN_PARAMS) as fh:
        frozen = json.load(fh)
    for path in paths:
        with open(path) as fh:
            d = json.load(fh)
        arm, seed, steps = d.get("arm"), d.get("seed"), d.get("steps")
        label = {s: lab for lab, s in J + REPORT_SLOTS}.get(steps, str(steps))
        bad, _ = validity_issues(d, arm=arm, seed=seed, steps=steps, label=label, expect_head=HEADS.get(arm), frozen=frozen,
                                 validation=False, needs_noise=label in NOISE_SLOTS)
        c = panel_counts(d) if d.get("panel") else {}
        keys = ("verdict", "n_clean", "lcb", "full", "falls", "hop_or_fall", "k2a", "k2a_seeds", "k2b", "k2b_seeds",
                "xc_median_mm", "statue_xc_median_mm", "xc_offset_mm", "com_mtp3_median_mm", "statue_com_mtp3_median_mm",
                "com_offset_mm", "statue_full", "statue_fall_seeds", "statue_com_footsite_mean_m", "roll_rail",
                "noise_joint_falls", "noise_joint_fall_seeds", "noise_own_falls", "noise_own_fall_seeds")
        print(json.dumps({"file": path, "mode": d.get("mode"), "arm": arm, "seed": seed, "steps": steps,
                          "counts": {k: c.get(k) for k in keys},
                          "common_noise": {k: (d.get("common_noise") or {}).get(k) for k in ("falls", "n", "fall_seeds")} if d.get("common_noise") else None,
                          "tail": {k: (d.get("tail") or {}).get(k) for k in ("falls", "n", "fall_seeds", "statue_falls")} if d.get("tail") else None,
                          "would_be_invalid_because": bad}, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("judged_root", nargs="?", help="directory holding <run id>/judge_<slot>.json")
    ap.add_argument("--runs-root", default=None, help="directory holding <run id>/[01_stance/]{eval,train}_episodes.csv")
    ap.add_argument("--json", default=None, help="also write the full result here (never overwritten)")
    ap.add_argument("--frozen-params", default=FROZEN_PARAMS)
    ap.add_argument("--head-L", default=HEADS["L"])
    ap.add_argument("--head-LT", default=HEADS["LT"])
    ap.add_argument("--validation", action="store_true", help="validation of this script only (see the docstring)")
    ap.add_argument("--inspect", nargs="+", default=None, help="print one or more judge files' counts and exit")
    ap.add_argument("--ledgers", action="append", default=[], metavar="RUN_ID=DIR",
                    help="read RUN_ID's eval/train_episodes.csv from DIR[/01_stance] (a run restarted as <RUN_ID>_b)")
    a = ap.parse_args()
    if a.inspect:
        return inspect(a.inspect)
    if not a.judged_root:
        ap.error("judged_root is required")
    if a.json and os.path.exists(a.json):
        raise SystemExit(f"refusing to overwrite {a.json}")
    ledger_dirs = {}
    for item in a.ledgers:
        run_id, sep, path = item.partition("=")
        if not sep or not run_id or not path:
            ap.error(f"--ledgers {item!r} is not RUN_ID=DIR")
        ledger_dirs[run_id] = path
    res = score(a.judged_root, runs_root=a.runs_root, frozen_path=a.frozen_params, heads={"L": a.head_L, "LT": a.head_LT},
                validation=a.validation, ledger_dirs=ledger_dirs)
    print(render(res))
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
