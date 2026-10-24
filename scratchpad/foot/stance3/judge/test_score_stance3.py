"""score_stance3.py on hand-made judge files and ledgers, each criterion just inside and just outside its bar.

    python3 test_score_stance3.py            (plain python; writes only under a temporary directory it removes)

Every case builds the three runs' judge files (J = 4.2M 5.2M 6.3M 8.4M 10.5M final) from a baseline that holds K1-K4
with margin, moves one quantity to the edge of its bar (inside) or one step past it (outside), scores the tree with
score_stance3.score and asserts the criterion's outcome per run.  Further cases: 3 against 4 episodes over a rail (K2a,
a fallen episode counting as over), K2b at exactly -0.929 and +0.571, an unmeasured foot, the K2c window and range at
their edges, K1 durability at 33 and 37 clean, missing and invalid files (smoke, relaxed binding, another HEAD, another
mode or arm, no K2 fields, a wrong control sigma, no noise at 6.3M, --allow-commit-mismatch or another recorded
git_commit, n_envs, hyperparameters_sha256 or seed, a checkpoint whose num_timesteps is not its slot's, an early-stop
final), --validation, every decision-table line, the best-final tie-break, the report-only K5, late-fall and S3
readouts on synthetic ledgers, and --ledgers for a run restarted under another RUN_ID.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_stance3 as SC  # noqa: E402

FROZEN = json.load(open(SC.FROZEN_PARAMS))
SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
SLOTS = dict(SC.J)
RUN_ARM = {run: (arm, seed) for run, arm, seed, _ in SC.RUNS}
R1, R2, R3 = (r[0] for r in SC.RUNS)
BARS = {"max_actuator_saturation_fraction": 0.1, "max_episode_yaw_change_deg": 25.0, "max_foot_contact_fraction": 0.02,
        "max_hop_or_fall_episodes": 1, "max_nonfoot_load_fraction": 0.02, "max_phantom_support_fraction": 0.05,
        "max_settle_airborne_substeps": 0, "max_settle_peak_floor_force_bw": 1.5, "max_settle_stance_width_change_m": 0.08,
        "max_settle_touchdowns": 2, "max_sole_cop_fore_aft": 0.8, "max_sole_corner_lift_m": 0.004, "max_sole_tilt_deg": 2.0,
        "max_touchdown_rate": 0.25, "max_window_displacement_m": 0.1, "min_all_feet_support": 0.98, "min_avg_reward": 2250.0,
        "min_avg_reward_statue_ratio": 0.6, "min_clean_stance_lcb": 0.8, "min_eval_episodes": 40, "min_foot_load_share": 0.4,
        "min_foot_load_share_windowed": 0.35, "min_full_horizon_fraction": 0.95, "min_sole_contacts": 1.5,
        "required_consecutive": 3, "settle_steps": 200}
STATUE_XC = 0.065


def episode(seed, *, full=True, clean=True, pad=0.4, w=(0.1, 0.1), xc=0.066, tilt=0.5, lift=0.001, hop=False,
            termination=None, foot=0.3):
    reasons = []
    if not full:
        reasons.append("horizon: 412 < 1000")
    if hop and full:
        reasons.append("min_all_feet_support: 0.9 < 0.98")
    if pad is None or pad > 0.8:
        reasons.append(f"max_sole_cop_fore_aft: {pad} > 0.8")
    if tilt > 2.0:
        reasons.append(f"max_sole_tilt_deg: {tilt} > 2.0")
    if lift > 0.004:
        reasons.append(f"max_sole_corner_lift_m: {lift} > 0.004")
    if not clean and not reasons:
        reasons.append("min_foot_load_share: 0.3 < 0.4")
    return {"seed": seed, "length": 1000 if full else 412, "reached_horizon": full, "full_horizon": full,
            "clean": not reasons, "reasons": reasons, "hop_or_fall": (not full) or hop, "reward": 3900.0,
            "termination_reason": "truncated" if full else (termination or "nosedive"),
            "max_sole_cop_fore_aft": pad, "max_foot_cop_fore_aft": foot, "max_sole_tilt_deg": tilt,
            "max_sole_corner_lift_m": lift, "foot_cop_fore_aft": {"r": w[0], "l": w[1]},
            "xc_both_feet_m": xc if full else None, "com_mtp3_head_m": xc if full else None,
            "com_fore_aft_footsite_m": -0.016 if full else None}


def statue_rows():
    return [episode(3042 + i, full=(i != 39), xc=STATUE_XC) for i in range(40)]


def noise(seed, space, falls):
    rows = [{"seed": 3042 + i, "length": 1000 if i >= falls else 300, "reached_horizon": i >= falls, "reward": 3800.0}
            for i in range(20)]
    d = {"n": 20, "falls": falls, "seeds": [3042, 3061], "space": space, "episodes_detail": rows}
    if space == "joint":
        ref = FROZEN["criteria"]["H4"]["control_sigma"][str(seed)]
        d.update({"sigma": "control", "sigma_source": ref["source"], "sigma_vector": list(ref["vector"])})
    else:
        d.update({"sigma": "own", "sigma_source": "own", "sigma_vector": [0.1] * 15})
    return d


def judge(run, label, *, eps=None, joint_falls=0, own_falls=0, verdict=None, terminations=None):
    """A valid judge file of *run* at slot *label*; eps defaults to 40 clean, full-horizon, centred episodes."""
    arm, seed = RUN_ARM[run]
    eps = eps if eps is not None else [episode(3042 + i) for i in range(40)]
    k = sum(e["clean"] for e in eps)
    full = sum(e["reached_horizon"] for e in eps)
    hop = sum(e["hop_or_fall"] for e in eps)
    lcb = SC.binomial_lcb(k, 40)
    if verdict is None:
        verdict = "PASS" if (k >= 37 and lcb >= 0.8 and full >= 38 and hop <= 1) else "FAIL"
    steps = SLOTS.get(label, label)
    stem = "stage1_final" if steps == "final" else f"{steps}_model" if steps in ("best", "robust_best") else f"stage1_{steps}_steps"
    terms = terminations or {"truncated": full, **({"nosedive": 40 - full} if full < 40 else {})}
    d = {"schema": SC.JUDGE_SCHEMA, "mode": "full", "smoke": False, "arm": arm, "seed": seed, "steps": steps, "checkpoint": stem,
         "git": {"head": SC.HEADS[arm], "status": "", "branch": "x"},
         "model_sha256": f"sha256:{run}{steps}", "plant_identity": {"physics_sha256": SC.R9_PHYSICS_SHA256, "physics_revision": 9},
         "arm_env_kwargs": {**SC.ARM_ENV_COMMON, **SC.ARM_ENV[arm]},
         "checks": {"relaxed": {"no_symmetry_check": False, "task_binding": "require", "allow_missing_records": False,
                                "allow_seed_mismatch": False, "allow_commit_mismatch": False},
                    "hard_failures": [], "plant_physics_sha256": {"match": True}, "symmetry": {"mirror_map_sha256_match": True},
                    "seed": {"label": seed, "checkpoint": seed, "match": True},
                    "checkpoint_num_timesteps": (SC.FINAL_TIMESTEPS if steps == "final" else steps if isinstance(steps, int)
                                                 else 3145728),
                    "stage_config": {"git_commit_match": True, "git_commit": SC.HEADS[arm], "clone_head": SC.HEADS[arm],
                                     "run_seed": seed, "run_n_envs": SC.N_ENVS,
                                     "run_hyperparameters_sha256": SC.HP_SHA256}},
         "panel": {"verdict": verdict, "passed": verdict == "PASS", "n_clean": k, "clean_lcb": lcb, "n_full_horizon": full,
                   "hop_or_fall_episodes": hop, "mean_reward": 3900.0, "statue_mean_reward": 3980.0,
                   "terminations": terms, "scored_gate_kind": "stance_quality/v2", "task_sha256": f"sha256:task{arm}",
                   "thresholds": dict(BARS), "statue_on_same_seeds": {"n": 40}, "statue_episodes_detail": statue_rows(),
                   "summary": {}, "episodes_detail": eps}}
    if label in ("6.3M", "final", "robust_best", "best") or isinstance(label, int):  # an int label: a selection slot
        d["noise_matched_control_joint"] = noise(seed, "joint", joint_falls)
        d["noise_matched_own"] = noise(seed, "policy", own_falls)
    return d


def baseline():
    return {run: {lab: judge(run, lab) for lab, _ in SC.J} for run, *_ in SC.RUNS}


def eps_with(n_bad=0, kind="unclean", start=0, **kw):
    """40 episodes, *n_bad* of them (from index *start*) altered by *kind*."""
    out = []
    for i in range(40):
        s = 3042 + i
        if not start <= i < start + n_bad:
            out.append(episode(s, **({"xc": kw["xc"]} if "xc" in kw else {})))
        elif kind == "unclean":
            out.append(episode(s, clean=False))
        elif kind == "fall":
            out.append(episode(s, full=False, **({"termination": kw["termination"]} if "termination" in kw else {})))
        elif kind == "hop":
            out.append(episode(s, hop=True))
        elif kind == "pad":
            out.append(episode(s, pad=kw.get("pad", 0.81)))
        elif kind == "w":
            out.append(episode(s, w=kw["w"]))
        elif kind == "tilt":
            out.append(episode(s, tilt=kw.get("tilt", 2.01)))
        elif kind == "lift":
            out.append(episode(s, lift=kw.get("lift", 0.0041)))
        elif kind == "fall_tilt":
            out.append(episode(s, full=False, tilt=3.0))
        else:
            raise ValueError(kind)
    return out


def with_xc(xc):
    return [episode(3042 + i, xc=xc) for i in range(40)]


def write_ledgers(root, run, *, evals=None, train=None):
    d = os.path.join(root, run, "01_stance")
    os.makedirs(d, exist_ok=True)
    if evals is not None:
        cols = ["evaluation", "timesteps", "episode", "env", "return", "length", "full_horizon", "cause",
                "max_sole_cop_fore_aft", "max_foot_cop_fore_aft", "com_fore_aft_mean_m"]
        with open(os.path.join(d, "eval_episodes.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for k, (ts, rows) in enumerate(evals):
                for i, (full, pad, foot, com) in enumerate(rows):
                    w.writerow([k, ts, i, 0, 4000.0, 1000 if full else 300, int(full), "truncated" if full else "fallen", pad, foot, com])
    if train is not None:
        with open(os.path.join(d, "train_episodes.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timesteps", "env", "length", "return", "cause"])
            for ts, length in train:
                w.writerow([ts, 0, length, 100.0, "x"])


def run(cells: dict, *, ledgers=None, validation=False, ledger_alias=None, map_alias=True):
    """Score *cells*; *ledger_alias* {run: dirname} writes that run's ledgers under dirname (a '<RUN_ID>_b' restart)
    and, with *map_alias*, maps it with score(ledger_dirs=...) as --ledgers does."""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    root = tempfile.mkdtemp(prefix="_score_unit_", dir=SCRATCH_DIR)
    alias = ledger_alias or {}
    try:
        for r, slots in cells.items():
            for lab, d in slots.items():
                if d is None:
                    continue
                os.makedirs(os.path.join(root, r), exist_ok=True)
                with open(os.path.join(root, r, f"judge_{SLOTS.get(lab, lab)}.json"), "w") as fh:
                    json.dump(d, fh)
        for r, kw in (ledgers or {}).items():
            write_ledgers(root, alias.get(r, r), **kw)
        ldirs = {r: os.path.join(root, name) for r, name in alias.items()} if map_alias else None
        return SC.score(root, validation=validation, ledger_dirs=ldirs)
    finally:
        shutil.rmtree(root)


RESULTS = []


def check(name, res, expect: dict, extra=None):
    """expect: {run: {criterion: bool}} with criterion in K1 K2 K3 K4 HOLDS FORE_AFT_HOLDS K2a K2b K2c."""
    got = {}
    for r, crits in expect.items():
        c = res["runs"][r]["criteria"]
        got[r] = {}
        for k in crits:
            if k in ("HOLDS", "FORE_AFT_HOLDS"):
                got[r][k] = c[k]
            elif k in ("K2a", "K2b", "K2c"):
                got[r][k] = c["K2"][k]["pass"]
            else:
                got[r][k] = c[k]["pass"]
    ok = got == expect
    if extra is not None:
        ok = ok and bool(extra(res))
    RESULTS.append((name, ok))
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok or extra is None else "  (extra check failed)"))
    return ok


def set_cell(c, r, lab, d):
    c[r][lab] = d
    return c


def main() -> int:
    ALL = {"K1": True, "K2": True, "K3": True, "K4": True, "HOLDS": True}
    check("baseline: every run holds", run(baseline()), {R1: ALL, R2: ALL, R3: ALL},
          lambda r: r["decision"]["lines"][0].startswith("1.") and not r["decision"]["provisional"])
    # ---- K2a: 3 against 4 over the pad rail at one checkpoint; a fallen episode counts as over; exactly 0.80 is not over
    c = baseline()
    set_cell(c, R2, "8.4M", judge(R2, "8.4M", eps=eps_with(3, "pad")))
    check("K2a inside: 3/40 over the pad rail (R2 8.4M)", run(c), {R2: {"K2a": True, "K2": True}})
    set_cell(c, R2, "8.4M", judge(R2, "8.4M", eps=eps_with(4, "pad")))
    check("K2a outside: 4/40 over the pad rail (R2 8.4M)", run(c), {R2: {"K2a": False, "K2": False, "FORE_AFT_HOLDS": False}},
          lambda r: r["runs"][R2]["criteria"]["K2"]["K2a"]["over"] == ["8.4M"])
    set_cell(c, R2, "8.4M", judge(R2, "8.4M", eps=eps_with(3, "pad")[:3] + eps_with(1, "fall", start=3)[3:4] + eps_with(0)[4:]))
    check("K2a outside: 3 over + 1 fallen (a non-full-horizon episode counts as over)", run(c), {R2: {"K2a": False}})
    set_cell(c, R2, "8.4M", judge(R2, "8.4M", eps=eps_with(10, "pad", pad=0.80)))
    check("K2a inside: 10 episodes at exactly 0.80 (strict >)", run(c), {R2: {"K2a": True}})
    set_cell(c, R2, "8.4M", judge(R2, "8.4M", eps=eps_with(3, "pad")[:3] + eps_with(1, "pad", start=3, pad=None)[3:4] + eps_with(0)[4:]))
    check("K2a outside: 3 over + 1 unmeasured", run(c), {R2: {"K2a": False}})
    # ---- K2b: exactly -0.929 and +0.571 inside; one step past outside; 3 against 4 outside; unmeasured and fallen
    c = baseline()
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(20, "w", w=(-0.929, 0.571))))
    check("K2b inside: 20 episodes at exactly -0.929 / +0.571", run(c), {R1: {"K2b": True, "HOLDS": True}})
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(3, "w", w=(0.0, 0.5711)) + []))
    check("K2b inside: 3 episodes at +0.5711", run(c), {R1: {"K2b": True}})
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(4, "w", w=(0.0, 0.5711))))
    check("K2b outside: 4 episodes at +0.5711", run(c), {R1: {"K2b": False, "K2": False}},
          lambda r: r["runs"][R1]["criteria"]["K2"]["K2b"]["seeds"]["5.2M"] == [3042, 3043, 3044, 3045])
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(4, "w", w=(-0.9291, 0.0))))
    check("K2b outside: 4 episodes at -0.9291 (the heel side)", run(c), {R1: {"K2b": False}})
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(3, "w", w=(-0.9291, 0.0))[:3] + eps_with(1, "w", start=3, w=(None, 0.1))[3:4] + eps_with(0)[4:]))
    check("K2b outside: 3 outside + 1 unmeasured foot", run(c), {R1: {"K2b": False}})
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(3, "w", w=(0.6, 0.0))[:3] + eps_with(1, "fall", start=3)[3:4] + eps_with(0)[4:]))
    check("K2b outside: 3 outside + 1 fallen", run(c), {R1: {"K2b": False}})
    # ---- K2c: window [-60, +30] mm of the statue median, at its edges; range <= 30 mm over the six
    c = baseline()
    set_cell(c, R3, "10.5M", judge(R3, "10.5M", eps=with_xc(STATUE_XC - 0.060)))
    check("K2c inside: offset exactly -60 mm (range 61 -> fails range only)", run(c), {R3: {"K2c": False}},
          lambda r: r["runs"][R3]["criteria"]["K2"]["K2c"]["window_pass"] and not r["runs"][R3]["criteria"]["K2"]["K2c"]["range_pass"])
    c = baseline()
    for lab, _ in SC.J:
        set_cell(c, R3, lab, judge(R3, lab, eps=with_xc(STATUE_XC - 0.060 + (0.030 if lab == "final" else 0.0)),
                                   joint_falls=0))
    check("K2c inside: offsets -60 mm x5 and -30 mm, range exactly 30 mm", run(c), {R3: {"K2c": True, "K2": True}},
          lambda r: r["runs"][R3]["criteria"]["K2"]["K2c"]["range_mm"] == 30.0)
    set_cell(c, R3, "final", judge(R3, "final", eps=with_xc(STATUE_XC - 0.060 + 0.030001)))
    check("K2c outside: range 30.001 mm", run(c), {R3: {"K2c": False}},
          lambda r: r["runs"][R3]["criteria"]["K2"]["K2c"]["window_pass"])
    c = baseline()
    set_cell(c, R3, "6.3M", judge(R3, "6.3M", eps=with_xc(STATUE_XC - 0.060001)))
    check("K2c outside: offset -60.001 mm", run(c), {R3: {"K2c": False}},
          lambda r: not r["runs"][R3]["criteria"]["K2"]["K2c"]["window_pass"])
    c = baseline()
    for lab, _ in SC.J:
        set_cell(c, R3, lab, judge(R3, lab, eps=with_xc(STATUE_XC + 0.030)))
    check("K2c inside: offset exactly +30 mm at all six", run(c), {R3: {"K2c": True}})
    set_cell(c, R3, "4.2M", judge(R3, "4.2M", eps=with_xc(STATUE_XC + 0.030001)))
    check("K2c outside: offset +30.001 mm", run(c), {R3: {"K2c": False}})
    c = baseline()
    set_cell(c, R3, "4.2M", judge(R3, "4.2M", eps=with_xc(STATUE_XC + 0.031)[:39] + [episode(3081, full=False, xc=0.5)]))
    check("K2c: the median is over full-horizon episodes (a fall is left out of it)", run(c), {R3: {"K2c": False}},
          lambda r: r["runs"][R3]["cells"]["4.2M"]["xc_offset_mm"] == 31.0)
    # ---- K1 at 6.3M / final, and durability across J at 33 and 37
    c = baseline()
    set_cell(c, R1, "6.3M", judge(R1, "6.3M", eps=eps_with(3, "unclean")))
    set_cell(c, R1, "final", judge(R1, "final", eps=eps_with(1, "fall")))
    check("K1 inside: 37/40 at 6.3M, one fall at final (full 39/40, hop-or-fall 1)", run(c), {R1: {"K1": True}})
    set_cell(c, R1, "final", judge(R1, "final", eps=eps_with(2, "fall")))
    check("K1 outside: full 38/40 at final (two falls are hop-or-fall 2)", run(c), {R1: {"K1": False}},
          lambda r: r["runs"][R1]["criteria"]["K1"]["detail"]["final"]["components"]["full_ge_38"])
    set_cell(c, R1, "final", judge(R1, "final"))
    set_cell(c, R1, "6.3M", judge(R1, "6.3M", eps=eps_with(4, "unclean")))
    check("K1 outside: 36/40 at 6.3M", run(c), {R1: {"K1": False, "HOLDS": False}})
    c = baseline()
    set_cell(c, R1, "final", judge(R1, "final", eps=eps_with(1, "fall")[:1] + eps_with(1, "hop", start=1)[1:2] + eps_with(0)[2:]))
    check("K1 outside: hop-or-fall 2 at final", run(c), {R1: {"K1": False}})
    c = baseline()
    set_cell(c, R1, "final", judge(R1, "final", verdict="FAIL"))
    check("K1 outside: repository verdict FAIL with passing components", run(c), {R1: {"K1": False}},
          lambda r: any("disagrees" in n for n in r["notes"]))
    c = baseline()
    set_cell(c, R2, "5.2M", judge(R2, "5.2M", eps=eps_with(7, "unclean")))
    check("K1 durability inside: one checkpoint at 33/40", run(c), {R2: {"K1": True}},
          lambda r: r["runs"][R2]["criteria"]["K1"]["below_37"] == ["5.2M"])
    set_cell(c, R2, "5.2M", judge(R2, "5.2M", eps=eps_with(8, "unclean")))
    check("K1 durability outside: one checkpoint at 32/40", run(c), {R2: {"K1": False}},
          lambda r: r["runs"][R2]["criteria"]["K1"]["below_33"] == ["5.2M"])
    c = baseline()
    set_cell(c, R2, "4.2M", judge(R2, "4.2M", eps=eps_with(3, "unclean")))
    set_cell(c, R2, "8.4M", judge(R2, "8.4M", eps=eps_with(3, "unclean")))
    set_cell(c, R2, "10.5M", judge(R2, "10.5M", eps=eps_with(4, "unclean")))
    check("K1 durability inside: 37, 37 and one 36", run(c), {R2: {"K1": True}})
    set_cell(c, R2, "4.2M", judge(R2, "4.2M", eps=eps_with(4, "unclean")))
    check("K1 durability outside: two checkpoints at 36/40", run(c), {R2: {"K1": False}})
    # ---- K3 and K4
    c = baseline()
    set_cell(c, R3, "final", judge(R3, "final", joint_falls=2, own_falls=1))
    check("K3 inside: joint 2/20, own 1/20 at final", run(c), {R3: {"K3": True}})
    set_cell(c, R3, "final", judge(R3, "final", joint_falls=3, own_falls=1))
    check("K3 outside: joint 3/20 at final", run(c), {R3: {"K3": False, "FORE_AFT_HOLDS": True}})
    set_cell(c, R3, "final", judge(R3, "final", joint_falls=0, own_falls=2))
    check("K3 outside: own 2/20 at final", run(c), {R3: {"K3": False}})
    c = baseline()
    set_cell(c, R1, "6.3M", judge(R1, "6.3M", eps=eps_with(3, "tilt")))
    set_cell(c, R1, "final", judge(R1, "final", eps=eps_with(2, "lift")[:2] + eps_with(1, "fall_tilt", start=2)[2:3] + eps_with(0)[3:]))
    check("K4 inside: 3 at 6.3M; 2 + a fallen roll at final", run(c), {R1: {"K4": True}})
    set_cell(c, R1, "final", judge(R1, "final", eps=eps_with(4, "tilt", tilt=2.0000001)))
    check("K4 outside: 4 at final", run(c), {R1: {"K4": False}})
    set_cell(c, R1, "final", judge(R1, "final", eps=eps_with(5, "tilt", tilt=2.0)[:5] + eps_with(10, "lift", lift=0.004)[5:10] + eps_with(0)[10:]))
    check("K4 inside: tilt exactly 2.0 / lift exactly 0.004 are not over", run(c), {R1: {"K4": True}})
    # ---- missing and invalid files
    c = baseline()
    c[R2]["8.4M"] = None
    check("missing R2 8.4M: K1 and K2 fail, K3 K4 hold, INCOMPLETE, decision provisional", run(c),
          {R2: {"K1": False, "K2": False, "K3": True, "K4": True}},
          lambda r: len(r["runs"][R2]["missing"]) == 1 and r["decision"]["provisional"] and not r["runs"][R2]["criteria"]["complete"])
    c = baseline()
    c[R3]["final"] = None
    check("missing R3 final: K1 K2 K3 K4 all fail", run(c), {R3: {"K1": False, "K2": False, "K3": False, "K4": False}})
    for name, mut in (
        ("smoke", lambda d: d.update(smoke=True)),
        ("quick mode", lambda d: d.update(mode="quick")),
        ("another clone HEAD", lambda d: d["git"].update(head=SC.HEADS["L"])),
        ("dirty clone", lambda d: d["git"].update(status=" M x")),
        ("judged as arm L", lambda d: d.update(arm="L")),
        ("arm L's leg weights", lambda d: d["arm_env_kwargs"].update(leg_home_pose_weight=0.5)),
        ("task binding relaxed", lambda d: d["checks"]["relaxed"].update(task_binding="report")),
        ("symmetry check off", lambda d: d["checks"]["relaxed"].update(no_symmetry_check=True)),
        ("hard binding failure", lambda d: d["checks"].update(hard_failures=["x"])),
        ("slot relabelled (seed 60 file in s52)", lambda d: d.update(seed=60)),
        ("wrong control sigma", lambda d: d["noise_matched_control_joint"].update(sigma_vector=[0.2] * 15)),
        ("no own-noise panel", lambda d: d.pop("noise_matched_own")),
        ("rows without K2 fields", lambda d: [e.pop("foot_cop_fore_aft") for e in d["panel"]["episodes_detail"]]),
        ("statue on 39 seeds", lambda d: d["panel"]["statue_episodes_detail"].pop()),
        ("whole-foot rail in the bars", lambda d: d["panel"]["thresholds"].update(max_foot_cop_fore_aft=0.8)),
        ("not the r9 plant", lambda d: d["plant_identity"].update(physics_revision=10)),
        ("judged with --allow-commit-mismatch, another git_commit",
         lambda d: (d["checks"]["relaxed"].update(allow_commit_mismatch=True),
                    d["checks"]["stage_config"].update(git_commit="0" * 40, git_commit_match=False))),
        ("judged with --allow-commit-mismatch alone", lambda d: d["checks"]["relaxed"].update(allow_commit_mismatch=True)),
        ("stage_config git_commit not the clone HEAD",
         lambda d: d["checks"]["stage_config"].update(git_commit="0" * 40, git_commit_match=False)),
        ("no stage_config record", lambda d: d["checks"].pop("stage_config")),
        ("n_envs 8", lambda d: d["checks"]["stage_config"].update(run_n_envs=8)),
        ("no n_envs recorded (an older judge)", lambda d: d["checks"]["stage_config"].pop("run_n_envs")),
        ("another hyperparameters_sha256", lambda d: d["checks"]["stage_config"].update(run_hyperparameters_sha256="sha256:" + "e" * 64)),
        ("stage_config seed 60 in s52", lambda d: d["checks"]["stage_config"].update(run_seed=60)),
        ("6.3M checkpoint holds 6,299,648 steps", lambda d: d["checks"].update(checkpoint_num_timesteps=6299648)),
        ("no num_timesteps recorded", lambda d: d["checks"].pop("checkpoint_num_timesteps")),
    ):
        c = baseline()
        d = copy.deepcopy(c[R3]["6.3M"])
        mut(d)
        c[R3]["6.3M"] = d
        check(f"invalid R3 6.3M ({name}): K1 K2 K3 K4 fail, listed", run(c), {R3: {"K1": False, "K2": False, "K3": False, "K4": False}},
              lambda r: len(r["runs"][R3]["invalid"]) == 1 and r["decision"]["provisional"])
    c = baseline()
    d = copy.deepcopy(c[R1]["4.2M"])
    d["checks"]["relaxed"]["task_binding"] = "report"
    c[R1]["4.2M"] = d
    check("--validation accepts a relaxed-binding file, noted", run(c, validation=True), {R1: ALL},
          lambda r: any("VALIDATION ONLY" in n for n in r["notes"]) and not r["runs"][R1]["invalid"])
    c = baseline()
    d = copy.deepcopy(c[R1]["4.2M"])
    d["checks"]["relaxed"]["allow_commit_mismatch"] = True
    d["checks"]["stage_config"].update(git_commit="0" * 40, git_commit_match=False, run_n_envs=None,
                                       run_hyperparameters_sha256=None)
    c[R1]["4.2M"] = d
    check("--validation accepts an --allow-commit-mismatch file without n_envs / recipe records, noted",
          run(c, validation=True), {R1: ALL},
          lambda r: any("VALIDATION ONLY" in n and "allow_commit_mismatch" in n and "n_envs" in n for n in r["notes"])
          and not r["runs"][R1]["invalid"])
    check("... and the same file is invalid without --validation", run(c), {R1: {"K1": False, "K2": False}},
          lambda r: len(r["runs"][R1]["invalid"]) == 1 and any("allow_commit_mismatch" in i for i in r["runs"][R1]["invalid"][0]["issues"]))
    # ---- 'final' is stage1_final at 12,001,280 steps: an early stop's stage1_final is not scored as final
    c = baseline()
    d = copy.deepcopy(c[R2]["final"])
    d["checks"]["checkpoint_num_timesteps"] = 9437184
    c[R2]["final"] = d
    check("invalid R2 final: an early-stop stage1_final at 9,437,184 steps fails K1 K2 K3 K4", run(c),
          {R2: {"K1": False, "K2": False, "K3": False, "K4": False}},
          lambda r: len(r["runs"][R2]["invalid"]) == 1 and "12,001,280" in r["runs"][R2]["invalid"][0]["issues"][0])
    c = baseline()
    c[R1]["robust_best"] = judge(R1, "robust_best")
    c[R1]["robust_best"]["checks"]["checkpoint_num_timesteps"] = 655360
    check("robust_best at any step is not refused for its num_timesteps", run(c), {R1: ALL},
          lambda r: r["runs"][R1]["cells"]["robust_best"] is not None and not r["runs"][R1]["invalid"])
    # ---- the decision table
    c = baseline()
    set_cell(c, R3, "final", judge(R3, "final", joint_falls=3))
    check("line 2: both L+T FORE-AFT HOLD, R3 fails K3", run(c), {R2: {"HOLDS": True}, R3: {"HOLDS": False, "FORE_AFT_HOLDS": True}},
          lambda r: [x[:2] for x in r["decision"]["lines"]] == ["2.", "5."])
    c = baseline()
    set_cell(c, R3, "6.3M", judge(R3, "6.3M", eps=eps_with(4, "w", w=(0.7, 0.0))))
    check("lines 3a and 5: R3 fails K2, Run 1 HOLDS, R2 HOLDS", run(c), {R1: {"HOLDS": True}, R3: {"K2": False}},
          lambda r: [x[:3] for x in r["decision"]["lines"]] == ["3a.", "5. "])
    c = baseline()
    set_cell(c, R2, "6.3M", judge(R2, "6.3M", eps=eps_with(4, "pad")))
    set_cell(c, R3, "6.3M", judge(R3, "6.3M", eps=eps_with(4, "pad")))
    set_cell(c, R1, "5.2M", judge(R1, "5.2M", eps=eps_with(4, "w", w=(0.7, 0.0))))
    check("line 3b: both L+T and Run 1 fail K2", run(c), {R1: {"K2": False}, R2: {"K2": False}, R3: {"K2": False}},
          lambda r: [x[:3] for x in r["decision"]["lines"]] == ["3b."])
    c = baseline()
    set_cell(c, R2, "6.3M", judge(R2, "6.3M", eps=eps_with(4, "unclean")))
    set_cell(c, R2, "final", judge(R2, "final", eps=[episode(3042 + i, hop=(i < 2)) for i in range(40)],
                                   terminations={"truncated": 38, "head_contact": 2}))
    check("line 4: R2 fails K1 (hop rails, head contact) while K2 holds", run(c), {R2: {"K1": False, "K2": True}},
          lambda r: any(x.startswith("4. Run 2") for x in r["decision"]["lines"]))
    c = baseline()
    set_cell(c, R3, "6.3M", judge(R3, "6.3M", eps=eps_with(4, "unclean")))
    check("line 4?: R3 fails K1 without the named modes; line 5 too", run(c), {R3: {"K1": False, "K2": True}},
          lambda r: any(x.startswith("4? Run 3") for x in r["decision"]["lines"]) and r["decision"]["lines"][-1].startswith("5."))
    # ---- best final
    c = baseline()
    set_cell(c, R1, "final", judge(R1, "final", joint_falls=1))
    set_cell(c, R2, "final", judge(R2, "final", joint_falls=0, own_falls=1))
    set_cell(c, R3, "final", judge(R3, "final", eps=eps_with(1, "unclean")))
    check("best final: tie at 40/40 and 1 fall -> tied", run(c), {R1: {"HOLDS": True}},
          lambda r: r["best_final"]["best"] is None and len(r["best_final"]["tied"]) == 2)
    set_cell(c, R2, "final", judge(R2, "final"))
    check("best final: Run 2 (40/40, 0 falls)", run(c), {R2: {"HOLDS": True}},
          lambda r: r["best_final"]["best"] == f"Run 2 {R2}")
    # ---- report only: K5, late falls, S3 on synthetic ledgers
    def ev(ts, com=-0.016, full=10, pad=0.3, foot=0.3, n=10):
        return (ts, [(i < full, pad, foot, com) for i in range(n)])
    step = 131072
    evals = [ev(k * step) for k in range(1, 41)]                          # 131072 .. 5242880
    # K5: 4 consecutive outside at/after 3,145,728 -> flag; 3 elsewhere -> no flag; window statue -0.016 + [-0.060, +0.030]
    for k in (25, 26, 27, 28):                                            # 3,276,800 .. 3,670,016
        evals[k - 1] = ev(k * step, com=-0.016 + 0.0301)
    for k in (33, 34, 35):
        evals[k - 1] = ev(k * step, com=-0.016 - 0.0601)
    evals[36] = ev(37 * step, com=-0.016 + 0.030)                         # exactly on the edge: inside
    # S3: evaluations 17..24 (up to 3,145,728) clean -> S3 candidate 3,145,728; 4,194,304 needs 25..32: 25-28 are clean
    # too (posture is not part of S3), so make 30 have a fall and 31 a pad over -> next eligible is 5,242,880 (33..40)
    evals[29] = ev(30 * step, full=9)
    evals[30] = ev(31 * step, pad=0.81)
    train = [(ts, 1000) for ts in range(1000, 262144 * 3, 1000)] + [(300000 + i, 500) for i in range(30)]
    c = baseline()
    res = run(c, ledgers={R1: {"evals": evals, "train": train}})
    k5, lf, s3 = res["runs"][R1]["K5"], res["runs"][R1]["late_falls"], res["runs"][R1]["S3"]
    ok_k5 = k5["flag"] and [(x["from"], x["to"], x["evaluations"]) for x in k5["runs"]] == [(25 * step, 28 * step, 4)]
    ok_s3 = s3["step"] == 5242880
    b1 = [t for t in lf["bins"] if t["bin_end"] == 2 * 262144][0]
    ok_lf = b1["late"] == 30 and abs(b1["fraction"] - 30 / b1["episodes"]) < 1e-12 and [t["bin_end"] for t in lf["flagged"]] == [2 * 262144]
    check("report only: K5 flags 4 consecutive (not 3, not the edge), S3 = 5,242,880, late falls flag one bin", res,
          {R1: {"HOLDS": True}}, lambda r: ok_k5 and ok_s3 and ok_lf)
    evals2 = list(evals)
    evals2[38] = ev(39 * step, foot=0.81)
    res = run(c, ledgers={R1: {"evals": evals2}})
    check("report only: S3 falls back to 3,670,016 when the 5,242,880 window has a whole-foot over", res, {R1: {"HOLDS": True}},
          lambda r: r["runs"][R1]["S3"]["step"] == 3670016 and r["runs"][R1]["late_falls"] is None)
    c = baseline()
    c[R1][3670016] = judge(R1, 3670016, own_falls=2)  # the S3 judge file
    c[R1][3145728] = judge(R1, 3145728)               # the report-only 3.1M slot
    c[R1]["robust_best"] = judge(R1, "robust_best", eps=eps_with(5, "unclean"), own_falls=7)
    res = run(c, ledgers={R1: {"evals": evals2}})
    sel = res["runs"][R1]["cells"]
    check("report only: the selection comparison reads the S0 and S3 judge files", res, {R1: {"HOLDS": True}},
          lambda r: sel["S3"]["noise_own_falls"] == 2 and sel["3.1M"]["n_clean"] == 40 and sel["robust_best"]["n_clean"] == 35
          and sel["best"] is None)
    # ---- --ledgers: a run restarted as '<RUN_ID>_b' keeps its judge files under <RUN_ID>/, its ledgers are elsewhere
    c = baseline()
    res = run(c, ledgers={R1: {"evals": evals, "train": train}}, ledger_alias={R1: R1 + "_b"})
    k5b, s3b = res["runs"][R1]["K5"], res["runs"][R1]["S3"]
    check("--ledgers: R1's ledgers read from stance3_armL_s60_b give the same K5, S3 and late falls", res, {R1: {"HOLDS": True}},
          lambda r: k5b == k5 and s3b["step"] == 5242880 and r["runs"][R1]["late_falls"] == lf
          and r["runs"][R1]["ledgers"]["eval_episodes"].endswith(os.path.join(R1 + "_b", "01_stance", "eval_episodes.csv"))
          and any("--ledgers" in n for n in r["notes"]))
    res = run(c, ledgers={R1: {"evals": evals, "train": train}}, ledger_alias={R1: R1 + "_b"}, map_alias=False)
    check("--ledgers: without the mapping the _b ledgers are not found (K5 not computed)", res, {R1: {"HOLDS": True}},
          lambda r: not r["runs"][R1]["K5"]["computed"] and r["runs"][R1]["late_falls"] is None)
    try:
        SC.score(SCRATCH_DIR, ledger_dirs={"stance3_armL_s52": "/nonexistent"})
        refused = False
    except SystemExit:
        refused = True
    RESULTS.append(("--ledgers: an unknown RUN_ID is refused", refused))
    print(f"{'ok  ' if refused else 'FAIL'} --ledgers: an unknown RUN_ID is refused")
    n_ok = sum(1 for r in RESULTS if r[1])
    print(f"\n{n_ok}/{len(RESULTS)} cases as expected")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
