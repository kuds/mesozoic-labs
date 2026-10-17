"""Score the r10 foot + arm L screen against its fixed criteria (foot/r10screen/criteria.txt), beside r9 arm L.

    python3 score_r10.py R10_ROOT [--json OUT.json]

R10_ROOT/s{seed}/judge_{steps}.json for seed 42, 52, 60 and steps 4194304 (4M), 5242880 (5M), 6291456 (6M): the files
judge_r10.py wrote.  Plain python (no repository import).  The verdict holds K1-K4 and nothing else:

  K1 gate        stance_quality/v2 PASS at 6M on >= 2 of 3 seeds.  A seed passes when the repository's verdict is
                 PASS and the criteria's components hold on its 40-episode panel (clean >= 37/40 with the exact
                 Clopper-Pearson LCB >= 0.80, full horizon >= 0.95, hop-or-fall <= 1); a disagreement between the two
                 is listed as a problem and the seed counts as failed.
  K2 fore-aft    every seed, each of 4M / 5M / 6M: full-horizon panel episodes with max_foot_cop_fore_aft > 0.80
                 <= 3 of 40 (an unmeasured value on a full-horizon episode counts as over and is listed).
  K3 robustness  at 6M: joint-space control-vector noise falls <= 7 of 60 pooled, and own-sigma noise falls <= 3 of 60
                 pooled (the r9 screen's H4: 20 spawns 3042-3061 per seed, frozen control sigma per seed).
  K4 roll        at 6M: full-horizon episodes with max_sole_tilt_deg > 2.0 or max_sole_corner_lift_m > 0.004
                 <= 6 of 120 pooled and <= 3 of 40 on each seed.
  overall        PASS when all four hold; otherwise FAIL naming the failed criteria.

A missing judge file, or one that fails the validity checks (wrong slot, smoke, relaxed bindings, another clone HEAD,
panel seeds / bars / noise configuration other than the screen's), counts as failed in every criterion that needs it
and is listed.  The r9 arm L numbers are recomputed from the r9 screen's judge files with the r9 rails (the pad's
max_sole_cop_fore_aft > 0.8 counted over all 40 episodes, as H3a did) and checked against its verdict.json.
Report only (never in the verdict): the r10 pad CoP counts, the whole-foot CoP means, mean reward against the statue,
terminations.  The best stance policy (criteria.txt "What follows"): the most clean panel episodes among the six 6M
checkpoints (r10 and r9); a tie goes to fewer noise falls (joint-space + own-sigma, summed).

--validation-synthetic (validation of this script only): accept judge files marked "synthetic" (one judge JSON copied
into several slots) and skip the two checks such a copy cannot meet (the checkpoint's recorded seed against the
slot, the frozen control sigma of the slot's seed); the output is marked VALIDATION ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

SCRATCH = "scratchpad"
R9_RUNS = SCRATCH + "/foot/symmetry/screen/runs"
R9_RUN_NAME = "sym_lean_settled"
R9_VERDICT = SCRATCH + "/foot/symmetry/screen/analysis/verdict.json"
FROZEN_PARAMS = SCRATCH + "/foot/symmetry/design/frozen_params.json"
FROZEN_PARAMS_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
EXPECT_HEAD = "026eb833a0c84989a3656f92b6229eab11febade"
JUDGE_SCHEMA = "r10screen-judge/v1"

SEEDS = (42, 52, 60)
CKPTS = (("4M", 4194304), ("5M", 5242880), ("6M", 6291456))
PANEL_SEEDS = list(range(3042, 3082))
NOISE_SEEDS = list(range(3042, 3062))
# criteria.txt, fixed before any r10 training
K1_MIN_SEEDS, K1_MIN_CLEAN, K1_MIN_LCB, K1_MIN_FULL_FRACTION, K1_MAX_HOP = 2, 37, 0.80, 0.95, 1
K2_BAR, K2_MAX = 0.80, 3                      # r10: max_foot_cop_fore_aft > 0.80 on full-horizon episodes
R9_PAD_BAR = 0.8                              # r9: the pad's max_sole_cop_fore_aft > 0.8, all episodes (H3a)
K3_JOINT_MAX, K3_OWN_MAX = 7, 3
K4_TILT, K4_LIFT, K4_POOLED_MAX, K4_SEED_MAX = 2.0, 0.004, 6, 3
PANEL_N, NOISE_N = 40, 20
#: The r10 stance stage's own bars the panel must have been scored at (the criteria's numbers among them).
R10_BARS = {"min_eval_episodes": 40, "min_clean_stance_lcb": 0.8, "min_full_horizon_fraction": 0.95,
            "max_hop_or_fall_episodes": 1, "max_foot_cop_fore_aft": 0.8, "max_sole_tilt_deg": 2.0,
            "max_sole_corner_lift_m": 0.004, "settle_steps": 200}
#: stance_gate_v2.HOP_OR_FALL_KEYS plus the early end ("horizon"); frozen_params gate.hop_or_fall_keys is the same list.
HOP_KEYS = {"min_all_feet_support", "max_touchdown_rate", "max_window_displacement_m", "max_actuator_saturation_fraction",
            "max_window_airborne_substeps", "max_window_peak_floor_force_bw", "horizon"}


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


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------------------------------------ one judge file
def panel_counts(d: dict, *, rail: str) -> dict:
    """Every count the criteria read from one judge file, recomputed from its per-episode rows.

    rail "foot" (r10): K2 counts full-horizon episodes with max_foot_cop_fore_aft > 0.80.
    rail "pad"  (r9):  K2 counts every episode with max_sole_cop_fore_aft > 0.8 (H3a's count)."""
    p = d["panel"]
    eps = p["episodes_detail"]
    n = len(eps)
    full = [bool(e["reached_horizon"]) for e in eps]
    clean = [bool(e["clean"]) for e in eps]
    hop = [(not f) or any(reason_key(r) in HOP_KEYS for r in (e.get("reasons") or [])) for e, f in zip(eps, full)]
    k = sum(clean)
    lcb = binomial_lcb(k, n) if n else 0.0
    roll = [f and (over(e.get("max_sole_tilt_deg"), K4_TILT) or over(e.get("max_sole_corner_lift_m"), K4_LIFT)) for e, f in zip(eps, full)]
    foot = [e.get("max_foot_cop_fore_aft") for e in eps]
    pad = [e.get("max_sole_cop_fore_aft") for e in eps]
    foot_fh = sum(1 for v, f in zip(foot, full) if f and (v is None or v > K2_BAR))
    out = {
        "n": n, "n_clean": k, "lcb": lcb, "lcb_recorded": p.get("clean_lcb"), "full": sum(full), "falls": n - sum(full),
        "hop_or_fall": sum(hop), "hop_or_fall_recorded": p.get("hop_or_fall_episodes"), "verdict": p.get("verdict"),
        "roll_rail": sum(roll),
        "foot_cop_fh": foot_fh, "foot_cop_any": sum(1 for v in foot if over(v, K2_BAR)),
        "foot_cop_unmeasured_fh": sum(1 for v, f in zip(foot, full) if f and v is None),
        "pad_cop_any": sum(1 for v in pad if over(v, R9_PAD_BAR)),
        "pad_cop_fh": sum(1 for v, f in zip(pad, full) if f and over(v, R9_PAD_BAR)),
        "mean_reward": p.get("mean_reward"), "statue_mean_reward": p.get("statue_mean_reward"),
        "terminations": p.get("terminations"),
        "noise_joint_falls": None, "noise_own_falls": None,
    }
    out["k2_count"] = foot_fh if rail == "foot" else out["pad_cop_any"]
    comp = {"n_eq_40": n == PANEL_N, "clean_ge_37": k >= K1_MIN_CLEAN, "lcb_ge_0.80": lcb >= K1_MIN_LCB,
            "full_ge_0.95": (sum(full) / n if n else 0.0) >= K1_MIN_FULL_FRACTION, "hop_or_fall_le_1": sum(hop) <= K1_MAX_HOP}
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
    # report only (r10): whole-foot CoP means per foot over the full-horizon episodes
    pf = (p.get("summary") or {}).get("per_foot") or {}
    out["foot_signed_cop_mean"] = {f: v.get("foot_signed_cop_mean") for f, v in pf.items()}
    out["foot_gate_cop_mean"] = {f: v.get("foot_gate_cop_mean") for f, v in pf.items()}
    return out


def validity_issues(d: dict, *, seed: int, steps: int, label: str, frozen: dict, expect_head: str, synthetic_ok: bool) -> list[str]:
    """Why one r10 judge file cannot be scored in its slot ([] when it can)."""
    bad = []
    if d.get("schema") != JUDGE_SCHEMA:
        bad.append(f"schema {d.get('schema')!r}")
    syn = d.get("synthetic")
    if syn and not synthetic_ok:
        bad.append("a synthetic copy (validation only)")
    if d.get("smoke"):
        bad.append("a smoke judge (shortened panels)")
    if d.get("seed") != seed or d.get("steps") != steps or d.get("checkpoint") != f"stage1_{steps}_steps":
        bad.append(f"slot s{seed}/{steps} holds seed {d.get('seed')} steps {d.get('steps')} ({d.get('checkpoint')})")
    git = d.get("git") or {}
    if git.get("head") != expect_head or git.get("status"):
        bad.append(f"judged on clone HEAD {git.get('head')} (dirty={bool(git.get('status'))}), not {expect_head}")
    checks = d.get("checks") or {}
    relaxed = checks.get("relaxed") or {}
    for key, ok in (("no_symmetry_check", False), ("allow_missing_records", False)):
        if relaxed.get(key) != ok:
            bad.append(f"judged with {key}")
    if relaxed.get("task_binding") != "require":
        bad.append(f"judged with task_binding {relaxed.get('task_binding')!r}")
    if relaxed.get("allow_seed_mismatch") and not (synthetic_ok and syn):
        bad.append("judged with allow_seed_mismatch")
    if checks.get("hard_failures"):
        bad.append(f"binding failures {checks.get('hard_failures')}")
    if not (checks.get("plant_physics_sha256") or {}).get("match"):
        bad.append("run plant_identity.json physics_sha256 does not match the clone")
    if not (checks.get("symmetry") or {}).get("mirror_map_sha256_match"):
        bad.append("checkpoint mesozoic_symmetry mirror_map_sha256 does not match the clone")
    if not ((checks.get("seed") or {}).get("match") or (synthetic_ok and syn)):
        bad.append(f"checkpoint seed {(checks.get('seed') or {}).get('checkpoint')} != slot seed {seed}")
    p = d.get("panel") or {}
    if [e.get("seed") for e in p.get("episodes_detail") or []] != PANEL_SEEDS:
        bad.append("panel episodes are not seeds 3042-3081 in order")
    if p.get("scored_gate_kind") != "stance_quality/v2":
        bad.append(f"panel scored as {p.get('scored_gate_kind')!r}")
    th = p.get("thresholds") or {}
    wrong = {k: th.get(k) for k, v in R10_BARS.items() if th.get(k) != v}
    if wrong:
        bad.append(f"panel bars differ from the r10 stance stage's: {wrong}")
    if "max_sole_cop_fore_aft" in th:
        bad.append("panel scored with the pad CoP rail (not the r10 stance stage)")
    if (p.get("statue_on_same_seeds") or {}).get("n") != PANEL_N:
        bad.append("no statue panel on the same seeds")
    if label == "6M":
        h4 = frozen["criteria"]["H4"]
        cj, ow = d.get("noise_matched_control_joint") or {}, d.get("noise_matched_own") or {}
        ref = h4["control_sigma"][str(seed)]
        checks6 = {
            "control joint present": bool(cj),
            "control joint space": cj.get("space") == "joint",
            "control joint n": cj.get("n") == NOISE_N and [r.get("seed") for r in cj.get("episodes_detail") or []] == NOISE_SEEDS,
            "control joint spawns": cj.get("seeds") == h4["spawn_range"],
            "own present": bool(ow),
            "own sigma_source": ow.get("sigma_source") == "own" and ow.get("space") == "policy",
            "own n": ow.get("n") == NOISE_N and [r.get("seed") for r in ow.get("episodes_detail") or []] == NOISE_SEEDS,
            "own spawns": ow.get("seeds") == h4["spawn_range"],
        }
        if not (synthetic_ok and syn):
            checks6["control joint sigma (frozen, this seed)"] = cj.get("sigma_source") == ref["source"] and cj.get("sigma_vector") == ref["vector"]
        bad += [f"H4 noise configuration: {k}" for k, ok in checks6.items() if not ok]
    return bad


# ------------------------------------------------------------------------------------------------ r10 and r9 grids
def load_r10(root: str, frozen: dict, expect_head: str, synthetic_ok: bool) -> tuple[dict, list, list, list]:
    cells, missing, invalid, notes = {}, [], [], []
    heads, tasks, plants = set(), set(), set()
    for seed in SEEDS:
        for label, steps in CKPTS:
            path = os.path.join(root, f"s{seed}", f"judge_{steps}.json")
            if not os.path.exists(path):
                cells[(seed, label)] = None
                missing.append(path)
                continue
            with open(path) as fh:
                d = json.load(fh)
            bad = validity_issues(d, seed=seed, steps=steps, label=label, frozen=frozen, expect_head=expect_head, synthetic_ok=synthetic_ok)
            if bad:
                cells[(seed, label)] = None
                invalid.append({"path": path, "issues": bad})
                continue
            c = panel_counts(d, rail="foot")
            c["path"] = path
            c["model_sha256"] = d.get("model_sha256")
            c["synthetic"] = bool(d.get("synthetic"))
            heads.add((d.get("git") or {}).get("head"))
            tasks.add((d.get("panel") or {}).get("task_sha256"))
            plants.add((d.get("plant_identity") or {}).get("physics_sha256"))
            if not c["k1_agree"]:
                notes.append(f"{path}: repository verdict {c['verdict']} disagrees with the criteria components {c['k1_components']} (seed counts as failed)")
            if c["lcb_recorded"] is not None and abs(c["lcb_recorded"] - c["lcb"]) > 1e-12:
                notes.append(f"{path}: recorded LCB {c['lcb_recorded']} != recomputed {c['lcb']}")
            if c["hop_or_fall_recorded"] is not None and c["hop_or_fall_recorded"] != c["hop_or_fall"]:
                notes.append(f"{path}: recorded hop_or_fall {c['hop_or_fall_recorded']} != recomputed {c['hop_or_fall']}")
            for name in ("noise_joint_falls", "noise_own_falls"):
                if c.get(name + "_recorded") is not None and c[name + "_recorded"] != c[name]:
                    notes.append(f"{path}: recorded {name} {c[name + '_recorded']} != recomputed {c[name]}")
            sc = (d.get("checks") or {}).get("stage_config") or {}
            if sc and not sc.get("git_commit_match"):
                notes.append(f"{path}: the run's stage_config.json git_commit {sc.get('git_commit')} != the judged clone HEAD "
                             f"{sc.get('clone_head')} (task, env, bars and symmetry keys bound; judged with allow_commit_mismatch)")
            if c["foot_cop_unmeasured_fh"]:
                notes.append(f"{path}: {c['foot_cop_unmeasured_fh']} full-horizon episode(s) without max_foot_cop_fore_aft (counted as over the rail)")
            cells[(seed, label)] = c
    if len(heads) > 1 or len(tasks) > 1 or len(plants) > 1:
        notes.append(f"the r10 judge files disagree: heads {sorted(map(str, heads))}, tasks {sorted(map(str, tasks))}, plants {sorted(map(str, plants))}")
    return cells, missing, invalid, notes


def load_r9(runs: str, verdict_path: str) -> tuple[dict, dict, list]:
    """r9 arm L: the judge files' counts with the r9 rails, and agreement with its verdict.json."""
    cells, notes = {}, []
    for seed in SEEDS:
        for label, steps in CKPTS:
            path = os.path.join(runs, f"{R9_RUN_NAME}_s{seed}", f"judge_{steps}.json")
            if not os.path.exists(path):
                cells[(seed, label)] = None
                notes.append(f"r9 judge file missing: {path}")
                continue
            with open(path) as fh:
                d = json.load(fh)
            c = panel_counts(d, rail="pad")
            c["path"] = path
            cells[(seed, label)] = c
    with open(verdict_path) as fh:
        v = json.load(fh)
    hd = v["arms"]["L"]["H_detail"]
    recomputed = {
        "H1_per_seed": {str(s): bool(cells[(s, "6M")] and cells[(s, "6M")]["k1_components_pass"] and cells[(s, "6M")]["k1_repo_pass"]) for s in SEEDS},
        "H2_roll_rail_per_seed": [cells[(s, "6M")]["roll_rail"] for s in SEEDS],
        "H3a_cop_any_4_5_6M": {str(s): [cells[(s, lab)]["pad_cop_any"] for lab, _ in CKPTS] for s in SEEDS},
        "H4_joint_falls": [cells[(s, "6M")]["noise_joint_falls"] for s in SEEDS],
        "H4_own_falls": [cells[(s, "6M")]["noise_own_falls"] for s in SEEDS],
    }
    agree = {k: recomputed[k] == hd[k] for k in recomputed}
    for k, ok in agree.items():
        if not ok:
            notes.append(f"r9 {k}: recomputed {recomputed[k]} != verdict.json {hd[k]}")
    meta = {"verdict": verdict_path, "params_sha256": v.get("params_sha256"), "H": v["arms"]["L"]["H"],
            "H_detail_used": {k: hd[k] for k in recomputed}, "recomputed": recomputed, "agrees_with_verdict_json": agree}
    return cells, meta, notes


def criteria(cells: dict, *, rail_name: str) -> dict:
    """K1-K4 on one 3-seed grid; a None cell (missing or invalid) fails every criterion that needs it."""
    c6 = {s: cells[(s, "6M")] for s in SEEDS}
    k1_seed = {s: bool(c6[s] and c6[s]["k1_pass"]) for s in SEEDS}
    K1 = {"pass": sum(k1_seed.values()) >= K1_MIN_SEEDS, "seeds_passing": [s for s in SEEDS if k1_seed[s]],
          "per_seed": {str(s): None if c6[s] is None else {"repo_verdict": c6[s]["verdict"], "clean": c6[s]["n_clean"], "lcb": c6[s]["lcb"],
                                                           "full": c6[s]["full"], "hop_or_fall": c6[s]["hop_or_fall"],
                                                           "components": c6[s]["k1_components"], "pass": k1_seed[s]} for s in SEEDS},
          "bar": f"v2 PASS on >= {K1_MIN_SEEDS} of 3 seeds (clean >= {K1_MIN_CLEAN}/40, LCB >= {K1_MIN_LCB}, full >= {K1_MIN_FULL_FRACTION}, hop-or-fall <= {K1_MAX_HOP})"}
    grid = {str(s): {lab: (None if cells[(s, lab)] is None else cells[(s, lab)]["k2_count"]) for lab, _ in CKPTS} for s in SEEDS}
    K2 = {"pass": all(v is not None and v <= K2_MAX for row in grid.values() for v in row.values()), "counts": grid,
          "over": [f"s{s} {lab}: {'missing/invalid' if v is None else v}" for s, row in grid.items() for lab, v in row.items() if v is None or v > K2_MAX],
          "bar": f"<= {K2_MAX} of 40 per seed at each of 4M/5M/6M ({rail_name})"}
    jf = [None if c6[s] is None else c6[s]["noise_joint_falls"] for s in SEEDS]
    of = [None if c6[s] is None else c6[s]["noise_own_falls"] for s in SEEDS]
    k3_complete = all(v is not None for v in jf + of)
    K3 = {"pass": k3_complete and sum(jf) <= K3_JOINT_MAX and sum(of) <= K3_OWN_MAX, "joint_falls": jf, "own_falls": of,
          "joint_pooled": sum(v for v in jf if v is not None), "own_pooled": sum(v for v in of if v is not None), "complete": k3_complete,
          "bar": f"joint <= {K3_JOINT_MAX}/60 pooled and own <= {K3_OWN_MAX}/60 pooled at 6M"}
    rr = [None if c6[s] is None else c6[s]["roll_rail"] for s in SEEDS]
    k4_complete = all(v is not None for v in rr)
    K4 = {"pass": k4_complete and sum(rr) <= K4_POOLED_MAX and max(rr) <= K4_SEED_MAX, "per_seed": rr,
          "pooled": sum(v for v in rr if v is not None), "complete": k4_complete,
          "bar": f"<= {K4_POOLED_MAX}/120 pooled and <= {K4_SEED_MAX}/40 per seed at 6M (tilt > {K4_TILT} deg or corner lift > {K4_LIFT} m, full horizon)"}
    out = {"K1": K1, "K2": K2, "K3": K3, "K4": K4}
    out["failed"] = [k for k in ("K1", "K2", "K3", "K4") if not out[k]["pass"]]
    out["overall"] = "PASS" if not out["failed"] else "FAIL"
    return out


def best_stance(r10: dict, r9: dict) -> dict:
    cands = []
    for tag, cells in (("r10", r10), ("r9", r9)):
        for s in SEEDS:
            c = cells.get((s, "6M"))
            if c is None:
                continue
            j, o = c["noise_joint_falls"], c["noise_own_falls"]
            cands.append({"run": f"{tag} arm L s{s} 6M", "clean": c["n_clean"], "joint_noise_falls": j,
                          "own_noise_falls": o, "noise_falls": None if j is None or o is None else j + o,
                          "path": c["path"]})
    if not cands:
        return {"best": None, "candidates": []}
    key = lambda x: (-x["clean"], x["noise_falls"] if x["noise_falls"] is not None else 10 ** 9)  # noqa: E731
    cands.sort(key=key)
    top = [c for c in cands if key(c) == key(cands[0])]
    return {"best": top[0]["run"] if len(top) == 1 else None, "tied": [c["run"] for c in top] if len(top) > 1 else [],
            "rule": "most clean panel episodes; tie -> fewer noise falls (joint-space + own-sigma)", "candidates": cands}


# ------------------------------------------------------------------------------------------------ printing
def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def lst(values) -> str:
    return "[" + ", ".join("missing" if v is None else str(v) for v in values) + "]"


def render(res: dict) -> str:
    L = []
    r10, r9 = res["r10"], res["r9"]
    L.append("r10 foot + arm L screen: score against foot/r10screen/criteria.txt (fixed 2026-10-16)")
    if res.get("validation_synthetic"):
        L.append("*** VALIDATION ONLY: synthetic judge copies accepted (--validation-synthetic); not a screen verdict ***")
    L.append(f"r10 judges: {res['r10_root']}")
    L.append(f"r9 arm L:   {R9_RUNS}/{R9_RUN_NAME}_s*/judge_*.json, checked against {res['r9_meta']['verdict']}: "
             + ("all r9 numbers agree" if all(res['r9_meta']['agrees_with_verdict_json'].values()) else "DISAGREEMENT (see notes)"))
    L.append("")
    g10, g9 = res["r10_cells"], res["r9_cells"]

    def cell(g, s, lab):
        return g.get(f"s{s}_{lab}")

    L.append("K1  stance_quality/v2 PASS at 6M on >= 2 of 3 seeds (clean >= 37/40 with LCB >= 0.80, full >= 0.95, hop-or-fall <= 1)")
    L.append("      seed |  r10: verdict clean  LCB    full hop  pass |  r9:  verdict clean  LCB    full hop  pass")
    for s in SEEDS:
        a, b = cell(g10, s, "6M"), cell(g9, s, "6M")
        sa = " missing/invalid                       " if a is None else f"{a['verdict']:>8} {a['n_clean']:>3}/40 {a['lcb']:.3f} {a['full']:>4} {a['hop_or_fall']:>3}  {'yes' if a['k1_pass'] else 'no ':>4}"
        sb = " missing" if b is None else f"{b['verdict']:>8} {b['n_clean']:>3}/40 {b['lcb']:.3f} {b['full']:>4} {b['hop_or_fall']:>3}  {'yes' if b['k1_pass'] else 'no ':>4}"
        L.append(f"      {s:>4} | {sa} | {sb}")
    L.append(f"   -> r10 K1 {'PASS' if r10['K1']['pass'] else 'FAIL'} ({len(r10['K1']['seeds_passing'])} of 3 seeds: {r10['K1']['seeds_passing']})"
             f"   | r9 K1 {'pass' if r9['K1']['pass'] else 'FAIL'} ({len(r9['K1']['seeds_passing'])} of 3: {r9['K1']['seeds_passing']})")
    L.append("")
    L.append("K2  fore-aft CoP: <= 3 of 40 panel episodes over the rail, every seed, at each of 4M / 5M / 6M")
    L.append("      r10 = full-horizon episodes with max_foot_cop_fore_aft > 0.80 [all-episode count in brackets, report only]")
    L.append("      r9  = episodes with the pad's max_sole_cop_fore_aft > 0.8, all 40 counted (H3a) [full-horizon count in brackets]")
    L.append("      seed |  r10   4M      5M      6M     |  r9   4M      5M      6M")
    for s in SEEDS:
        ra = []
        rb = []
        for lab, _ in CKPTS:
            a, b = cell(g10, s, lab), cell(g9, s, lab)
            ra.append("  miss  " if a is None else f"{a['foot_cop_fh']:>3} [{a['foot_cop_any']:>2}]")
            rb.append("  miss  " if b is None else f"{b['pad_cop_any']:>3} [{b['pad_cop_fh']:>2}]")
        L.append(f"      {s:>4} |      {'  '.join(ra)} |      {'  '.join(rb)}")
    L.append(f"   -> r10 K2 {'PASS' if r10['K2']['pass'] else 'FAIL'}" + ("" if r10["K2"]["pass"] else f" (over 3 or missing: {', '.join(r10['K2']['over'])})")
             + f"   | r9 K2 {'pass' if r9['K2']['pass'] else 'FAIL'}" + ("" if r9["K2"]["pass"] else f" ({', '.join(r9['K2']['over'])})"))
    L.append("")
    L.append("K3  at 6M: joint-space control-vector noise falls <= 7 of 60 pooled; own-sigma noise falls <= 3 of 60 pooled")
    L.append(f"      r10 joint {lst(r10['K3']['joint_falls'])} = {r10['K3']['joint_pooled']}/60, own {lst(r10['K3']['own_falls'])} = {r10['K3']['own_pooled']}/60"
             f"   | r9 joint {lst(r9['K3']['joint_falls'])} = {r9['K3']['joint_pooled']}/60, own {lst(r9['K3']['own_falls'])} = {r9['K3']['own_pooled']}/60")
    L.append(f"   -> r10 K3 {'PASS' if r10['K3']['pass'] else 'FAIL'}{'' if r10['K3']['complete'] else ' (incomplete: a 6M checkpoint is missing or invalid)'}"
             f"   | r9 K3 {'pass' if r9['K3']['pass'] else 'FAIL'}")
    L.append("")
    L.append("K4  at 6M: full-horizon roll-rail failures (tilt > 2.0 deg or corner lift > 0.004 m) <= 6 of 120 pooled, <= 3 of 40 per seed")
    L.append(f"      r10 {lst(r10['K4']['per_seed'])} = {r10['K4']['pooled']}/120   | r9 {lst(r9['K4']['per_seed'])} = {r9['K4']['pooled']}/120")
    L.append(f"   -> r10 K4 {'PASS' if r10['K4']['pass'] else 'FAIL'}{'' if r10['K4']['complete'] else ' (incomplete: a 6M checkpoint is missing or invalid)'}"
             f"   | r9 K4 {'pass' if r9['K4']['pass'] else 'FAIL'}")
    L.append("")
    L.append("OVERALL (r10 + arm L): " + ("PASS -- all four criteria hold" if r10["overall"] == "PASS" else f"FAIL -- failed: {', '.join(r10['failed'])}")
             + f"   (r9 arm L on the same tests: {r9['overall']}{'' if r9['overall'] == 'PASS' else ' -- ' + ', '.join(r9['failed'])})")
    if res["missing"]:
        L.append("Missing r10 judge files (count as failed): " + "; ".join(res["missing"]))
    if res["invalid"]:
        L.append("Invalid r10 judge files (count as failed):")
        for x in res["invalid"]:
            L.append(f"   {x['path']}: {'; '.join(x['issues'])}")
    b = res["best_stance_policy"]
    L.append("")
    L.append("Best stance policy among the six 6M checkpoints (most clean panel episodes; tie -> fewer noise falls, joint-space + own-sigma): "
             + (b["best"] if b["best"] else ("TIE: " + ", ".join(b["tied"]) if b.get("tied") else "none available")))
    for c in b["candidates"]:
        L.append(f"   {c['run']:<22} clean {c['clean']:>2}/40  joint noise falls {fmt(c['joint_noise_falls'])}")
    L.append("")
    L.append("Report only (never in the verdict):")
    L.append("      seed ckpt  r10 pad CoP>0.8 all [fh]  foot CoP mean (signed r / l; |.| window r / l)   reward / statue       terminations")
    for s in SEEDS:
        for lab, _ in CKPTS:
            a = cell(g10, s, lab)
            if a is None:
                continue
            sm, gm = a["foot_signed_cop_mean"], a["foot_gate_cop_mean"]
            ratio = (a["mean_reward"] / a["statue_mean_reward"]) if a["mean_reward"] and a["statue_mean_reward"] else None
            L.append(f"      {s:>4} {lab:>4}  {a['pad_cop_any']:>4} [{a['pad_cop_fh']:>2}]                "
                     f"{fmt(sm.get('r'))} / {fmt(sm.get('l'))}; {fmt(gm.get('r'))} / {fmt(gm.get('l'))}       "
                     f"{fmt(a['mean_reward'], 1)} / {fmt(a['statue_mean_reward'], 1)} ({fmt(ratio)})   {a['terminations']}")
    L.append("      (r9 arm L rewards are on the r9 stage's production [env], without the arm's reward terms; r10's include them)")
    if res["notes"]:
        L.append("")
        L.append("Notes:")
        for n in res["notes"]:
            L.append(f"   {n}")
    return "\n".join(L)


def score(root: str, *, runs: str = R9_RUNS, verdict: str = R9_VERDICT, frozen_path: str = FROZEN_PARAMS,
          expect_head: str = EXPECT_HEAD, synthetic_ok: bool = False) -> dict:
    fsha = sha256_file(frozen_path)
    if fsha != FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{frozen_path} sha256 {fsha} != {FROZEN_PARAMS_SHA256}")
    with open(frozen_path) as fh:
        frozen = json.load(fh)
    r10_cells, missing, invalid, notes = load_r10(root, frozen, expect_head, synthetic_ok)
    r9_cells, r9_meta, r9_notes = load_r9(runs, verdict)
    if r9_meta["params_sha256"] != FROZEN_PARAMS_SHA256:
        r9_notes.append(f"r9 verdict params_sha256 {r9_meta['params_sha256']} != frozen_params {FROZEN_PARAMS_SHA256}")
    r10 = criteria(r10_cells, rail_name="r10: full-horizon, max_foot_cop_fore_aft > 0.80")
    r9 = criteria(r9_cells, rail_name="r9: all episodes, pad max_sole_cop_fore_aft > 0.8 (H3a)")
    # report only: would the all-episode count (H3a's way) have decided K2 differently on r10?
    alt = all(c is not None and c["foot_cop_any"] <= K2_MAX for (s, lab), c in r10_cells.items())
    if alt != r10["K2"]["pass"] and not missing and not invalid:
        notes.append(f"K2 counted over all 40 episodes (as r9's H3a counted) would read {'PASS' if alt else 'FAIL'} (report only)")
    if any(c and c.get("synthetic") for c in r10_cells.values()):
        notes.append("synthetic r10 judge copies scored (--validation-synthetic): VALIDATION ONLY")
    ser = lambda g: {f"s{s}_{lab}": c for (s, lab), c in g.items()}  # noqa: E731
    return {"schema": "r10screen-score/v1", "criteria_file": SCRATCH + "/foot/r10screen/criteria.txt",
            "r10_root": os.path.abspath(root), "validation_synthetic": synthetic_ok, "expect_head": expect_head,
            "frozen_params_sha256": fsha, "r10": r10, "r9": r9, "overall": r10["overall"], "failed": r10["failed"],
            "missing": missing, "invalid": invalid, "best_stance_policy": best_stance(r10_cells, r9_cells),
            "r9_meta": r9_meta, "notes": notes + r9_notes, "r10_cells": ser(r10_cells), "r9_cells": ser(r9_cells)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("r10_root", help="directory holding s{seed}/judge_{steps}.json")
    ap.add_argument("--json", default=None, help="also write the full result here (never overwritten)")
    ap.add_argument("--r9-runs", default=R9_RUNS)
    ap.add_argument("--r9-verdict", default=R9_VERDICT)
    ap.add_argument("--frozen-params", default=FROZEN_PARAMS)
    ap.add_argument("--expect-head", default=EXPECT_HEAD)
    ap.add_argument("--validation-synthetic", action="store_true", help="validation of this script only (see the docstring)")
    a = ap.parse_args()
    if a.json and os.path.exists(a.json):
        raise SystemExit(f"refusing to overwrite {a.json}")
    res = score(a.r10_root, runs=a.r9_runs, verdict=a.r9_verdict, frozen_path=a.frozen_params, expect_head=a.expect_head,
                synthetic_ok=a.validation_synthetic)
    print(render(res))
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
