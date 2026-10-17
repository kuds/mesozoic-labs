"""Validation (c): score_r10.py on hand-made judge JSONs, each criterion just inside and just outside its bar.

    python3 test_score_r10.py            (plain python; writes only under a temporary directory it removes)

Every case builds nine valid judge files (s42/s52/60 x 4M/5M/6M) from a baseline that passes K1-K4 with margin, moves
one quantity to the edge of its bar (inside) or one step past it (outside), scores the tree with score_r10.score and
asserts the criterion's outcome, the overall verdict and the failed list.  Further cases: a missing file, invalid
files (smoke, another clone HEAD, a wrong control sigma, a relabelled slot, relaxed bindings), the repository verdict
disagreeing with the criteria components, a fallen episode over a rail (not counted), an unmeasured foot CoP on a
full-horizon episode (counted), bars on the exact threshold (strict >), and the best-stance tie-break.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_r10 as SC  # noqa: E402

FROZEN = json.load(open(SC.FROZEN_PARAMS))
STEPS = dict(SC.CKPTS)
BARS = {"max_actuator_saturation_fraction": 0.1, "max_episode_yaw_change_deg": 25.0, "max_foot_contact_fraction": 0.02,
        "max_foot_cop_fore_aft": 0.8, "max_hop_or_fall_episodes": 1, "max_nonfoot_load_fraction": 0.02,
        "max_phantom_support_fraction": 0.05, "max_settle_airborne_substeps": 0, "max_settle_peak_floor_force_bw": 1.5,
        "max_settle_stance_width_change_m": 0.08, "max_settle_touchdowns": 2, "max_sole_corner_lift_m": 0.004,
        "max_sole_tilt_deg": 2.0, "max_touchdown_rate": 0.25, "max_window_displacement_m": 0.1, "min_all_feet_support": 0.98,
        "min_avg_reward": 2250.0, "min_avg_reward_statue_ratio": 0.6, "min_clean_stance_lcb": 0.8, "min_eval_episodes": 40,
        "min_foot_load_share": 0.4, "min_foot_load_share_windowed": 0.35, "min_full_horizon_fraction": 0.95,
        "required_consecutive": 3, "settle_steps": 200}


def episode(seed, *, full=True, clean=True, cop=0.3, tilt=0.5, lift=0.001, hop=False, pad=0.4):
    reasons = []
    if not full:
        reasons.append("horizon: 412 < 1000")
    if hop and full:
        reasons.append("min_all_feet_support: 0.9 < 0.98")
    if cop is None or cop > 0.8:
        reasons.append(f"max_foot_cop_fore_aft: {cop} > 0.8")
    if tilt > 2.0:
        reasons.append(f"max_sole_tilt_deg: {tilt} > 2.0")
    if lift > 0.004:
        reasons.append(f"max_sole_corner_lift_m: {lift} > 0.004")
    if not clean and not reasons:
        reasons.append("min_foot_load_share: 0.3 < 0.4")
    return {"seed": seed, "length": 1000 if full else 412, "reached_horizon": full, "full_horizon": full,
            "clean": not reasons, "reasons": reasons, "failing_rails": sorted({r.split(":")[0] for r in reasons}),
            "hop_or_fall": (not full) or hop, "reward": 3900.0, "termination_reason": "truncated" if full else "nosedive",
            "max_foot_cop_fore_aft": cop, "max_sole_cop_fore_aft": pad, "max_sole_tilt_deg": tilt, "max_sole_corner_lift_m": lift,
            "roll_rail": False, "pad": {"r_foot_cop_fore_aft": 0.1, "l_foot_cop_fore_aft": -0.1,
                                        "r_foot_cop_fore_aft_gate": 0.3, "l_foot_cop_fore_aft_gate": 0.3}}


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


def judge(seed, label, *, eps=None, joint_falls=0, own_falls=0, verdict=None):
    """A valid judge file; eps defaults to 40 clean, full-horizon, centred episodes."""
    eps = eps if eps is not None else [episode(3042 + i) for i in range(40)]
    k = sum(e["clean"] for e in eps)
    full = sum(e["reached_horizon"] for e in eps)
    hop = sum(e["hop_or_fall"] for e in eps)
    lcb = SC.binomial_lcb(k, 40)
    comp_pass = k >= 37 and lcb >= 0.8 and full / 40 >= 0.95 and hop <= 1
    if verdict is None:
        verdict = "PASS" if comp_pass else "FAIL"
    steps = STEPS[label]
    d = {"schema": SC.JUDGE_SCHEMA, "seed": seed, "steps": steps, "checkpoint": f"stage1_{steps}_steps", "smoke": False,
         "git": {"head": SC.EXPECT_HEAD, "status": "", "branch": "claude/screen-trex-r10-arm-l"},
         "model_sha256": f"sha256:{seed}{steps}", "plant_identity": {"physics_sha256": "sha256:r10"},
         "checks": {"relaxed": {"no_symmetry_check": False, "task_binding": "require", "allow_missing_records": False,
                                "allow_seed_mismatch": False},
                    "hard_failures": [], "plant_physics_sha256": {"match": True}, "symmetry": {"mirror_map_sha256_match": True},
                    "seed": {"label": seed, "checkpoint": seed, "match": True}},
         "panel": {"verdict": verdict, "passed": verdict == "PASS", "n_clean": k, "clean_lcb": lcb, "n_full_horizon": full,
                   "hop_or_fall_episodes": hop, "mean_reward": 3900.0, "statue_mean_reward": 3982.0,
                   "terminations": {"truncated": full, **({"nosedive": 40 - full} if full < 40 else {})},
                   "scored_gate_kind": "stance_quality/v2", "task_sha256": "sha256:task", "thresholds": dict(BARS),
                   "statue_on_same_seeds": {"n": 40}, "summary": {"per_foot": {}}, "episodes_detail": eps}}
    if label == "6M":
        d["noise_matched_control_joint"] = noise(seed, "joint", joint_falls)
        d["noise_matched_own"] = noise(seed, "policy", own_falls)
    return d


def baseline():
    return {(s, lab): judge(s, lab) for s in SC.SEEDS for lab, _ in SC.CKPTS}


def run(cells: dict, synthetic_ok=False):
    root = tempfile.mkdtemp(prefix="_score_unit_", dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        for (s, lab), d in cells.items():
            if d is None:
                continue
            os.makedirs(os.path.join(root, f"s{s}"), exist_ok=True)
            with open(os.path.join(root, f"s{s}", f"judge_{STEPS[lab]}.json"), "w") as fh:
                json.dump(d, fh)
        return SC.score(root, synthetic_ok=synthetic_ok)
    finally:
        shutil.rmtree(root)


def eps_with(n_bad=0, kind="clean", start=0, **kw):
    """40 episodes, *n_bad* of them (from index *start*) altered by *kind*."""
    out = []
    for i in range(40):
        bad = start <= i < start + n_bad
        if not bad:
            out.append(episode(3042 + i))
        elif kind == "unclean":
            out.append(episode(3042 + i, clean=False))
        elif kind == "fall":
            out.append(episode(3042 + i, full=False))
        elif kind == "hop":
            out.append(episode(3042 + i, hop=True))
        elif kind == "cop":
            out.append(episode(3042 + i, cop=kw.get("cop", 0.81)))
        elif kind == "fall_cop":
            out.append(episode(3042 + i, full=False, cop=0.95))
        elif kind == "roll_tilt":
            out.append(episode(3042 + i, tilt=kw.get("tilt", 2.01)))
        elif kind == "roll_lift":
            out.append(episode(3042 + i, lift=kw.get("lift", 0.0041)))
        elif kind == "fall_roll":
            out.append(episode(3042 + i, full=False, tilt=3.0))
        else:
            raise ValueError(kind)
    return out


RESULTS = []


def check(name, res, expect: dict, overall=None, extra=None):
    got = {k: res["r10"][k]["pass"] for k in expect}
    ok = got == expect
    if overall is not None:
        ok = ok and res["overall"] == overall
    if extra is not None:
        ok = ok and bool(extra(res))
    RESULTS.append((name, ok, got, res["overall"], res["failed"]))
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} overall {res['overall']} failed {res['failed']}")
    return ok


def main() -> int:
    # 0. baseline: everything passes
    check("baseline passes all four", run(baseline()), {"K1": True, "K2": True, "K3": True, "K4": True}, "PASS")
    # K1: clean 37/40 on two seeds (third failing) inside; 36/40 on one of them outside
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(3, "unclean"))
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(3, "unclean"))
    c[(60, "6M")] = judge(60, "6M", eps=eps_with(10, "unclean"))
    check("K1 inside: 37/40, 37/40, 30/40", run(c), {"K1": True}, "PASS", lambda r: r["r10"]["K1"]["seeds_passing"] == [42, 52])
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(4, "unclean"))
    check("K1 outside: 37/40, 36/40, 30/40", run(c), {"K1": False}, "FAIL", lambda r: r["failed"] == ["K1"])
    # K1 hop-or-fall: 1 fall inside (full 39/40), 2 hops outside
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(1, "fall"))
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(1, "hop"))
    check("K1 inside: hop-or-fall 1 on two seeds", run(c), {"K1": True}, "PASS")
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(2, "hop"))
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(2, "fall"))
    check("K1 outside: hop-or-fall 2 on two seeds (one of them full 38/40 = 0.95)", run(c), {"K1": False}, "FAIL",
          lambda r: r["r10"]["K1"]["seeds_passing"] == [60])
    # K1: repository verdict disagreeing with the components -> that seed fails, with a note
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", verdict="FAIL")
    c[(52, "6M")] = judge(52, "6M", verdict="FAIL")
    check("K1 repo FAIL with passing components -> seed fails (2 seeds)", run(c), {"K1": False}, "FAIL",
          lambda r: any("disagrees" in n for n in r["notes"]))
    # K2: 3 full-horizon foot CoP > 0.80 inside, 4 outside, on one cell (s52 5M)
    c = baseline()
    c[(52, "5M")] = judge(52, "5M", eps=eps_with(3, "cop"))
    check("K2 inside: 3/40 at s52 5M", run(c), {"K2": True}, "PASS")
    c[(52, "5M")] = judge(52, "5M", eps=eps_with(4, "cop"))
    check("K2 outside: 4/40 at s52 5M", run(c), {"K2": False}, "FAIL", lambda r: r["r10"]["K2"]["over"] == ["s52 5M: 4"])
    # K2: a fallen episode over the rail is not counted (full horizon only); the all-episode reading is noted
    c = baseline()
    c[(60, "4M")] = judge(60, "4M", eps=eps_with(3, "cop") [:3] + eps_with(1, "fall_cop", start=3)[3:4] + eps_with(0)[4:])
    check("K2 inside: 3 full-horizon + 1 fallen over the rail", run(c), {"K2": True}, "PASS",
          lambda r: any("all 40 episodes" in n for n in r["notes"]))
    # K2: exactly 0.80 is not over (strict >); an unmeasured value on a full-horizon episode counts as over
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(10, "cop", cop=0.80))
    check("K2 inside: 10 episodes at exactly 0.80", run(c), {"K2": True}, "PASS")
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(3, "cop") [:3] + eps_with(1, "cop", start=3, cop=None)[3:4] + eps_with(0)[4:])
    check("K2 outside: 3 over + 1 unmeasured on full horizon", run(c), {"K2": False}, "FAIL",
          lambda r: any("without max_foot_cop_fore_aft" in n for n in r["notes"]))
    # K3: joint pooled 7 inside / 8 outside; own pooled 3 inside / 4 outside
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", joint_falls=3, own_falls=1)
    c[(52, "6M")] = judge(52, "6M", joint_falls=2, own_falls=1)
    c[(60, "6M")] = judge(60, "6M", joint_falls=2, own_falls=1)
    check("K3 inside: joint 7/60, own 3/60", run(c), {"K3": True}, "PASS")
    c[(60, "6M")] = judge(60, "6M", joint_falls=3, own_falls=1)
    check("K3 outside: joint 8/60", run(c), {"K3": False}, "FAIL")
    c[(60, "6M")] = judge(60, "6M", joint_falls=2, own_falls=2)
    check("K3 outside: own 4/60", run(c), {"K3": False}, "FAIL")
    # K4: pooled 6 with 3/3/0 inside; 3/3/1 = 7 outside; 4/0/0 per seed outside; a fallen roll episode not counted
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(3, "roll_tilt"))
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(3, "roll_lift"))
    c[(60, "6M")] = judge(60, "6M", eps=eps_with(1, "fall_roll"))
    check("K4 inside: 3 / 3 / 0 (+1 fallen roll)", run(c), {"K4": True}, None, lambda r: r["r10"]["K4"]["per_seed"] == [3, 3, 0])
    c[(60, "6M")] = judge(60, "6M", eps=eps_with(1, "roll_tilt"))
    check("K4 outside: 3 / 3 / 1 = 7 pooled", run(c), {"K4": False}, None)
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(4, "roll_tilt", tilt=2.0000001))
    check("K4 outside: 4 on one seed (tilt 2.0000001)", run(c), {"K4": False}, None)
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(10, "roll_tilt", tilt=2.0) [:5] + eps_with(10, "roll_lift", lift=0.004)[5:])
    check("K4 inside: tilt exactly 2.0 / lift exactly 0.004 not over", run(c), {"K4": True}, "PASS")
    # missing and invalid files
    c = baseline()
    c[(60, "5M")] = None
    check("missing s60 5M -> K2 fails, listed", run(c), {"K1": True, "K2": False, "K3": True, "K4": True}, "FAIL",
          lambda r: len(r["missing"]) == 1)
    c = baseline()
    c[(52, "6M")] = None
    check("missing s52 6M -> K2 K3 K4 fail; K1 2/3", run(c), {"K1": True, "K2": False, "K3": False, "K4": False}, "FAIL")
    for name, mut in (
        ("smoke", lambda d: d.update(smoke=True)),
        ("another clone HEAD", lambda d: d["git"].update(head="964b70750f0f1afcda186a515fd56df93fb1cffc")),
        ("wrong control sigma", lambda d: d["noise_matched_control_joint"].update(sigma_vector=[0.2] * 15)),
        ("slot relabelled (seed 42 file in s52)", lambda d: d.update(seed=42)),
        ("task binding relaxed", lambda d: d["checks"]["relaxed"].update(task_binding="report")),
        ("synthetic without --validation-synthetic", lambda d: d.update(synthetic={"of": "x"})),
        ("noise panel with 19 episodes", lambda d: d["noise_matched_own"]["episodes_detail"].pop()),
        ("pad CoP rail in the bars", lambda d: d["panel"]["thresholds"].update(max_sole_cop_fore_aft=0.8)),
    ):
        c = baseline()
        d = copy.deepcopy(c[(52, "6M")])
        mut(d)
        c[(52, "6M")] = d
        check(f"invalid s52 6M ({name}) -> K2 K3 K4 fail, listed", run(c), {"K1": True, "K2": False, "K3": False, "K4": False}, "FAIL",
              lambda r: len(r["invalid"]) == 1)
    # synthetic copies are accepted only with --validation-synthetic, and then skip the seed / sigma binding
    c = baseline()
    for key in list(c):
        d = copy.deepcopy(c[(42, key[1])])
        d.update(seed=key[0], synthetic={"of": "s42"})
        d["checks"]["seed"]["match"] = False
        d["checks"]["relaxed"]["allow_seed_mismatch"] = False
        c[key] = d
    check("synthetic copies with --validation-synthetic", run(c, synthetic_ok=True), {"K1": True, "K2": True, "K3": True, "K4": True}, "PASS",
          lambda r: any("VALIDATION ONLY" in n for n in r["notes"]))
    # best stance policy: most clean, tie -> fewer noise falls, joint-space + own-sigma (r9 s60 has 40/40, 6 joint and 0 own falls)
    c = baseline()
    for s, jf in ((42, 5), (52, 6), (60, 7)):
        c[(s, "6M")] = judge(s, "6M", joint_falls=jf)
    check("best stance: r10 s42 (40/40, 5 falls) over r9 s60 (40/40, 6)", run(c), {"K1": True}, None,
          lambda r: r["best_stance_policy"]["best"] == "r10 arm L s42 6M")
    c[(42, "6M")] = judge(42, "6M", joint_falls=6)
    c[(52, "6M")] = judge(52, "6M", joint_falls=7)
    check("best stance: tie r10 s42 and r9 s60 (40/40, 6 falls each)", run(c), {"K1": True}, None,
          lambda r: r["best_stance_policy"]["best"] is None and r["best_stance_policy"]["tied"] == ["r10 arm L s42 6M", "r9 arm L s60 6M"])
    c[(42, "6M")] = judge(42, "6M", eps=eps_with(1, "unclean"), joint_falls=0)
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(1, "unclean"), joint_falls=0)
    c[(60, "6M")] = judge(60, "6M", eps=eps_with(1, "unclean"), joint_falls=0)
    check("best stance: r9 s60 (40/40) over r10 39/40 with 0 falls", run(c), {"K1": True}, None,
          lambda r: r["best_stance_policy"]["best"] == "r9 arm L s60 6M")
    # the tie-break sums joint-space and own-sigma falls: r10 s42 at 40/40 with 5 joint + 3 own (8) loses to
    # r9 s60 at 40/40 with 6 joint + 0 own (6)
    c = baseline()
    c[(42, "6M")] = judge(42, "6M", joint_falls=5, own_falls=3)
    c[(52, "6M")] = judge(52, "6M", eps=eps_with(1, "unclean"), joint_falls=0)
    c[(60, "6M")] = judge(60, "6M", eps=eps_with(1, "unclean"), joint_falls=0)
    check("best stance: tie on clean broken by joint + own falls (r9 s60 6 beats r10 s42 5 + 3)", run(c), {"K1": True}, None,
          lambda r: r["best_stance_policy"]["best"] == "r9 arm L s60 6M")
    n_ok = sum(1 for r in RESULTS if r[1])
    print(f"\n{n_ok}/{len(RESULTS)} cases as expected")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
