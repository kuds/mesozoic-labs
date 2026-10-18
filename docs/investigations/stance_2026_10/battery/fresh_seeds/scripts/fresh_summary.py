"""Fresh-seed six-heading certificates (5042, 9042) beside the earlier 3042 / 7042 runs: per-run facts and pooled
clean-seed rates.  OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen); report only,
certifies nothing.  Reads stance_heading_report.json / run_summary.json files; writes fresh_seed_summary.{json,txt}.
"""
import json, sys, collections
from pathlib import Path

import environments
assert "main_ro" in environments.__file__, environments.__file__
from environments.shared.curriculum.recovery_gate import binomial_lcb, binomial_ucb

S = Path("scratchpad")
F = S / "foot/battery/fresh_seeds"
H = S / "foot/next_signals/howclose"

CKPTS = {  # label -> (run seed, ckpt key, step)
    "s60_6M": (60, "6291456", 6291456), "s52_6M": (52, "6291456", 6291456),
    "s60_robust_best": (60, "robustbest", 524288), "s52_robust_best": (52, "robustbest", 917504),
}
PANELS = (3042, 5042, 7042, 9042)

BARS = [("max_sole_cop_fore_aft", "max", 0.80), ("max_sole_tilt_deg", "max", 2.0), ("max_sole_corner_lift_m", "max", 0.004),
        ("min_sole_contacts", "min", 1.5), ("min_foot_load_share", "min", 0.40), ("min_foot_load_share_windowed", "min", 0.35),
        ("settle_peak_floor_force_bw", "max", 1.5), ("settle_stance_width_change_m", "max", 0.08), ("settle_touchdowns", "max", 2),
        ("episode_yaw_change_deg", "max", 25.0), ("window_displacement_m", "max", 0.10), ("max_actuator_saturation_fraction", "max", 0.10),
        ("all_feet_support", "min", 0.98)]
CEILING = {"all_feet_support": 1.0}  # worst == ceiling carries no margin information


def run_dir(seed, key, panel):
    if panel == 3042:
        return H / f"cert_r9armL_s{seed}_{key}_optoff"
    if panel == 7042:
        return H / f"audit7042_r9armL_s{seed}_{key}_optoff"
    return F / f"cert{panel}_r9armL_s{seed}_{key}_optoff"


def analyse(d):
    rep = json.loads((d / "stance_heading_report.json").read_text())
    summ = json.loads((d / "run_summary.json").read_text())
    res = rep["result"]
    horizon = rep["horizon"]
    seeds = rep["seeds"]
    pol = [r for h in rep["headings"] for r in h["episode_evidence"]]
    unclean = collections.defaultdict(lambda: {"headings": [], "criteria": set()})
    crit_seeds = collections.defaultdict(set)
    labels = collections.defaultdict(set)
    for h in rep["headings"]:
        for r in h["episode_evidence"]:
            labels[r["seed"]].add(bool(r["clean"]))
            if not r["clean"]:
                unclean[r["seed"]]["headings"].append(h["yaw_deg"])
                for x in r["reasons"]:
                    c = x.split(":")[0].strip()
                    if "unmeasured" in x:
                        continue  # a fallen episode leaves later windows unmeasured; not a separate failure
                    unclean[r["seed"]]["criteria"].add(c)
                    crit_seeds[c].add(r["seed"])
    mixed = sorted(s for s, v in labels.items() if len(v) > 1)
    # falls: any heading that did not reach the horizon
    falls = collections.defaultdict(list)
    for h in rep["headings"]:
        for r in h["episode_evidence"]:
            if not r["reached_horizon"]:
                falls[r["seed"]].append({"yaw_deg": h["yaw_deg"], "length": r["length"], "terminated": r["terminated"]})
    terms = {h["yaw_deg"]: {k: v for k, v in (h["terminations"] or {}).items() if k != "truncated"} for h in rep["headings"]}
    fall_rows = []
    for s, ev in sorted(falls.items()):
        st = [r for h in rep["headings"] for r in (h["statue"] or {}).get("episode_evidence", []) if r["seed"] == s]
        fall_rows.append({"seed": s, "n_headings_fallen": len(ev), "lengths": sorted({e["length"] for e in ev}),
                          "statue_reached_horizon_all_headings": all(r["reached_horizon"] for r in st) if st else None,
                          "statue_max_pad_cop": max((r["max_sole_cop_fore_aft"] for r in st if r.get("max_sole_cop_fore_aft") is not None), default=None)})
    statue_cop = collections.defaultdict(float); statue_fh = {}
    for h in rep["headings"]:
        for r in (h["statue"] or {}).get("episode_evidence", []):
            if r.get("max_sole_cop_fore_aft") is not None:
                statue_cop[r["seed"]] = max(statue_cop[r["seed"]], r["max_sole_cop_fore_aft"])
            statue_fh[r["seed"]] = statue_fh.get(r["seed"], True) and bool(r["reached_horizon"])
    statue_n_fh = min(sum(1 for r in (h["statue"] or {}).get("episode_evidence", []) if r["reached_horizon"]) for h in rep["headings"])
    # pad CoP
    full = [r for r in pol if r["length"] >= horizon and r.get("max_sole_cop_fore_aft") is not None]
    worst = max(full, key=lambda r: r["max_sole_cop_fore_aft"]) if full else None
    cop_by_seed = collections.defaultdict(float)
    for r in full:
        cop_by_seed[r["seed"]] = max(cop_by_seed[r["seed"]], r["max_sole_cop_fore_aft"])
    past_cop_seeds = {s: round(v, 4) for s, v in sorted(cop_by_seed.items()) if v > 0.80}
    # bar margins over full-horizon policy episodes (cert_margins convention)
    margins = []
    for key, kind, bar in BARS:
        v = [float(r[key]) for r in pol if r.get(key) is not None and r["length"] >= horizon]
        if not v:
            continue
        w = max(v) if kind == "max" else min(v)
        rel = (bar - w) / bar if kind == "max" else (w - bar) / bar
        margins.append({"bar": key, "kind": kind, "limit": bar, "policy_worst": w, "relative_margin": rel,
                        "episodes_past": sum(1 for x in v if (x > bar if kind == "max" else x < bar)),
                        "at_ceiling": CEILING.get(key) is not None and w == CEILING[key]})
    informative = sorted([m for m in margins if not m["at_ceiling"] and m["episodes_past"] == 0], key=lambda m: m["relative_margin"])
    violated = sorted([m for m in margins if m["episodes_past"] > 0], key=lambda m: m["relative_margin"])
    ph = res["per_heading"]
    per_heading = [{"yaw_deg": p["yaw_deg"], "n_clean": p["n_clean"], "clean_lcb": p["clean_lcb"], "n_full_horizon": p["n_full_horizon"],
                    "hop_or_fall": p["hop_or_fall_episodes"], "mean_reward": p["mean_reward"], "statue_mean_reward": p["statue_mean_reward"],
                    "reward_ratio": p["mean_reward"] / p["statue_mean_reward"], "passed": p["passed"], "failures": p["failures"]} for p in ph]
    # binding bars: what decided the verdict
    binding = []
    if not rep["passed"]:
        binding += sorted({f.split(" ")[0] if not f.startswith("heading") else f.split(": ")[1].split(" ")[0] for f in rep["failures"]})
    if crit_seeds:
        binding_episode = {c: sorted(v) for c, v in sorted(crit_seeds.items(), key=lambda kv: -len(kv[1]))}
    else:
        binding_episode = {}
    max_hof = max(p["hop_or_fall_episodes"] for p in ph)
    n_clean = res["n_clean_seeds"]
    out = {
        "dir": str(d), "report": str(d / "stance_heading_report.json"), "run_summary": str(d / "run_summary.json"),
        "first_seed": rep["seeds"][0], "last_seed": rep["seeds"][-1], "options": summ["options"],
        "model": summ["model"], "model_sha256": summ["model_sha256"], "vecnorm_sha256": summ["vecnorm_sha256"],
        "task_sha256": rep["task_sha256"], "gate_sha256": rep["gate_sha256"], "seconds": summ["seconds"],
        "passed": rep["passed"], "n_clean_seeds": n_clean, "n_seeds": res["n_seeds"], "clean_seed_lcb": res["clean_seed_lcb"],
        "failures": rep["failures"],
        "unclean_seeds": {str(s): {"headings_unclean": len(v["headings"]), "criteria": sorted(v["criteria"]),
                                   "policy_max_pad_cop": max((r["max_sole_cop_fore_aft"] for r in pol if r["seed"] == s and r.get("max_sole_cop_fore_aft") is not None), default=None),
                                   "statue_max_pad_cop": statue_cop.get(s), "statue_full_horizon_all_headings": statue_fh.get(s)}
                          for s, v in sorted(unclean.items())},
        "seeds_with_mixed_labels_across_headings": mixed,
        "episode_criteria_failing_seeds": binding_episode,
        "verdict_level_failing_rails": binding,
        "slack_unclean_seeds_before_lcb_fail": n_clean - 37,  # 37/40 passes, 36/40 fails (stance_heading_gate.py:45-46)
        "max_hop_or_fall_per_heading": max_hof, "hop_or_fall_rail": rep["thresholds"]["max_hop_or_fall_episodes"],
        "falls": fall_rows, "terminations_by_heading": {str(k): v for k, v in terms.items()},
        "worst_pad_cop_full_horizon": None if worst is None else {"value": worst["max_sole_cop_fore_aft"], "seed": worst["seed"], "yaw_deg": worst["yaw_deg"]},
        "seeds_past_pad_cop_bar": past_cop_seeds,
        "tightest_unviolated_bars": informative[:3],
        "violated_bars_full_horizon": violated,
        "bar_margins": margins,
        "per_heading": per_heading,
        "statue_min_full_horizon_over_headings": statue_n_fh,
        "statue_worst_pad_cop": max(statue_cop.values()) if statue_cop else None,
        "statue_fall_seeds": sorted(sd for sd, ok in statue_fh.items() if not ok),
    }
    return out


def describe_binding(r):
    if not r["passed"]:
        parts = [f"verdict rails failed: {', '.join(r['verdict_level_failing_rails'])}"]
    else:
        parts = [f"verdict passes with {r['slack_unclean_seeds_before_lcb_fail']} unclean seed(s) of slack on the grouped LCB bar 0.80"]
        if r["max_hop_or_fall_per_heading"] >= r["hop_or_fall_rail"]:
            parts.append(f"hop_or_fall {r['max_hop_or_fall_per_heading']} per heading = rail {r['hop_or_fall_rail']} (one more fall fails)")
    if r["episode_criteria_failing_seeds"]:
        parts.append("episode bars that made seeds unclean: " + ", ".join(f"{c} ({len(v)} seed{'s' if len(v) != 1 else ''})" for c, v in r["episode_criteria_failing_seeds"].items()))
    t = r["tightest_unviolated_bars"][0]
    parts.append(f"tightest unviolated bar (full-horizon episodes): {t['bar']} worst {t['policy_worst']:.4g} vs {t['limit']:g} (margin {100 * t['relative_margin']:.1f}%)")
    return "; ".join(parts)


def fmt(x):
    return 'unmeasured' if x is None else f'{x:.4f}'


def interval(k, n):
    return {"k": k, "n": n, "rate": k / n, "two_sided_95_clopper_pearson": [binomial_lcb(k, n, alpha=0.025), binomial_ucb(k, n, alpha=0.025)],
            "one_sided_95_lcb_gate_convention": binomial_lcb(k, n, alpha=0.05)}


runs = {}
for label, (seed, key, step) in CKPTS.items():
    runs[label] = {"step": step, "panels": {}}
    for p in PANELS:
        d = run_dir(seed, key, p)
        if (d / "stance_heading_report.json").exists() and (d / "run_summary.json").exists():
            runs[label]["panels"][str(p)] = analyse(d)

pooled = {}
for label, v in runs.items():
    P = v["panels"]
    shas = {r["model_sha256"] for r in P.values()}
    k = sum(r["n_clean_seeds"] for r in P.values()); n = sum(r["n_seeds"] for r in P.values())
    pooled[label] = {"panels": sorted(P), "model_sha256_consistent": len(shas) == 1, "model_sha256": sorted(shas),
                     "clean_by_panel": {p: f"{r['n_clean_seeds']}/{r['n_seeds']}" for p, r in sorted(P.items())},
                     **interval(k, n)}
    fresh = [p for p in ("5042", "9042") if p in P]
    if "3042" in P:
        pooled[label]["verdict_3042"] = "PASS" if P["3042"]["passed"] else "FAIL"
        pooled[label]["fresh_verdicts"] = {p: ("PASS" if P[p]["passed"] else "FAIL") for p in fresh}
        pooled[label]["replicates_3042_verdict"] = {p: P[p]["passed"] == P["3042"]["passed"] for p in fresh}
    fr = [P[p] for p in fresh]
    if fr:
        pooled[label]["fresh_only"] = interval(sum(r["n_clean_seeds"] for r in fr), sum(r["n_seeds"] for r in fr))

overlap = {}
for p in PANELS:
    sets = {label: set(v["panels"][str(p)]["unclean_seeds"]) for label, v in runs.items() if str(p) in v["panels"]}
    allseeds = sorted(set().union(*sets.values())) if sets else []
    overlap[str(p)] = {sd: sorted(l for l, st in sets.items() if sd in st) for sd in allseeds}

doc = {"label": "OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen, MirrorKLPPO, experiment_* keys); "
                "report only; certifies nothing. main 2e734e29, options off, runner foot/next_signals/howclose/scripts/run_cert.py.",
       "pooling_note": "pooled over disjoint 40-seed panels 3042-3081, 5042-5081, 7042-7081, 9042-9081 where a run exists; a seed is one trial "
                       "(clean at all six headings); interval = exact Clopper-Pearson (main's binomial_lcb/binomial_ucb, alpha 0.025 each side); "
                       "one-sided 95% LCB = the gate's statistic form applied to the pool (not a gate quantity: the gate reads one 40-seed panel).",
       "runs": runs, "pooled": pooled, "unclean_seed_overlap_by_panel": overlap}
(F / "fresh_seed_summary.json").write_text(json.dumps(doc, indent=1, default=str))

L = [doc["label"], ""]
for label, v in runs.items():
    L.append(f"=== {label} (step {v['step']:,})")
    for p, r in sorted(v["panels"].items()):
        src = Path(r["report"]).relative_to(S)
        L.append(f" panel {r['first_seed']}-{r['last_seed']}: {'PASS' if r['passed'] else 'FAIL'} {r['n_clean_seeds']}/{r['n_seeds']} clean at all six, "
                 f"LCB {r['clean_seed_lcb']:.4f}   [{src}]")
        L.append(f"   binding: {describe_binding(r)}")
        w = r["worst_pad_cop_full_horizon"]
        L.append(f"   worst pad CoP (full-horizon): {w['value']:.4f} seed {w['seed']} heading {w['yaw_deg']:+g}; seeds past 0.80: {r['seeds_past_pad_cop_bar'] or 'none'}")
        if r["falls"]:
            L.append("   falls: " + "; ".join(f"seed {f['seed']} at {f['n_headings_fallen']}/6 headings, length {f['lengths']}, statue full horizon {f['statue_reached_horizon_all_headings']}"
                                         for f in r["falls"]) + f"; terminations {r['terminations_by_heading']}")
        else:
            L.append("   falls: none (240/240 policy episodes full horizon)")
        if r["unclean_seeds"]:
            L.append("   unclean seeds: " + "; ".join(f"{s} ({u['headings_unclean']}/6: {', '.join(u['criteria'])}; policy pad CoP {fmt(u['policy_max_pad_cop'])}, statue {fmt(u['statue_max_pad_cop'])}, statue full horizon {u['statue_full_horizon_all_headings']})" for s, u in r["unclean_seeds"].items()))
        if r["seeds_with_mixed_labels_across_headings"]:
            L.append(f"   seeds whose clean label differs across headings: {r['seeds_with_mixed_labels_across_headings']}")
        rr = [h["reward_ratio"] for h in r["per_heading"]]
        L.append(f"   statue: min full horizon {r['statue_min_full_horizon_over_headings']}/40 over headings, statue fall seeds {r['statue_fall_seeds'] or 'none'}, worst pad CoP {r['statue_worst_pad_cop']:.4f}")
        L.append(f"   reward/statue ratio {min(rr):.3f}-{max(rr):.3f}; min full horizon {min(h['n_full_horizon'] for h in r['per_heading'])}/40; "
                 f"sha {r['model_sha256'][:12]}; {r['seconds']:.0f} s")
    q = pooled[label]
    lo, hi = q["two_sided_95_clopper_pearson"]
    L.append(f" POOLED {q['k']}/{q['n']} = {q['rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}], one-sided 95% LCB {q['one_sided_95_lcb_gate_convention']:.4f} "
             f"over panels {', '.join(q['panels'])} ({q['clean_by_panel']}); model sha consistent {q['model_sha256_consistent']}")
    if "replicates_3042_verdict" in q:
        L.append(f" 3042 verdict {q['verdict_3042']}; fresh verdicts {q['fresh_verdicts']}; replicates {q['replicates_3042_verdict']}")
    if "fresh_only" in q:
        f_ = q["fresh_only"]
        L.append(f" fresh panels only (5042+9042): {f_['k']}/{f_['n']}, 95% CI [{f_['two_sided_95_clopper_pearson'][0]:.4f}, {f_['two_sided_95_clopper_pearson'][1]:.4f}]")
    L.append("")
L.append("unclean seeds by panel -> checkpoints unclean on that seed: " + json.dumps(overlap))
L.append("")
L.append(doc["pooling_note"])
(F / "fresh_seed_summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
