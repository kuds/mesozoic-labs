"""Tabulate the own-sigma (stochastic) heading-0 panels against the deterministic ones -- report-only.

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints; certifies nothing.
Inputs: panels/<label>_{stochastic,deterministic}/panel.json and panels/statue_statue/panel.json (this battery),
and the earlier certificates' 0 deg cells in foot/next_signals/howclose/cert_r9armL_*_optoff/ (read-only).
"""
import json, math, statistics, sys
from pathlib import Path

S = Path(sys.argv[1])
O = S / "foot/battery/own_sigma"
H = S / "foot/next_signals/howclose"
P = O / "panels"
LABELS = [("s60_6M", "cert_r9armL_s60_6291456_optoff"), ("s52_6M", "cert_r9armL_s52_6291456_optoff"),
          ("s42_6M", "cert_r9armL_s42_6291456_optoff"), ("s60_rb524288", "cert_r9armL_s60_robustbest_optoff"),
          ("s52_rb917504", "cert_r9armL_s52_robustbest_optoff"), ("s42_rb655360", "cert_r9armL_s42_robustbest_optoff")]
BAR = 0.80
# the bars as the earlier margins runner (howclose/scripts/margins.py) lists them
BARS = [("max_sole_cop_fore_aft", "max", 0.80), ("max_sole_tilt_deg", "max", 2.0), ("max_sole_corner_lift_m", "max", 0.004),
        ("min_sole_contacts", "min", 1.5), ("min_foot_load_share", "min", 0.40), ("min_foot_load_share_windowed", "min", 0.35),
        ("settle_peak_floor_force_bw", "max", 1.5), ("settle_stance_width_change_m", "max", 0.08), ("settle_touchdowns", "max", 2),
        ("episode_yaw_change_deg", "max", 25.0), ("window_displacement_m", "max", 0.10), ("max_actuator_saturation_fraction", "max", 0.10),
        ("all_feet_support", "min", 0.98)]


def margins(eps):
    """Per bar over full-horizon episodes: worst, median, count past the bar (as howclose/scripts/margins.py)."""
    res = {}
    for key, kind, bar in BARS:
        v = [float(e[key]) for e in eps if e["reached_horizon"] and e.get(key) is not None and math.isfinite(float(e[key]))]
        if not v:
            res[key] = None
            continue
        res[key] = {"bar": bar, "kind": kind, "worst": r4(max(v) if kind == "max" else min(v)), "median": r4(statistics.median(v)),
                    "past_bar": sum(1 for x in v if (x > bar if kind == "max" else x < bar))}
    return res


def rel(p):
    return str(p)


def r4(x):
    return None if x is None else round(float(x), 4)


def panel_stats(path):
    d = json.loads(path.read_text())
    eps = d["episodes"]
    v = d["v2_result"]
    full = [e for e in eps if e["reached_horizon"]]
    pad = [e["max_sole_cop_fore_aft"] for e in full if e.get("max_sole_cop_fore_aft") is not None and math.isfinite(e["max_sole_cop_fore_aft"])]
    signed = [e["pad"].get("signed_pad_cop_gate_foot") for e in full if e["pad"].get("signed_pad_cop_gate_foot") is not None]
    worst_e = max(full, key=lambda e: e["max_sole_cop_fore_aft"]) if full else None
    ext_signed = max(signed, key=abs) if signed else None
    unclean = {e["seed"]: [r.split(":")[0] for r in e["reasons"]] for e in eps if not e["clean"]}
    falls = [{"seed": e["seed"], "length": e["length"], "termination_reason": e["termination_reason"]} for e in eps if not e["reached_horizon"]]
    toe = [max(e["pad"]["toe_edge_fraction_per_foot"]) for e in full if e["pad"].get("toe_edge_fraction_per_foot")]
    heel = [max(e["pad"]["heel_edge_fraction_per_foot"]) for e in full if e["pad"].get("heel_edge_fraction_per_foot")]
    out = {
        "file": rel(path),
        "n_episodes": v["n_episodes"], "falls": len(falls), "fall_detail": falls,
        "n_full_horizon": v["n_full_horizon"], "full_horizon_fraction": v["full_horizon_fraction"],
        "n_clean_v2": v["n_clean"], "clean_lcb_v2": v["clean_lcb"], "v2_single_panel_passed": v["passed"],
        "v2_failures": v["failures"], "criterion_failures": {k: c for k, c in v["criterion_failures"].items() if c},
        "hop_or_fall_episodes": v.get("hop_or_fall_episodes"),
        "unclean_seeds": unclean, "reward_mean": d["reward_mean"],
        "pad_cop_gate_metric_full_horizon": {"n": len(pad), "median": r4(statistics.median(pad)) if pad else None,
                                             "worst": r4(max(pad)) if pad else None,
                                             "worst_seed": worst_e["seed"] if worst_e else None,
                                             "past_bar_0.80": sum(1 for x in pad if x > BAR)},
        "signed_pad_cop_gate_foot_full_horizon": {"n": len(signed), "median": r4(statistics.median(signed)) if signed else None,
                                                  "most_extreme": r4(ext_signed), "min": r4(min(signed)) if signed else None,
                                                  "max": r4(max(signed)) if signed else None,
                                                  "n_negative_heel_side": sum(1 for x in signed if x < 0)},
        "toe_edge_step_fraction_max_foot_median": r4(statistics.median(toe)) if toe else None,
        "heel_edge_step_fraction_max_foot_median": r4(statistics.median(heel)) if heel else None,
        "bar_margins_full_horizon": margins(eps),
        "recompute_matches_gate_metric_all": all(e["pad"].get("recompute_matches_gate_metric", True) for e in eps),
        "per_seed": {e["seed"]: {"len": e["length"], "clean": e["clean"], "pad": e.get("max_sole_cop_fore_aft"),
                                  "signed": e["pad"].get("signed_pad_cop_gate_foot"), "reward": e["reward"]} for e in eps},
    }
    m = d["meta"]
    if "std_per_actuator" in m:
        out["std_per_actuator"] = {k: round(v_, 5) for k, v_ in m["std_per_actuator"].items()}
        out["std_mean"] = round(m["std_mean"], 5)
        out["log_std"] = [round(x, 4) for x in m["log_std"]]
    for k in ("check_max_abs_clip_mu_minus_repo_deterministic_predict", "check_max_abs_dist_std_minus_exp_log_std",
              "model_sha256", "vecnorm_sha256", "model", "seconds"):
        if k in m:
            out[k] = m[k]
    if m["mode"] == "stochastic":
        clip = [e["noise"]["clip_fraction_per_actuator"] for e in eps]
        names = m["actuator_names"]
        out["clip_fraction_per_actuator_mean"] = {n: r4(statistics.mean(c[i] for c in clip)) for i, n in enumerate(names)}
    return out, d


def cert_zero(dirname):
    rep = json.loads((H / dirname / "stance_heading_report.json").read_text())
    zero = [h for h in rep["headings"] if h["yaw_deg"] == 0][0]
    ev = zero["episode_evidence"]
    full = [e for e in ev if e["reached_horizon"]]
    pad = [e["max_sole_cop_fore_aft"] for e in full]
    rs = json.loads((H / dirname / "run_summary.json").read_text())
    return {
        "file": rel(H / dirname / "stance_heading_report.json"),
        "zero_deg_n_clean": zero["metrics"]["n_clean"], "zero_deg_full_horizon_fraction": zero["metrics"]["full_horizon_fraction"],
        "zero_deg_pad_median": r4(statistics.median(pad)), "zero_deg_pad_worst": r4(max(pad)),
        "zero_deg_unclean": {e["seed"]: [r.split(":")[0] for r in e["reasons"]] for e in ev if not e["clean"]},
        "six_heading_passed": rs["passed"], "six_heading_n_clean_seeds": rs["n_clean_seeds"],
        "six_heading_file": rel(H / dirname / "run_summary.json"), "model_sha256": rs["model_sha256"],
        "statue_reference_zero_deg": (zero["statue"] or {}).get("reference"),
        "rows": {e["seed"]: e for e in ev},
    }, rep


summary = {"REPORT_ONLY": "offline evaluation of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen); certifies nothing",
           "panel": "heading 0 (v2 panel, spawn_yaw None), seeds 3042-3081, 40 episodes, main 2e734e29 stance stage config options off",
           "checkpoints": {}}
lines = []
w = lines.append
statue_path = P / "statue_statue/panel.json"
st, std_ = panel_stats(statue_path)
summary["statue"] = {k: st[k] for k in ("file", "falls", "fall_detail", "full_horizon_fraction", "pad_cop_gate_metric_full_horizon",
                                         "signed_pad_cop_gate_foot_full_horizon", "reward_mean")}
statue_ref = std_["statue_reference"]

for label, cdir in LABELS:
    entry = {}
    for mode in ("stochastic", "deterministic"):
        p = P / f"{label}_{mode}/panel.json"
        if p.exists():
            entry[mode], _ = panel_stats(p)
    cz, _ = cert_zero(cdir)
    rows = cz.pop("rows")
    entry["cert_zero_deg"] = {k: v for k, v in cz.items() if k != "statue_reference_zero_deg"}
    entry["statue_reference_matches_cert_zero_deg"] = cz["statue_reference_zero_deg"] == statue_ref
    # harness check: deterministic re-run vs the certificate's 0 deg cell, row for row
    if "deterministic" in entry:
        det = entry["deterministic"]["per_seed"]
        mism = []
        for seed, r in rows.items():
            mine = det.get(seed)
            if mine is None or mine["len"] != r["length"] or mine["clean"] != r["clean"] or \
                    abs(mine["reward"] - r["reward"]) > 1e-6 or abs((mine["pad"] or 0) - (r["max_sole_cop_fore_aft"] or 0)) > 1e-9:
                mism.append(seed)
        entry["harness_check_deterministic_vs_cert_zero_deg"] = {"seeds_compared": len(rows), "mismatched_seeds": mism,
                                                                  "identical": not mism}
        entry["deterministic"]["model_sha_matches_cert"] = entry["deterministic"].get("model_sha256") == cz["model_sha256"]
    if "stochastic" in entry and "deterministic" in entry:
        s_, d_ = entry["stochastic"]["per_seed"], entry["deterministic"]["per_seed"]
        lost = sorted(k for k in d_ if d_[k]["clean"] and not s_[k]["clean"])
        gained = sorted(k for k in d_ if not d_[k]["clean"] and s_[k]["clean"])
        both_full = [k for k in d_ if d_[k]["len"] >= 1000 and s_[k]["len"] >= 1000]
        dpad = [s_[k]["pad"] - d_[k]["pad"] for k in both_full]
        entry["paired_stochastic_vs_deterministic"] = {
            "clean_lost_under_noise": lost, "clean_gained_under_noise": gained,
            "pad_cop_diff_stoch_minus_det_median": r4(statistics.median(dpad)) if dpad else None,
            "pad_cop_diff_min": r4(min(dpad)) if dpad else None, "pad_cop_diff_max": r4(max(dpad)) if dpad else None,
            "n_paired_full_horizon": len(both_full)}
    for mode in ("stochastic", "deterministic"):
        if mode in entry:
            entry[mode].pop("per_seed")
    summary["checkpoints"][label] = entry

(O / "own_sigma_summary.json").write_text(json.dumps(summary, indent=1, default=str))

# ---- text report ----
w("OWN-SIGMA (STOCHASTIC) HEADING-0 PANELS -- REPORT-ONLY OFFLINE EVALUATION")
w("Uncertified scratch-recipe checkpoints (r9 arm L symmetry screen, MirrorKLPPO). This certifies nothing.")
w("Panel: main 2e734e29 stance stage config, options off, v2 panel at heading 0 (spawn_yaw None), seeds 3042-3081, 40 episodes,")
w("judged with main's evaluate_stance_v2_gate against the zero-action statue rolled on the same seeds.")
w("Stochastic: a = clip(mu + sigma*z, -1, 1), sigma = exp(log_std); z from default_rng(SeedSequence(episode_seed, spawn_key=(1,))),")
w("same z stream for every checkpoint. Deterministic: main's _load_policy predict (re-rolled here as a harness check).")
w("pad CoP = v2 gate metric max_sole_cop_fore_aft (bar 0.80; unsigned). signed pad CoP = window mean of the per-step signed fore-aft")
w("pad CoP over loaded steps, of the foot that sets the gate metric (+ toe, - heel; half-lengths). Both over full-horizon episodes.")
w(f"Summary JSON: {O / 'own_sigma_summary.json'}")
w("")
w(f"Statue ({statue_path}): falls {st['falls']} {[f['seed'] for f in st['fall_detail']]}, full horizon {st['full_horizon_fraction']:.3f}, "
  f"pad CoP median {st['pad_cop_gate_metric_full_horizon']['median']} worst {st['pad_cop_gate_metric_full_horizon']['worst']}, "
  f"signed median {st['signed_pad_cop_gate_foot_full_horizon']['median']} extreme {st['signed_pad_cop_gate_foot_full_horizon']['most_extreme']}")
w("")
hdr = f"{'checkpoint':14s} {'mode':6s} {'falls':>5s} {'fullH':>6s} {'clean':>6s} {'LCB':>6s} {'padMed':>7s} {'padWorst':>8s} {'>0.80':>5s} {'sgnMed':>7s} {'sgnExt':>7s} {'reward':>8s}"
w(hdr)
for label, _ in LABELS:
    e = summary["checkpoints"][label]
    cz = e["cert_zero_deg"]
    for mode, tag in (("stochastic", "stoch"), ("deterministic", "det")):
        if mode not in e:
            continue
        x = e[mode]
        pc, sg = x["pad_cop_gate_metric_full_horizon"], x["signed_pad_cop_gate_foot_full_horizon"]
        w(f"{label:14s} {tag:6s} {x['falls']:>5d} {x['full_horizon_fraction']:>6.3f} {x['n_clean_v2']:>3d}/40 {x['clean_lcb_v2']:>6.3f} "
          f"{pc['median']:>7.4f} {pc['worst']:>8.4f} {pc['past_bar_0.80']:>5d} {sg['median']:>7.4f} {sg['most_extreme']:>7.4f} {x['reward_mean']:>8.1f}")
    w(f"{'':14s} cert0  {'':>5s} {cz['zero_deg_full_horizon_fraction']:>6.3f} {cz['zero_deg_n_clean']:>3d}/40 {'':>6s} {cz['zero_deg_pad_median']:>7.4f} "
      f"{cz['zero_deg_pad_worst']:>8.4f}   six-heading cert: {'PASS' if cz['six_heading_passed'] else 'FAIL'} {cz['six_heading_n_clean_seeds']}/40")
w("")
# per-actuator std table (exp(log_std), from the stochastic panel's meta; identical in both modes)
names = list(summary["checkpoints"][LABELS[0][0]]["stochastic"]["std_per_actuator"])
w("Per-actuator policy std, exp(log_std) (action units; Box [-1, 1]); from each stochastic panel.json meta.std_per_actuator:")
w(f"{'actuator':14s} " + " ".join(f"{l:>13s}" for l, _ in LABELS))
for n in names:
    w(f"{n:14s} " + " ".join(f"{summary['checkpoints'][l]['stochastic']['std_per_actuator'][n]:>13.4f}" for l, _ in LABELS))
LEG = [n for n in names if n[:2] in ("r_", "l_")]
summary["std_group_means"] = {}
for l, _ in LABELS:
    sd = summary["checkpoints"][l]["stochastic"]["std_per_actuator"]
    summary["std_group_means"][l] = {"all": r4(statistics.mean(sd.values())), "legs": r4(statistics.mean(sd[n] for n in LEG)),
                                     "neck_head_tail": r4(statistics.mean(sd[n] for n in names if n not in LEG))}
w(f"{'mean (all)':14s} " + " ".join(f"{summary['std_group_means'][l]['all']:>13.4f}" for l, _ in LABELS))
w(f"{'mean (legs)':14s} " + " ".join(f"{summary['std_group_means'][l]['legs']:>13.4f}" for l, _ in LABELS))
w(f"{'mean (axial)':14s} " + " ".join(f"{summary['std_group_means'][l]['neck_head_tail']:>13.4f}" for l, _ in LABELS))
w("")
(O / "own_sigma_summary.json").write_text(json.dumps(summary, indent=1, default=str))
for label, _ in LABELS:
    e = summary["checkpoints"][label]
    w(f"== {label}")
    if "stochastic" in e:
        x = e["stochastic"]
        w(f"   stochastic file: {x['file']}")
        w(f"   falls: {x['fall_detail']}")
        w(f"   unclean seeds (stochastic): {x['unclean_seeds']}")
        w(f"   criterion failures (stochastic): {x['criterion_failures']}; v2 single-panel verdict {'PASS' if x['v2_single_panel_passed'] else 'FAIL'}")
        w(f"   std per actuator: {x['std_per_actuator']} (mean {x['std_mean']})")
        w(f"   clip fraction per actuator (mean over episodes): {x['clip_fraction_per_actuator_mean']}")
        w(f"   checks: |clip(mu) - repo deterministic predict| max {x['check_max_abs_clip_mu_minus_repo_deterministic_predict']}, "
          f"|dist std - exp(log_std)| max {x['check_max_abs_dist_std_minus_exp_log_std']:.2e}, gate metric recomputed from trace: {x['recompute_matches_gate_metric_all']}")
        w("   bar margins (stochastic, full-horizon): " + "; ".join(f"{k} worst {v['worst']} med {v['median']} past {v['past_bar']}" for k, v in x["bar_margins_full_horizon"].items() if v))
        w(f"   toe-edge step fraction (worse foot) median {x['toe_edge_step_fraction_max_foot_median']}, heel-edge {x['heel_edge_step_fraction_max_foot_median']}")
    if "deterministic" in e:
        x = e["deterministic"]
        w(f"   deterministic file: {x['file']}; unclean {x['unclean_seeds']}; falls {x['fall_detail']}")
        w("   bar margins (deterministic, full-horizon): " + "; ".join(f"{k} worst {v['worst']} med {v['median']} past {v['past_bar']}" for k, v in x["bar_margins_full_horizon"].items() if v))
        w(f"   harness check vs cert 0 deg cell ({e['cert_zero_deg']['file']}): {e.get('harness_check_deterministic_vs_cert_zero_deg')}; "
          f"model sha matches cert: {x.get('model_sha_matches_cert')}; statue reference matches cert 0 deg: {e['statue_reference_matches_cert_zero_deg']}")
    if "paired_stochastic_vs_deterministic" in e:
        w(f"   paired: {e['paired_stochastic_vs_deterministic']}")
    w("")
(O / "own_sigma_report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
