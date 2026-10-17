"""foot/symmetry screen (final design, r2): compute every predeclared criterion from the raw per-episode records and apply
the binding decision logic (decide_final.py).  Read-only on its inputs; no rounding before any comparison (float64 as
computed; rounding is for display only); every threshold comes from frozen_params.json, whose sha256 is checked
against the value passed on the command line (READY-v1 records it).

Inputs per run (templates in frozen_params.json 'layout', overridable with --layout for validation on older screens):
  judge_{steps}.json        judge.py panel (40 episodes, seeds 3042-3081) and the H4 noise panels
  gate_progress.npz         in-training evals (resume6m/ carries 0-6M natively since PR #621)
  roll_diag_{steps}/measure.json, statue.npz     H5 rows and the statue rolls they are measured against
  judge_sym_{steps}/{raw,sym}/     judge_sym.py --log-obs (raw) and --symmetrize (sym) at 6M (S1, S2) and 3M (S1 report)
  validity.json             the harness's record of method failures / collapse-backstop firings / infrastructure
                            failures (imputed checkpoints) / exploratory label; absent = none
INVALID-infrastructure (exit status 3, nothing decided): in the J1 / J2 modes every input the protocol says must exist
for a non-imputed checkpoint is required (3M judges and evals for every arm; 4M-6M judges and the complete 3-6M evals
for every arm that continued or is the comparator; the 6M roll_diag for every arm that continued; the 6M judge_sym
raw / sym / obslog for every H-PASS arm), and every judge must carry the frozen configuration (panel seeds, the
statue's reward, and at 6M the H4 noise panels: control sigma source and vector per seed, 20 episodes on 3042-3061,
joint space; own sigma, 20 episodes on 3042-3061); every roll_diag statue.npz must have the READY-v1 sha256.  A
missing or mis-configured input is regenerated and verify is re-run; it never becomes a criterion failure.
Usage:
  verify_final.py --params frozen_params.json --params-sha256 <hex> --runs <screen/runs> --interim --out interim.json      (J1)
  verify_final.py --params frozen_params.json --params-sha256 <hex> --runs <screen/runs> --s2-bars <S2_bars.json>
                  --s2-bars-sha256 <hex> --statue-rolls-sha256 <hex> --out verdict.json                                   (J2)
  verify_final.py --params ... --layout L --single-arm KEY=RUN [--no-symmetry] (validation on older screens only; not a verdict)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decide_final as D  # noqa: E402

LABELS = ("1M", "2M", "3M", "4M", "5M", "6M")
EXIT_INVALID_INFRA = 3


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ helpers
def reason_key(r: str) -> str:
    return r.split(":")[0].strip()


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def judge_config_issues(d: dict, P: dict, seed: int, lab: str, path: str) -> list:
    """the frozen judge configuration: panel seeds, the statue's reward and, at 6M, the H4 noise panels."""
    bad = []
    eps = d["panel"]["episodes_detail"]
    if [e["seed"] for e in eps] != P["panel"]["seeds"]:
        bad.append(f"{path}: panel seeds differ from the frozen panel")
    if d["panel"].get("statue_mean_reward") != P["files"]["statue"]["statue_mean_reward"]:
        bad.append(f"{path}: statue_mean_reward {d['panel'].get('statue_mean_reward')} != frozen {P['files']['statue']['statue_mean_reward']}")
    if lab == "6M":
        h4 = P["criteria"]["H4"]
        cj, ow = d.get("noise_matched_control_joint") or {}, d.get("noise_matched_own") or {}
        ref = h4["control_sigma"][str(seed)]
        checks = {"control joint sigma_source": cj.get("sigma_source") == ref["source"],
                  "control joint sigma_vector": cj.get("sigma_vector") == ref["vector"],
                  "control joint n": cj.get("n") == h4["episodes_per_seed"] and len(cj.get("episodes_detail", [])) == h4["episodes_per_seed"],
                  "control joint spawns": cj.get("seeds") == h4["spawn_range"], "control joint space": cj.get("space") == h4["space"],
                  "own sigma_source": ow.get("sigma_source") == "own",
                  "own n": ow.get("n") == h4["episodes_per_seed"] and len(ow.get("episodes_detail", [])) == h4["episodes_per_seed"],
                  "own spawns": ow.get("seeds") == h4["spawn_range"]}
        bad += [f"{path}: H4 noise configuration: {k}" for k, ok in checks.items() if not ok]
    return bad


def panel_cell(d: dict, P: dict) -> dict:
    """counts and per-episode pad statistics from one judge file's panel."""
    c = P["criteria"]
    eps = d["panel"]["episodes_detail"]
    full = [bool(e["reached_horizon"]) for e in eps]
    rr = [f and ((e.get("max_sole_tilt_deg") is not None and e["max_sole_tilt_deg"] > c["roll_rail_tilt_deg"]) or
                 (e.get("max_sole_corner_lift_m") is not None and e["max_sole_corner_lift_m"] > c["roll_rail_corner_lift_m"]))
          for e, f in zip(eps, full)]
    cop = [(e.get("max_sole_cop_fore_aft") is not None and e["max_sole_cop_fore_aft"] > c["cop_bar"]) for e in eps]
    gr = [e.get("pad", {}).get("r_sole_cop_fore_aft_gate") for e in eps]
    gl = [e.get("pad", {}).get("l_sole_cop_fore_aft_gate") for e in eps]
    even = [(a + b) / 2 for a, b, f in zip(gr, gl, full) if f and a is not None and b is not None]
    gap = [a - b for a, b, f in zip(gr, gl, full) if f and a is not None and b is not None]   # signed; the class uses |median|
    gmin_fail = [min(a, b) for a, b, f, x in zip(gr, gl, full, cop) if f and x and a is not None and b is not None]
    clean = [bool(e["clean"]) for e in eps]
    hop = [any(reason_key(r) in set(P["gate"]["hop_or_fall_keys"]) | {"horizon"} for r in e["reasons"]) or not f for e, f in zip(eps, full)]
    p = d["panel"]
    k, n = sum(clean), len(eps)
    lcb = float(stats.beta.ppf(P["gate"]["lcb_alpha"], k, n - k + 1)) if k > 0 else 0.0
    rails_ok = p["mean_reward"] >= P["gate"]["min_avg_reward"] and (p.get("statue_mean_reward") is None or p["mean_reward"] >= P["gate"]["min_avg_reward_statue_ratio"] * p["statue_mean_reward"])
    comp = {"n_ge_min": n >= P["gate"]["min_eval_episodes"], "lcb": lcb >= P["gate"]["min_clean_stance_lcb"],
            "full_fraction": sum(full) / n >= P["gate"]["min_full_horizon_fraction"], "hop_or_fall": sum(hop) <= P["gate"]["max_hop_or_fall_episodes"],
            "reward_rails": bool(rails_ok)}
    unclean_rails = {}
    for e, f in zip(eps, full):
        if e["clean"]:
            continue
        keys = {reason_key(r) for r in e["reasons"]}
        fam = set()
        for kk in keys:
            if kk in ("max_sole_tilt_deg", "max_sole_corner_lift_m"):
                fam.add("ROLL")
            elif kk == "max_sole_cop_fore_aft":
                fam.add("COP")
            else:
                fam.add("ROBUSTNESS")
        if not f:
            fam.add("ROBUSTNESS")
        for x in fam:
            unclean_rails[x] = unclean_rails.get(x, 0) + 1
    return {"n": n, "falls": n - sum(full), "full": sum(full), "roll_rail": sum(rr), "cop_any": sum(cop),
            "cop_fh": sum(1 for a, f in zip(cop, full) if a and f), "cop_unmeasured": sum(1 for e in eps if e.get("max_sole_cop_fore_aft") is None),
            "n_clean": k, "lcb": lcb, "hop_or_fall": sum(hop), "verdict": p["verdict"], "gate_components": comp,
            "gate_components_pass": all(comp.values()), "agrees_with_gate": all(comp.values()) == (p["verdict"] == "PASS"),
            "median_even_g": float(np.median(even)) if even else None, "pad_bias": float(abs(np.median(gap))) if gap else None,
            "median_min_g_cop_failing (report only)": float(np.median(gmin_fail)) if gmin_fail else None,
            "even_g_per_episode": even, "unclean_rail_families": unclean_rails,
            "noise_joint_falls": (d.get("noise_matched_control_joint") or {}).get("falls"),
            "noise_own_falls": (d.get("noise_matched_own") or {}).get("falls"),
            "action_scale": d.get("action_scale"), "action_scale_nonunit": d.get("action_scale_nonunit")}


def ols_proj(x, y, x0, t_q):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = len(x)
    X = np.stack([x, np.ones_like(x)], 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res / (n - 2))
    sxx = float(((x - x.mean()) ** 2).sum())
    pred = float(beta[1] + beta[0] * x0)
    sem = math.sqrt(s2 * (1 / n + (x0 - x.mean()) ** 2 / sxx))
    t = float(stats.t.ppf(t_q, n - 2))
    xc = x - x.mean()
    A = np.stack([xc ** 2, xc, np.ones_like(xc)], 1)
    cq, *_ = np.linalg.lstsq(A, y, rcond=None)
    rq = y - A @ cq
    cov = float(rq @ rq / (n - 3)) * np.linalg.inv(A.T @ A)
    return {"n": n, "slope": float(beta[0]), "slope_se": math.sqrt(s2 / sxx), "proj": pred, "sem": sem, "ci90_lo": pred - t * sem, "ci90_hi": pred + t * sem,
            "quad_curv_t": float(cq[0] / math.sqrt(cov[0, 0]))}


def gate_series(paths):
    """concatenate the fresh file's rows with the resume file's later rows (or the resume file alone if it carries all)."""
    zs = [dict(np.load(p)) for p in paths if os.path.exists(p)]
    if not zs:
        return None
    if len(zs) == 1:
        return zs[0]
    z6, z3 = zs[0], zs[1]
    t3 = z3["timesteps"]
    late = z6["timesteps"] > t3.max()
    early = ~late
    if early.sum() == len(t3):
        return z6
    return {k: np.concatenate([z3[k], z6[k][late]]) for k in z3 if k in z6}


def h3b_window(Z, P):
    h = P["criteria"]["H3b"]
    t = Z["timesteps"].astype(np.int64)
    lo, hi = P["checkpoints"]["3M"], P["checkpoints"]["6M"]
    late = (t > lo) & (t <= hi)
    expected = np.arange(lo + h["eval_every"], hi + 1, h["eval_every"])
    return t, late, bool(np.array_equal(t[late], expected))


def h3b(Z, P):
    h = P["criteria"]["H3b"]
    lo = P["checkpoints"]["3M"]
    t, late, complete = h3b_window(Z, P)
    out = {"n_evals": int(late.sum()), "complete": complete}
    x = t / 1e6
    g = {f: Z[f"{f}_sole_cop_fore_aft_gate"].astype(np.float64) for f in "rl"}
    for f in "rl":
        fit = ols_proj(x[late], g[f][late], h["x_eval_M"], h["t_quantile"])
        sub = t > P["checkpoints"]["4M"]
        fit["fit_4.2_6.3_proj"] = ols_proj(x[sub & late], g[f][sub & late], h["x_eval_M"], h["t_quantile"])["proj"] if (sub & late).sum() > 3 else None
        fit["pass"] = fit["proj"] <= h["bar"]
        fit["ci_fail (report only since r2)"] = fit["ci90_lo"] > h["bar"]
        out[f] = fit
    # fa_needed rule (ii) (r2): the EVEN series (g_r + g_l)/2, one test per seed
    ev = ols_proj(x[late], (g["r"][late] + g["l"][late]) / 2, h["x_eval_M"], h["t_quantile"])
    ev["ci_fail"] = ev["ci90_lo"] > h["bar"]
    out["even"] = ev
    # v1's edge-pad version (report only): the pad with the larger mean over the last 8 evals <= 3M
    m3 = t <= lo
    edge = "r" if Z["r_sole_cop_fore_aft_gate"][m3][-8:].mean() >= Z["l_sole_cop_fore_aft_gate"][m3][-8:].mean() else "l"
    out["v1_edge_pad"] = edge
    out["pass"] = out["complete"] and out["r"]["pass"] and out["l"]["pass"]
    return out


def last8_evals(Z, steps):
    t = Z["timesteps"].astype(np.int64)
    m = t <= steps
    gr = Z["r_sole_cop_fore_aft_gate"][m][-8:].astype(np.float64)
    gl = Z["l_sole_cop_fore_aft_gate"][m][-8:].astype(np.float64)
    return float(((gr + gl) / 2).mean()), float(abs(gr.mean() - gl.mean()))


def h5(measures, P):
    focus = set(P["criteria"]["H5"]["focus_spawns"])
    rows, per = [], {}
    for s, m in measures.items():
        sel = [abs(r["policy_added"]) for r in m["h5_rows"] if r["focus"] and r["base_full"] and math.isfinite(r["policy_added"])]
        assert {r["seed"] for r in m["h5_rows"] if r["focus"]} == focus, "roll_diag focus spawns differ from the frozen list"
        per[s] = {"median": float(np.median(sel)) if sel else None, "n": len(sel)}
        rows += sel
    med = float(np.median(rows)) if rows else float("inf")
    return {"per_seed": per, "pooled_median_deg": med, "n": len(rows), "pass": med <= P["criteria"]["H5"]["bar_deg"]}


# ------------------------------------------------------------------------------------------------ symmetry (S1, S2)
class MirrorMaps:
    def __init__(self, P):
        m = json.load(open(P["files"]["mirror_map"]["path"]))
        assert sha_file(P["files"]["mirror_map"]["path"]) == P["files"]["mirror_map"]["sha256"]
        a = sorted(m["action_map"], key=lambda e: e["index"])
        o = sorted(m["obs_map"], key=lambda e: e["index"])
        self.ap, self.asg = np.array([e["mirror_index"] for e in a]), np.array([e["sign"] for e in a], float)
        self.op, self.osg = np.array([e["mirror_index"] for e in o]), np.array([e["sign"] for e in o], float)

    def Ma(self, x):
        return self.asg * x[..., self.ap]

    def Mo(self, x):
        return self.osg * x[..., self.op]


def mu_bar_forward(obs_bar, model_zip, vecnorm_pkl):
    """mu(s_bar) by a forward pass on the checkpoint (stock SB3 load; used when the obslog lacks 'mu_bar')."""
    import pickle

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import BaseModel
    torch.set_num_threads(1)
    pol = PPO.load(model_zip, device="cpu").policy
    vn = pickle.load(open(vecnorm_pkl, "rb"))
    x = np.clip((obs_bar - np.asarray(vn.obs_rms.mean)) / np.sqrt(np.asarray(vn.obs_rms.var) + vn.epsilon), -vn.clip_obs, vn.clip_obs).astype(np.float32)
    with torch.no_grad():
        f = BaseModel.extract_features(pol, torch.as_tensor(x), pol.pi_features_extractor)
        return pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy().astype(np.float64)


def s1_seed(obslog_path, judge_raw, mm, P, model_zip=None, vecnorm=None):
    S = P["criteria"]["S1"]
    z = np.load(obslog_path)
    rows = judge_raw["panel"]["episodes_detail"]
    ep, t = z["episode"], z["t"]
    full = np.array([bool(rows[e]["reached_horizon"]) for e in ep])
    win = full & (t >= S["window_start"]) & (t <= S["window_end"])
    n_full_eps = sum(1 for r in rows if r["reached_horizon"])
    mu, mum = z["mu"].astype(np.float64), z["mu_m"].astype(np.float64)
    Dd = mu - mm.Ma(mum)
    out = {"n_full_horizon_episodes": n_full_eps, "window_steps": int(win.sum()), "groups": {}, "channels": {}}
    if n_full_eps < S["min_full_horizon_episodes"]:
        out["pass"] = False
        out["reason"] = "fewer full-horizon episodes than the minimum"
        return out
    deg = np.array(S["deg_per_unit_by_actuator"], float)
    for g, spec in S["groups"].items():
        idx = spec["actuators"]
        v = np.max(np.abs(Dd[win][:, idx]) / 2 * deg[idx], axis=1)
        med, p90 = float(np.median(v)), float(np.percentile(v, 90))
        out["groups"][g] = {"median_deg": med, "p90_deg": p90, "bound_deg": spec["bound_deg"],
                            "pass": med <= spec["bound_deg"] and p90 <= S["k90"] * spec["bound_deg"]}
    if "mu_bar" in z.files:
        mub = z["mu_bar"].astype(np.float64)
    else:
        obs = z["obs"].astype(np.float64)
        mub = mu_bar_forward((obs + mm.Mo(obs)) / 2.0, model_zip, vecnorm)
    a = np.clip(mub[win], -1, 1)
    odd = (a - mm.Ma(a)) / 2
    for c, spec in S["channels"].items():
        val = abs(float(odd[:, spec["odd_index"]].mean())) * spec["deg_per_unit"]
        out["channels"][c] = {"abs_mean_deg": val, "bound_deg": spec["bound_deg"], "pass": val <= spec["bound_deg"]}
    out["pass"] = all(x["pass"] for x in out["groups"].values()) and all(x["pass"] for x in out["channels"].values())
    out["ankle_common_dc_deg"] = float(((np.clip(mu[win], -1, 1)[:, 6] + np.clip(mu[win], -1, 1)[:, 10]) / 2).mean() * deg[6])
    return out


def s2(raw_judges, sym_judges, bars):
    flips = dclean = 0
    for s in raw_judges:
        r = {e["seed"]: bool(e["clean"]) for e in raw_judges[s]["panel"]["episodes_detail"]}
        y = {e["seed"]: bool(e["clean"]) for e in sym_judges[s]["panel"]["episodes_detail"]}
        assert set(r) == set(y)
        flips += sum(1 for k in r if r[k] != y[k])
        dclean += sum(r.values()) - sum(y.values())
    return {"flips": flips, "delta_clean": dclean, "F_S2": bars["F_S2"], "D_S2": bars["D_S2"],
            "pass": flips <= bars["F_S2"] and abs(dclean) <= bars["D_S2"]}


# ------------------------------------------------------------------------------------------------ one arm
def arm_record(arm_key, run_name, P, L, runs, comparator=False, expect=None, statue_rolls_sha=None):
    """expect: None (validation: nothing required) or {"resume": bool, "h5": bool}: the protocol's required inputs, whose
    absence (for a non-imputed checkpoint) is listed in rec['missing_inputs'] (-> INVALID-infrastructure)."""
    seeds = P["seeds"]
    cks = P["checkpoints"]
    cells, gp, validity, missing, config = {}, {}, {}, [], []
    for s in seeds:
        run = L["run"].format(runs=runs, arm=run_name, seed=s)
        validity[s] = load_json(L["validity"].format(run=run, arm=run_name, seed=s, runs=runs)) or {}
    imputed = {s: list(validity[s].get("imputed_checkpoints", [])) for s in seeds}
    for s in seeds:
        run = L["run"].format(runs=runs, arm=run_name, seed=s)
        cells[s] = {}
        for lab in LABELS:
            path = L["judge"].format(run=run, arm=run_name, seed=s, runs=runs, steps=cks[lab])
            d = load_json(path)
            cells[s][lab] = panel_cell(d, P) if d is not None else None
            if d is not None and lab in ("3M", "4M", "5M", "6M"):
                config += judge_config_issues(d, P, s, lab, path)
            required = expect is not None and lab not in imputed[s] and (lab == "3M" or (lab in ("4M", "5M", "6M") and expect["resume"]))
            if d is None and required:
                missing.append(path)
        gp[s] = gate_series([p.format(run=run, arm=run_name, seed=s, runs=runs) for p in L["gate_progress"]])
        if expect is not None:
            if gp[s] is None and "3M" not in imputed[s]:
                missing.append(f"{run}: gate_progress.npz")
            elif gp[s] is not None and expect["resume"] and "6M" not in imputed[s] and not h3b_window(gp[s], P)[2]:
                missing.append(f"{run}: gate_progress.npz does not hold the 24 evals in (3M, 6M]")
    rec = {"arm": arm_key, "run_name": run_name, "seeds": seeds, "comparator": comparator, "seeds_imputed": {str(s): v for s, v in imputed.items()},
           "exploratory": any(bool(validity[s].get("exploratory")) for s in seeds),
           "exploratory_reason": [f"s{s}: {validity[s].get('exploratory_reason')}" for s in seeds if validity[s].get("exploratory")]}
    c = P["criteria"]

    def cell(s, lab):
        x = cells[s][lab]
        return None if (x is None or lab in imputed[s]) else x
    # interim (an imputed / missing 3M checkpoint counts as failed in every criterion)
    c3 = {s: cell(s, "3M") for s in seeds}
    miss3 = [s for s in seeds if c3[s] is None]
    rr3 = sum(x["roll_rail"] for x in c3.values() if x)
    f3 = sum(x["falls"] for x in c3.values() if x)
    i3_bad = sum(1 for x in c3.values() if x and x["cop_any"] > c["I3"]["per_seed_bar"]) + len(miss3)
    rec["interim"] = {"I1": not miss3 and rr3 <= c["I1"]["pooled_max"], "I2": not miss3 and f3 <= c["I2"]["pooled_max"],
                      "I3": i3_bad <= c["I3"]["max_seeds_over"], "roll_rail_per_seed": [c3[s]["roll_rail"] if c3[s] else None for s in seeds],
                      "falls_per_seed": [c3[s]["falls"] if c3[s] else None for s in seeds],
                      "cop_any_per_seed": [c3[s]["cop_any"] if c3[s] else None for s in seeds], "missing_or_imputed_3M": miss3}
    have6 = all(cells[s]["6M"] is not None or "6M" in imputed[s] for s in seeds)
    rec["reached_6M"] = bool(have6 and any(cells[s]["6M"] is not None for s in seeds))
    # 6M H criteria
    if rec["reached_6M"]:
        c6 = {s: cell(s, "6M") for s in seeds}
        h1_seed = {s: bool(c6[s] and c6[s]["gate_components_pass"]) for s in seeds}
        agree = all(c6[s]["agrees_with_gate"] for s in seeds if c6[s])
        rr6 = [c6[s]["roll_rail"] if c6[s] else None for s in seeds]
        H2 = all(v is not None for v in rr6) and sum(rr6) <= c["H2"]["pooled_max"] and max(rr6) <= c["H2"]["per_seed_max"]
        h3a_rows = {s: [cell(s, lab)["cop_any"] if cell(s, lab) else None for lab in ("4M", "5M", "6M")] for s in seeds}
        H3a = all(v is not None and v <= c["H3a"]["per_seed_max"] for r in h3a_rows.values() for v in r)
        h3b_rows = {s: (h3b(gp[s], P) if gp[s] is not None and "6M" not in imputed[s] else None) for s in seeds}
        H3b = all(r is not None and r["pass"] for r in h3b_rows.values())
        jf = [c6[s]["noise_joint_falls"] if c6[s] else None for s in seeds]
        of = [c6[s]["noise_own_falls"] if c6[s] else None for s in seeds]
        H4 = all(v is not None for v in jf + of) and sum(jf) <= c["H4"]["joint_pooled_max"] and sum(of) <= c["H4"]["own_pooled_max"]
        meas = {}
        for s in seeds:
            run = L["run"].format(runs=runs, arm=run_name, seed=s)
            mp = L["roll_diag_measure"].format(run=run, arm=run_name, seed=s, runs=runs, steps=cks["6M"])
            m = load_json(mp)
            if "6M" in imputed[s]:
                continue
            if m is not None:
                meas[s] = m
                sp = L["roll_diag_statue"].format(run=run, arm=run_name, seed=s, runs=runs, steps=cks["6M"])
                if statue_rolls_sha is not None and (not os.path.exists(sp) or sha_file(sp) != statue_rolls_sha):
                    config.append(f"{sp}: statue rolls missing or sha256 != READY-v1 value")
            elif expect is not None and expect["h5"]:
                missing.append(mp)
        h5r = h5(meas, P) if len(meas) == len(seeds) else {"pass": False, "reason": "roll_diag missing or imputed"}
        f6 = [c6[s]["falls"] if c6[s] else None for s in seeds]
        H6 = all(v is not None for v in f6) and sum(f6) <= c["H6"]["pooled_max"]
        rec["H"] = {"H1": sum(h1_seed.values()) >= c["H1"]["min_seeds"], "H2": bool(H2), "H3a": bool(H3a), "H3b": bool(H3b),
                    "H3": bool(H3a and H3b), "H4": bool(H4), "H5": bool(h5r["pass"]), "H6": bool(H6)}
        rec["H_detail"] = {"H1_per_seed": h1_seed, "H1_components_agree_with_gate": agree, "H2_roll_rail_per_seed": rr6,
                           "H3a_cop_any_4_5_6M": h3a_rows, "H3b": h3b_rows, "H4_joint_falls": jf, "H4_own_falls": of, "H5": h5r, "H6_falls": f6,
                           "action_scale_unit": all((c6[s] or {}).get("action_scale_nonunit") in (None, {}, []) for s in seeds)}
        if not agree:
            rec["INVALID_analysis"] = "gate verdict disagrees with its recomputed components -> arbiter"
    else:
        rec["H"] = None
    # families failed
    fam = set()
    if not rec["interim"]["I1"]:
        fam.add("ROLL")
    if not rec["interim"]["I2"]:
        fam.add("ROBUSTNESS")
    if not rec["interim"]["I3"]:
        fam.add("COP")
    if rec["H"]:
        for k, f in (("H2", "ROLL"), ("H5", "ROLL"), ("H3", "COP"), ("H4", "ROBUSTNESS"), ("H6", "ROBUSTNESS")):
            if not rec["H"][k]:
                fam.add(f)
        if not rec["H"]["H1"]:
            tally = {}
            for s in seeds:
                cc = cell(s, "6M")
                if cc and not cc["gate_components_pass"]:
                    for k, v in cc["unclean_rail_families"].items():
                        tally[k] = tally.get(k, 0) + v
            if tally:
                top = max(tally.values())
                fam |= {k for k, v in tally.items() if v == top}
            else:
                fam.add("ROBUSTNESS")
    if any(validity[s].get("method_failure") for s in seeds):
        fam.add("ROBUSTNESS")
    rec["families_failed"] = sorted(fam)
    # fa evidence (observed checkpoints, full-horizon CoP only; class per seed and checkpoint)
    ev = {"per_seed": {}, "h3b_even_ci_fail": {}, "h3b_pad_ci_fail (report only)": {}, "class_6M": {}}
    for s in seeds:
        per = {}
        for lab in ("3M", "4M", "5M", "6M"):
            x = cell(s, lab)
            if x is None:
                per[lab] = {"observed": False}
                continue
            e8 = last8_evals(gp[s], cks[lab]) if gp[s] is not None else (None, None)
            cl = D.classify_cop(x["median_even_g"], x["pad_bias"], e8[0], e8[1], P) if x["median_even_g"] is not None else {"class": None, "evals_agree": None}
            per[lab] = {"observed": True, "cop_fh": x["cop_fh"], "cop_any": x["cop_any"], "class": cl["class"], "evals_agree": cl["evals_agree"],
                        "median_even_g": x["median_even_g"], "pad_bias": x["pad_bias"],
                        "median_min_g_cop_failing (report only)": x["median_min_g_cop_failing (report only)"]}
        ev["per_seed"][str(s)] = per
        hb = rec.get("H_detail", {}).get("H3b", {}).get(s) if rec["H"] else None
        if hb is None and gp[s] is not None and "6M" not in imputed[s] and cell(s, "6M") is not None:
            hb = h3b(gp[s], P)
        ok6 = bool(hb and hb["complete"] and cell(s, "6M") is not None)
        ev["h3b_even_ci_fail"][str(s)] = bool(ok6 and hb["even"]["ci_fail"])
        ev["h3b_pad_ci_fail (report only)"][str(s)] = [f for f in "rl" if ok6 and hb[f]["ci_fail (report only since r2)"]]
        ev["class_6M"][str(s)] = per["6M"].get("class") if per["6M"]["observed"] else None
    rec["cop_evidence"] = ev
    rec["_cells"] = {str(s): {lab: (None if v is None else {k: vv for k, vv in v.items() if k != "even_g_per_episode"}) for lab, v in cells[s].items()} for s in seeds}
    rec["_even_g_pooled"] = {lab: [g for s in seeds if cells[s][lab] for g in cells[s][lab]["even_g_per_episode"]] for lab in LABELS}
    rec["missing_inputs"] = missing
    rec["judge_config_issues"] = config
    return rec


def symmetry(run_name, P, L, runs, mm, bars, required):
    """S1 / S2 at 6M.  required (an H-PASS arm): missing inputs are returned for INVALID-infrastructure; otherwise
    (report only) a missing input gives S1 = S2 = None."""
    seeds = P["seeds"]
    cks = P["checkpoints"]
    s1, raw_j, sym_j, missing, config = {}, {}, {}, [], []
    for s in seeds:
        run = L["run"].format(runs=runs, arm=run_name, seed=s)
        pr = L["judge_sym"].format(run=run, arm=run_name, seed=s, runs=runs, steps=cks["6M"], mode="raw")
        ps = L["judge_sym"].format(run=run, arm=run_name, seed=s, runs=runs, steps=cks["6M"], mode="sym")
        ob = L["obslog"].format(run=run, arm=run_name, seed=s, runs=runs, steps=cks["6M"], mode="raw")
        jr, js = load_json(pr), load_json(ps)
        missing += [p for p, x in ((pr, jr), (ps, js), (ob, os.path.exists(ob) or None)) if x is None]
        if jr is None or js is None or not os.path.exists(ob):
            continue
        for p, d in ((pr, jr), (ps, js)):
            if [e["seed"] for e in d["panel"]["episodes_detail"]] != P["panel"]["seeds"] or d["panel"].get("statue_mean_reward") != P["files"]["statue"]["statue_mean_reward"]:
                config.append(f"{p}: panel seeds or statue reward differ from the frozen values")
        raw_j[s], sym_j[s] = jr, js
        s1[s] = s1_seed(ob, jr, mm, P, jr.get("model"), jr.get("vecnorm"))
    if missing or config:
        return {"S1": None, "S2": None, "missing_inputs": missing if required else [], "judge_config_issues": config if required else [],
                "symmetry_unavailable (report only)": missing + config}
    sd = s2(raw_j, sym_j, bars)
    return {"S1": all(v["pass"] for v in s1.values()), "S1_detail": s1, "S2": sd["pass"], "S2_detail": sd, "missing_inputs": [], "judge_config_issues": []}


def view_3M(rec: dict) -> dict:
    """L's record restricted to its 3M evidence (the comparator rule asks whether the 3M evidence established fa_needed)."""
    r = copy.deepcopy(rec)
    ev = r["cop_evidence"]
    ev["per_seed"] = {s: {"3M": v["3M"]} for s, v in ev["per_seed"].items()}
    ev["h3b_even_ci_fail"] = {s: False for s in ev["h3b_even_ci_fail"]}
    return r


def check_params(path, sha):
    h = sha_file(path)
    assert sha is None or h == sha, f"frozen_params sha256 {h} != {sha}"
    return json.load(open(path)), h


def invalid_infra(out, a, problems):
    out["INVALID-infrastructure"] = problems
    out["decision"] = None
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"INVALID-infrastructure": problems}, indent=1))
    sys.exit(EXIT_INVALID_INFRA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--params-sha256", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--layout")
    ap.add_argument("--s2-bars")
    ap.add_argument("--s2-bars-sha256")
    ap.add_argument("--statue-rolls-sha256")
    ap.add_argument("--out")
    ap.add_argument("--single-arm", help="KEY=RUN_NAME: one arm's record only (validation on older screens; NOT a verdict)")
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--interim", action="store_true", help="J1: the 3M interim decision, the comparator rule and the 3M classes only")
    a = ap.parse_args()
    P, h = check_params(a.params, a.params_sha256)
    L = json.load(open(a.layout)) if a.layout else P["layout"]
    mm = MirrorMaps(P)
    assert sha_file(P["files"]["statue"]["path"]) == P["files"]["statue"]["sha256"], "statue reference changed"
    out = {"params_sha256": h, "runs": a.runs, "arms": {}}
    if a.single_arm:
        out["mode"] = "validation (single arm): not a verdict"
        bars = json.load(open(a.s2_bars)) if a.s2_bars else P["criteria"]["S2"]["fallback_bars"]
        key, name = a.single_arm.split("=")
        rec = arm_record(key, name, P, L, a.runs, statue_rolls_sha=a.statue_rolls_sha256)
        if not a.no_symmetry:
            rec.update(symmetry(name, P, L, a.runs, mm, bars, required=False))
        out["arms"][key] = rec
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1, default=str)
        print(json.dumps({k: rec.get(k) for k in ("interim", "H", "families_failed", "S1", "S2", "judge_config_issues")}, indent=1, default=str))
        return
    # J1 / J2: the protocol's inputs are required
    out["mode"] = "J1 interim" if a.interim else "J2 verdict"
    recs0 = {key: arm_record(key, name, P, L, a.runs, expect={"resume": False, "h5": False}) for key, name in P["arms"].items()}
    cont = {k: D.interim(r) for k, r in recs0.items()}
    comp = D.comparator_continuation(view_3M(recs0["L"]), recs0["C"], P)
    if a.interim:
        probs = [p for r in recs0.values() for p in r["missing_inputs"] + r["judge_config_issues"]]
        if probs:
            invalid_infra(out, a, probs)
        out["interim"] = {key: {"continue": cont[key], "I": r["interim"], "families_failed": r["families_failed"], "exploratory": r["exploratory"],
                                "classes_3M": {s: r["cop_evidence"]["per_seed"][s]["3M"].get("class") for s in r["cop_evidence"]["per_seed"]}}
                          for key, r in recs0.items()}
        out["fa_needed_from_3M"] = D.fa_needed(view_3M(recs0["L"]), P)
        out["comparator_continuation"] = comp
        out["arms"] = recs0
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1, default=str)
        print(json.dumps({k: out[k] for k in ("interim", "fa_needed_from_3M", "comparator_continuation")}, indent=1, default=str))
        return
    for need in ("s2_bars", "s2_bars_sha256", "statue_rolls_sha256"):
        if getattr(a, need) is None:
            raise SystemExit(f"J2 requires --{need.replace('_', '-')} (READY-v1 value); there is no fallback")
    assert sha_file(a.s2_bars) == a.s2_bars_sha256, "S2 bars file sha256 differs from the READY-v1 value"
    bars = json.load(open(a.s2_bars))
    S2c = P["criteria"]["S2"]
    assert S2c["floor"]["F_S2"] <= bars["F_S2"] <= S2c["cap"]["F_S2"] and S2c["floor"]["D_S2"] <= bars["D_S2"] <= S2c["cap"]["D_S2"], "S2 bars outside floor / cap"
    recs = {}
    for key, name in P["arms"].items():
        resume = cont[key] or (key == "L" and comp)
        recs[key] = arm_record(key, name, P, L, a.runs, comparator=False, expect={"resume": resume, "h5": cont[key]},
                               statue_rolls_sha=a.statue_rolls_sha256)
    if not cont["L"] and recs["L"]["reached_6M"]:
        recs["L"]["comparator"] = True
        recs["L"]["comparator_rule_held"] = comp
    probs = [p for r in recs.values() for p in r["missing_inputs"] + r["judge_config_issues"]]
    for key, name in P["arms"].items():
        req = D.h_pass(recs[key])
        if req or (recs[key]["reached_6M"] and not recs[key]["comparator"]):
            sy = symmetry(name, P, L, a.runs, mm, bars, required=req)
            probs += sy["missing_inputs"] + sy["judge_config_issues"]
            recs[key].update(sy)
        else:
            recs[key].update({"S1": None, "S2": None})
    if probs:
        out["arms"] = recs
        invalid_infra(out, a, probs)
    out["arms"] = recs
    out["decision"] = D.outcome(recs["L"], recs["C"], P)
    out["comparator_continuation_rule (3M evidence)"] = comp
    out["inputs"] = {"s2_bars": a.s2_bars, "s2_bars_sha256": a.s2_bars_sha256, "statue_rolls_sha256": a.statue_rolls_sha256}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(out["decision"], indent=1, default=str))


if __name__ == "__main__":
    main()
