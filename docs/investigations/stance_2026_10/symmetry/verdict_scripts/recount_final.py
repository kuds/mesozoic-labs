"""foot/symmetry screen (final design, r2): INDEPENDENT recount of every count, statistic, verdict-level boolean and the
outcome that verify_final.py reports, from the same raw files but through separately written code (vectorised arrays
instead of per-episode dicts; scipy.stats.linregress and an explicit prediction-interval formula for H3(b) and the
even-series CI; binomtest's exact interval for H1's LCB; SB3's policy.get_distribution for mu(s_bar)).  It shares only
decide_final.py (the pure, unit-tested rule table), to which it hands its OWN per-arm records.

Recomputed and compared: per seed and checkpoint the counts (falls, roll rail, CoP any / full horizon, clean), median
even_g and pad bias, the class; I1-I3; H1 (per seed: the five gate components incl. the LCB) and H1; H2; H3(a); H3(b)
(both pads' projections and CIs) and the even-series projection, CI and ci_fail; H4; H5 (median |policy-added roll| over
the focus full-horizon rows, and its pass); H6; the families failed; S1 (per group median / p90, per channel |mean|);
S2 (flips, delta clean, pass); exploratory; and, when both arms are present, the outcome class, labels, PR arm, 12M
arms and fa_needed with its evidence seeds.  Any disagreement with verify_final.py's output is INVALID-analysis and goes
to arbiter_final.py.

Usage: recount_final.py --params frozen_params.json --params-sha256 <hex> --runs <dir> --verify verdict.json [--layout L]
                        [--s2-bars S2_bars.json --s2-bars-sha256 <hex>] [--with-symmetry] [--arms L,C]
Exit status 0 iff every compared quantity agrees exactly (counts, booleans, strings) or to 1e-9 (floats).
"""
from __future__ import annotations

import argparse
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

LABS = ("3M", "4M", "5M", "6M")


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def arrays(judge, P):
    E = judge["panel"]["episodes_detail"]
    f = np.array([1 if e["reached_horizon"] else 0 for e in E], bool)
    tilt = np.array([np.nan if e.get("max_sole_tilt_deg") is None else e["max_sole_tilt_deg"] for e in E], float)
    lift = np.array([np.nan if e.get("max_sole_corner_lift_m") is None else e["max_sole_corner_lift_m"] for e in E], float)
    cop = np.array([np.nan if e.get("max_sole_cop_fore_aft") is None else e["max_sole_cop_fore_aft"] for e in E], float)
    gr = np.array([np.nan if e.get("pad", {}).get("r_sole_cop_fore_aft_gate") is None else e["pad"]["r_sole_cop_fore_aft_gate"] for e in E], float)
    gl = np.array([np.nan if e.get("pad", {}).get("l_sole_cop_fore_aft_gate") is None else e["pad"]["l_sole_cop_fore_aft_gate"] for e in E], float)
    clean = np.array([e["clean"] for e in E], bool)
    c, G = P["criteria"], P["gate"]
    with np.errstate(invalid="ignore"):
        rr = f & ((np.nan_to_num(tilt, nan=-1) > c["roll_rail_tilt_deg"]) | (np.nan_to_num(lift, nan=-1) > c["roll_rail_corner_lift_m"]))
        cp = np.nan_to_num(cop, nan=-1) > c["cop_bar"]
    ok = f & np.isfinite(gr) & np.isfinite(gl)
    # H1's gate components, written independently of verify_final.panel_cell
    hopkeys = set(G["hop_or_fall_keys"]) | {"horizon"}
    hop = np.array([(not e["reached_horizon"]) or any(r.split(":")[0].strip() in hopkeys for r in e["reasons"]) for e in E], bool)
    n, k = len(E), int(clean.sum())
    lcb = float(stats.binomtest(k, n).proportion_ci(confidence_level=1 - 2 * G["lcb_alpha"], method="exact").low) if k > 0 else 0.0
    pm, st = judge["panel"]["mean_reward"], judge["panel"].get("statue_mean_reward")
    comp = {"n_ge_min": n >= G["min_eval_episodes"], "lcb": lcb >= G["min_clean_stance_lcb"], "full_fraction": f.mean() >= G["min_full_horizon_fraction"],
            "hop_or_fall": int(hop.sum()) <= G["max_hop_or_fall_episodes"],
            "reward_rails": bool(pm >= G["min_avg_reward"] and (st is None or pm >= G["min_avg_reward_statue_ratio"] * st))}
    fam = {"ROLL": 0, "COP": 0, "ROBUSTNESS": 0}
    for e, ff in zip(E, f):
        if e["clean"]:
            continue
        ks = {r.split(":")[0].strip() for r in e["reasons"]}
        got = {"ROLL" if x in ("max_sole_tilt_deg", "max_sole_corner_lift_m") else "COP" if x == "max_sole_cop_fore_aft" else "ROBUSTNESS" for x in ks}
        if not ff:
            got.add("ROBUSTNESS")
        for x in got:
            fam[x] += 1
    return {"falls": int((~f).sum()), "roll_rail": int(rr.sum()), "cop_any": int(cp.sum()), "cop_fh": int((cp & f).sum()),
            "n_clean": k, "lcb": lcb, "gate_components": comp, "gate_pass": all(comp.values()), "unclean_fam": fam,
            "median_even_g": float(np.median(((gr + gl) / 2)[ok])) if ok.any() else None,
            "pad_bias": float(np.abs(np.median((gr - gl)[ok]))) if ok.any() else None,
            "joint_falls": (judge.get("noise_matched_control_joint") or {}).get("falls"), "own_falls": (judge.get("noise_matched_own") or {}).get("falls")}


def pred_ci(x, g, P):
    h = P["criteria"]["H3b"]
    lr = stats.linregress(x, g)
    n = len(x)
    yhat = lr.intercept + lr.slope * h["x_eval_M"]
    resid = g - (lr.intercept + lr.slope * x)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    se_mean = s * np.sqrt(1.0 / n + (h["x_eval_M"] - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    q = stats.t.ppf(h["t_quantile"], n - 2)
    return float(yhat), float(yhat - q * se_mean)


def series(paths):
    got = [np.load(p) for p in paths if os.path.exists(p)]
    if not got:
        return None
    if len(got) == 1:
        return {k: got[0][k] for k in got[0].files}
    a, b = got[0], got[1]
    if len(a["timesteps"]) > len(b["timesteps"]) and np.array_equal(a["timesteps"][:len(b["timesteps"])], b["timesteps"]):
        return {k: a[k] for k in a.files}
    keep = a["timesteps"] > b["timesteps"][-1]
    return {k: np.concatenate([b[k], a[k][keep]]) for k in b.files if k in a.files}


def last8(Z, steps):
    m = Z["timesteps"].astype(np.int64) <= steps
    r = Z["r_sole_cop_fore_aft_gate"][m][-8:].astype(np.float64)
    q = Z["l_sole_cop_fore_aft_gate"][m][-8:].astype(np.float64)
    return float(((r + q) * 0.5).mean()), float(abs(r.mean() - q.mean()))


def mirror(P):
    m = json.load(open(P["files"]["mirror_map"]["path"]))
    perm = np.zeros(15, int)
    sign = np.zeros(15)
    for e in m["action_map"]:
        perm[e["index"]], sign[e["index"]] = e["mirror_index"], e["sign"]
    operm = np.zeros(63, int)
    osign = np.zeros(63)
    for e in m["obs_map"]:
        operm[e["index"]], osign[e["index"]] = e["mirror_index"], e["sign"]
    return perm, sign, operm, osign


def s1(obslog, judge, P, mp):
    perm, sign, operm, osign = mp
    S = P["criteria"]["S1"]
    z = np.load(obslog)
    full_ep = np.array([e["reached_horizon"] for e in judge["panel"]["episodes_detail"]])
    w = full_ep[z["episode"]] & (z["t"] >= S["window_start"]) & (z["t"] <= S["window_end"])
    if full_ep.sum() < S["min_full_horizon_episodes"]:
        return {"pass": False, "groups": {}, "channels": {}}
    deg = np.asarray(S["deg_per_unit_by_actuator"], float)
    mu = z["mu"][w].astype(np.float64)
    mm = z["mu_m"][w].astype(np.float64)
    e = np.abs(mu - sign[None, :] * mm[:, perm]) * 0.5 * deg[None, :]
    out = {"groups": {}, "channels": {}}
    for g, spec in S["groups"].items():
        v = e[:, spec["actuators"][0]]
        for i in spec["actuators"][1:]:
            v = np.maximum(v, e[:, i])
        out["groups"][g] = [float(np.median(v)), float(np.quantile(v, 0.9))]
    if "mu_bar" in z.files:
        mb = z["mu_bar"][w].astype(np.float64)
    else:
        import pickle

        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        pol = PPO.load(judge["model"], device="cpu").policy
        vn = pickle.load(open(judge["vecnorm"], "rb"))
        o = z["obs"][w].astype(np.float64)
        ob = 0.5 * (o + osign[None, :] * o[:, operm])
        xn = np.clip((ob - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
        with torch.no_grad():
            mb = pol.get_distribution(torch.as_tensor(xn, dtype=torch.float32)).distribution.mean.numpy().astype(np.float64)
    a = np.clip(mb, -1.0, 1.0)
    odd = 0.5 * (a - sign[None, :] * a[:, perm])
    for c, spec in S["channels"].items():
        out["channels"][c] = abs(float(odd[:, spec["odd_index"]].mean())) * spec["deg_per_unit"]
    out["pass"] = all(v[0] <= S["groups"][g]["bound_deg"] and v[1] <= S["k90"] * S["groups"][g]["bound_deg"] for g, v in out["groups"].items()) and \
        all(v <= S["channels"][c]["bound_deg"] for c, v in out["channels"].items())
    return out


def recount_arm(key, name, P, L, runs, mp, bars, with_sym):
    seeds, cks, c = P["seeds"], P["checkpoints"], P["criteria"]
    rec = {"cells": {}, "h3b": {}, "imputed": {}, "validity": {}}
    for s in seeds:
        run = L["run"].format(runs=runs, arm=name, seed=s)
        v = jload(L["validity"].format(run=run, arm=name, seed=s, runs=runs)) or {}
        rec["validity"][s] = v
        imp = set(v.get("imputed_checkpoints", []))
        rec["imputed"][s] = imp
        rec["cells"][s] = {}
        for lab in LABS:
            j = jload(L["judge"].format(run=run, arm=name, seed=s, runs=runs, steps=cks[lab]))
            rec["cells"][s][lab] = arrays(j, P) if (j is not None and lab not in imp) else None
        Z = series([p.format(run=run, arm=name, seed=s, runs=runs) for p in L["gate_progress"]])
        rec["h3b"][s] = None
        rec["Z"] = rec.get("Z", {})
        rec["Z"][s] = Z
        if Z is not None and "6M" not in imp and rec["cells"][s]["6M"] is not None:
            t = Z["timesteps"].astype(np.int64)
            m = (t > cks["3M"]) & (t <= cks["6M"])
            x = t[m].astype(np.float64) / 1e6
            gr, gl = Z["r_sole_cop_fore_aft_gate"][m].astype(np.float64), Z["l_sole_cop_fore_aft_gate"][m].astype(np.float64)
            rec["h3b"][s] = {"r": pred_ci(x, gr, P), "l": pred_ci(x, gl, P), "even": pred_ci(x, 0.5 * (gr + gl), P), "n": int(m.sum()),
                             "complete": bool(np.array_equal(t[m], np.arange(cks["3M"] + c["H3b"]["eval_every"], cks["6M"] + 1, c["H3b"]["eval_every"])))}
    cell = rec["cells"]
    c3 = [cell[s]["3M"] for s in seeds]
    ok3 = all(x is not None for x in c3)
    I = {"I1": ok3 and sum(x["roll_rail"] for x in c3) <= c["I1"]["pooled_max"], "I2": ok3 and sum(x["falls"] for x in c3) <= c["I2"]["pooled_max"],
         "I3": sum(1 for x in c3 if x is None or x["cop_any"] > c["I3"]["per_seed_bar"]) <= c["I3"]["max_seeds_over"]}
    rec["interim"] = I
    reached = all(cell[s]["6M"] is not None or "6M" in rec["imputed"][s] for s in seeds) and any(cell[s]["6M"] is not None for s in seeds)
    rec["reached_6M"] = bool(reached)
    H = None
    if reached:
        c6 = [cell[s]["6M"] for s in seeds]
        allc6 = all(x is not None for x in c6)
        H = {"H1_per_seed": {s: bool(cell[s]["6M"] is not None and cell[s]["6M"]["gate_pass"]) for s in seeds}}
        H["H1"] = sum(H["H1_per_seed"].values()) >= c["H1"]["min_seeds"]
        H["H2"] = allc6 and sum(x["roll_rail"] for x in c6) <= c["H2"]["pooled_max"] and max(x["roll_rail"] for x in c6) <= c["H2"]["per_seed_max"]
        H["H3a"] = all(cell[s][lab] is not None and cell[s][lab]["cop_any"] <= c["H3a"]["per_seed_max"] for s in seeds for lab in ("4M", "5M", "6M"))
        H["H3b"] = all(rec["h3b"][s] is not None and rec["h3b"][s]["complete"] and rec["h3b"][s]["r"][0] <= c["H3b"]["bar"] and rec["h3b"][s]["l"][0] <= c["H3b"]["bar"] for s in seeds)
        H["H3"] = H["H3a"] and H["H3b"]
        H["H4"] = allc6 and all(x["joint_falls"] is not None and x["own_falls"] is not None for x in c6) and \
            sum(x["joint_falls"] for x in c6) <= c["H4"]["joint_pooled_max"] and sum(x["own_falls"] for x in c6) <= c["H4"]["own_pooled_max"]
        rows = []
        have = 0
        focus = np.array(sorted(c["H5"]["focus_spawns"]))
        for s in seeds:
            run = L["run"].format(runs=runs, arm=name, seed=s)
            m = jload(L["roll_diag_measure"].format(run=run, arm=name, seed=s, runs=runs, steps=cks["6M"]))
            if m is None or "6M" in rec["imputed"][s]:
                continue
            have += 1
            seed_ = np.array([r["seed"] for r in m["h5_rows"]])
            foc = np.array([bool(r["focus"]) for r in m["h5_rows"]]) & np.isin(seed_, focus)
            full = np.array([bool(r["base_full"]) for r in m["h5_rows"]])
            pa = np.array([r["policy_added"] if r["policy_added"] is not None else np.nan for r in m["h5_rows"]], float)
            sel = foc & full & np.isfinite(pa)
            rows.append(np.abs(pa[sel]))
        H["H5_median"] = float(np.median(np.concatenate(rows))) if (have == len(seeds) and sum(len(r) for r in rows)) else None
        H["H5"] = H["H5_median"] is not None and H["H5_median"] <= c["H5"]["bar_deg"]
        H["H6"] = allc6 and sum(x["falls"] for x in c6) <= c["H6"]["pooled_max"]
    rec["H"] = H
    fam = set()
    for k, f in (("I1", "ROLL"), ("I2", "ROBUSTNESS"), ("I3", "COP")):
        if not I[k]:
            fam.add(f)
    if H:
        for k, f in (("H2", "ROLL"), ("H5", "ROLL"), ("H3", "COP"), ("H4", "ROBUSTNESS"), ("H6", "ROBUSTNESS")):
            if not H[k]:
                fam.add(f)
        if not H["H1"]:
            tally = {"ROLL": 0, "COP": 0, "ROBUSTNESS": 0}
            for s in seeds:
                x = cell[s]["6M"]
                if x is not None and not x["gate_pass"]:
                    for kk, v in x["unclean_fam"].items():
                        tally[kk] += v
            top = max(tally.values())
            fam |= {kk for kk, v in tally.items() if v == top} if top > 0 else {"ROBUSTNESS"}
    if any(rec["validity"][s].get("method_failure") for s in seeds):
        fam.add("ROBUSTNESS")
    rec["families_failed"] = sorted(fam)
    rec["exploratory"] = any(bool(rec["validity"][s].get("exploratory")) for s in seeds)
    # classes and fa evidence
    rec["classes"] = {}
    for s in seeds:
        rec["classes"][s] = {}
        for lab in LABS:
            x = cell[s][lab]
            if x is None or x["median_even_g"] is None:
                rec["classes"][s][lab] = None
                continue
            e8 = last8(rec["Z"][s], cks[lab]) if rec["Z"][s] is not None else (None, None)
            rec["classes"][s][lab] = D.classify_cop(x["median_even_g"], x["pad_bias"], e8[0], e8[1], P)["class"]
    # symmetry
    rec["S1"] = rec["S2"] = None
    if with_sym:
        s1d, flips, dclean, okall = {}, 0, 0, True
        for s in seeds:
            run = L["run"].format(runs=runs, arm=name, seed=s)
            jr = jload(L["judge_sym"].format(run=run, arm=name, seed=s, runs=runs, steps=cks["6M"], mode="raw"))
            js = jload(L["judge_sym"].format(run=run, arm=name, seed=s, runs=runs, steps=cks["6M"], mode="sym"))
            ob = L["obslog"].format(run=run, arm=name, seed=s, runs=runs, steps=cks["6M"], mode="raw")
            if jr is None or js is None or not os.path.exists(ob):
                okall = False
                break
            s1d[s] = s1(ob, jr, P, mp)
            cr = np.array([e["clean"] for e in sorted(jr["panel"]["episodes_detail"], key=lambda e: e["seed"])], bool)
            cs = np.array([e["clean"] for e in sorted(js["panel"]["episodes_detail"], key=lambda e: e["seed"])], bool)
            flips += int((cr != cs).sum())
            dclean += int(cr.sum()) - int(cs.sum())
        if okall:
            rec["S1_detail"] = s1d
            rec["S1"] = all(v["pass"] for v in s1d.values())
            rec["S2_detail"] = {"flips": flips, "delta_clean": dclean}
            rec["S2"] = flips <= bars["F_S2"] and abs(dclean) <= bars["D_S2"]
    return rec


def to_decide_record(key, rc, P, comparator):
    """recount's own per-arm record in decide_final's schema."""
    seeds = P["seeds"]
    per = {}
    for s in seeds:
        per[str(s)] = {}
        for lab in LABS:
            x = rc["cells"][s][lab]
            per[str(s)][lab] = {"observed": False} if x is None else {"observed": True, "cop_fh": x["cop_fh"], "class": rc["classes"][s][lab]}
    evci = {str(s): bool(rc["h3b"][s] is not None and rc["h3b"][s]["complete"] and rc["h3b"][s]["even"][1] > P["criteria"]["H3b"]["bar"]) for s in seeds}
    H = None if rc["H"] is None else {k: bool(rc["H"][k]) for k in ("H1", "H2", "H3", "H3a", "H3b", "H4", "H5", "H6")}
    return {"arm": key, "seeds": seeds, "comparator": comparator, "interim": rc["interim"], "reached_6M": rc["reached_6M"], "H": H,
            "S1": rc["S1"], "S2": rc["S2"], "families_failed": rc["families_failed"], "exploratory": rc["exploratory"],
            "seeds_imputed": {str(s): sorted(rc["imputed"][s]) for s in seeds},
            "cop_evidence": {"per_seed": per, "h3b_even_ci_fail": evci, "class_6M": {str(s): rc["classes"][s]["6M"] for s in seeds}}}


def compare(ver, rc, P):
    """every compared quantity: (name, verify value, recount value, agrees)."""
    rows = []
    seeds = P["seeds"]

    def add(n, a, b, tol=0.0):
        if isinstance(a, float) or isinstance(b, float):
            ok = (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= max(tol, 1e-9))
        else:
            ok = a == b
        rows.append([n, a, b, bool(ok)])
    for k in ("I1", "I2", "I3"):
        add(k, bool(ver["interim"][k]), bool(rc["interim"][k]))
    add("reached_6M", bool(ver["reached_6M"]), bool(rc["reached_6M"]))
    add("exploratory", bool(ver.get("exploratory")), bool(rc["exploratory"]))
    add("families_failed", sorted(ver["families_failed"]), rc["families_failed"])
    for s in seeds:
        for lab in LABS:
            vc = ver["_cells"][str(s)].get(lab)
            rcc = rc["cells"][s][lab]
            imp = lab in ver["seeds_imputed"].get(str(s), [])
            if vc is None or rcc is None or imp:
                add(f"s{s} {lab} observed", vc is not None and not imp, rcc is not None)
                continue
            for q in ("falls", "roll_rail", "cop_any", "cop_fh", "n_clean"):
                add(f"s{s} {lab} {q}", vc[q], rcc[q])
            add(f"s{s} {lab} median_even_g", vc["median_even_g"], rcc["median_even_g"])
            add(f"s{s} {lab} pad_bias", vc["pad_bias"], rcc["pad_bias"])
            add(f"s{s} {lab} class", ver["cop_evidence"]["per_seed"][str(s)][lab].get("class"), rc["classes"][s][lab])
            add(f"s{s} {lab} lcb", vc["lcb"], rcc["lcb"], 1e-9)
            for g, v in vc["gate_components"].items():
                add(f"s{s} {lab} gate {g}", bool(v), bool(rcc["gate_components"][g]))
    if ver.get("H"):
        for k in ("H1", "H2", "H3a", "H3b", "H3", "H4", "H5", "H6"):
            add(k, bool(ver["H"][k]), bool(rc["H"][k]))
        for s in seeds:
            add(f"s{s} H1 per seed", bool(ver["H_detail"]["H1_per_seed"].get(str(s), ver["H_detail"]["H1_per_seed"].get(s))), rc["H"]["H1_per_seed"][s])
        h5v = ver["H_detail"]["H5"].get("pooled_median_deg")
        add("H5 pooled median", h5v, rc["H"]["H5_median"], 1e-9)
        for s in seeds:
            hv = ver["H_detail"]["H3b"].get(s) or ver["H_detail"]["H3b"].get(str(s))
            if hv and rc["h3b"][s]:
                for f in ("r", "l", "even"):
                    add(f"s{s} H3b {f} proj", hv[f]["proj"], rc["h3b"][s][f][0], 1e-9)
                    add(f"s{s} H3b {f} ci90_lo", hv[f]["ci90_lo"], rc["h3b"][s][f][1], 1e-9)
    for s in seeds:
        add(f"s{s} h3b_even_ci_fail", bool(ver["cop_evidence"]["h3b_even_ci_fail"].get(str(s))),
            bool(rc["h3b"][s] is not None and rc["h3b"][s]["complete"] and rc["h3b"][s]["even"][1] > P["criteria"]["H3b"]["bar"]))
    if rc["S1"] is not None or ver.get("S1") is not None:
        add("S1", ver.get("S1"), rc["S1"])
        add("S2", ver.get("S2"), rc["S2"])
        if ver.get("S2_detail") and rc.get("S2_detail"):
            add("S2 flips", ver["S2_detail"]["flips"], rc["S2_detail"]["flips"])
            add("S2 delta_clean", ver["S2_detail"]["delta_clean"], rc["S2_detail"]["delta_clean"])
        if ver.get("S1_detail") and rc.get("S1_detail"):
            for s in seeds:
                vd = ver["S1_detail"].get(str(s)) or ver["S1_detail"].get(s)
                r = rc["S1_detail"][s]
                for g, (med, p90) in r["groups"].items():
                    add(f"s{s} S1a {g} median", vd["groups"][g]["median_deg"], med, 1e-9)
                    add(f"s{s} S1a {g} p90", vd["groups"][g]["p90_deg"], p90, 1e-9)
                for c_, val in r["channels"].items():
                    add(f"s{s} S1b {c_}", vd["channels"][c_]["abs_mean_deg"], val, 1e-6)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--params-sha256", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--layout")
    ap.add_argument("--s2-bars")
    ap.add_argument("--s2-bars-sha256")
    ap.add_argument("--arms", default=None, help="comma list of arm keys (default: all in the verify output)")
    ap.add_argument("--with-symmetry", action="store_true")
    a = ap.parse_args()
    h = hashlib.sha256(open(a.params, "rb").read()).hexdigest()
    assert h == a.params_sha256, "frozen_params sha256 mismatch"
    P = json.load(open(a.params))
    L = json.load(open(a.layout)) if a.layout else P["layout"]
    V = json.load(open(a.verify))
    if V.get("INVALID-infrastructure"):
        raise SystemExit("verify reported INVALID-infrastructure: nothing to recount")
    if a.s2_bars:
        assert a.s2_bars_sha256 and hashlib.sha256(open(a.s2_bars, "rb").read()).hexdigest() == a.s2_bars_sha256, "S2 bars sha256 mismatch"
        bars = json.load(open(a.s2_bars))
    else:
        bars = P["criteria"]["S2"]["fallback_bars"]   # validation on older screens only (no J2 verdict file is recounted without --s2-bars)
        if V.get("mode") == "J2 verdict":
            raise SystemExit("a J2 verdict is recounted with --s2-bars and --s2-bars-sha256")
    mp = mirror(P)
    keys = a.arms.split(",") if a.arms else list(V["arms"])
    allrows, own = [], {}
    for key in keys:
        name = V["arms"][key]["run_name"]
        with_sym = a.with_symmetry and V["arms"][key].get("S1_detail") is not None
        rc = recount_arm(key, name, P, L, a.runs, mp, bars, with_sym)
        own[key] = rc
        rows = compare(V["arms"][key], rc, P)
        allrows += [[f"{key} {r[0]}"] + r[1:] for r in rows]
    if V.get("decision") and set(keys) == {"L", "C"}:
        compL = (not D.interim({"interim": own["L"]["interim"]})) and own["L"]["reached_6M"]
        recs = {k: to_decide_record(k, own[k], P, comparator=(k == "L" and compL)) for k in ("L", "C")}
        o = D.outcome(recs["L"], recs["C"], P)
        vd = V["decision"]
        for k in ("outcome", "pr_arm", "arms_12M", "labels", "L_6M_families"):
            allrows.append([f"decision {k}", vd.get(k), o.get(k), vd.get(k) == o.get(k)])
        allrows.append(["decision fa_needed", vd["fa_needed"]["fa_needed"], o["fa_needed"]["fa_needed"], vd["fa_needed"]["fa_needed"] == o["fa_needed"]["fa_needed"]])
        allrows.append(["decision fa evidence seeds", vd["fa_needed"]["evidence_seeds"], o["fa_needed"]["evidence_seeds"],
                        vd["fa_needed"]["evidence_seeds"] == o["fa_needed"]["evidence_seeds"]])
    bad = [r for r in allrows if not r[3]]
    print(json.dumps({"compared": len(allrows), "disagreements": bad}, indent=1, default=str))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
