"""foot/symmetry screen (final design): the ARBITER for INVALID-analysis disputes between verify_final.py and
recount_final.py.  Written and hashed before any screen data exist (frozen_params.json 'scripts').  It recomputes ONE
disputed quantity from the raw per-episode records against the written definition in frozen_params.json (and the
design's criteria.definitions); its printed value is final and logged; nothing else is recomputed and no training is
re-run for it.  Deliberately minimal: plain loops, no shared code with verify_final.py or recount_final.py.

Quantities (arm = run name, e.g. sym_lean_settled; ckpt = 3M..6M):
  count:<arm>:<seed>:<ckpt>:<falls|roll_rail|cop_any|cop_fh|n_clean>
  pads:<arm>:<seed>:<ckpt>          median even_g and |median (g_r - g_l)| (pad bias) over the full-horizon episodes
  noise:<arm>:<seed>                joint-space and own-sigma noise-panel falls at 6M
  h3b:<arm>:<seed>:<r|l|even>       12M projection and its 90% CI lower bound (OLS over the 24 evals in (3M, 6M]); 'even' =
                                    the series (g_r + g_l)/2 (fa_needed rule (ii), r2)
  gate:<arm>:<seed>:<ckpt>          H1's gate components: n, clean, exact one-sided LCB, full-horizon fraction,
                                    hop-or-fall episodes, mean reward vs the reward rails
  h5:<arm>                          pooled median |policy-added roll| (6M roll_diag)
  s1a:<arm>:<seed>:<group>          median and p90 of the per-step S1a value (degrees)
  s2:<arm>                          pooled per-episode clean flips and delta clean (raw vs --symmetrize, 6M)
Usage: arbiter_final.py --params frozen_params.json --params-sha256 <hex> --runs <dir> --q <quantity> [--layout L]
"""
import argparse
import hashlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def quantile(v, q):
    """linear interpolation between order statistics (numpy's default 'linear' method)."""
    v = sorted(v)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--params-sha256")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--layout")
    a = ap.parse_args()
    assert a.params_sha256 is None or hashlib.sha256(open(a.params, "rb").read()).hexdigest() == a.params_sha256
    P = json.load(open(a.params))
    L = json.load(open(a.layout)) if a.layout else P["layout"]
    c = P["criteria"]
    parts = a.q.split(":")
    kind = parts[0]

    def run(arm, seed):
        return L["run"].format(runs=a.runs, arm=arm, seed=seed)

    def judge(arm, seed, ckpt):
        return json.load(open(L["judge"].format(run=run(arm, seed), arm=arm, seed=seed, runs=a.runs, steps=P["checkpoints"][ckpt])))
    if kind == "count":
        arm, seed, ckpt, what = parts[1], int(parts[2]), parts[3], parts[4]
        n = 0
        for e in judge(arm, seed, ckpt)["panel"]["episodes_detail"]:
            full = e["reached_horizon"]
            cop = e.get("max_sole_cop_fore_aft")
            if what == "falls":
                n += 0 if full else 1
            elif what == "roll_rail":
                t, lft = e.get("max_sole_tilt_deg"), e.get("max_sole_corner_lift_m")
                n += 1 if full and ((t is not None and t > c["roll_rail_tilt_deg"]) or (lft is not None and lft > c["roll_rail_corner_lift_m"])) else 0
            elif what == "cop_any":
                n += 1 if (cop is not None and cop > c["cop_bar"]) else 0
            elif what == "cop_fh":
                n += 1 if (full and cop is not None and cop > c["cop_bar"]) else 0
            elif what == "n_clean":
                n += 1 if e["clean"] else 0
        val = n
    elif kind == "pads":
        arm, seed, ckpt = parts[1], int(parts[2]), parts[3]
        ev, gp = [], []
        for e in judge(arm, seed, ckpt)["panel"]["episodes_detail"]:
            g = e.get("pad", {})
            if e["reached_horizon"] and g.get("r_sole_cop_fore_aft_gate") is not None and g.get("l_sole_cop_fore_aft_gate") is not None:
                ev.append((g["r_sole_cop_fore_aft_gate"] + g["l_sole_cop_fore_aft_gate"]) / 2)
                gp.append(g["r_sole_cop_fore_aft_gate"] - g["l_sole_cop_fore_aft_gate"])
        val = {"median_even_g": median(ev), "pad_bias": abs(median(gp))}
    elif kind == "noise":
        arm, seed = parts[1], int(parts[2])
        d = judge(arm, seed, "6M")
        val = {"joint": sum(1 for e in d["noise_matched_control_joint"]["episodes_detail"] if not e["reached_horizon"]),
               "own": sum(1 for e in d["noise_matched_own"]["episodes_detail"] if not e["reached_horizon"])}
    elif kind == "h3b":
        import numpy as np
        from scipy import stats
        arm, seed, pad = parts[1], int(parts[2]), parts[3]
        Z = None
        for p in L["gate_progress"]:
            p = p.format(run=run(arm, seed), arm=arm, seed=seed, runs=a.runs)
            if os.path.exists(p):
                z = np.load(p)
                if Z is None or len(z["timesteps"]) > len(Z[0]):
                    if pad == "even":
                        Z = (list(z["timesteps"]), [(float(a) + float(b)) / 2 for a, b in zip(z["r_sole_cop_fore_aft_gate"], z["l_sole_cop_fore_aft_gate"])])
                    else:
                        Z = (list(z["timesteps"]), list(z[f"{pad}_sole_cop_fore_aft_gate"]))
        lo, hi = P["checkpoints"]["3M"], P["checkpoints"]["6M"]
        pts = [(t / 1e6, float(g)) for t, g in zip(*Z) if lo < t <= hi]
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        b = sum((x - mx) * (y - my) for x, y in pts) / sxx
        a0 = my - b * mx
        s2 = sum((y - a0 - b * x) ** 2 for x, y in pts) / (n - 2)
        x0 = c["H3b"]["x_eval_M"]
        proj = a0 + b * x0
        lo90 = proj - stats.t.ppf(c["H3b"]["t_quantile"], n - 2) * math.sqrt(s2 * (1 / n + (x0 - mx) ** 2 / sxx))
        val = {"n": n, "proj": proj, "ci90_lo": lo90}
    elif kind == "gate":
        from scipy import stats
        arm, seed, ckpt = parts[1], int(parts[2]), parts[3]
        d = judge(arm, seed, ckpt)
        G = P["gate"]
        eps = d["panel"]["episodes_detail"]
        n = len(eps)
        k = sum(1 for e in eps if e["clean"])
        full = sum(1 for e in eps if e["reached_horizon"])
        hk = set(G["hop_or_fall_keys"]) | {"horizon"}
        hop = sum(1 for e in eps if (not e["reached_horizon"]) or any(r.split(":")[0].strip() in hk for r in e["reasons"]))
        lcb = float(stats.beta.ppf(G["lcb_alpha"], k, n - k + 1)) if k > 0 else 0.0
        mr, st = d["panel"]["mean_reward"], d["panel"].get("statue_mean_reward")
        val = {"n": n, "clean": k, "lcb": lcb, "full_fraction": full / n, "hop_or_fall": hop, "mean_reward": mr, "statue_mean_reward": st,
               "pass": n >= G["min_eval_episodes"] and lcb >= G["min_clean_stance_lcb"] and full / n >= G["min_full_horizon_fraction"]
               and hop <= G["max_hop_or_fall_episodes"] and mr >= G["min_avg_reward"] and (st is None or mr >= G["min_avg_reward_statue_ratio"] * st)}
    elif kind == "h5":
        arm = parts[1]
        v = []
        for s in P["seeds"]:
            m = json.load(open(L["roll_diag_measure"].format(run=run(arm, s), arm=arm, seed=s, runs=a.runs, steps=P["checkpoints"]["6M"])))
            v += [abs(r["policy_added"]) for r in m["h5_rows"] if r["focus"] and r["base_full"] and math.isfinite(r["policy_added"])]
        val = {"n": len(v), "median": median(v)}
    elif kind == "s1a":
        import numpy as np
        arm, seed, group = parts[1], int(parts[2]), parts[3]
        S = c["S1"]
        mp = json.load(open(P["files"]["mirror_map"]["path"]))["action_map"]
        perm = {e["index"]: (e["mirror_index"], e["sign"]) for e in mp}
        jr = json.load(open(L["judge_sym"].format(run=run(arm, seed), arm=arm, seed=seed, runs=a.runs, steps=P["checkpoints"]["6M"], mode="raw")))
        z = np.load(L["obslog"].format(run=run(arm, seed), arm=arm, seed=seed, runs=a.runs, steps=P["checkpoints"]["6M"], mode="raw"))
        full = [e["reached_horizon"] for e in jr["panel"]["episodes_detail"]]
        idx = S["groups"][group]["actuators"]
        deg = S["deg_per_unit_by_actuator"]
        ep, tt, mu, mum = z["episode"].tolist(), z["t"].tolist(), z["mu"].tolist(), z["mu_m"].tolist()  # read each array once
        v = []
        for k in range(len(tt)):
            if not full[int(ep[k])] or not (S["window_start"] <= int(tt[k]) <= S["window_end"]):
                continue
            v.append(max(abs(mu[k][i] - perm[i][1] * mum[k][perm[i][0]]) / 2 * deg[i] for i in idx))
        val = {"n": len(v), "median": median(v), "p90": quantile(v, 0.9)}
    elif kind == "s2":
        arm = parts[1]
        flips = dc = 0
        for s in P["seeds"]:
            r = {e["seed"]: e["clean"] for e in json.load(open(L["judge_sym"].format(run=run(arm, s), arm=arm, seed=s, runs=a.runs, steps=P["checkpoints"]["6M"], mode="raw")))["panel"]["episodes_detail"]}
            y = {e["seed"]: e["clean"] for e in json.load(open(L["judge_sym"].format(run=run(arm, s), arm=arm, seed=s, runs=a.runs, steps=P["checkpoints"]["6M"], mode="sym")))["panel"]["episodes_detail"]}
            flips += sum(1 for k in r if r[k] != y[k])
            dc += sum(r.values()) - sum(y.values())
        val = {"flips": flips, "delta_clean": dc}
    else:
        raise SystemExit(f"unknown quantity {a.q}")
    print(json.dumps({"quantity": a.q, "value": val}, default=float))


if __name__ == "__main__":
    main()
