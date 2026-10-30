"""EARLY-BIAS PROBE: aggregate the per-rollout JSONs of eb_update.py across seeds (no simulation).

Every per-epoch prediction is averaged over the rollouts (seeds) of a checkpoint; uncertainty is the SE across
rollouts (n = number of seeds), the honest unit, since rows within a rollout are not independent.
Per-update scaling uses the measured epochs per update of the following 1,048,576-step interval (eb_actual.json).
Writes eb_aggregate.txt and eb_aggregate.json.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
CKPTS = [("s42", 1048576), ("s42", 2097152), ("s42", 3145728), ("s42", 3670016),
         ("s80", 1048576), ("s80", 2097152), ("s80", 3145728),
         ("s60", 5242880), ("s60", 6291456), ("s60", 7340032), ("s60", 8388608)]
TARGETS = ("bank_mean", "statue_settled", "rollout_mean")
GROUPS = ("tail3", "tail12", "legs", "neckhead")


def ms(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return (math.nan, math.nan, 0)
    return (float(x.mean()), float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan, n)


def fmt(m, s, scale=1.0, w=9, p=5):
    return f"{m * scale:+{w}.{p}f} ± {s * scale:.{p}f}"


def epochs_per_update(actual, run, N):
    for k, v in actual["epochs"].items():
        r, a, b = k.split("_")
        if r == run and int(a) == N:
            return v["epochs_per_update"]
    # the interval ending at N (for the last local checkpoint before a gap)
    for k, v in actual["epochs"].items():
        r, a, b = k.split("_")
        if r == run and int(b) == N:
            return v["epochs_per_update"]
    return math.nan


def train_episodes(run, N, half=131072):
    p = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "train_episodes.csv.copy1"
    R, L, falls = [], [], 0
    with open(p) as fh:
        for row in csv.DictReader(fh):
            t = int(row["timesteps"])
            if N - half <= t <= N + half:
                R.append(float(row["return"]))
                L.append(int(row["length"]))
                falls += int(row["cause"] not in ("truncated", "time_limit", ""))
    return {"n": len(R), "return_mean": float(np.mean(R)) if R else None, "length_mean": float(np.mean(L)) if L else None,
            "falls": falls, "causes_note": "cause != truncated/time_limit counted as a fall"}


def main():
    actual = json.loads((HERE / "eb_actual.json").read_text())
    probe_rows = {}
    import sys
    sys.path.insert(0, str(HERE))
    import eb_common as E  # noqa: E402
    out = {}
    L = []
    P = L.append
    P("EARLY-BIAS PROBE: aggregate over rollouts (per-epoch first-order changes of the deterministic tail_3 mean mu14)")
    P("units: action units per epoch (one pass of 64 mini-batches at the checkpoint's parameters), mean ± SE over rollouts")
    P("")
    for run, N in CKPTS:
        fs = sorted(glob.glob(str(HERE / "eb_out2" / f"{run}_{N}_*.json")))
        if not fs:
            continue
        D = [json.loads(Path(f).read_text()) for f in fs]
        D = [d for d in D if "terms_gae_sb" in d["split_per_epoch"]["adam_rollout_mean"]]  # the final format only
        n = len(D)
        if n < 2:
            continue
        if run not in probe_rows:
            probe_rows[run] = E.probe(run)
        pr = probe_rows[run]
        i0 = int(np.flatnonzero(pr["update_timesteps"] == N)[0])
        epu = epochs_per_update(actual, run, N)
        key = f"{run}_{N}"
        o = {"n": n, "seeds": [d["seed"] for d in D], "epochs_per_update": epu}
        P("=" * 118)
        P(f"{run} at {N:,} steps: n = {n} rollouts (seeds {min(o['seeds'])}-{max(o['seeds'])}); lr {D[0]['lr']:.3e}; ent_coef "
          f"{D[0]['ent_coef']:.5f}; epochs/update in the next interval {epu:.2f}; ret scale {D[0]['ret_scale']:.2f}")
        # ---- checks
        ck = defaultdict(list)
        for d in D:
            for k, v in d["checks"].items():
                if isinstance(v, list):
                    ck[k].append(abs(v[0] - v[1]))
                elif isinstance(v, bool):
                    ck[k].append(0.0 if v else 1.0)
                else:
                    ck[k].append(float(v))
        P("  checks (max over rollouts): " + "; ".join(f"{k} {max(v):.2g}" for k, v in ck.items()))
        o["checks_max"] = {k: max(v) for k, v in ck.items()}
        # ---- validation of the rollout against the run's records
        ro = defaultdict(list)
        for d in D:
            for k in ("mu14_mean", "mu14_q50", "mu14_q95", "mu14_beyond_high", "a14_clip_high", "a14_sat", "rew_raw_mean"):
                ro[k].append(d["rollout"][k])
            ro["ep_return"].extend([e[0] for e in d["rollout"]["episodes"]])
            ro["ep_len"].extend([e[1] for e in d["rollout"]["episodes"]])
        te = train_episodes(run, N)
        o["validation"] = {"rollout_mu14_mean": ms(ro["mu14_mean"]), "probe_rollout_mu_mean": float(pr["rollout_mu_mean"][i0, 14]),
                           "rollout_mu14_q50": ms(ro["mu14_q50"]), "probe_rollout_mu_q50": float(pr["rollout_mu_q50"][i0, 14]),
                           "rollout_mu14_q95": ms(ro["mu14_q95"]), "probe_rollout_mu_q95": float(pr["rollout_mu_q95"][i0, 14]),
                           "a14_clip_high": ms(ro["a14_clip_high"]), "probe_clip_high": float(pr["rollout_action_clip_high_share"][i0, 14]),
                           "a14_sat": ms(ro["a14_sat"]), "probe_sat": float(pr["rollout_action_saturated_share"][i0, 14]),
                           "episodes": len(ro["ep_return"]), "ep_return_mean": float(np.mean(ro["ep_return"])) if ro["ep_return"] else None,
                           "ep_len_mean": float(np.mean(ro["ep_len"])) if ro["ep_len"] else None,
                           "ep_falls": int(sum(1 for x in ro["ep_len"] if x < 1000)), "train_episodes_csv": te}
        v = o["validation"]
        P(f"  VALIDATION rollout mu14 mean {fmt(*v['rollout_mu14_mean'][:2], p=4)} vs probe row {v['probe_rollout_mu_mean']:+.4f}; "
          f"median {v['rollout_mu14_q50'][0]:+.4f} vs {v['probe_rollout_mu_q50']:+.4f}; q95 {v['rollout_mu14_q95'][0]:+.4f} vs "
          f"{v['probe_rollout_mu_q95']:+.4f}; sampled a14 > 1 {v['a14_clip_high'][0]:.4f} vs {v['probe_clip_high']:.4f}; |a14|>=0.9 "
          f"{v['a14_sat'][0]:.4f} vs {v['probe_sat']:.4f}")
        P(f"  VALIDATION episodes completed {v['episodes']}: return {v['ep_return_mean']}, length {v['ep_len_mean']}, falls "
          f"{v['ep_falls']}  | train_episodes.csv within +-131,072 steps: n {te['n']}, return {te['return_mean']}, length "
          f"{te['length_mean']}, non-truncated {te['falls']}")
        # ---- components per target
        o["components"] = {}
        P("  COMPONENTS (per epoch; Adam = the checkpoint's exp_avg_sq preconditioning; plain = -lr x clipped gradient):")
        for tname in TARGETS:
            row = []
            for pre in ("adam", "plain"):
                for c in ("full", "pg", "kl", "ent", "vf"):
                    x = [d["target_delta_per_epoch"][f"{pre}_{tname}_{c}"] for d in D]
                    o["components"][f"{pre}_{tname}_{c}"] = ms(x)
                    if c in ("full", "pg", "kl") or pre == "adam":
                        if pre == "adam":
                            row.append(f"adam.{c} {fmt(*ms(x)[:2])}")
                        else:
                            row.append(f"plain.{c} {fmt(*ms(x)[:2], scale=1e6, p=3)} e-6")
            mom = [d["target_delta_per_epoch"][f"adam_{tname}_momentum1"] for d in D]
            o["components"][f"adam_{tname}_momentum1"] = ms(mom)
            P(f"   {tname:15s} " + " | ".join(row[:5]))
            P(f"   {'':15s} " + " | ".join(row[5:]) + f" | Adam in-flight momentum, one step {mom[0]:+.6f} (identical across seeds)")
        # ---- noise group
        P("  POLICY TERM BY WHOSE SAMPLED NOISE CARRIES IT (Adam, per epoch): " + "")
        o["noise_groups"] = {}
        for tname in TARGETS:
            parts = []
            for g in GROUPS:
                x = [d["split_per_epoch"][f"adam_{tname}"]["pg_by_noise_group"][g] for d in D]
                o["noise_groups"][f"{tname}_{g}"] = ms(x)
                parts.append(f"{g} {fmt(*ms(x)[:2])}")
            P(f"   {tname:15s} " + " | ".join(parts))
        # ---- reward terms
        keys = D[0]["reward_keys"]
        o["terms"] = {}
        P("  POLICY TERM BY REWARD TERM (Adam, per epoch).  t = mean / SE over rollouts.  Columns:")
        P("    GAE-sb  = the update's own advantage split exactly, A = sum_k GAE(r_k, V=0) + critic bootstraps - V(s), each part")
        P("              minus a cross-fitted state-only baseline (same expected pull, lower variance; the parts sum to the")
        P("              actual pull in expectation).  CRITIC_BOOTSTRAP = the critic's bootstrapped values (beyond the")
        P("              gamma*lambda horizon), not attributable to a reward term.  PRIMARY.")
        P("    MC-b    = lambda 1 per-term discounted returns (gamma 0.98) minus the same kind of baseline: the Monte Carlo")
        P("              policy gradient split by term, with no critic (rows >= 200 steps from a cut).  CHECK.")
        P("    GAE-raw = the exact split without baselines (high variance).  Sorted by |t| of GAE-sb.")
        for tname in ("rollout_mean", "bank_mean", "statue_settled"):
            sp = [d["split_per_epoch"][f"adam_{tname}"] for d in D]
            tot = ms([x["pg"] for x in sp])
            sbt = ms([sum(x["terms_gae_sb"].values()) for x in sp])
            mct = ms([sum(x["terms_mc_base"].values()) for x in sp])
            o["terms"][f"{tname}__total"] = {"pg": tot, "gae_sb_sum": sbt, "mc_base_sum": mct}
            P(f"   target {tname}: policy term {fmt(*tot[:2])}; GAE-sb parts' sum {fmt(*sbt[:2])}; MC-b sum {fmt(*mct[:2])}")
            rows = []
            for k in keys + ["CRITIC_BOOTSTRAP"]:
                g0 = ms([x["terms_gae_sb"][k] for x in sp])
                g0t = ms([x["terms_gae_sb_by_group"]["tail3"][k] for x in sp])
                g1 = ms([x["terms_gae"][k] for x in sp]) if k in sp[0]["terms_gae"] else ms([x["critic_gae"] for x in sp])
                g3 = ms([x["terms_mc_base"][k] for x in sp]) if k in sp[0]["terms_mc_base"] else (math.nan, math.nan, 0)
                o["terms"][f"{tname}_{k}"] = {"gae_sb": g0, "gae_sb_tail3_noise": g0t, "gae_raw": g1, "mc_base": g3}
                rows.append((k, g0, g0t, g3, g1))

            def tval(g):
                return g[0] / g[1] if g[1] and g[1] > 0 else 0.0
            rows.sort(key=lambda r: -abs(tval(r[1])))
            for k, g0, g0t, g3, g1 in rows:
                if g0[0] == 0 and g1[0] == 0:
                    continue
                nm = k[7:] if k.startswith("reward_") else k
                P(f"     {nm:22s} GAE-sb {g0[0]:+.6f} ± {g0[1]:.6f} (t {tval(g0):+5.1f}) [via tail_3 noise {g0t[0]:+.6f} ± {g0t[1]:.6f}]"
                  f" | MC-b {g3[0]:+.6f} ± {g3[1]:.6f} (t {tval(g3):+5.1f}) | GAE-raw {g1[0]:+.6f} (t {tval(g1):+4.1f})")
        # ---- regions
        o["regions"] = {}
        P("  BY STATE REGION (Adam, per epoch): policy term, mirror term; top reward terms (GAE-sb) by |t|")
        for tname in ("rollout_mean", "bank_mean"):
            P(f"   target {tname}")
            for rn in ("tail3_deg", "pitch_deg", "feet_loaded", "ep_step"):
                inner = D[0]["split_per_epoch"][f"adam_{tname}"]["regions"][rn]["inner_edges"]
                edges = ["-inf"] + [f"{e:g}" for e in inner] + ["inf"]
                for j in range(len(inner) + 1):
                    b = [d["split_per_epoch"][f"adam_{tname}"]["regions"][rn]["bins"][str(j)] for d in D]
                    pg, kl = ms([x["pg"] for x in b]), ms([x["kl"] for x in b])
                    tt = []
                    for k in keys + ["CRITIC_BOOTSTRAP"]:
                        g = ms([x["terms_gae_sb"][k] for x in b])
                        if g[1] and g[1] > 0:
                            tt.append((g[0] / g[1], k, g))
                    tt.sort(key=lambda z: -abs(z[0]))
                    o["regions"][f"{tname}_{rn}_{j}"] = {"n_rows_mean": float(np.mean([x["n"] for x in b])), "pg": pg, "kl": kl,
                                                         "terms_gae_sb": {k: g for _, k, g in tt}}
                    top = ", ".join(f"{k[7:] if k.startswith('reward_') else k} {g[0]:+.5f} (t {t:+.1f})" for t, k, g in tt[:3])
                    P(f"     {rn:11s} [{edges[j]:>5s},{edges[j + 1]:>5s}) rows {np.mean([x['n'] for x in b]):7.1f}  pg {fmt(*pg[:2])} "
                      f"(t {pg[0] / pg[1] if pg[1] else 0:+4.1f}) kl {fmt(*kl[:2])} | {top}")
        # ---- like-for-like GAE vs MC on the same rows, and the critic's error against realised returns by tail angle
        DC = [d for d in D if "critic_check" in d]
        if len(DC) >= 2:
            o["critic_test"] = {"n": len(DC)}
            P(f"  CRITIC TEST (n = {len(DC)} rollouts with the added fields; rows >= 200 steps from a cut):")
            for tname in ("rollout_mean", "bank_mean", "statue_settled"):
                sp = [d["split_per_epoch"][f"adam_{tname}"] for d in DC]
                g = ms([x["pg_on_mc_rows"] for x in sp])
                cb = ms([x["gae_sb_on_mc_rows"]["CRITIC_BOOTSTRAP"] for x in sp])
                rw = ms([sum(v for k, v in x["gae_sb_on_mc_rows"].items() if k != "CRITIC_BOOTSTRAP") for x in sp])
                mc = ms([sum(x["terms_mc_base"].values()) for x in sp])
                dlt = ms([x["pg_on_mc_rows"] - sum(x["terms_mc_base"].values()) for x in sp])
                o["critic_test"][tname] = {"gae_pg_same_rows": g, "gae_sb_critic_same_rows": cb, "gae_sb_rewards_same_rows": rw,
                                           "mc_b_sum": mc, "gae_minus_mc": dlt}
                P(f"   {tname:15s} GAE pull on these rows {fmt(*g[:2])} = rewards (gamma*lambda horizon) {fmt(*rw[:2])} + critic "
                  f"bootstrap {fmt(*cb[:2])} | critic-free MC (gamma 0.98) {fmt(*mc[:2])} | GAE - MC {fmt(*dlt[:2])} (t "
                  f"{dlt[0] / dlt[1] if dlt[1] else 0:+.1f})")
            sl = ms([d["critic_check"]["resid_vs_tail_deg_slope"] for d in DC])
            rm_ = ms([d["critic_check"]["resid_mean"] for d in DC])
            o["critic_test"]["resid_slope_per_deg"] = sl
            bins = {}
            for j in range(6):
                vals = [d["critic_check"]["resid_by_tail_bin"][str(j)] for d in DC]
                xs = [v[0] for v in vals if v[0] is not None]
                ns = sum(v[1] for v in vals)
                bins[str(j)] = (ms(xs), ns)
            o["critic_test"]["resid_by_tail_bin"] = {k: [v[0], v[1]] for k, v in bins.items()}
            P(f"   critic error (realised discounted return - V(s), normalised units): mean {fmt(*rm_[:2])}; slope on the tail_3 "
              f"angle {fmt(*sl[:2])} per degree (negative = the critic over-values a higher tail)")
            lab = ["<-1", "-1..0", "0..1", "1..2", "2..3", ">3"]
            P("   by tail angle (deg): " + "; ".join(f"{lab[int(k)]}: {v[0][0]:+.4f}±{v[0][1]:.4f} (rows {v[1]})"
                                                   for k, v in bins.items() if v[0][2] >= 2))
        # ---- r9-style bias pull
        g = [d["mb"]["G_pg_b14_mean"] for d in D]
        grms = [d["mb"]["G_pg_b14_rms"] for d in D]
        gkl = [d["mb"]["G_kl_b14_mean"] for d in D]
        o["r9_style"] = {"G_pg_b14_mean": ms(g), "G_pg_b14_rms_mean": float(np.mean(grms)), "G_kl_b14_mean": ms(gkl),
                         "neg_rollouts": int(sum(1 for x in g if x < 0))}
        P(f"  R9-STYLE pull on action_net.bias[14] (d clipped surrogate / d bias, per mini-batch, first epoch): mean "
          f"{fmt(*ms(g)[:2], p=4)} (SE over rollouts; {o['r9_style']['neg_rollouts']}/{n} rollouts negative), RMS "
          f"{np.mean(grms):.3f}; mirror term {fmt(*ms(gkl)[:2], p=4)}.  Negative = descending raises the bias.")
        # ---- per-update prediction against the actual
        o["per_update"] = {}
        for W in (32, 64):
            a = actual["windows"].get(f"{run}_{N}_W{W}")
            if a is None:
                continue
            ws = np.array(a["weights_share_per_update"])
            act = np.array(a["actual_per_update"])
            pred_full = np.mean([np.array(d["bank_delta_per_epoch"]["adam_full"]) for d in D], axis=0)
            pred_se = np.std([np.array(d["bank_delta_per_epoch"]["adam_full"]) for d in D], axis=0, ddof=1) / math.sqrt(n)
            pb = pred_full[:, 14].mean() * epu
            pbse = math.sqrt(np.var([np.array(d["bank_delta_per_epoch"]["adam_full"])[:, 14].mean() for d in D], ddof=1) / n) * epu
            ps = pred_full[[0, 5], 14].mean() * epu
            psse = math.sqrt(np.var([np.array(d["bank_delta_per_epoch"]["adam_full"])[[0, 5], 14].mean() for d in D], ddof=1) / n) * epu
            rm = ms([d["target_delta_per_epoch"]["adam_rollout_mean_full"] for d in D])
            # 150-output calibration: actual weights share vs predicted mean, all actuators and bank states
            x = pred_full.ravel()
            y = ws.ravel()
            r_all = float(np.corrcoef(x, y)[0, 1])
            slope_all = float((x * y).sum() / (x * x).sum())
            mask = np.ones_like(pred_full, dtype=bool)
            mask[:, 14] = False
            r_ex = float(np.corrcoef(pred_full[mask], ws[mask])[0, 1])
            sign_agree = float(np.mean(np.sign(pred_full[mask]) == np.sign(ws[mask])))
            sd1 = float(np.mean(np.array(a["single_update_sd"])[:, 14]))  # mean over bank states of per-state SD
            bank_mean_sd1 = a.get("bank_mean_single_update_sd")
            o["per_update"][f"W{W}"] = {
                "pred_bank_mean_tail3_per_update": [pb, pbse], "actual_bank_mean_tail3_per_update": float(act[:, 14].mean()),
                "actual_bank_mean_weights_share": float(ws[:, 14].mean()),
                "pred_statue_settled_tail3_per_update": [ps, psse], "actual_statue_settled_weights_share": float(ws[[0, 5], 14].mean()),
                "pred_rollout_mean_tail3_per_update": [rm[0] * epu, rm[1] * epu],
                "actual_probe_rollout_mu14_per_update": a["rollout_mu14_per_update"],
                "calib_r_all150": r_all, "calib_slope_actual_on_pred": slope_all, "calib_r_excl_tail3": r_ex,
                "sign_agree_excl_tail3": sign_agree,
                "pred_noise_sd_bank_mean_per_update": float(np.std([np.array(d["bank_delta_per_epoch"]["adam_full"])[:, 14].mean()
                                                                    for d in D], ddof=1) * epu),
            }
            # noise calibration: effective epochs k = actual single-update SD / predicted single-rollout per-epoch SD
            sd_pe = float(np.std([np.array(d["bank_delta_per_epoch"]["adam_full"])[:, 14].mean() for d in D], ddof=1))
            k_eff = bank_mean_sd1 / sd_pe if bank_mean_sd1 and sd_pe > 0 else math.nan
            pb_e = pred_full[:, 14].mean()
            pb_ese = pbse / epu
            rm_e = rm
            u_k = {"k_eff": k_eff, "actual_bank_mean_single_update_sd": bank_mean_sd1, "pred_bank_mean_per_epoch_sd": sd_pe,
                   "calibrated_bank_mean_per_update": [pb_e * k_eff, pb_ese * k_eff],
                   "calibrated_statue_settled_per_update": [ps / epu * k_eff, psse / epu * k_eff],
                   "calibrated_rollout_mean_per_update": [rm_e[0] * k_eff, rm_e[1] * k_eff]}
            o["per_update"][f"W{W}"].update(u_k)
            u = o["per_update"][f"W{W}"]
            P(f"  PER UPDATE (x {epu:.2f} epochs) vs ACTUAL over N -> N + {W} updates (probe; weights share = actual minus the "
              f"normaliser's share):")
            P(f"    bank mean tail_3: predicted {pb:+.5f} ± {pbse:.5f}; actual {act[:, 14].mean():+.5f}, weights share "
              f"{ws[:, 14].mean():+.5f}")
            P(f"    statue_settled tail_3: predicted {ps:+.5f} ± {psse:.5f}; weights share {ws[[0, 5], 14].mean():+.5f}")
            P(f"    rollout-row mean tail_3 (fixed states): predicted {rm[0] * epu:+.5f} ± {rm[1] * epu:.5f}; probe's training-row "
              f"mean (states move too) {a['rollout_mu14_per_update']:+.5f}")
            P(f"    150 bank outputs: r(pred, actual weights share) {r_all:+.3f}; LS slope actual/pred {slope_all:+.3f}; "
              f"excluding tail_3 r {r_ex:+.3f}, sign agreement {sign_agree:.2f}; one rollout's per-update SD (bank-mean tail_3) "
              f"{u['pred_noise_sd_bank_mean_per_update']:.5f}")
            P(f"    NOISE-CALIBRATED (effective epochs k = actual single-update SD {bank_mean_sd1:.5f} / one rollout's per-epoch "
              f"SD {sd_pe:.5f} = {k_eff:.2f}): bank mean {pb_e * k_eff:+.5f} ± {pb_ese * k_eff:.5f} vs weights share "
              f"{ws[:, 14].mean():+.5f}; statue_settled {ps / epu * k_eff:+.5f} ± {psse / epu * k_eff:.5f} vs "
              f"{ws[[0, 5], 14].mean():+.5f}; rollout rows {rm_e[0] * k_eff:+.5f} ± {rm_e[1] * k_eff:.5f} vs training-row "
              f"{a['rollout_mu14_per_update']:+.5f}")
        out[key] = o
    (HERE / "eb_aggregate.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "eb_aggregate.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
