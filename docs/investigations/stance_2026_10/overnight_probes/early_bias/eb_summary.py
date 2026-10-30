"""EARLY-BIAS PROBE: compact table from eb_aggregate.json (no simulation).  Writes eb_summary.txt."""
import json
import math
from pathlib import Path

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
A = json.loads((HERE / "eb_aggregate.json").read_text())
R = json.loads((HERE / "eb_realized.json").read_text()) if (HERE / "eb_realized.json").exists() else {}
L = []
P = L.append


def f(x, p=5):
    m, s = x[0], x[1]
    t = m / s if s and s > 0 else float("nan")
    return f"{m:+.{p}f}±{s:.{p}f} (t{t:+.1f})"


P("Per-epoch first-order change of the deterministic tail_3 mean (Adam-preconditioned, the checkpoint's own exp_avg_sq), mean ± SE over n rollouts")
P("")
for key, o in A.items():
    run, N = key.split("_")
    c = o["components"]
    P(f"{run} {int(N):>9,d}  n={o['n']}")
    for tname in ("rollout_mean", "bank_mean", "statue_settled"):
        P(f"   {tname:15s} full {f(c[f'adam_{tname}_full'])} | policy {f(c[f'adam_{tname}_pg'])} | mirror {f(c[f'adam_{tname}_kl'])} | "
          f"plain(clip) full {c[f'plain_{tname}_full'][0]*1e6:+.3f}e-6±{c[f'plain_{tname}_full'][1]*1e6:.3f}e-6 | in-flight momentum (1 step) "
          f"{c[f'adam_{tname}_momentum1'][0]:+.2e}")
    ng = o["noise_groups"]
    P(f"   policy term by noise carrier (rollout_mean): tail3 {f(ng['rollout_mean_tail3'])} legs {f(ng['rollout_mean_legs'])} "
      f"tail12 {f(ng['rollout_mean_tail12'])} neck/head {f(ng['rollout_mean_neckhead'])}")
    for tname in ("rollout_mean", "bank_mean", "statue_settled"):
        T = {k[len(tname) + 1:]: v for k, v in o["terms"].items() if k.startswith(tname + "_") and not k.endswith("__total")}
        tot = o["terms"][f"{tname}__total"]
        crit = T.get("CRITIC_BOOTSTRAP", {}).get("gae_sb")
        rew = [(k, v["gae_sb"]) for k, v in T.items() if k != "CRITIC_BOOTSTRAP"]
        rew_sum = sum(v[0] for _, v in rew)
        sig = [(k, v) for k, v in rew if v[1] and abs(v[0] / v[1]) >= 2.0]
        sig.sort(key=lambda z: -abs(z[1][0]))
        P(f"   [{tname}] GAE-sb: critic bootstrap {f(crit)}; reward terms' sum {rew_sum:+.5f}; MC-b sum {f(tot['mc_base_sum'])}")
        P("      |t|>=2: " + ", ".join(f"{k.replace('reward_', '')} {v[0]:+.5f} (t{v[0] / v[1]:+.1f})" for k, v in sig[:8]))
    r9 = o["r9_style"]
    P(f"   r9-style dL_pg/d bias14 per mini-batch {f(r9['G_pg_b14_mean'], 4)}, {r9['neg_rollouts']}/{o['n']} rollouts negative, RMS "
      f"{r9['G_pg_b14_rms_mean']:.3f}")
    for W in ("W32",):
        u = o["per_update"].get(W)
        if u:
            P(f"   per update ({W}): k_eff {u['k_eff']:.2f}; bank mean predicted {u['calibrated_bank_mean_per_update'][0]:+.5f}±"
              f"{u['calibrated_bank_mean_per_update'][1]:.5f} vs actual weights {u['actual_bank_mean_weights_share']:+.5f}; statue_settled "
              f"{u['calibrated_statue_settled_per_update'][0]:+.5f}±{u['calibrated_statue_settled_per_update'][1]:.5f} vs "
              f"{u['actual_statue_settled_weights_share']:+.5f}; rollout rows {u['calibrated_rollout_mean_per_update'][0]:+.5f}±"
              f"{u['calibrated_rollout_mean_per_update'][1]:.5f} vs training-row {u['actual_probe_rollout_mu14_per_update']:+.5f}; "
              f"x epochs: bank {u['pred_bank_mean_tail3_per_update'][0]:+.5f}; 150-output r {u['calib_r_all150']:+.2f}")
    rr = [v for k, v in R.items() if k.startswith(f"{run}_{N}_")]
    if rr:
        v = rr[0]
        P(f"   realised over the next local checkpoint ({v['updates']:.0f} updates), weights' share per update on own on-policy states "
          f"{v['rollout_states']['weights_per_update']:+.5f}, bank {v['bank']['weights_per_update']:+.5f}, statue_settled "
          f"{v['statue_settled']['weights_per_update']:+.5f}; critic dV/dq3 {v['sens_N']['dV_dq3_per_deg']:+.5f}/deg; G {v['sens_N']['loop_gain_G_5p16']:+.3f}")
    P("")
(HERE / "eb_summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
