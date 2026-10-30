"""EARLY-BIAS PROBE: the ACTUAL change per update, from the run's own policy probe (read only, no simulation).

For each checkpoint N used by eb_update.py, from policy_probe.npz (row N = the policy that collected rollout N, i.e.
exactly the checkpoint's parameters; row N + 8192 k = after k more updates):
  - actual per-update change of the bank's deterministic mu (10 states x 15 actuators), over the next W updates
    (W = 32, i.e. N -> N + 262,144, and W = 64), as (mu[N + W] - mu[N]) / W and as an OLS slope over rows N..N+W;
  - the per-update noise: SD of the single-update differences mu[k + 1] - mu[k] over the window;
  - the normaliser's share at fixed weights: mu(theta_N, stats[N + W]) - mu(theta_N, stats[N]) (the probe's recorded
    obs_rms, VecNormalize's formula), so weights' share = actual - normaliser's share (first order);
  - the checkpoint's _n_updates (epochs run) and Adam step count: epochs per update between checkpoints.
Writes eb_actual.json and eb_actual.txt.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
sys.path.insert(0, str(HERE))
import eb_common as E  # noqa: E402

CKPTS = {"s42": [1048576, 2097152, 3145728, 3670016],
         "s80": [1048576, 2097152, 3145728],
         "s60": [5242880, 6291456, 7340032, 8388608]}
ALL_LOCAL = {r: sorted(int(p.name.split("_")[1]) for p in (E.copies(r) / "models").glob("stage1_*_steps.zip.copy1"))
             for r in CKPTS}


def n_updates(run, steps):
    zp, _ = E.ckpt_paths(run, steps)
    with zipfile.ZipFile(zp) as z:
        d = json.loads(z.read("data"))
    opt = torch.load(__import__("io").BytesIO(zipfile.ZipFile(zp).read("policy.optimizer.pth")), map_location="cpu",
                     weights_only=True)
    st = opt["state"][0]["step"]
    return int(d["_n_updates"]), float(st)


def main():
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid, cfg = E.load_code_config()

    def factory():
        return env_class(mid)(**cfg["env_kwargs"])

    out = {"epochs": {}, "windows": {}}
    lines = []
    for run in CKPTS:
        pr = E.probe(run)
        t = pr["update_timesteps"]
        # epochs per update between local checkpoints
        prev = None
        for s in ALL_LOCAL[run]:
            nu, st = n_updates(run, s)
            if prev is not None:
                ds, dnu, dst = s - prev[0], nu - prev[1], st - prev[2]
                out["epochs"][f"{run}_{prev[0]}_{s}"] = {"updates": ds / 8192, "epochs": dnu, "epochs_per_update": dnu / (ds / 8192),
                                                         "adam_steps": dst, "adam_steps_per_update": dst / (ds / 8192)}
            prev = (s, nu, st)
        for N in CKPTS[run]:
            i0 = int(np.flatnonzero(t == N)[0])
            zp, pp = E.ckpt_paths(run, N)
            model, norm, _ = load_sb3_checkpoint(str(zp), str(pp), factory)
            pol = model.policy
            raw = pr["bank_raw_obs"].astype(np.float64)
            eps = float(norm.epsilon)
            clip = float(norm.clip_obs)

            def mu_at(mean, var):
                z = np.clip((raw - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
                with torch.no_grad():
                    return pol.get_distribution(torch.as_tensor(z)).mode().numpy().astype(np.float64)

            mu0 = mu_at(pr["obs_rms_mean"][i0].astype(np.float64), pr["obs_rms_var"][i0].astype(np.float64))
            chk = float(np.abs(mu0 - pr["bank_action_now"][i0]).max())
            for W in (32, 64):
                i1 = i0 + W
                if i1 >= len(t):
                    continue
                a = pr["bank_action_now"][i0:i1 + 1].astype(np.float64)  # (W+1, 10, 15)
                per = (a[-1] - a[0]) / W
                x = np.arange(W + 1) - W / 2
                slope = (x[:, None, None] * (a - a.mean(0))).sum(0) / (x ** 2).sum()
                d1 = np.diff(a, axis=0)
                sd1 = d1.std(0, ddof=1)
                mu_norm = mu_at(pr["obs_rms_mean"][i1].astype(np.float64), pr["obs_rms_var"][i1].astype(np.float64))
                norm_share = (mu_norm - mu0) / W
                weights_share = per - norm_share
                rm = pr["rollout_mu_mean"][i0:i1 + 1, 14].astype(np.float64)
                key = f"{run}_{N}_W{W}"
                out["windows"][key] = {
                    "run": run, "N": N, "W": W, "probe_ckpt_err": chk,
                    "actual_per_update": per.tolist(), "slope_per_update": slope.tolist(),
                    "single_update_sd": sd1.tolist(), "normaliser_share_per_update": norm_share.tolist(),
                    "weights_share_per_update": weights_share.tolist(),
                    "rollout_mu14_per_update": float((rm[-1] - rm[0]) / W),
                    "bank_mean_single_update_sd": float(np.diff(a[:, :, 14].mean(1)).std(ddof=1)),
                    "statue_settled_single_update_sd": float(np.diff(a[:, [0, 5], 14].mean(1)).std(ddof=1)),
                    "rollout_mu14_single_update_sd": float(np.diff(rm).std(ddof=1)),
                    "rollout_mu14_start": float(rm[0]), "rollout_mu14_end": float(rm[-1]),
                }
                ss = [0, 5]
                lines.append(
                    f"{run} N={N:>8d} W={W}: bank-mean tail_3 per update actual {per[:, 14].mean():+.5f} "
                    f"(slope {slope[:, 14].mean():+.5f}; single-update SD {sd1[:, 14].mean():.5f}); normaliser share "
                    f"{norm_share[:, 14].mean():+.5f}; weights share {weights_share[:, 14].mean():+.5f} | statue_settled "
                    f"{per[ss, 14].mean():+.5f} (weights {weights_share[ss, 14].mean():+.5f}) | rollout-row mean "
                    f"{(rm[-1] - rm[0]) / W:+.5f}/update ({rm[0]:+.3f} -> {rm[-1]:+.3f}) | ckpt-vs-probe {chk:.1e}")
    lines.append("")
    lines.append("epochs per update between local checkpoints (from _n_updates and the Adam step count in the zips):")
    for k, v in out["epochs"].items():
        lines.append(f"  {k}: {v['epochs_per_update']:.3f} epochs/update, {v['adam_steps_per_update']:.1f} Adam steps/update")
    (HERE / "eb_actual.json").write_text(json.dumps(out, indent=1))
    (HERE / "eb_actual.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
