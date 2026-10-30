"""ADVERSARIAL VERIFIER of the early-warning curve (role verify_early_warning).  My own re-measurement; it does not
import or call any ew_* code.  NO TRAINING, no optimizer step: checkpoints opened read only, deterministic simulation
and network evaluation only.

    PYTHONPATH=<r10round/ecs/code> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 venv/bin/python -B vew_measure.py OUT.jsonl

Per checkpoint:
  - the harness loader (load_sb3_checkpoint) is used ONLY to drive deterministic rollouts with model.predict, exactly as
    the eval / probe path does;
  - every sweep uses MY OWN forward pass: actor weights read from policy.pth inside the zip (torch.load, weights only)
    -> float64 numpy, tanh MLP 63-512-256-15; MY OWN normaliser formula clip((x - mean) / sqrt(var + eps), +-clip) with
    mean / var / eps / clip read from the pickle.  It is cross-checked against SB3's get_distribution mean.
  - state sets: A = seeds 5001-5010 (the early-warning set, for reproduction), B = seeds 6001-6020 (held out), raw
    observations after steps 500, 600, ..., 1000 (6 per episode); REF = the probe bank's 2 statue_settled rows.
  - tail sweeps on obs[6] (tail_3 angle): P = 0, .25, .5, .75, 1 deg (the early-warning primary); H = 0, .25, .5 deg;
    L = each state's own angle +-0.25 deg (local derivative at the operating point); LEG = 0..6 deg (legacy).
    G = median over states of the least-squares slope (per deg) x 5.16.  c0 = median of mu14 at 0 deg.  FP = c0 / (1-G).
  - clipping / saturation: the tail_3 z at each swept point (own formula), the number of swept state-points at the
    clip, and the share of hidden units with |tanh| > 0.99 at 0 and at 1 deg (layer 1, layer 2).
  - bootstrap over episodes (cluster = episode, 2,000 resamples, rng 12345) for G and c0 of P.
"""
from __future__ import annotations

import io
import json
import math
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
HERE = SCR / "com_revisit/notes/verify_early_warning"
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ECS = SCR / "r10round/ecs/runs"
PLANT = 5.16
IT3, AT3 = 6, 14
SEEDS_A = list(range(5001, 5011))
SEEDS_B = list(range(6001, 6021))
SNAPS = (500, 600, 700, 800, 900, 1000)
P = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
H = np.array([0.0, 0.25, 0.5])
LEG = np.arange(0.0, 6.01, 1.0)

CKPTS = [("s42", 1048576), ("s42", 2097152), ("s42", 3145728), ("s42", 3670016),
         ("s60", 6291456), ("s60", 7340032), ("s60", 8388608),
         ("s80", 3145728), ("s80", 6291456), ("s80", 7340032), ("s80", 8388608), ("s80", 10485760),
         ("s80", 11534336), ("s80", 12001280),
         ("s80_B_1401", 5767168), ("s42_A_1151", 4718592),
         ("s60", 1048576), ("s80", 1048576),  # second pass (added while the first ran): the F8 1.05M values
         ("s80", 2097152), ("s80", 4194304), ("s80", 5242880), ("s80", 9437184)]  # third pass: s80's max G_L


def paths(run, steps):
    if run in ("s42", "s60", "s80"):
        m = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "models"
        if steps == 12001280:
            return m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1"
        return m / f"stage1_{steps}_steps.zip.copy1", m / f"stage1_vecnormalize_{steps}_steps.pkl.copy1"
    m = ECS / run / "01_stance" / "models"
    return m / f"stage1_{steps}_steps.zip", m / f"stage1_vecnormalize_{steps}_steps.pkl"


class MyNet:
    """Own float64 actor + own normaliser (no SB3 forward)."""

    def __init__(self, z, p):
        with zipfile.ZipFile(z) as zf:
            sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
        g = lambda k: sd[k].detach().cpu().numpy().astype(np.float64)  # noqa: E731
        self.W0, self.b0 = g("mlp_extractor.policy_net.0.weight"), g("mlp_extractor.policy_net.0.bias")
        self.W1, self.b1 = g("mlp_extractor.policy_net.2.weight"), g("mlp_extractor.policy_net.2.bias")
        self.W2, self.b2 = g("action_net.weight"), g("action_net.bias")
        with open(p, "rb") as fh:
            vn = pickle.load(fh)
        self.mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
        self.var = np.asarray(vn.obs_rms.var, dtype=np.float64)
        self.eps = float(vn.epsilon)
        self.clip = float(vn.clip_obs)
        self.std = np.sqrt(self.var + self.eps)

    def z(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def hidden(self, X):
        zc = np.clip(self.z(X), -self.clip, self.clip)
        h1 = np.tanh(zc @ self.W0.T + self.b0)
        h2 = np.tanh(h1 @ self.W1.T + self.b1)
        return h1, h2

    def mu(self, X):
        _, h2 = self.hidden(X)
        return h2 @ self.W2.T + self.b2


def set_t3(X, deg):
    Y = np.array(X, dtype=np.float64, copy=True)
    Y[:, IT3] = math.radians(deg) if np.isscalar(deg) else np.radians(deg)
    return Y


def ls_slopes(xs, ys):
    """xs (k,) or (k, n); ys (k, n) -> per-state LS slope."""
    xs = np.asarray(xs, dtype=np.float64)
    if xs.ndim == 1:
        xs = np.repeat(xs[:, None], ys.shape[1], axis=1)
    xc = xs - xs.mean(axis=0)
    return (xc * (ys - ys.mean(axis=0))).sum(axis=0) / (xc ** 2).sum(axis=0)


def sweep(net, X, degs):
    ys = np.array([net.mu(set_t3(X, d))[:, AT3] for d in degs])
    zt = np.array([net.z(set_t3(X, d))[:, IT3] for d in degs])
    sl = ls_slopes(degs, ys)
    return ys, sl, zt


def local_sweep(net, X):
    th = np.degrees(X[:, IT3])
    degs = np.array([th - 0.25, th, th + 0.25])
    ys = np.array([net.mu(set_t3(X, degs[i]))[:, AT3] for i in range(3)])
    return ls_slopes(degs, ys), th


def summarize(net, X, ep_index):
    out = {"n_states": int(len(X))}
    ys, sl, zt = sweep(net, X, P)
    G = float(np.median(sl)) * PLANT
    c0 = float(np.median(ys[0]))
    out["P"] = {"G": G, "c0": c0, "FP": c0 / (1 - G) if G < 1 else float("inf"),
                "z_max": float(zt.max()), "z_min": float(zt.min()), "n_clipped": int((np.abs(zt) >= net.clip).sum()),
                "slope_q25": float(np.quantile(sl, 0.25)) * PLANT, "slope_q75": float(np.quantile(sl, 0.75)) * PLANT,
                "frac_states_G_pos": float(np.mean(sl > 0)),
                "lin_ratio_hi_lo": float(np.median((ys[4] - ys[2]) / 0.5) / np.median((ys[2] - ys[0]) / 0.5))
                if abs(np.median((ys[2] - ys[0]) / 0.5)) > 1e-9 else None,
                "mu_at_states": float(np.median(net.mu(X)[:, AT3]))}
    # curvature: quadratic fit coefficient (per deg^2) median
    q = np.polyfit(P, ys, 2)[0]
    out["P"]["quad_coef_median"] = float(np.median(q))
    # hidden saturation at 0 and 1 deg
    for d in (0.0, 1.0):
        h1, h2 = net.hidden(set_t3(X, d))
        out["P"][f"sat_h1_at{d:g}"] = float(np.mean(np.abs(h1) > 0.99))
        out["P"][f"sat_h2_at{d:g}"] = float(np.mean(np.abs(h2) > 0.99))
    ysH, slH, ztH = sweep(net, X, H)
    out["H"] = {"G": float(np.median(slH)) * PLANT, "c0": float(np.median(ysH[0]))}
    slL, th = local_sweep(net, X)
    out["L"] = {"G": float(np.median(slL)) * PLANT, "theta_deg_median": float(np.median(th)),
                "z_max": float(np.max(net.z(set_t3(X, th + 0.25))[:, IT3]))}
    ysG, slG, ztG = sweep(net, X, LEG)
    out["LEG"] = {"G": float(np.median(slG)) * PLANT, "c0": float(np.median(ysG[0])),
                  "first_clipped_deg": float(LEG[np.argmax((ztG >= net.clip).any(axis=1))]) if (ztG >= net.clip).any() else None}
    # bootstrap over episodes
    if ep_index is not None:
        rng = np.random.default_rng(12345)
        eps = np.unique(ep_index)
        bg, bc = [], []
        for _ in range(2000):
            pick = rng.choice(eps, size=len(eps), replace=True)
            idx = np.concatenate([np.flatnonzero(ep_index == e) for e in pick])
            bg.append(np.median(sl[idx]) * PLANT)
            bc.append(np.median(ys[0][idx]))
        out["P"]["G_boot_q025"], out["P"]["G_boot_q975"] = (float(x) for x in np.quantile(bg, [0.025, 0.975]))
        out["P"]["G_boot_sd"] = float(np.std(bg))
        out["P"]["c0_boot_q025"], out["P"]["c0_boot_q975"] = (float(x) for x in np.quantile(bc, [0.025, 0.975]))
    return out


def rollouts(env, model, norm, seeds):
    X, epi, eps = [], [], []
    for j, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        cmd, t3 = [], []
        k = 0
        term = trunc = False
        for k in range(1, 1001):
            a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            cmd.append(float(np.clip(a[AT3], -1, 1)))
            obs, r, term, trunc, _ = env.step(a)
            t3.append(float(obs[IT3]))
            if k in SNAPS:
                X.append(np.array(obs, dtype=np.float64, copy=True))
                epi.append(j)
            if term or trunc:
                break
        w = slice(500, 1000)
        eps.append({"seed": seed, "steps": k, "terminated": bool(term),
                    "own14": float(np.mean(cmd[w])) if len(cmd) > 500 else None,
                    "frac99": float(np.mean(np.abs(np.array(cmd[w])) >= 0.99)) if len(cmd) > 500 else None,
                    "tail_deg": math.degrees(float(np.mean(t3[w]))) if len(t3) > 500 else None})
    return np.array(X), np.array(epi), eps


def main() -> int:
    out_path = Path(sys.argv[1])
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    factory = lambda: env_class(mid)(**ctor)  # noqa: E731
    env = factory()
    assert env.unwrapped.model.actuator(AT3).name == "tail_3_pitch_act"
    with np.load(F2 / "stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1") as d:
        bank = np.asarray(d["bank_raw_obs"], dtype=np.float64)
        ss = np.asarray(d["bank_labels"]) == "statue_settled"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["run"], json.loads(l)["steps"]) for l in out_path.read_text().splitlines()}
    (HERE / "states").mkdir(exist_ok=True)
    with open(out_path, "a") as fh:
        for run, steps in CKPTS:
            if (run, steps) in done or (only and f"{run}:{steps}" not in only):
                continue
            z, p = paths(run, steps)
            t0 = time.process_time()
            net = MyNet(z, p)
            model, norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
            # cross-check my forward and normaliser against SB3
            with torch.no_grad():
                nb = norm.normalize_obs(bank.astype(np.float32))
                sb3 = model.policy.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32))).distribution.mean.numpy()
            xchk = float(np.max(np.abs(sb3 - net.mu(bank))))
            nchk = float(np.max(np.abs(np.asarray(norm.normalize_obs(bank), dtype=np.float64)
                                       - np.clip(net.z(bank), -net.clip, net.clip))))
            row = {"run": run, "steps": steps, "zip": str(z), "xcheck_mu_vs_sb3": xchk, "xcheck_norm": nchk,
                   "clip": net.clip, "eps": net.eps,
                   "t3_mean_deg": math.degrees(net.mean[IT3]), "t3_sd_deg": math.degrees(net.std[IT3]),
                   "deg_at_clip": math.degrees(net.mean[IT3] + net.clip * net.std[IT3])}
            row["REF"] = summarize(net, bank[ss], None)
            t1 = time.process_time()
            for name, seeds in (("A", SEEDS_A), ("B", SEEDS_B)):
                X, epi, eps = rollouts(env, model, norm, seeds)
                np.savez_compressed(HERE / "states" / f"{run}_{steps}_{name}.npz", X=X, epi=epi)
                s = summarize(net, X, epi)
                full = [e for e in eps if e["own14"] is not None]
                s["own14_median"] = float(np.median([e["own14"] for e in full]))
                s["own14_mean"] = float(np.mean([e["own14"] for e in full]))
                s["frac99_median"] = float(np.median([e["frac99"] for e in full]))
                s["n_over_bar"] = int(sum(e["frac99"] > 0.10 for e in full))
                s["n_terminated"] = int(sum(e["terminated"] for e in eps))
                s["episodes"] = eps
                row[name] = s
            # pooled A+B
            XA = np.load(HERE / "states" / f"{run}_{steps}_A.npz")
            XB = np.load(HERE / "states" / f"{run}_{steps}_B.npz")
            Xab = np.concatenate([XA["X"], XB["X"]])
            eab = np.concatenate([XA["epi"], XB["epi"] + 100])
            row["AB"] = summarize(net, Xab, eab)
            row["cpu_s"] = {"load_and_ref": t1 - t0, "rollouts_and_sweeps": time.process_time() - t1}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{run} {steps}: xchk {xchk:.1e}/{nchk:.1e} | A G {row['A']['P']['G']:+.3f} "
                  f"[{row['A']['P']['G_boot_q025']:+.3f},{row['A']['P']['G_boot_q975']:+.3f}] c0 {row['A']['P']['c0']:+.3f} "
                  f"own {row['A']['own14_median']:+.3f} | B G {row['B']['P']['G']:+.3f} "
                  f"[{row['B']['P']['G_boot_q025']:+.3f},{row['B']['P']['G_boot_q975']:+.3f}] c0 {row['B']['P']['c0']:+.3f} "
                  f"own {row['B']['own14_median']:+.3f} | REF G {row['REF']['P']['G']:+.3f} | zmax {row['A']['P']['z_max']:.2f} "
                  f"| cpu {row['cpu_s']['rollouts_and_sweeps']:.0f}s", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
