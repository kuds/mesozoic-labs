"""Which running average r_heat_peak (D29) should price the hottest drive with: candidate averages compared offline on
the drive heat the four D22 128 mm policies produce. Simulation only; nothing here trains.

Each episode (20 s, flat floor, per-body randomisation, delay 10 ms, seeds 100000-100002, commands 0.25 and 0.15 m/s,
Gaussian action noise of std 0, 0.1 and 0.2 on the deterministic policy) logs each control step's per-drive mean of
(torque / continuous rating)^2 through a substep hook. Every candidate starts at 2 s (control step 101, F2's window)
and low-passes with tau 4 s; the cost per step at weight 1 is max(0, sqrt(the largest of the 12) - 0.8). Candidates:
- bc: bias-corrected by 1 - (1 - alpha)^n (the version reviewed on 27 Nov 2026);
- floorN: bias-corrected with n floored at N steps;
- warm: started at margin^2, no correction (adopted);
- cold: started at 0, no correction;
- bc_price4: bias-corrected, priced from 4 s only.
Measures: the cost over 2-20 s as a share of F2's own price (900 steps x max(0, F2 - 0.8), F2 = sqrt of the largest
drive's 2-20 s mean); the share of the cost in 2-4 s (11 % of the window's time); the cost per step on episodes
whose F2 is under the margin (ideally 0) and the share of it in 2-4 s; the extra cost of a 0.5 s burst at twice the
rating at 3 s against 10 s, on the hottest and on the second-hottest drive, deterministic and with noise; and the
linearised discounted weight (gamma 0.99) of one step's heat in the average at several times (the cost's hinge and
maximum over drives are left out of that last measure; they make early heat on a cool or not-hottest drive cost more
than later heat, which the burst rows show).

Usage:
  python steer_heat_average.py --runs-dir DIR --out OUT_DIR [--logs LOGS.npz]
DIR as for steer_heat_weight.py. Writes OUT_DIR/heat_average_comparison.md; the logs (about 6.6 MB) are kept only
with --logs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RL = Path(__file__).resolve().parents[1]
for p in (str(RL), str(RL / "analysis"), str(RL.parents[2])):
    if p not in sys.path:
        sys.path.insert(0, p)
import gait_pipe1 as gp  # noqa: E402
from trike_env import TrikeEnv, TrikeEnvConfig, flat_floor  # noqa: E402

SEEDS = (100000, 100001, 100002)
ALPHA = 1.0 - math.exp(-0.02 / 4.0)
MARGIN, K0, GAMMA = 0.8, 100, 0.99
ESTS = ("bc", "floor50", "floor100", "floor200", "warm", "cold", "bc_price4")


def log_episode(env, policy, stats, seed, command, std):
    import torch

    mean, var, clip, eps = stats
    rng = np.random.default_rng(seed + 7)
    acc = []
    env.substep_hook = lambda e: acc.append(e.data.actuator_force[e._leg_ctrl_index] ** 2 * e._inv_cont2)
    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": 10})
    xs = []
    try:
        while True:
            o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            with torch.no_grad():
                act, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
            act = act.numpy()[0] + (rng.normal(0.0, std, 12) if std else 0.0)
            acc.clear()
            obs, _, term, trunc, _ = env.step(act)
            xs.append(np.mean(acc, axis=0))
            if term or trunc:
                break
    finally:
        env.substep_hook = None
    return np.array(xs), bool(term)


def job(args):
    import torch

    runs_dir, run = args
    torch.set_num_threads(1)
    rd = Path(runs_dir) / run
    s = json.loads((rd / "run_record.json").read_text())["settings"]
    ck = sorted((rd / "checkpoints").glob("trike60_[0-9]*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    pkl = ck.with_name(ck.name.replace("trike60_", "trike60_vecnormalize_").replace(".zip", ".pkl"))
    ec = {**s["env_config"], "model_path": str(gp.GEN / Path(s["env_config"]["model_path"]).name)}
    env = TrikeEnv(flat_floor(TrikeEnvConfig.from_dict(ec)))
    stats = gp.load_obs_stats(pkl)
    policy = gp.build_policy(env, ck, s["net"], s["log_std_init"])
    out = {}
    for command in (0.25, 0.15):
        for std in (0.0, 0.1, 0.2):
            for seed in SEEDS:
                out[f"{run}|{command}|{std}|{seed}"] = log_episode(env, policy, stats, seed, command, std)
    return out


def estimate(x, kind):
    """Each drive's estimated mean square per control step (nan before the window)."""
    est = np.full(x.shape, np.nan)
    ema = np.full(12, MARGIN ** 2) if kind == "warm" else np.zeros(12)
    n = 0
    for k in range(K0, len(x)):
        n += 1
        ema += ALPHA * (x[k] - ema)
        if kind == "bc":
            est[k] = ema / (1.0 - (1.0 - ALPHA) ** n)
        elif kind.startswith("floor"):
            est[k] = ema / (1.0 - (1.0 - ALPHA) ** max(n, int(kind[5:])))
        elif kind == "bc_price4":
            est[k] = ema / (1.0 - (1.0 - ALPHA) ** n) if k >= 200 else 0.0
        else:
            est[k] = ema
    return est


def cost(x, kind):
    est = estimate(x, kind)
    c = np.zeros(len(x))
    ok = ~np.isnan(est[:, 0])
    c[ok] = np.maximum(np.sqrt(est[ok].max(1)) - MARGIN, 0.0)
    return c


def f2(x):
    return math.sqrt(x[K0:].mean(0).max())


def weight(kind, t):
    """d(discounted return of the mean-square estimate) / d(one step's heat at time t), ignoring the hinge."""
    k = int(round(t / 0.02)) - 1
    if kind in ("warm", "cold"):
        return sum((GAMMA * (1.0 - ALPHA)) ** j * ALPHA for j in range(1000 - k))
    s = 0.0
    for nn in range(k, 1000):
        if kind == "bc_price4" and nn < 200:
            continue
        n0 = int(kind[5:]) if kind.startswith("floor") else 1
        s += GAMMA ** (nn - k) * ALPHA * (1.0 - ALPHA) ** (nn - k) / (1.0 - (1.0 - ALPHA) ** max(nn - K0 + 1, n0))
    return s


def report(logs: dict) -> str:
    groups: dict = {}
    for key, (x, _fell) in logs.items():
        _, cmd, std, _ = key.split("|")
        groups.setdefault((cmd, std), []).append(x)
    L = ["# Running averages for r_heat_peak compared (steer_heat_average.py)", "",
         f"{len(logs)} episodes, {sum(f for _, f in logs.values())} falls. Four D22 128 mm policies x 3 seeds per "
         "group. Cost at weight 1, from 2 s; 'warm' is adopted (trike_env.heat_peak_update).", "",
         "## Cost over 2-20 s as a share of F2's own price", "",
         "| command, noise | F2 (12 episodes) | F2 price per step | " + " | ".join(ESTS) + " |",
         "|---" * (3 + len(ESTS)) + "|"]
    for (cmd, std), xs in sorted(groups.items()):
        fs = [f2(x) for x in xs]
        ideal = sum(900 * max(0.0, f - MARGIN) for f in fs)
        cells = [f"{sum(cost(x, e).sum() for x in xs) / ideal:.2f}" for e in ESTS]
        L.append(f"| {cmd} m/s, std {std} | {min(fs):.2f}-{max(fs):.2f} | {ideal / 900 / len(xs):.3f} | "
                 + " | ".join(cells) + " |")
    L += ["", "## Share of the cost in 2-4 s (11 % of the window's time)", "",
          "| noise | " + " | ".join(ESTS) + " |", "|---" * (1 + len(ESTS)) + "|"]
    for std in ("0.0", "0.1", "0.2"):
        xs = [x for (c, s), v in groups.items() if s == std for x in v]
        cells = []
        for e in ESTS:
            cs = [cost(x, e) for x in xs]
            tot = sum(c.sum() for c in cs)
            cells.append(f"{sum(c[100:200].sum() for c in cs) / tot:.2f}" if tot else "-")
        L.append(f"| std {std} | " + " | ".join(cells) + " |")
    cool = [x for v in groups.values() for x in v if f2(x) < MARGIN]
    L += ["", f"## Episodes whose F2 is under the margin ({len(cool)} episodes): cost per step over all 1000 steps "
          "(ideally 0), and the share of it in 2-4 s", "",
          "| measure | " + " | ".join(ESTS) + " |", "|---" * (1 + len(ESTS)) + "|",
          "| cost per step | " + " | ".join(f"{np.mean([cost(x, e).mean() for x in cool]):.4f}" for e in ESTS) + " |"]
    cells = []
    for e in ESTS:
        cs = [cost(x, e) for x in cool]
        tot = sum(c.sum() for c in cs)
        cells.append(f"{sum(c[100:200].sum() for c in cs) / tot:.2f}" if tot else "-")
    L.append("| share in 2-4 s | " + " | ".join(cells) + " |")
    L += ["", "## Extra cost of a 0.5 s burst at 2x rating (25 steps at (torque / rating)^2 = 4), mean over the "
          "episodes, at 3 s and at 10 s", "",
          "| average | drive | noise | at 3 s | at 10 s | ratio |", "|---|---|---|---|---|---|"]
    for e in ESTS:
        for rank, rank_name in ((0, "hottest"), (1, "2nd hottest")):
            for std in ("0.0", "0.1", "0.2"):
                xs = [x for (c, s), v in groups.items() if s == std for x in v]
                d = {}
                for t in (3.0, 10.0):
                    tot = 0.0
                    for x in xs:
                        j = int(np.argsort(x[K0:].mean(0))[::-1][rank])
                        y = x.copy()
                        k = int(t / 0.02)
                        y[k:k + 25, j] = 4.0
                        tot += cost(y, e).sum() - cost(x, e).sum()
                    d[t] = tot / len(xs)
                L.append(f"| {e} | {rank_name} | {std} | {d[3.0]:.1f} | {d[10.0]:.1f} | "
                         f"{d[3.0] / d[10.0] if d[10.0] else float('nan'):.2f} |")
    ts = (2.02, 2.5, 3.0, 4.0, 6.0, 10.0, 15.0)
    L += ["", "## Discounted weight (gamma 0.99) of one step's heat in the average, by time (hinge and maximum "
          "over drives left out)", "",
          "| average | " + " | ".join(f"{t:g} s" for t in ts) + " |", "|---" * (1 + len(ts)) + "|"]
    for e in ESTS:
        L.append(f"| {e} | " + " | ".join(f"{weight(e, t):.2f}" for t in ts) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.workers) as p:
        logs = {}
        for d in p.map(job, [(a.runs_dir, f"fore128-s{i}") for i in range(4)]):
            logs.update(d)
    if a.logs:
        np.savez_compressed(a.logs, **{k: v[0] for k, v in logs.items()})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heat_average_comparison.md").write_text(report(logs) + "\n")
    print(f"{len(logs)} episodes; wrote {out / 'heat_average_comparison.md'}")


if __name__ == "__main__":
    main()
