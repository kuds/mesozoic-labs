"""How strong the hottest-drive heat cost (r_heat_peak, D29) can be before standing still out-earns a noisy walking
gait: the README's reward-per-step method ("Reward per step at three stages of training") applied to the four D22
128 mm policies. Simulation only; nothing here trains.

Each case is one 20 s episode per seed (100000-100002), command 0.25 or 0.15 m/s, servo delay 10 ms, randomisation on
(per body only: the flat floor), penalty and late scale 1 (full scale, from 3 M steps on), in the run's own env with
the steering presets' heat settings (margin 0.8, tau 4 s, start 2 s) at weight 1. r_heat_peak is linear in its weight,
so the base reward (every term but r_heat_peak) and the cost at weight 1 are recorded, and any weight w is evaluated
afterwards as base - w x cost. Cases: standing (zero action); noise (Gaussian, std 0.2, around zero: the initial
policy's); each policy deterministic, and with Gaussian action noise of std 0.05, 0.1, 0.15 and 0.2 added (PPO's
exploration noise starts at 0.2, log_std_init -1.6; the D22 runs ended at about 0.085). Also the whole robot x1.1
(standing, the policy, the policy + noise 0.2). The policies are the straight-task D22 policies; the steering task's
r_track differs, so the margins are a guide for choosing the dev runs' weights, not a prediction. For reference, the
cost of the alternative D29 did not choose, every drive priced (the sum over the 12 drives of max(0, RMS / rating -
0.8), from the env's own averages), is recorded beside it as cost_sum_w1.

Usage:
  python steer_heat_weight.py --runs-dir DIR --out OUT_DIR
DIR holds fore128-s0 ... fore128-s3, each with run_record.json and checkpoints/ (the final pair is used), loaded with
the gait analysis's safe loaders. Writes OUT_DIR/heat_weight_episodes.json and OUT_DIR/heat_weight_margins.md.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
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
COMMANDS = (0.25, 0.15)
HEAT = dict(w_heat_peak=1.0, heat_peak_margin=0.8, heat_peak_tau_s=4.0, heat_peak_start_s=2.0)
STDS = (0.0, 0.05, 0.1, 0.15, 0.2)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0)
TERMS = ("r_track", "r_heat", "r_saturation", "r_air_time", "r_slip", "r_energy")


def episode(env, policy, stats, seed, command, std):
    import torch

    mean, var, clip, eps = stats
    rng = np.random.default_rng(seed + 7)
    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": 10})
    base = cost = cost_sum = 0.0
    terms = dict.fromkeys(TERMS, 0.0)
    n = 0
    while True:
        if policy is None:
            act = np.zeros(12)
        else:
            o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            with torch.no_grad():
                act, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
            act = act.numpy()[0]
        if std:
            act = act + rng.normal(0.0, std, 12)
        obs, r, terminated, truncated, info = env.step(act)
        base += r - info["r_heat_peak"]
        cost += -info["r_heat_peak"]  # weight 1, late scale 1
        if env._heat_n:  # every drive priced, from the same bias-corrected averages (heat_peak_update)
            rms = np.sqrt(env._heat_ema / (1.0 - (1.0 - env._heat_alpha) ** env._heat_n))
            cost_sum += float(np.maximum(rms - HEAT["heat_peak_margin"], 0.0).sum())
        for k in TERMS:
            terms[k] += info[k]
        n += 1
        if terminated or truncated:
            break
    return {"base": base / n, "cost_w1": cost / n, "cost_sum_w1": cost_sum / n, "steps": n, "fell": bool(terminated),
            "termination": info["termination"] if terminated else "", "terms": {k: v / n for k, v in terms.items()},
            "hottest_at_end": float(env.heat_peak_rms), "mass_kg": float(env.model.body_mass.sum())}


def job(args):
    import torch

    runs_dir, run = args
    torch.set_num_threads(1)
    rd = Path(runs_dir) / run
    s = json.loads((rd / "run_record.json").read_text())["settings"]
    zips = sorted((rd / "checkpoints").glob("trike60_[0-9]*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    ck = zips[-1]
    pkl = ck.with_name(ck.name.replace("trike60_", "trike60_vecnormalize_").replace(".zip", ".pkl"))
    ec = {**s["env_config"], "model_path": str(gp.GEN / Path(s["env_config"]["model_path"]).name)}
    cfg = flat_floor(TrikeEnvConfig.from_dict(ec)).replace(**HEAT)
    nominal, heavy = TrikeEnv(cfg), TrikeEnv(cfg.replace(mass_scale_whole_range=(1.1, 1.1)))
    stats = gp.load_obs_stats(pkl)
    policy = gp.build_policy(nominal, ck, s["net"], s["log_std_init"])
    specs = [("nominal", "standing", None, 0.0), ("nominal", "noise", None, 0.2)]
    specs += [("nominal", f"policy+{std:g}", policy, std) for std in STDS]
    specs += [("x1.1", "standing", None, 0.0), ("x1.1", "policy+0", policy, 0.0), ("x1.1", "policy+0.2", policy, 0.2)]
    out = []
    for cond, case, pol, std in specs:
        env = nominal if cond == "nominal" else heavy
        for command in COMMANDS:
            for seed in SEEDS:
                out.append({"run": run, "checkpoint": ck.name, "cond": cond, "case": case, "command": command,
                            "seed": seed, **episode(env, pol, stats, seed, command, std)})
    return out


def margins_md(rows, key="cost_w1") -> str:
    g = defaultdict(list)
    for r in rows:
        g[(r["cond"], r["command"], r["case"], r["run"])].append(r)
    runs = sorted({r["run"] for r in rows})
    m = lambda rs, k: st.mean(x[k] for x in rs)  # noqa: E731
    title = "the hottest drive priced (D29)" if key == "cost_w1" else "every drive priced (reference, not used)"
    L = [f"# Reward margin over standing at full scale, by heat weight: {title} (steer_heat_weight.py)", "",
         f"Falls: {sum(r['fell'] for r in rows)} of {len(rows)} episodes. Per-step means over 3 seeds; margin = "
         "(case's base - w x cost at weight 1) - (standing's base - w x standing's cost); standing pays no "
         "r_heat_peak. The worst of the four policies is given, then all four.", ""]
    for cond in ("nominal", "x1.1"):
        for cmd in COMMANDS:
            cases = [c for c in ["noise"] + [f"policy+{s:g}" for s in STDS] if (cond, cmd, c, runs[0]) in g]
            L += [f"## {cond}, {cmd} m/s", "",
                  "| case | hottest drive at 20 s (4 runs) | " + " | ".join(f"w {w:g}" for w in WEIGHTS) + " |",
                  "|---" * (2 + len(WEIGHTS)) + "|"]
            for c in cases:
                hot = " ".join(f"{m(g[(cond, cmd, c, r)], 'hottest_at_end'):.2f}" for r in runs)
                cells = []
                for w in WEIGHTS:
                    ms = [m(g[(cond, cmd, c, r)], "base") - w * m(g[(cond, cmd, c, r)], key)
                          - (m(g[(cond, cmd, "standing", r)], "base") - w * m(g[(cond, cmd, "standing", r)], key))
                          for r in runs]
                    cells.append(f"{min(ms):+.3f} ({' '.join(f'{x:+.2f}' for x in ms)})")
                L.append(f"| {c} | {hot} | " + " | ".join(cells) + " |")
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    runs = [f"fore128-s{i}" for i in range(4)]
    with Pool(a.workers) as p:
        rows = [x for chunk in p.map(job, [(a.runs_dir, r) for r in runs]) for x in chunk]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heat_weight_episodes.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "heat_weight_margins.md").write_text(margins_md(rows) + "\n")
    (out / "heat_weight_margins_every_drive.md").write_text(margins_md(rows, "cost_sum_w1") + "\n")
    print(f"{len(rows)} episodes, {sum(r['fell'] for r in rows)} falls; wrote {out}")


if __name__ == "__main__":
    main()
