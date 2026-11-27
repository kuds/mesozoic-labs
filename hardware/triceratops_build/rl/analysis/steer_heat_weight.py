"""How strong the hottest-drive heat cost (r_heat_peak, D29) can be before standing still out-earns a noisy walking
gait: the README's reward-per-step method ("Reward per step at three stages of training") applied to the four D22
128 mm policies. Simulation only; nothing here trains.

Each case is one 20 s episode per seed (100000-100002), command 0.25 or 0.15 m/s, servo delay 10 ms, randomisation on
(per body only: the flat floor), penalty scale 1 and late scale 1 (full scale, from 3 M steps on) unless --late
says otherwise, in the run's own env with
the steering presets' heat settings (margin 0.8, tau 4 s, start 2 s) at weight 1. r_heat_peak is linear in its weight,
so the base reward (every term but r_heat_peak) and the cost at weight 1 are recorded, and any weight w is evaluated
afterwards as base - w x cost. Cases: standing (zero action); noise (Gaussian, std 0.2, around zero: the initial
policy's); each policy deterministic, and with Gaussian action noise of std 0.05, 0.1, 0.15, 0.175 and 0.2 added
(PPO's exploration noise starts at 0.2, log_std_init -1.6; shorter earlier runs had 0.17-0.18 at 1-2 M steps, and
the D22 runs ended at about 0.085 at 10 M). Also the whole robot x1.1 (standing, and the policy at the same noise
levels). The policies are the straight-task D22 policies at their final (10 M) checkpoints; the steering task's
r_track differs, so the margins are a guide for choosing the dev runs' weights, not a prediction. For reference, the
cost of the alternative D29 did not choose, every drive priced (the sum over the 12 drives of max(0, RMS / rating -
0.8), from the env's own averages, times the late scale), is recorded beside it as cost_sum_w1.

Usage:
  python steer_heat_weight.py --runs-dir DIR --out OUT_DIR [--stds 0,0.05,0.1] [--late 0.5] [--tag _x] [--runs a,b]
--stds sets the policies' action-noise levels (default the six above; distinct, finite, >= 0), --late the late
scale (default 1, full scale: r_heat, r_saturation, the negative part of r_air_time and r_heat_peak are multiplied by
it, as train.py's late ramp does; 0.5-1, the stages from 2 M steps on, where train.py's penalty scale is 1), --runs
the run folders (default fore128-s0 ... s3); the x1.1 cases use the same noise levels, and the output file names
carry --tag. "Cost at weight 1" is at the given late scale.
DIR holds the run folders, each with run_record.json and checkpoints/ (the newest pair is used), loaded with the gait
analysis's safe loaders. Writes OUT_DIR/heat_weight_episodes<tag>.json, OUT_DIR/heat_weight_margins<tag>.md and
OUT_DIR/heat_weight_margins_every_drive<tag>.md. Each margin is given for the worst of the policies, with the largest
of their standard errors (per-seed margins, paired by seed with standing), then for each policy.
"""
from __future__ import annotations

import argparse
import json
import math
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
STDS = (0.0, 0.05, 0.1, 0.15, 0.175, 0.2)
WEIGHTS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
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
        cost += -info["r_heat_peak"]  # weight 1, at the env's late scale (r_heat_peak includes it)
        if env._step_count > env._heat_start_step:  # every drive priced, from the same averages, same late scale
            cost_sum += env.late_scale * float(np.maximum(np.sqrt(env._heat_ema) - HEAT["heat_peak_margin"], 0.0).sum())
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

    runs_dir, run, stds, late = args
    torch.set_num_threads(1)
    rd = Path(runs_dir) / run
    s = json.loads((rd / "run_record.json").read_text())["settings"]
    zips = sorted((rd / "checkpoints").glob("trike60_[0-9]*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    ck = zips[-1]
    pkl = ck.with_name(ck.name.replace("trike60_", "trike60_vecnormalize_").replace(".zip", ".pkl"))
    ec = {**s["env_config"], "model_path": str(gp.GEN / Path(s["env_config"]["model_path"]).name)}
    cfg = flat_floor(TrikeEnvConfig.from_dict(ec)).replace(**HEAT)
    nominal, heavy = TrikeEnv(cfg), TrikeEnv(cfg.replace(mass_scale_whole_range=(1.1, 1.1)))
    for e in (nominal, heavy):  # train.py's schedules at this stage (the penalty ramp is full from 2 M steps)
        e.late_scale, e.penalty_scale = late, 1.0
    stats = gp.load_obs_stats(pkl)
    policy = gp.build_policy(nominal, ck, s["net"], s["log_std_init"])
    specs = [("nominal", "standing", None, 0.0), ("nominal", "noise", None, 0.2)]
    specs += [("nominal", f"policy+{std:g}", policy, std) for std in stds]
    specs += [("x1.1", "standing", None, 0.0)] + [("x1.1", f"policy+{std:g}", policy, std) for std in stds]
    out = []
    for cond, case, pol, std in specs:
        env = nominal if cond == "nominal" else heavy
        for command in COMMANDS:
            for seed in SEEDS:
                out.append({"run": run, "checkpoint": ck.name, "cond": cond, "case": case, "command": command,
                            "seed": seed, "late_scale": late, **episode(env, pol, stats, seed, command, std)})
    return out


def margins_md(rows, key="cost_w1", stds=STDS, late=1.0) -> str:
    g = defaultdict(list)
    for r in rows:
        g[(r["cond"], r["command"], r["case"], r["run"])].append(r)
    runs = sorted({r["run"] for r in rows})
    m = lambda rs, k: st.mean(x[k] for x in rs)  # noqa: E731
    title = "the hottest drive priced (D29)" if key == "cost_w1" else "every drive priced (reference, not used)"
    stage = "at full scale (late scale 1)" if late == 1.0 else f"at late scale {late:g}"
    L = [f"# Reward margin over standing {stage}, by heat weight: {title} (steer_heat_weight.py)", "",
         f"Falls: {sum(r['fell'] for r in rows)} of {len(rows)} episodes. Per-step means over 3 seeds; margin = "
         "(case's base - w x cost at weight 1) - (standing's base - w x standing's cost), paired by seed; standing "
         f"pays no r_heat_peak; 'cost at weight 1' is at this late scale. Each cell: the worst of the {len(runs)} "
         f"policies' 3-seed means, +- the largest of their {len(runs)} standard errors, then all {len(runs)} means. "
         "'hottest drive' is the env's average (tau 4 s) at 20 s.", ""]
    for cond in ("nominal", "x1.1"):
        for cmd in COMMANDS:
            cases = [c for c in ["noise"] + [f"policy+{s:g}" for s in stds] if (cond, cmd, c, runs[0]) in g]
            L += [f"## {cond}, {cmd} m/s", "",
                  "| case | hottest drive at 20 s (4 policies) | " + " | ".join(f"w {w:g}" for w in WEIGHTS) + " |",
                  "|---" * (2 + len(WEIGHTS)) + "|"]
            for c in cases:
                hot = " ".join(f"{m(g[(cond, cmd, c, r)], 'hottest_at_end'):.2f}" for r in runs)
                cells = []
                for w in WEIGHTS:
                    ms, ses = [], []
                    for r in runs:
                        cs = sorted(g[(cond, cmd, c, r)], key=lambda x: x["seed"])
                        sd = sorted(g[(cond, cmd, "standing", r)], key=lambda x: x["seed"])
                        per = [(a["base"] - w * a[key]) - (b["base"] - w * b[key]) for a, b in zip(cs, sd)]
                        ms.append(st.mean(per))
                        ses.append(st.stdev(per) / len(per) ** 0.5)
                    cells.append(f"{min(ms):+.2f} +-{max(ses):.2f} ({' '.join(f'{x:+.2f}' for x in ms)})")
                L.append(f"| {c} | {hot} | " + " | ".join(cells) + " |")
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stds", default=",".join(f"{x:g}" for x in STDS), help="action-noise stds, comma-separated")
    ap.add_argument("--late", type=float, default=1.0, help="late scale (train.py: 0 until 1 M, 1 from 3 M)")
    ap.add_argument("--runs", default="fore128-s0,fore128-s1,fore128-s2,fore128-s3", help="run folders in --runs-dir")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    a = ap.parse_args()
    stds = tuple(float(x) for x in a.stds.split(","))
    labels = [f"{x:g}" for x in stds]
    if not 0.5 <= a.late <= 1.0:
        raise SystemExit("--late must lie in [0.5, 1] (train.py's stages from 2 M steps on, where the penalty scale is 1)")
    if any(not math.isfinite(x) or x < 0 for x in stds) or len(set(labels)) != len(labels):
        raise SystemExit("--stds must be finite, >= 0 and distinct (after formatting with :g)")
    runs = a.runs.split(",")
    with Pool(a.workers) as p:
        rows = [x for chunk in p.map(job, [(a.runs_dir, r, stds, a.late) for r in runs]) for x in chunk]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"heat_weight_episodes{a.tag}.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / f"heat_weight_margins{a.tag}.md").write_text(margins_md(rows, stds=stds, late=a.late) + "\n")
    (out / f"heat_weight_margins_every_drive{a.tag}.md").write_text(
        margins_md(rows, "cost_sum_w1", stds=stds, late=a.late) + "\n")
    print(f"{len(rows)} episodes, {sum(r['fell'] for r in rows)} falls; wrote {out}")


if __name__ == "__main__":
    main()
