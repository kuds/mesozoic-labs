"""Walking pilot 3's report-only hidden-clock readout: V - G by episode-step bucket (criteria_pilot3.txt).

Scratch, never merged; REPORT ONLY: never part of the outcome, the existence reading or Q4. Built at the FIX2 stage
(2026-11-03), before any pilot-3 training step, and pinned by sha256 before launch.

What it measures (the criteria's "Hidden-clock audit"): at one checkpoint pair, 32 on-policy STOCHASTIC episodes of
the T. rex locomotion stage under the pair's FROZEN statistics (a 4-env DummyVecEnv, the first 8 complete episodes of
each env; envs seeded seed_base + rank, torch and numpy seeded seed_base; seed_base 6271, the PC round's V - G tool's),
actions as SB3 2.9.0's collect_rollouts takes them (the policy's forward on the normalised observation, sampled,
clipped to the action space). For every state s_k (k = the env's own _step_count when the observation was made, 0 at
reset; the time-limit terminal observation is not a state):
  V_k = the stock critic's value of the normalised observation (policy.predict_values);
  G_k = sum_{j=k}^{T-1} gamma^(j-k) r~_j + [time-limit truncation] gamma^(T-k) V(s_T), no bootstrap after a true
        termination; gamma = the stage's [ppo] gamma (0.995), checked equal to the model's and the sidecar's; r~ = the
        reward as the training buffer holds it: VecNormalize.normalize_reward with the pair's frozen ret_rms.
G is computed under the env as create_vec_env builds it, so with the stage's final reward weights (forward_vel_weight
2.0; no reward-ramp callback runs here). Training's reward equals it from ramp_timesteps (2,000,000) on, so at the
four judged checkpoints V and G share one reward; at a checkpoint before 2,000,000 (the smoke's 65,536, trained at
ramped weights 0.20-0.26) V - G mixes two rewards and is read only roughly (FIX3, by code reading of
RewardRampCallback and locomotion.toml).
Buckets by k: [0,100) [100,200) [200,500) [500,800) [800,1000]. Per bucket: the episodes contributing, the states, the
mean of V - G and of |V - G| over episodes (each episode contributes its mean over the bucket's steps; an episode that
ends before a bucket contributes nothing to it) with the SE across episodes (sample SD / sqrt(n)), mean V and mean G
(these two pooled over the bucket's states, not weighted by episode, so mean V - mean G need not equal the mean of
V - G; FIX3 note).
A bucket is printed "step-localised" when, against every neighbour it has (one for [0,100) and for [800,1000], two
otherwise), the mean of the per-episode differences (bucket minus neighbour, over the episodes that reach both)
exceeds 4 times its SE (their sample SD over the square root of their number) in absolute value; with fewer than 2
such episodes the comparison is "n/a" and the bucket is not flagged. The expectation is no step-localised error.
The loop gain and the critic's tail_3 slope (the PC round's ported critic tools) are printed "not run": this tool does
not run those tools, and whether they run on the locomotion env unchanged was not tried.
Self-checks (each a PROBLEM, exit 3, the report still written): every truncation is at step 1,000; k runs 0 ... T-1
within each episode; the loop's own forward values equal the recomputed V (max |dV| <= 1e-4); gamma agrees.

    cd TREE && ISO_WT=TREE python3 -I TOOLS/iso.py TOOLS/pilot3_vg.py --cell OUT/s42/8000000 --json VG/s42_8000000.json
    cd TREE && ISO_WT=TREE python3 -I TOOLS/iso.py TOOLS/pilot3_vg.py --checkpoint ZIP --vecnorm PKL --json X.json
--cell (the judged cell, pilot3_judge.py's OUT/s<seed>/<steps>) reads the pair from its binding.json and refuses
(exit 2, nothing written) when this tree is not clean or not at the cell's head, or a file is no longer the judged
one (cell_problems); --checkpoint/--vecnorm reads any locomotion pair (a check of the tool). One simulation process;
no training. FIX3: an existing --json target is refused (exit 2, nothing written); the per-episode V, G and
bootstrap step is episode_rows and the seeding seed_all (both unchanged in what they compute), so tests reach them;
the record's "seeding" holds the seeds as set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCHEMA = "walking-pilot3-vg/v1"
BUCKETS = ((0, 100), (100, 200), (200, 500), (500, 800), (800, 1001))
BUCKET_NAMES = ("[0,100)", "[100,200)", "[200,500)", "[500,800)", "[800,1000]")
HORIZON = 1000
GAMMA = 0.995
N_ENVS = 4
EPISODES_PER_ENV = 8
SEED_BASE = 6271
LOCALISED_SE = 4.0
DV_TOL = 1e-4
NOT_RUN_BESIDE = ("loop gain and the critic's tail_3 slope: not run (this tool does not run the PC round's ported "
                  "critic tools; whether they run on the locomotion env unchanged was not tried)")


class Problem(Exception):
    pass


def sha256_file(path: "str | Path") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def bucket_of(k: int) -> int:
    """Index into BUCKETS of a state's step counter k (0 <= k <= 1000; 1000 belongs to [800,1000])."""
    if isinstance(k, bool) or int(k) != k:
        raise ValueError(f"k {k!r} is not an integer")
    k = int(k)
    if k < 0 or k > HORIZON:
        raise ValueError(f"k {k} outside [0, {HORIZON}]")
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= k < hi:
            return i
    raise AssertionError("unreachable")


def returns_to_go(r_norm, gamma: float, bootstrap: "float | None" = None) -> np.ndarray:
    """G_t = sum_{j>=t} gamma^(j-t) r_j (+ gamma^(T-t) bootstrap at a time-limit truncation; None after a true
    termination), float64."""
    r = np.asarray(r_norm, dtype=np.float64).reshape(-1)
    G = np.empty_like(r)
    acc = 0.0 if bootstrap is None else float(bootstrap)
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + float(gamma) * acc
        G[t] = acc
    return G


def _sd(xs) -> "float | None":
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else None


def episode_bucket_means(ep: dict) -> dict:
    """{bucket index: (mean V - G, mean |V - G|, n states)} of one episode {"k", "V", "G"}."""
    k = np.asarray(ep["k"], dtype=np.int64)
    e = np.asarray(ep["V"], dtype=np.float64) - np.asarray(ep["G"], dtype=np.float64)
    bi = np.array([bucket_of(int(x)) for x in k], dtype=np.int64)
    out = {}
    for b in range(len(BUCKETS)):
        sel = bi == b
        if sel.any():
            out[b] = (float(e[sel].mean()), float(np.abs(e[sel]).mean()), int(sel.sum()))
    return out


def bucket_stats(episodes: list) -> list:
    """Per bucket: n_episodes, n_states, mean_err (SE), mean_abs_err (SE), mean_V, mean_G, over the episodes."""
    per_ep = [episode_bucket_means(ep) for ep in episodes]
    rows = []
    for b, name in enumerate(BUCKET_NAMES):
        errs = [m[b][0] for m in per_ep if b in m]
        abss = [m[b][1] for m in per_ep if b in m]
        n_states = sum(m[b][2] for m in per_ep if b in m)
        vs, gs = [], []
        for ep in episodes:
            k = np.asarray(ep["k"], dtype=np.int64)
            sel = np.array([bucket_of(int(x)) == b for x in k], dtype=bool)
            vs.append(np.asarray(ep["V"], dtype=np.float64)[sel])
            gs.append(np.asarray(ep["G"], dtype=np.float64)[sel])
        n = len(errs)
        row = {"bucket": name, "n_episodes": n, "n_states": int(n_states)}
        if n:
            sd_e, sd_a = _sd(errs), _sd(abss)
            row.update(mean_err=float(np.mean(errs)), se_err=None if sd_e is None else sd_e / math.sqrt(n),
                       mean_abs_err=float(np.mean(abss)), se_abs_err=None if sd_a is None else sd_a / math.sqrt(n),
                       mean_V=float(np.concatenate(vs).mean()), mean_G=float(np.concatenate(gs).mean()))
        else:
            row.update(mean_err=None, se_err=None, mean_abs_err=None, se_abs_err=None, mean_V=None, mean_G=None)
        rows.append(row)
    return rows


def localisation(episodes: list, factor: float = LOCALISED_SE) -> list:
    """Per bucket, against each neighbour: the paired per-episode difference (bucket minus neighbour) over the
    episodes that reach both, its mean, SE and whether |mean| > factor * SE; "step_localised" when that holds
    against every neighbour it has (a comparison with fewer than 2 episodes is n/a and flags nothing)."""
    per_ep = [episode_bucket_means(ep) for ep in episodes]
    out = []
    for b, name in enumerate(BUCKET_NAMES):
        comps = []
        for nb in (b - 1, b + 1):
            if not 0 <= nb < len(BUCKETS):
                continue
            d = [m[b][0] - m[nb][0] for m in per_ep if b in m and nb in m]
            sd = _sd(d)
            if len(d) < 2 or sd is None:
                comps.append({"neighbour": BUCKET_NAMES[nb], "n": len(d), "mean_diff": float(np.mean(d)) if d else None,
                              "se": None, "over": None})
                continue
            se = sd / math.sqrt(len(d))
            mean = float(np.mean(d))
            comps.append({"neighbour": BUCKET_NAMES[nb], "n": len(d), "mean_diff": mean, "se": se,
                          "over": bool(abs(mean) > factor * se)})
        flagged = bool(comps) and all(c["over"] is True for c in comps)
        out.append({"bucket": name, "comparisons": comps, "step_localised": flagged})
    return out


def episode_problems(episodes: list) -> list:
    """Self-checks on collected episodes: k = 0 ... T-1; every truncation at T = 1,000."""
    problems = []
    for i, ep in enumerate(episodes):
        k = np.asarray(ep["k"], dtype=np.int64)
        if not np.array_equal(k, np.arange(len(k))):
            problems.append(f"episode {i}: step counters are not 0 ... {len(k) - 1}")
        if ep.get("trunc") and len(k) != HORIZON:
            problems.append(f"episode {i}: truncated at {len(k)} steps, not {HORIZON}")
    return problems


# ---------------------------------------------------------------- the rollout (needs the tree)

def build(zip_path: str, pkl_path: str, seed: int):
    """The locomotion stage's 4-env DummyVecEnv (Monitor-wrapped, as training builds it), the pair's VecNormalize
    frozen (training False, norm_reward True), the stock PPO model; vn, torch and numpy seeded seed."""
    import copy

    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.config import load_stage_config
    from environments.shared.policy_loading import load_sb3_model
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.train_base import create_vec_env

    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    cfg = copy.deepcopy(load_stage_config(mid, 2))
    gamma = float(cfg["ppo_kwargs"]["gamma"])
    venv = create_vec_env(get_species_config(mid), {2: {"env_kwargs": dict(cfg["env_kwargs"])}}, 2, N_ENVS, seed=seed,
                          use_subproc=False, algorithm="ppo", gamma=gamma)
    vn = VecNormalize.load(str(pkl_path), venv.venv)
    vn.training = False
    vn.norm_reward = True
    model = load_sb3_model(str(zip_path), algorithm="ppo", env=vn, device="cpu")
    seeding = seed_all(vn, seed)
    return cfg, gamma, vn, model, seeding


def seed_all(vn, seed: int) -> dict:
    """Seed the envs (vn.seed: env rank r gets seed + r), torch and numpy with seed; return the seeds as set."""
    import torch

    vn.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return {"vecenv_seed": int(seed), "torch_initial_seed": int(torch.initial_seed()),
            "numpy_mt19937_key0": int(np.random.get_state()[1][0])}


def collect(vn, model, n_per_env: int) -> list:
    """Complete episodes per env (the first n_per_env of each): SB3 2.9.0 collect_rollouts' action path with the
    inner DummyVecEnv stepped and VecNormalize's own normalize_obs / normalize_reward applied (its step_wait with
    training False), so the raw terminal observation is kept for the bootstrap."""
    import torch

    policy = model.policy
    policy.set_training_mode(False)
    low, high = model.action_space.low, model.action_space.high
    envs = [e.unwrapped for e in vn.venv.envs]
    n = vn.num_envs
    raw = vn.venv.reset()
    cur = [{"k": [], "X": [], "r_norm": [], "V_loop": []} for _ in range(n)]
    done: list = [[] for _ in range(n)]
    while min(len(d) for d in done) < n_per_env:
        ks = [int(u._step_count) for u in envs]
        z = vn.normalize_obs(raw)
        with torch.no_grad():
            actions, values, _logp = policy(torch.as_tensor(z))
        actions = actions.cpu().numpy()
        new_raw, rew, dones, infos = vn.venv.step(np.clip(actions, low, high))
        rn = np.asarray(vn.normalize_reward(np.asarray(rew, dtype=np.float32)), dtype=np.float32)
        for i in range(n):
            c = cur[i]
            c["k"].append(ks[i])
            c["X"].append(np.array(raw[i], dtype=np.float32, copy=True))
            c["r_norm"].append(np.float32(rn[i]))
            c["V_loop"].append(float(values[i, 0]))
            if dones[i]:
                trunc = bool(infos[i].get("TimeLimit.truncated", False))
                tobs = infos[i].get("terminal_observation")
                if len(done[i]) < n_per_env:
                    done[i].append({"env": i, "k": np.asarray(c["k"], dtype=np.int64), "X": np.stack(c["X"]),
                                    "r_norm": np.asarray(c["r_norm"], dtype=np.float32),
                                    "V_loop": np.asarray(c["V_loop"], dtype=np.float64), "trunc": trunc,
                                    "term_X": None if tobs is None else np.array(tobs, dtype=np.float32, copy=True)})
                cur[i] = {"k": [], "X": [], "r_norm": [], "V_loop": []}
        raw = new_raw
    return [ep for d in done for ep in d]


def critic_values(policy, vn, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    import torch

    z = np.asarray(vn.normalize_obs(np.asarray(X, dtype=np.float32)), dtype=np.float32)
    out = np.empty(len(z), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(z), chunk):
            out[s:s + chunk] = policy.predict_values(torch.as_tensor(z[s:s + chunk]))[:, 0].numpy()
    return out


def episode_rows(eps: list, value_fn, gamma: float) -> tuple:
    """(rows, max |V - V_loop|, problems) of collected episodes: V = value_fn(X); G = returns_to_go of r_norm with the
    time-limit bootstrap value_fn(terminal observation) after a truncation and none after a termination (SB3 hands a
    terminal observation for both; only a truncation reads it)."""
    rows, max_dv, problems = [], 0.0, []
    for ep in eps:
        V = value_fn(ep["X"])
        max_dv = max(max_dv, float(np.max(np.abs(V - ep["V_loop"]))))
        boot = None
        if ep["trunc"]:
            if ep["term_X"] is None:
                problems.append(f"env {ep['env']}: a truncated episode without its terminal observation")
            else:
                boot = float(value_fn(ep["term_X"][None, :])[0])
        rows.append({"k": ep["k"], "V": V, "G": returns_to_go(ep["r_norm"], gamma, boot), "trunc": ep["trunc"]})
    return rows, max_dv, problems


def measure(zip_path: str, pkl_path: str, *, seed: int = SEED_BASE, n_per_env: int = EPISODES_PER_ENV) -> dict:
    cfg, gamma, vn, model, seeding = build(zip_path, pkl_path, seed)
    problems = []
    if not (gamma == float(model.gamma) == float(vn.gamma) == GAMMA):
        problems.append(f"gamma: stage {gamma}, model {model.gamma}, VecNormalize {vn.gamma} (expected {GAMMA})")
    eps = collect(vn, model, n_per_env)
    problems += episode_problems(eps)
    rows, max_dv, more = episode_rows(eps, lambda X: critic_values(model.policy, vn, X), gamma)
    problems += more
    if max_dv > DV_TOL:
        problems.append(f"the loop's forward values differ from the recomputed V by up to {max_dv:.3g} (> {DV_TOL})")
    units = {"gamma": gamma, "ret_var": float(np.asarray(vn.ret_rms.var).reshape(-1)[0]), "epsilon": float(vn.epsilon),
             "clip_reward": float(vn.clip_reward), "num_timesteps": int(model.num_timesteps)}
    vn.close()
    return {"episodes": len(rows), "truncated": sum(1 for r in rows if r["trunc"]),
            "terminated": sum(1 for r in rows if not r["trunc"]), "lengths": [int(len(r["k"])) for r in rows],
            "units": units, "max_abs_dV_loop": max_dv, "buckets": bucket_stats(rows),
            "localisation": localisation(rows), "problems": problems, "seeding": seeding}


def _f(x, fmt="+.4f") -> str:
    return "n/a" if x is None else format(x, fmt)


def render(rec: dict) -> list:
    lines = [f"V - G (report only) at {rec['label']}: {rec['episodes']} episodes ({rec['truncated']} truncated at "
             f"1000, {rec['terminated']} terminated), seed base {rec['settings']['seed_base']}, gamma "
             f"{rec['units']['gamma']}, num_timesteps {rec['units']['num_timesteps']:,}",
             "  bucket        n_ep  states  mean V-G (SE)          mean |V-G| (SE)        mean V    mean G   "
             "step-localised"]
    loc = {r["bucket"]: r for r in rec["localisation"]}
    for r in rec["buckets"]:
        lr = loc[r["bucket"]]
        cmp = "; ".join(f"vs {c['neighbour']} {_f(c['mean_diff'])} (SE {_f(c['se'], '.4f')}, n {c['n']})"
                        for c in lr["comparisons"])
        lines.append(f"  {r['bucket']:<12} {r['n_episodes']:>5} {r['n_states']:>7}  {_f(r['mean_err'])} "
                     f"({_f(r['se_err'], '.4f')})    {_f(r['mean_abs_err'], '.4f')} ({_f(r['se_abs_err'], '.4f')})"
                     f"    {_f(r['mean_V'], '.3f')}  {_f(r['mean_G'], '.3f')}   "
                     f"{'STEP-LOCALISED' if lr['step_localised'] else 'no'}  [{cmp}]")
    lines.append(f"  {NOT_RUN_BESIDE}")
    lines.append("  PROBLEMS: " + ("; ".join(rec["problems"]) if rec["problems"] else "none"))
    return lines


def cell_problems(binding: dict, *, head: str, dirty: str) -> list:
    """Every reason --cell refuses: the tree not at the cell's head, a pair file missing or no longer the judged one
    (its sha256), a tree with local edits (git status --porcelain --untracked-files=no)."""
    problems = []
    if binding.get("head") != head:
        problems.append(f"the cell was judged at {binding.get('head')}, this tree is {head}")
    for key in ("checkpoint", "vecnorm"):
        want = binding.get(f"{key}_sha256")
        p = Path(str(binding.get(key)))
        if not binding.get(key) or not p.is_file() or sha256_file(p) != want:
            problems.append(f"{p} is missing or no longer the judged file ({want})")
    if dirty:
        problems.append("this tree has local edits")
    return problems


def arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", default=None, help="the judged cell OUT/s<seed>/<steps> (its binding.json)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--episodes-per-env", type=int, default=EPISODES_PER_ENV)
    return ap


def main(argv=None) -> int:
    a = arg_parser().parse_args(argv)
    if bool(a.cell) == bool(a.checkpoint or a.vecnorm) or (a.checkpoint is None) != (a.vecnorm is None):
        print("refusing (nothing written): give --cell, or --checkpoint with --vecnorm", flush=True)
        return 2
    if Path(a.json).exists():  # FIX3: never overwrite (a mistyped path could name a cell's binding.json)
        print(f"refusing (nothing written): --json {a.json} exists (never overwritten)", flush=True)
        return 2
    import environments

    tree = Path(environments.__file__).resolve().parents[1]
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True, env=env).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=tree, text=True,
                                    env=env).strip()
    binding = None
    if a.cell:
        binding = json.loads((Path(a.cell) / "binding.json").read_text())
        ckpt, vec = Path(binding["checkpoint"]), Path(binding["vecnorm"])
        problems = cell_problems(binding, head=head, dirty=dirty)
        if problems:
            print("refusing (nothing written): " + "; ".join(problems), flush=True)
            return 2
        label = f"s{binding.get('seed')} {int(binding['steps']):,}" + (" (final-model fallback)"
                                                                       if binding.get("final_fallback") else "")
    else:
        ckpt, vec = Path(a.checkpoint), Path(a.vecnorm)
        label = ckpt.name
    rec = measure(str(ckpt), str(vec), seed=a.seed_base, n_per_env=a.episodes_per_env)
    rec.update(schema=SCHEMA, label=label, report_only=True, tool_sha256=sha256_file(__file__), tree=str(tree),
               head=head, tree_clean=not dirty, cell=a.cell, checkpoint=str(ckpt), checkpoint_sha256=sha256_file(ckpt),
               vecnorm=str(vec), vecnorm_sha256=sha256_file(vec),
               settings={"seed_base": a.seed_base, "n_envs": N_ENVS, "episodes_per_env": a.episodes_per_env,
                         "buckets": list(BUCKET_NAMES), "gamma": GAMMA, "localised_se_factor": LOCALISED_SE},
               beside=NOT_RUN_BESIDE)
    print("\n".join(render(rec)), flush=True)
    Path(a.json).write_text(json.dumps(rec, indent=1, default=str))
    return 3 if rec["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
