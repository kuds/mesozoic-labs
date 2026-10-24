"""verify_1 (independent re-roll, REPORT ONLY): the judge's joint-space noise test (judge_stance{4,5}.noise_matched,
space "joint", sigma = frozen control_sigma[seed]) with the noise vector n = sigma * z transformed per condition on the
same draws.  Counts falls (length < horizon).  The judge module is imported read-only; checkpoints are loaded only via
J.load_checkpoint (deterministic mean, VecNormalize stats applied).  Output: one JSON line per condition.

Conditions (n = sigma * z, all 15 drawn every step as in the judge):
  full          n                                        (must reproduce the judge file's lengths and terms)
  keep:<idx,..> n masked to those actuators
  hpA / hpS     k3roll's counterfactual: z_l = -z_r / z_l = z_r, hip pitch only (doubles one mode's variance)
  hpD / hpC     projection of the ACTUAL hip-pitch noise onto its differential / common mode, hip pitch only:
                d = (n_r - n_l)/2 -> (d, -d);  c = (n_r + n_l)/2 -> (c, c);  hpD + hpC = hip pitch noise exactly
  fullD / fullC full noise with the hip-pitch pair replaced by its differential / common projection
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
JUDGES = {"LTSB": SCRATCH + "/foot/stance5/judge/judge_stance5.py", "LTS": SCRATCH + "/foot/stance4/judge/judge_stance4.py"}
ROOTS = {"LTSB": SCRATCH + "/foot/stance5", "LTS": SCRATCH + "/foot/stance4"}
RHP, LHP = 3, 7
NAMED = {"tail": [11, 12, 13, 14], "neckhead": [0, 1, 2], "legs": [3, 4, 5, 6, 7, 8, 9, 10], "hip_pitch": [3, 7],
         "legs_no_hippitch": [4, 5, 6, 8, 9, 10], "knee": [5, 9], "legs_no_knee": [3, 4, 6, 7, 8, 10],
         "nohp": [i for i in range(15) if i not in (3, 7)], "noknee": [i for i in range(15) if i not in (5, 9)]}


def transform(cond, z, sigma):
    z = z.copy()
    if cond == "hpA":
        z[LHP] = -z[RHP]
    elif cond == "hpS":
        z[LHP] = z[RHP]
    n = sigma * z
    if cond in ("full",):
        return n
    if cond in NAMED:
        m = np.zeros(15)
        m[NAMED[cond]] = 1
        return n * m
    if cond in ("hpA", "hpS"):
        m = np.zeros(15)
        m[[RHP, LHP]] = 1
        return n * m
    d = 0.5 * (n[RHP] - n[LHP])
    c = 0.5 * (n[RHP] + n[LHP])
    if cond in ("hpD", "hpC"):
        out = np.zeros(15)
    elif cond in ("fullD", "fullC"):
        out = n.copy()
    else:
        raise SystemExit("unknown cond " + cond)
    if cond.endswith("D"):
        out[RHP], out[LHP] = d, -d
    else:
        out[RHP], out[LHP] = c, c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("LTSB", "LTS"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = "judge_stance5" if a.arm == "LTSB" else "judge_stance4"
    spec = importlib.util.spec_from_file_location(name, JUDGES[a.arm])
    J = importlib.util.module_from_spec(spec)
    sys.modules[name] = J
    spec.loader.exec_module(J)
    import torch
    from stable_baselines3.common.policies import BaseModel

    torch.set_num_threads(1)
    clone = J.ARMS[a.arm]["clone"]
    J.isolate(clone)
    git = J.git_state(clone)
    assert git["head"] == J.ARMS[a.arm]["head"] and not git["status"], git
    ctx = J.Ctx(a.arm)
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder

    seed = int(a.run.rsplit("_s", 1)[1])
    with open(J.FROZEN_PARAMS, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == J.FROZEN_PARAMS_SHA256
    sigma = np.asarray(J.read_json(J.FROZEN_PARAMS)["criteria"]["H4"]["control_sigma"][str(seed)]["vector"], dtype=np.float64)
    stage_dir = os.path.join(ROOTS[a.arm], "runs", a.run, "01_stance")
    for st in a.steps:
        steps = J.parse_steps(st)
        _, zp, vp = J.checkpoint_paths(os.path.join(stage_dir, "models"), steps)
        jd = J.read_json(os.path.join(ROOTS[a.arm], "judged", a.run, f"judge_{st}.json"))
        jn = jd["noise_matched_control_joint"]
        assert jd["model_sha256"] == "sha256:" + J.sha256_file(zp).split(":")[-1], "judge file is another checkpoint"
        assert np.array_equal(np.asarray(jn["sigma_vector"]), sigma)
        model, normalizer = J.load_checkpoint(ctx, zp, vp)
        pol = model.policy
        vmean, vvar = np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var)
        eps, clip_obs = normalizer.epsilon, normalizer.clip_obs
        for cond in a.conds:
            t0 = time.monotonic()
            env = ctx.env_class(**ctx.env_kwargs)
            low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
            rec = SubstepContactRecorder(env, Morphology.from_env(env, ctx.mid)).attach()
            lengths, terms = [], []
            try:
                for i in range(J.NOISE_EPISODES):
                    s = J.NOISE_SEED + i
                    rng = np.random.default_rng(s)
                    obs, _ = env.reset(seed=s)
                    nst = 0
                    while True:
                        o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
                        with torch.no_grad():
                            f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
                            mu = pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)
                        cmd = np.clip(mu, low, high)
                        z = rng.standard_normal(cmd.shape)
                        act = np.clip(cmd + transform(cond, z, sigma), low, high)
                        obs, r, term, trunc, info = env.step(act)
                        rec.end_step(act, float(r))
                        nst += 1
                        if term or trunc:
                            rec.end_episode(terminated=bool(term), truncated=bool(trunc),
                                            termination_reason=str(info.get("termination_reason", "")))
                            terms.append(str(info.get("termination_reason", "")))
                            break
                    lengths.append(nst)
            finally:
                rec.detach()
                env.close()
            seeds = list(range(J.NOISE_SEED, J.NOISE_SEED + J.NOISE_EPISODES))
            row = {"run": a.run, "steps": st, "cond": cond, "falls": sum(L < ctx.horizon for L in lengths),
                   "fall_seeds": [s for s, L in zip(seeds, lengths) if L < ctx.horizon], "lengths": lengths,
                   "terms": terms, "seconds": round(time.monotonic() - t0, 1)}
            if cond == "full":
                jl = [(r["seed"], r["length"], r["termination_reason"]) for r in jn["episodes_detail"]]
                row["matches_judge"] = jl == list(zip(seeds, lengths, terms))
                row["judge_falls"] = jn["falls"]
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in row if k not in ("lengths", "terms")}), flush=True)
        del model, normalizer


if __name__ == "__main__":
    main()
