"""Realised weights-only change of the deterministic tail_3 mean on a checkpoint's OWN on-policy states (no training).

For run in (s42, s80) and the checkpoints 1,048,576 / 1,310,720 (Drive copy) / 2,097,152: collect stochastic on-policy
raw observations with checkpoint A (frozen statistics, 4 envs x 2048 steps, seeds as given), then evaluate
mu14(theta_B, stats_A) - mu14(theta_A, stats_A) for later checkpoints B.  Also the probe bank.  Writes vb_realized.txt/json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

V = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
STEPS = (1048576, 1310720, 2097152)
SEEDS = (7501, 7502)


def paths(run, st):
    if st == 1310720:
        return V / "ckpt" / f"{run}_stage1_1310720_steps.zip", V / "ckpt" / f"{run}_stage1_vecnormalize_1310720_steps.pkl"
    m = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "models"
    return m / f"stage1_{st}_steps.zip.copy1", m / f"stage1_vecnormalize_{st}_steps.pkl.copy1"


def main():
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    kw = dict(load_stage_config(mid, "stance")["env_kwargs"])
    cls = env_class(mid)
    co = {"learning_rate": 0.0, "lr_schedule": (lambda _: 0.0), "clip_range": (lambda _: 0.2)}
    out, lines = {}, []
    for run in ("s42", "s80"):
        pol, nrm = {}, {}
        for st in STEPS:
            z, p = paths(run, st)
            pol[st] = PPO.load(str(z), device="cpu", custom_objects=co).policy
            with open(p, "rb") as fh:
                nrm[st] = pickle.load(fh)
        for ia, A in enumerate(STEPS[:-1]):
            raws = []
            for seed in SEEDS:
                def mk(rank):
                    def _f():
                        set_random_seed(seed + rank)
                        e = Monitor(cls(**kw))
                        e.reset(seed=seed + rank)
                        return e
                    return _f
                venv = DummyVecEnv([mk(i) for i in range(4)])
                z, p = paths(run, A)
                vn = VecNormalize.load(str(p), venv)
                vn.training = False
                vn.seed(seed)
                set_random_seed(seed)
                obs = vn.reset()
                rr = []
                policy = pol[A]
                for _ in range(2048):
                    rr.append(vn.get_original_obs().copy())
                    with torch.no_grad():
                        a, _v, _lp = policy(torch.as_tensor(obs, dtype=torch.float32), deterministic=False)
                    a = np.clip(a.numpy(), -1, 1)
                    obs, _r, _d, _i = vn.step(a)
                raws.append(np.concatenate(rr, 0))
                vn.close()
            raw = np.concatenate(raws, 0).astype(np.float32)

            def mu(w, s, x):
                with torch.no_grad():
                    return pol[w].get_distribution(torch.as_tensor(nrm[s].normalize_obs(x), dtype=torch.float32)).mode().numpy()[:, 14]

            m0 = mu(A, A, raw)
            rec = {"n": int(len(raw)), "mu_A": float(m0.mean()), "tail_deg": float(np.degrees(raw[:, 6]).mean())}
            for B in STEPS[ia + 1:]:
                upd = (B - A) / 8192
                mw = mu(B, A, raw)
                mb = mu(B, B, raw)
                rec[str(B)] = {"weights_only": float(mw.mean()), "actual": float(mb.mean()), "w_per_update": float((mw - m0).mean() / upd),
                               "rows_up": float(((mw - m0) > 0).mean())}
                lines.append(f"{run} own {A:,} states (n {len(raw)}, tail {rec['tail_deg']:+.2f} deg): mu14 {m0.mean():+.4f} -> theta_{B:,} "
                             f"(stats_A) {mw.mean():+.4f} [per update {(mw - m0).mean() / upd:+.5f}, rows up {((mw - m0) > 0).mean():.2f}]; "
                             f"with stats_B {mb.mean():+.4f}")
                print(lines[-1], flush=True)
            out[f"{run}_{A}"] = rec
    (V / "vb_realized.json").write_text(json.dumps(out, indent=1))
    (V / "vb_realized.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
