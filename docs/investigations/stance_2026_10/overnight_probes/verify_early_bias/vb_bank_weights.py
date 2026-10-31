"""Weights' share of the change of the deterministic tail_3 mean on the probe bank (no simulation, no training):
mu14(theta_B, stats_A) - mu14(theta_A, stats_A) for checkpoint pairs, s42 and s80, including the 1,310,720 Drive copies."""
import json, sys, pickle
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
V = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
torch.set_num_threads(1)
def paths(run, st):
    if st == 1310720:
        return V / "ckpt" / f"{run}_stage1_1310720_steps.zip", V / "ckpt" / f"{run}_stage1_vecnormalize_1310720_steps.pkl"
    m = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "models"
    return m / f"stage1_{st}_steps.zip.copy1", m / f"stage1_vecnormalize_{st}_steps.pkl.copy1"
lines = []
for run in ("s42", "s80"):
    pr = np.load(F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "policy_probe.npz.copy1")
    bank = pr["bank_raw_obs"].astype(np.float32)
    ss = [i for i, l in enumerate(pr["bank_labels"]) if l == "statue_settled"]
    pol, nrm = {}, {}
    for st in (1048576, 1310720, 2097152):
        z, p = paths(run, st)
        pol[st] = PPO.load(str(z), device="cpu", custom_objects={"learning_rate": 0.0, "lr_schedule": (lambda _: 0.0), "clip_range": (lambda _: 0.2)}).policy
        with open(p, "rb") as fh:
            nrm[st] = pickle.load(fh)
    def mu(st_w, st_n):
        with torch.no_grad():
            return pol[st_w].get_distribution(torch.as_tensor(nrm[st_n].normalize_obs(bank), dtype=torch.float32)).mode().numpy()[:, 14]
    for a, b in ((1048576, 1310720), (1310720, 2097152), (1048576, 2097152)):
        upd = (b - a) / 8192
        m0 = mu(a, a); mw = mu(b, a); mb = mu(b, b)
        lines.append(f"{run} {a:>8d}->{b:>8d} ({upd:.0f} upd) bank-mean mu14 {m0.mean():+.4f} -> weights-only {mw.mean():+.4f} "
                     f"(per upd {(mw-m0).mean()/upd:+.5f}) -> actual {mb.mean():+.4f}; statue_settled {m0[ss].mean():+.4f} -> "
                     f"weights-only {mw[ss].mean():+.4f} (per upd {(mw-m0)[ss].mean()/upd:+.5f}) -> actual {mb[ss].mean():+.4f}")
        print(lines[-1], flush=True)
(V / "vb_bank_weights.txt").write_text("\n".join(lines) + "\n")
