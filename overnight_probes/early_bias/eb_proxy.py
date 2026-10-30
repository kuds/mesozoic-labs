"""EARLY-BIAS PROBE: is the critic's tail-angle sensitivity a stand-in for episode phase?  (no training)

One on-policy rollout per checkpoint (as eb_update.py builds it; seeds 9501, 9502).  Per row: the raw tail_3 angle,
the episode step, the critic's V(s), its local derivative dV/d(tail_3 angle), and the realised gamma = 0.98 discounted
return-to-go G (rows >= 200 steps from a time limit or the rollout end).  Reports, in normalised-return units per
degree of tail angle:
  - corr(tail angle, episode step) and corr(V, episode step) over the rows;
  - OLS slope of V on the tail angle, alone and with episode-step bins (20 bins of 50 steps) as controls;
  - the same two slopes for the realised G (what V should track);
  - the mean local derivative dV/dq3 (the quantity GAE's bootstrap uses when an action moves the tail).
Correlational only.  Writes eb_proxy.txt / .json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
sys.path.insert(0, str(HERE))
import eb_common as E  # noqa: E402

CK = [("s42", 1048576), ("s42", 2097152), ("s42", 3145728), ("s80", 1048576), ("s80", 2097152), ("s80", 3145728),
      ("s60", 5242880), ("s60", 6291456)]
SEEDS = (9501, 9502)


def ols_slope(y, x, controls=None):
    cols = [np.ones_like(x), x]
    if controls is not None:
        cols += [controls[:, j] for j in range(controls.shape[1])]
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def main():
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger

    out, lines = {}, []
    for run, N in CK:
        pr = E.probe(run)
        acc = {k: [] for k in ("tail", "tep", "V", "dV", "G", "keep")}
        for seed in SEEDS:
            b = E.build(run, N, env_seed=seed)
            E.set_probe_schedule(b, pr)
            m, vn, pol = b.model, b.vn, b.model.policy
            vn.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            m.set_logger(Logger(folder=None, output_formats=[]))
            m._last_obs = None

            class Rec(BaseCallback):
                def __init__(self):
                    super().__init__(0)
                    self.raw, self.tep, self.r, self.d, self.tr = [], [], [], [], []

                def _on_rollout_start(self):
                    self.prev = vn.get_original_obs().copy()
                    self.t = np.zeros(vn.num_envs, dtype=np.int64)

                def _on_step(self):
                    L = self.locals
                    dn = np.asarray(L["dones"]).copy()
                    self.raw.append(self.prev.copy())
                    self.tep.append(self.t.copy())
                    self.r.append(np.asarray(vn.get_original_reward(), dtype=np.float64).copy())
                    self.d.append(dn)
                    self.tr.append(np.array([bool(i.get("TimeLimit.truncated", False)) for i in L["infos"]]))
                    self.prev = vn.get_original_obs().copy()
                    self.t = np.where(dn, 0, self.t + 1)
                    return True

            rec = Rec()
            _t, cb = m._setup_learn(m.n_steps * m.n_envs, rec, False, "eb", False)
            assert m.collect_rollouts(m.env, cb, m.rollout_buffer, n_rollout_steps=m.n_steps)
            raw = np.array(rec.raw)  # (nS, nE, 63)
            tep = np.array(rec.tep)
            rew = np.array(rec.r) / math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon))
            dn = np.array(rec.d)
            tr = np.array(rec.tr)
            nS, nE = rew.shape
            G = np.zeros((nS, nE))
            cut = np.zeros((nS, nE))
            nxt = np.zeros(nE)
            nc = np.zeros(nE)
            for s in reversed(range(nS)):
                last = s == nS - 1
                G[s] = rew[s] + 0.98 * (~dn[s]) * (0.0 if last else nxt)
                nc = np.where(tr[s] | last, 0.0, np.where(dn[s] & ~tr[s], 1e9, nc + 1.0))
                cut[s] = nc
                nxt = G[s]
            x = raw.reshape(-1, 63).astype(np.float32)
            z = torch.as_tensor(np.asarray(vn.normalize_obs(x), dtype=np.float32)).requires_grad_(True)
            v = pol.predict_values(z)
            g = torch.autograd.grad(v.sum(), z)[0][:, 6].numpy()
            scale = 1.0 / math.sqrt(float(vn.obs_rms.var[6]) + float(vn.epsilon))
            acc["tail"].append(np.degrees(x[:, 6]))
            acc["tep"].append(tep.ravel())
            acc["V"].append(v.detach().numpy().ravel())
            acc["dV"].append(g * scale * math.pi / 180.0)
            acc["G"].append(G.ravel())
            acc["keep"].append(cut.ravel() >= 200)
        a = {k: np.concatenate(v) for k, v in acc.items()}
        keep = a["keep"]
        bins = np.minimum(a["tep"] // 50, 19)
        C = np.stack([(bins == j).astype(float) for j in range(1, 20)], 1)
        res = {
            "n": int(len(a["tail"])), "n_keep": int(keep.sum()),
            "tail_deg_mean": float(a["tail"].mean()), "tail_deg_sd": float(a["tail"].std()),
            "corr_tail_tep": float(np.corrcoef(a["tail"], a["tep"])[0, 1]),
            "corr_V_tep": float(np.corrcoef(a["V"], a["tep"])[0, 1]),
            "V_on_tail": ols_slope(a["V"], a["tail"]),
            "V_on_tail_ctrl_phase": ols_slope(a["V"], a["tail"], C),
            "G_on_tail": ols_slope(a["G"][keep], a["tail"][keep]),
            "G_on_tail_ctrl_phase": ols_slope(a["G"][keep], a["tail"][keep], C[keep]),
            "V_on_tail_keep_ctrl_phase": ols_slope(a["V"][keep], a["tail"][keep], C[keep]),
            "dV_dq3_local": float(a["dV"].mean()),
            "tail_by_phase": [float(a["tail"][bins == j].mean()) if (bins == j).any() else None for j in range(20)],
            "V_by_phase": [float(a["V"][bins == j].mean()) if (bins == j).any() else None for j in range(20)],
        }
        out[f"{run}_{N}"] = res
        lines.append(f"{run} {N:>8d} (n {res['n']}): corr(tail, ep step) {res['corr_tail_tep']:+.2f}; corr(V, ep step) "
                     f"{res['corr_V_tep']:+.2f}; per degree of tail angle: V slope {res['V_on_tail']:+.4f} (phase-controlled "
                     f"{res['V_on_tail_ctrl_phase']:+.4f}), realised G slope {res['G_on_tail']:+.4f} (phase-controlled "
                     f"{res['G_on_tail_ctrl_phase']:+.4f}; V on the same rows {res['V_on_tail_keep_ctrl_phase']:+.4f}); local "
                     f"dV/dq3 {res['dV_dq3_local']:+.4f}; tail by phase (deg) 0-50 {res['tail_by_phase'][0]:+.2f}, 200-250 "
                     f"{res['tail_by_phase'][4]:+.2f}, 950-1000 {res['tail_by_phase'][19]:+.2f}")
        print(lines[-1], flush=True)
    (HERE / "eb_proxy.json").write_text(json.dumps(out, indent=1))
    (HERE / "eb_proxy.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
