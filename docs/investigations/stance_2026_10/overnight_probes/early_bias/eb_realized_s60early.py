"""EARLY-BIAS PROBE: the REALISED change of the deterministic tail_3 mean on a checkpoint's own on-policy states.

For each checkpoint N and the next local checkpoint N2 (F2 copies), collect stochastic on-policy states with the
checkpoint N policy exactly as eb_update.py does (frozen statistics, n_envs 4 x 2048 steps, seeds SEED + rank), then
evaluate the raw observations under
    mu14(theta_N,  stats_N)          the checkpoint itself,
    mu14(theta_N2, stats_N)          the later weights, the checkpoint's normalisation (weights' share),
    mu14(theta_N2, stats_N2)         the later checkpoint as it would act on these states,
and report the means per update (divided by (N2 - N) / 8192).  Also on the probe bank.  No training.
Writes eb_realized_s60early.json / .txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
sys.path.insert(0, str(HERE))
import eb_common as E  # noqa: E402

PAIRS = [("s60", 1048576, 2097152), ("s60", 2097152, 3145728), ("s60", 3145728, 4194304), ("s60", 4194304, 5242880)]
SEEDS = (9301, 9302)


def main():
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger

    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid, cfg = E.load_code_config()

    def factory():
        return env_class(mid)(**cfg["env_kwargs"])

    out, lines = {}, []
    for run, N, N2 in PAIRS:
        pr = E.probe(run)
        z2, p2 = E.ckpt_paths(run, N2)
        m2, norm2, _ = load_sb3_checkpoint(str(z2), str(p2), factory)
        res = {"seeds": list(SEEDS)}
        raws = []
        for seed in SEEDS:
            b = E.build(run, N, env_seed=seed)
            E.set_probe_schedule(b, pr)
            m, vn = b.model, b.vn
            vn.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            m.set_logger(Logger(folder=None, output_formats=[]))
            m._last_obs = None

            class Rec(BaseCallback):
                def __init__(self):
                    super().__init__(0)
                    self.raw = []

                def _on_rollout_start(self):
                    self.prev = vn.get_original_obs().copy()

                def _on_step(self):
                    self.raw.append(self.prev.copy())
                    self.prev = vn.get_original_obs().copy()
                    return True

            rec = Rec()
            _t, cb = m._setup_learn(m.n_steps * m.n_envs, rec, False, "eb", False)
            assert m.collect_rollouts(m.env, cb, m.rollout_buffer, n_rollout_steps=m.n_steps)
            raws.append(np.concatenate(rec.raw, 0))
            n1 = b.vn
            pol1 = m.policy
        raw = np.concatenate(raws, 0).astype(np.float32)
        bank = pr["bank_raw_obs"].astype(np.float32)

        def mu(pol, norm, x):
            with torch.no_grad():
                return pol.get_distribution(torch.as_tensor(np.asarray(norm.normalize_obs(x), dtype=np.float32))).mode().numpy()[:, 14]

        upd = (N2 - N) / 8192
        for name, x in (("rollout_states", raw), ("bank", bank), ("statue_settled", bank[[0, 5]])):
            a0 = mu(pol1, n1, x)
            aw = mu(m2.policy, n1, x)
            a2 = mu(m2.policy, norm2, x)
            res[name] = {"n": int(len(x)), "mu_N": float(a0.mean()), "mu_N2_statsN": float(aw.mean()), "mu_N2": float(a2.mean()),
                         "weights_per_update": float((aw - a0).mean() / upd), "total_per_update": float((a2 - a0).mean() / upd),
                         "weights_share_rows_positive": float(((aw - a0) > 0).mean())}
        res["updates"] = upd
        # the critic's and the policy's sensitivity to the observed tail_3 angle (obs 6), per degree, at these states
        import math as _m

        def sens(pol, norm, x):
            z = torch.as_tensor(np.asarray(norm.normalize_obs(x), dtype=np.float32), dtype=torch.float32).requires_grad_(True)
            v = pol.predict_values(z).sum()
            gv = torch.autograd.grad(v, z)[0][:, 6].numpy()
            z2 = torch.as_tensor(np.asarray(norm.normalize_obs(x), dtype=np.float32), dtype=torch.float32).requires_grad_(True)
            mu = pol.get_distribution(z2).mode()[:, 14].sum()
            gm = torch.autograd.grad(mu, z2)[0][:, 6].numpy()
            scale = 1.0 / _m.sqrt(float(norm.obs_rms.var[6]) + float(norm.epsilon))
            clipped = np.abs(np.asarray(norm.normalize_obs(x))[:, 6]) >= float(norm.clip_obs)
            k = _m.pi / 180.0
            return {"dV_dq3_per_deg": float((gv * scale * k)[~clipped].mean()),
                    "dV_dq3_per_deg_sd": float((gv * scale * k)[~clipped].std()),
                    "dmu14_dq3_per_deg": float((gm * scale * k)[~clipped].mean()),
                    "loop_gain_G_5p16": float((gm * scale * k)[~clipped].mean() * 5.16),
                    "V_mean": float(pol.predict_values(torch.as_tensor(np.asarray(norm.normalize_obs(x), dtype=np.float32))).detach().numpy().mean()),
                    "tail3_deg_mean": float(np.degrees(x[:, 6]).mean()), "n_clipped": int(clipped.sum())}

        res["sens_N"] = sens(pol1, n1, raw)
        res["sens_N2_on_N_states"] = sens(m2.policy, norm2, raw)
        out[f"{run}_{N}_{N2}"] = res
        r = res["rollout_states"]
        lines.append(f"{run} {N:>8d} -> {N2:>8d} ({upd:.0f} updates): on its own on-policy states (n {r['n']}): mu14 {r['mu_N']:+.4f} -> "
                     f"{r['mu_N2']:+.4f}; weights' share per update {r['weights_per_update']:+.5f} (rows up {r['weights_share_rows_positive']:.2f}), "
                     f"total {r['total_per_update']:+.5f} | bank weights {res['bank']['weights_per_update']:+.5f} | statue_settled "
                     f"weights {res['statue_settled']['weights_per_update']:+.5f}")
        sN = res["sens_N"]
        s2 = res["sens_N2_on_N_states"]
        lines.append(f"      critic dV/d(tail_3 angle) per degree at these states: theta_N {sN['dV_dq3_per_deg']:+.5f} (SD over rows "
                     f"{sN['dV_dq3_per_deg_sd']:.5f}), theta_N2 {s2['dV_dq3_per_deg']:+.5f}; policy loop slope x 5.16 (G) theta_N "
                     f"{sN['loop_gain_G_5p16']:+.3f}, theta_N2 {s2['loop_gain_G_5p16']:+.3f}; mean V {sN['V_mean']:.3f}; tail angle "
                     f"{sN['tail3_deg_mean']:+.2f} deg")
        print(lines[-2], flush=True)
        print(lines[-1], flush=True)
    (HERE / "eb_realized_s60early.json").write_text(json.dumps(out, indent=1))
    (HERE / "eb_realized_s60early.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
