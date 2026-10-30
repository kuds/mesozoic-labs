"""EARLY-BIAS PROBE: which loss component sets the gradient-norm clip (the only path by which the value and entropy
terms reach the tail_3 mean).  One rollout per checkpoint as eb_update.py builds it; per mini-batch of the first epoch
the norms of the full gradient and of each component's gradient (policy / entropy / value / mirror), and the norms
over the actor-mean parameters only.  No training.  Writes eb_gradnorm.txt."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
sys.path.insert(0, str(HERE))
import eb_common as E  # noqa: E402


def main():
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    lines = []
    for run, N in (("s42", 1048576), ("s80", 1048576), ("s42", 3145728), ("s60", 6291456)):
        seed = 9401
        pr = E.probe(run)
        b = E.build(run, N, env_seed=seed)
        E.set_probe_schedule(b, pr)
        m, vn, pol = b.model, b.vn, b.model.policy
        vn.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
        m.set_logger(Logger(folder=None, output_formats=[])); m._last_obs = None

        class Nop(BaseCallback):
            def _on_step(self):
                return True
        _t, cb = m._setup_learn(m.n_steps * m.n_envs, Nop(), False, "eb", False)
        assert m.collect_rollouts(m.env, cb, m.rollout_buffer, n_rollout_steps=m.n_steps)
        names = [n for n, _ in pol.named_parameters()]
        params = [p for _, p in pol.named_parameters()]
        actor = [i for i, n in enumerate(names) if n.startswith(("mlp_extractor.policy_net.", "action_net."))]
        mirror = m._mirror_begin_update()
        clip = float(m.clip_range(m._current_progress_remaining))
        rec = {k: [] for k in ("full", "pg", "ent", "vf", "kl", "pg_actor", "kl_actor", "c")}
        for data in m.rollout_buffer.get(m.batch_size):
            vals, lp, ent = pol.evaluate_actions(data.observations, data.actions)
            a = data.advantages
            an = (a - a.mean()) / (a.std() + 1e-8)
            r = torch.exp(lp - data.old_log_prob)
            pg = -torch.min(an * r, an * torch.clamp(r, 1 - clip, 1 + clip)).mean()
            vl = m._mirror_value_loss(mirror, data, F.mse_loss(data.returns, vals.flatten()))
            el = -ent.mean()
            base = pg + m.ent_coef * el + m.vf_coef * vl
            full = m._mirror_loss(mirror, data, base)
            for k, t in (("full", full), ("pg", pg), ("ent", m.ent_coef * el), ("vf", m.vf_coef * vl), ("kl", full - base)):
                g = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
                nrm = math.sqrt(sum(float((x ** 2).sum()) for x in g if x is not None))
                rec[k].append(nrm)
                if k in ("pg", "kl"):
                    rec[k + "_actor"].append(math.sqrt(sum(float((g[i] ** 2).sum()) for i in actor if g[i] is not None)))
            rec["c"].append(min(1.0, float(m.max_grad_norm) / (rec["full"][-1] + 1e-6)))
        med = {k: float(np.median(v)) for k, v in rec.items()}
        lines.append(f"{run} {N:>8d} (seed {seed}, 64 mini-batches): median |grad| full {med['full']:.3f}; policy {med['pg']:.3f} "
                     f"(actor-mean params {med['pg_actor']:.3f}); entropy {med['ent']:.5f}; value {med['vf']:.3f}; mirror {med['kl']:.4f} "
                     f"(actor {med['kl_actor']:.4f}); clip factor c median {med['c']:.4f} (range {min(rec['c']):.4f}-{max(rec['c']):.4f})")
        print(lines[-1], flush=True)
    (HERE / "eb_gradnorm.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
