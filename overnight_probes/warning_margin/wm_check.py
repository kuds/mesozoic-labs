"""Pre-run checks (no training): channel identities, predict clipping, timing, reproduction of one early-warning
episode (ew_r10.jsonl, s80 12,001,280, seed 5001: own14 window mean and tail_3 angle), and the normaliser's tail_3 clip
angle for the seven checkpoints."""
from __future__ import annotations

import json
import math
import time

import numpy as np

import wm_common as C

factory, ctor = C.make_factory()
env = factory()
u = env.unwrapped
m = u.model
print("ctor env_kwargs:", json.dumps(ctor, default=str))
print("actuator 14:", m.actuator(C.A_T3).name, "| nu", m.nu)
assert m.actuator(C.A_T3).name == "tail_3_pitch_act"
obs, _ = env.reset(seed=3042)
qa = m.jnt_qposadr[m.joint("tail_3_pitch").id]
print("obs[6] after reset %.9f, qpos tail_3_pitch %.9f" % (obs[C.I_T3], u.data.qpos[qa]))
for _ in range(5):
    obs, *_ = env.step(np.zeros(m.nu))
    print("  step: obs[6] %.9f qpos %.9f" % (obs[C.I_T3], u.data.qpos[qa]))
print("action space:", env.action_space)

pol = C.Policy("s80", 12_001_280, factory)
big = pol.model.predict(np.full(pol.mean.shape, 10.0, dtype=np.float32), deterministic=True)[0]
print("predict on an extreme input: min %.3f max %.3f (clipped to [-1,1] if both within)" % (big.min(), big.max()))
for run, steps in C.CKPTS:
    p = C.Policy(run, steps, factory)
    print(f"{run} {steps}: tail_3 normaliser mean {math.degrees(p.mean[C.I_T3]):+.3f} deg sd {math.degrees(p.std[C.I_T3]):.3f} "
          f"deg clip {p.clip} -> observed angle capped at {p.t3_clip_deg:.2f} deg")

ew = None
for line in open(C.SCR / "com_revisit/notes/early_warning/ew_r10.jsonl"):
    r = json.loads(line)
    if r["run"] == "s80" and r["steps"] == 12_001_280:
        ew = {e["seed"]: e for e in r["episodes"]}
t = time.time()
e = C.episode(env, pol, 5001, "none", 0.0)
print("seed 5001 none: own_win %.6f (ew own14_win %.6f), tail3 %.4f (ew %.4f), pitch %.4f (ew %.4f); wall %.2f s cpu %.2f s"
      % (e["own_win"], ew[5001]["own14_win"], e["tail3_deg_win"], ew[5001]["tail3_deg_win"], e["pitch_deg_win"],
         ew[5001]["pitch_deg_win"], time.time() - t, e["cpu_s"]))
t = time.time()
e1 = C.episode(env, pol, 3042, "bias", 0.3)
e2 = C.episode(env, pol, 3042, "bias", 0.3)
print("seed 3042 bias 0.3 twice: cmd_win %.9f / %.9f, n_rail %d / %d; wall %.2f s for 2" % (
    e1["cmd_win"], e2["cmd_win"], e1["n_rail_win"], e2["n_rail_win"], time.time() - t))
e3 = C.episode(env, pol, 3042, "obs", 3.0)
print("seed 3042 obs 3.0 deg: cmd_win %.4f own_win %.4f n_rail %d tail3 %.3f cpu %.2f" % (
    e3["cmd_win"], e3["own_win"], e3["n_rail_win"], e3["tail3_deg_win"], e3["cpu_s"]))
env.close()
