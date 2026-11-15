"""Round C scratch experiment C2 (not a record): step rate of a stub R5 env (Stage 1's env with R5's base: R2-mode
tracker, no feedback term, no ankle cap; Stage 1's observation and reward as a stand-in for R5's) with action noise
sigma 0.1 (R5's initial exploration), per floor; survival of the zero action per task; the cost of one legs.LegGap pose.
Usage: exp_c2_rate.py [body] [floor] [env_steps]"""
import os
import sys
import time

os.environ["MUJOCO_GL"] = "disable"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import resource  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage1 import control, env, legs, plant  # noqa: E402

body = sys.argv[1] if len(sys.argv) > 1 else "75-M-H-10"
floor = sys.argv[2] if len(sys.argv) > 2 else "hard"
n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
what = sys.argv[4] if len(sys.argv) > 4 else "rate"


class StubR5(env.ResidualEnv):
    mode = "R5stub"

    def _gains(self):
        return None, dict(source="R5: no tracker gains", status="not used", flag=None, candidate=None, cma_seed=None)

    def _setup(self):
        self.trackers = {t: control.Tracker(self.p, self.tracks[t], "R2", gains=None, terms=(), ankle_cap=(),
                                            rate=self.cfg.rate, model=self.nominal_model) for t in self.tasks}
        self.residual_scale = np.where(self._legs, 0.35, 0.15)
        self.tk = self.trackers[self.tasks[0]]


t0 = time.perf_counter()
cfg = env.EnvConfig(body=body, floor=floor, latency_steps=(0, 12), kp=(0.6, 1.2), friction=(0.4, 1.2))
e = StubR5(cfg)
print(f"{body} {floor}: env built in {time.perf_counter() - t0:.1f} s; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB",
      flush=True)
if what == "rate":
    for sigma in (0.0, 0.1):
        rng = np.random.default_rng(0)
        obs, _ = e.reset(seed=1)
        t0, c0, resets, falls = time.perf_counter(), time.process_time(), 0, 0
        for _ in range(n_steps):
            a = np.clip(sigma * rng.standard_normal(env.ACTION_DIM), -1, 1).astype(np.float32)
            obs, r, term, trunc, info = e.step(a)
            if term or trunc:
                falls += term
                obs, _ = e.reset()
                resets += 1
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        print(f"  sigma {sigma:g}: {n_steps / wall:.0f} env steps/s wall, {n_steps / cpu:.0f} per CPU s; {resets} resets "
              f"({falls} falls); load {os.getloadavg()[0]:.2f}", flush=True)
    lg = legs.LegGap(body)
    q = e.p.d.qpos.copy()
    t0 = time.perf_counter()
    for _ in range(300):
        d = lg.at(q)
    print(f"  legs.LegGap.at: {1e3 * (time.perf_counter() - t0) / 300:.3f} ms per pose; min gap {1e3 * d.min():.1f} mm; "
          f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
else:  # survival of the zero action per task, nominal plant, a given delay
    k = int(what)
    a = np.zeros(env.ACTION_DIM, np.float32)
    for task in ("walk", "brisk", "walking_turn_left", "in_place_turn_left", "stand"):
        up = 0
        times = []
        for ep in range(5):
            e.reset(seed=0, options=dict(task=task, episode_seed=100 + ep, push=None,
                                         factors=dict(friction=None, mass=1.0, kp=1.0, latency_steps=k, noise_scale=1.0)))
            while True:
                _, _, term, trunc, info = e.step(a)
                if term or trunc:
                    break
            up += not term
            times.append(round(e.k_env * e.dt_env, 2))
        print(f"  zero action, k={k}, {task}: {up}/5 up; seconds {times} of {e.seconds:.1f}", flush=True)
