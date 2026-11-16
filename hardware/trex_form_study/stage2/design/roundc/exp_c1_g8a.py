"""Round C scratch experiment C1 (not a record): can a Tracker in R2 mode with no feedback term and no ankle cap,
stepped 10 physics steps per env step with residual None and rate_factor 1 inside speedfit.servo_delay, reproduce the
panels' R0 row (R1 mode, terms=(), ankle_cap=()) under control.run, digest per physics step, at k = 0 and k = 5?
The two sides run on two separately built plants.  Stock Stage 1 arrays (the read-only S1 view), 75-M-H-10 by default.
Usage: exp_c1_g8a.py [body] [seconds]"""
import os
import sys
import time

os.environ["MUJOCO_GL"] = "disable"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage1 import control, env, plant, speedfit, tune  # noqa: E402
from designlab.stage1.refs import Reference  # noqa: E402

body = sys.argv[1] if len(sys.argv) > 1 else "75-M-H-10"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
run_dir = tune.RUN / body
plan = [("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_brisk"), ("ref", "brisk", 8)]
RATE = tune.COMMAND_RATE
seed = 3042
s_start, s_noise = tune.derived_seed(seed, 0, 0), tune.derived_seed(seed, 0, 1)


def build():
    p = plant.build(body, "hard")
    omega = control.omega0(p, Reference.load(run_dir / "refs" / "nominal.npz").q_home)
    return p, control.build_track(p, run_dir, plan, omega)


def r0_row(p, track, k):
    """The panels' R0 row's episode (panels.py:841-852) inside the bar's delay wrapper."""
    tk = control.Tracker(p, track, "R1", gains=None, terms=(), ankle_cap=(), rate=RATE)
    rec = env._Digests(p)
    with speedfit.servo_delay(p, k):
        out = control.run(tk, seconds, start="stance", seed=s_start, noise_seed=s_noise, recorder=rec,
                          joint_noise=tune.START_JOINT_NOISE_RAD, root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S)
    return rec.rows, out


def r5_zero(p, track, k, nominal, scratch, mu0, gain0, bias0):
    """R5's zero action: R2 mode, no terms, no cap; env.Stage1Env's reset and step order (env.py:470-564)."""
    m = p.m
    m.geom_friction[:] = mu0  # _apply_factors (env.py:447-458) with the nominal factors
    m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = gain0, bias0
    env.set_trunk_mass(m, p.trunk, 1.0, nominal, scratch)
    tk = control.Tracker(p, track, "R2", gains=None, terms=(), ankle_cap=(), rate=RATE, model=nominal)
    tk.residual, tk.rate_factor = None, 1.0
    rows = []
    n_env = int(round(seconds / (env.ENV_SUBSTEPS * tk.dt)))
    with speedfit.servo_delay(p, k):
        tk.reset("stance", joint_noise=tune.START_JOINT_NOISE_RAD, seed=s_start,
                 root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S, noise_seed=s_noise)
        fell = None
        t0 = time.perf_counter()
        for i in range(n_env):
            a = np.zeros(tk.nj + 1)
            off = a[: tk.nj] * 0.35
            tk.residual = off if off.any() else None
            tk.rate_factor = 1.0 + 0.25 * float(a[tk.nj])
            for _ in range(env.ENV_SUBSTEPS):
                tk.step()
                rows.append(env._state_digest(p.d))
            if p.fall_cause() is not None:
                fell = i + 1
                break
        wall = time.perf_counter() - t0
    return rows, fell, (i + 1) / wall


pa, ta = build()
pb, tb = build()
nominal = copy.deepcopy(pb.m)
import mujoco  # noqa: E402

scratch = mujoco.MjData(pb.m)
mu0, g0, b0 = pb.m.geom_friction.copy(), pb.m.actuator_gainprm[:, 0].copy(), pb.m.actuator_biasprm[:, 1].copy()
for k in (0, 5):
    ref, out = r0_row(pa, ta, k)
    got, fell, rate = r5_zero(pb, tb, k, nominal, scratch, mu0, g0, b0)
    n = min(len(ref), len(got))
    first = next((i for i in range(n) if ref[i] != got[i]), None)
    print(f"{body} k={k}: R0 row {len(ref)} steps (fell {out['fell']} at {out['fall_time_s']}), R5 zero {len(got)} steps "
          f"(terminated at env step {fell}); compared {n}; first difference {first}; "
          f"zero-action loop {rate:.0f} env steps/s (digests on)", flush=True)
