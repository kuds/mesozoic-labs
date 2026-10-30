"""Creep scan harness (never committed): a noisy spawn (or the noise-free statue) of any CoM test class under the
neutral action, with the creep lens's detection generalised from creeplib.py (copied here unchanged) and a
per-physics-substep floor-load recorder.

Detection per spawn (all three reported separately; a "creep" is all three together):
  speed  : the test reduction's com_speed_end_rms_mps (RMS horizontal CoM speed over the window's last 2 s) > 1 mm/s
           (creeplib.summary's "creeps").
  travel : the largest horizontal travel of any foot reference site from where it stood at t = 1 s, over t >= 1 s
           (control-rate samples of the test's sampler) > 5 mm.
  oscill : a per-substep floor-load oscillation: over the last 2 s, the linearly detrended per-foot floor normal
           load (the sampler's own foot attribution: floor contacts of each foot's leg geoms, mj_contactForce) has
           a standard deviation above 1 % of the body weight on either foot.  Its frequency is the peak (above
           1 Hz) of the Hann-windowed spectrum of that foot's load over the whole window (window_start_s to the end).

Everything here changes only the in-memory MjModel of one env instance (or that instance's frame_skip); no file of
the tree is written.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import mujoco
import numpy as np

from environments.compsognathus_longipes.tests.test_com_stance_quality import (
    TestAnatomicalComStanceQuality,
    TestRobotComStanceQuality,
)
from environments.shared.gait.morphology import Morphology
from environments.shared.tests.com_stance_helpers import FAMILIES, check, run_tracked_episode
from environments.velociraptor_mongoliensis.tests.test_com_stance_quality import TestComStanceQuality as TestVelo

CLASSES = {
    "compy_anat": TestAnatomicalComStanceQuality,
    "velo": TestVelo,
    "robot": TestRobotComStanceQuality,  # the detector's positive and negative controls only
}
#: Each model's own training reset noise, read from its stance config's [env] reset_noise_scale
#: (configs/compsognathus_longipes/anatomical/stance.toml 0.01; configs/velociraptor_mongoliensis/anatomical/
#: stance.toml 0.05; the robot's 0.01 per the verifier's M2).
TRAIN_NOISE = {"compy_anat": 0.01, "velo": 0.05, "robot": 0.01}

SPEED_BAR_MPS = 1e-3
TRAVEL_BAR_M = 5e-3
OSC_BAR_BW = 0.01
TAIL_S = 2.0


# ── in-memory perturbations (the robot fix's analogues) ──


def _dofs_of(model, joints):
    return [int(model.jnt_dofadr[model.joint(name).id]) for name in joints]


def perturb(env, spec: dict | None) -> None:
    """spec keys: arm_joints + arm_factor (armature x factor on the named joints), arm_all (x factor on every
    non-root DOF), hdiv (timestep / k, frame_skip * k: control period kept)."""
    if not spec:
        return
    u = env.unwrapped
    m = u.model
    if "arm_joints" in spec:
        for dof in _dofs_of(m, spec["arm_joints"]):
            m.dof_armature[dof] *= float(spec["arm_factor"])
    if "arm_all" in spec:
        m.dof_armature[6:] *= float(spec["arm_all"])
    if "hdiv" in spec:
        k = int(spec["hdiv"])
        dt0 = u.dt
        m.opt.timestep /= k
        u.frame_skip = int(u.frame_skip * k)
        assert abs(u.dt - dt0) < 1e-12, (u.dt, dt0)


# ── per-substep recorder ──


class LoadRecorder:
    """Per physics substep: time, per-foot floor normal load (the sampler's attribution), the floor load on animal
    geoms of no foot, root vertical velocity, contact count."""

    def __init__(self, env, morphology: Morphology):
        u = env.unwrapped
        self.u, self.m = u, u.model
        self.geom_foot = np.asarray(morphology.geom_foot)
        self.is_floor = np.asarray(morphology.is_floor)
        self.is_animal = np.asarray(morphology.is_animal)
        self.nfeet = len(morphology.feet)
        self.f6 = np.zeros(6)
        self.t: list[float] = []
        self.load: list[np.ndarray] = []
        self.other: list[float] = []
        self.vz: list[float] = []
        self.ncon: list[int] = []

    def __call__(self):
        d, m = self.u.data, self.m
        load = np.zeros(self.nfeet)
        other = 0.0
        n = int(d.ncon)
        if n:
            geoms = d.contact.geom[:n]
            f1, f2 = self.is_floor[geoms[:, 0]], self.is_floor[geoms[:, 1]]
            og = np.where(f1, geoms[:, 1], geoms[:, 0])
            sel = np.flatnonzero((f1 != f2) & self.is_animal[og])
            for i in sel:
                mujoco.mj_contactForce(m, d, int(i), self.f6)
                foot = self.geom_foot[og[i]]
                if foot >= 0:
                    load[foot] += self.f6[0]
                else:
                    other += self.f6[0]
        self.t.append(float(d.time))
        self.load.append(load)
        self.other.append(other)
        self.vz.append(float(d.qvel[2]))
        self.ncon.append(n)

    def arrays(self):
        return (np.asarray(self.t), np.asarray(self.load), np.asarray(self.other), np.asarray(self.vz),
                np.asarray(self.ncon))


def _detrended_sd(x: np.ndarray, t: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(np.std(x - A @ coef))


def _peak(x: np.ndarray, h: float, fmin: float = 1.0) -> tuple[float, float, float]:
    """(peak frequency above fmin, its amplitude, the share of non-DC power within +-2 bins of the peak)."""
    if len(x) < 16:
        return float("nan"), float("nan"), float("nan")
    y = (x - x.mean()) * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(x), h)
    ok = f > fmin
    if not ok.any() or spec[ok].sum() <= 0:
        return float("nan"), 0.0, float("nan")
    i = int(np.flatnonzero(ok)[np.argmax(spec[ok])])
    share = float(spec[max(i - 2, 1): i + 3].sum() / spec[1:].sum())
    return float(f[i]), float(math.sqrt(spec[i])), share


def substep_summary(rec: LoadRecorder, weight_n: float, window_start_s: float, h: float) -> dict[str, Any]:
    t, load, other, vz, ncon = rec.arrays()
    T = t[-1]
    tail = t >= T - TAIL_S - 1e-9
    win = t >= window_start_s - 1e-9
    sds = [_detrended_sd(load[tail, k], t[tail]) for k in range(load.shape[1])]
    peaks = [_peak(load[win, k], h) for k in range(load.shape[1])]
    k_big = int(np.nanargmax(sds)) if np.isfinite(sds).any() else 0
    total = load.sum(axis=1)
    # root vz RMS per whole second (the landing ring-down)
    sec = np.floor(t - 1e-12).astype(int)
    vz_sec = [1e3 * float(np.sqrt(np.mean(vz[sec == s] ** 2))) for s in range(int(math.ceil(T))) if (sec == s).any()]
    return {
        "load_sd_tail_bw": [s / weight_n for s in sds],
        "load_sd_tail_bw_max": float(np.nanmax(sds) / weight_n),
        "load_tail_min_n": [float(load[tail, k].min()) for k in range(load.shape[1])],
        "load_tail_max_n": [float(load[tail, k].max()) for k in range(load.shape[1])],
        "unloaded_tail_frac": [float(np.mean(load[tail, k] < 1e-9)) for k in range(load.shape[1])],
        "total_tail_sd_bw": float(np.std(total[tail]) / weight_n),
        "total_tail_mean_bw": float(np.mean(total[tail]) / weight_n),
        "other_floor_tail_max_n": float(other[tail].max()),
        "osc_freq_hz": peaks[k_big][0],
        "osc_peak_share": peaks[k_big][2],
        "osc_freq_hz_per_foot": [p[0] for p in peaks],
        "root_vz_tail_rms_mmps": 1e3 * float(np.sqrt(np.mean(vz[tail] ** 2))),
        "root_vz_rms_per_s_mmps": [round(v, 4) for v in vz_sec],
        "ncon_tail_mean": float(np.mean(ncon[tail])),
    }


# ── one episode ──


def run(model: str, seed: int, noise: float | None, spec: dict | None = None, record: bool = True,
        keep_recorder: bool = False):
    """One episode under the neutral action through the model's CoM test class.  noise None = the noise-free statue
    (the class's env_kwargs, reset noise 0).  Returns (row, episode, metrics, recorder or None)."""
    CLS = CLASSES[model]
    inst = CLS()
    env = inst.make_env() if noise is None else inst.make_env(reset_noise_scale=noise)
    perturb(env, spec)
    u = env.unwrapped
    rec = None
    t0 = time.time()
    try:
        morph = Morphology.from_env(env, CLS.model_id)
        weight = float(u.model.body_subtreemass[morph.root_body]) * abs(float(u.model.opt.gravity[2]))
        h = float(u.model.opt.timestep)
        if record:
            rec = LoadRecorder(env, morph)
            u._substep_probe_hook = rec
        ep = run_tracked_episode(env, CLS.model_id, seed=seed, morphology=morph)
    finally:
        env.close()
    row: dict[str, Any] = {"model": model, "seed": seed, "noise": noise, "spec": spec or {}, "stood": bool(ep.stood),
                           "steps": ep.steps, "termination": ep.termination_reason}
    metrics = None
    if ep.stood:
        series, metrics = inst.reduce(ep)
        problems = inst.support_failures(ep, series, {k: inst.spread_bar(k) for k in inst.keys("support")})
        keys = [k for fam in FAMILIES if fam != "support" for k in inst.keys(fam)]
        problems += [v.describe() for v in check(metrics, [inst.spread_bar(k) for k in keys])]
        tr = ep.track
        t = np.asarray(tr.t)
        com = np.asarray(tr.com)
        feet = np.asarray(tr.foot_pos)[:, :, :2]
        i1 = int(np.argmax(t >= 1.0 - 1e-9))
        iw = int(np.argmax(t >= CLS.window_start_s - 1e-9))
        travel = np.linalg.norm(feet[i1:] - feet[i1], axis=2)
        row.update({
            "n_problems": len(problems),
            "problems": [p[:160] for p in problems[:3]],
            "com_speed_end_rms_mps": float(metrics["com_speed_end_rms_mps"]),
            "com_speed_rms_mps": float(metrics["com_speed_rms_mps"]),
            "foot_travel_after1s_m": float(travel.max()),
            "foot_travel_after1s_per_foot_m": [float(x) for x in travel.max(axis=0)],
            "com_travel_window_mm": 1e3 * float(np.linalg.norm(com[-1, :2] - com[iw, :2])),
            "com_travel_after1s_mm": 1e3 * float(np.linalg.norm(com[-1, :2] - com[i1, :2])),
            "com_dx_after1s_mm": 1e3 * float(com[-1, 0] - com[i1, 0]),
            "metrics": {k: float(v) for k, v in metrics.items()},
        })
        if rec is not None:
            row.update(substep_summary(rec, weight, CLS.window_start_s, h))
        row["flag_speed"] = bool(row["com_speed_end_rms_mps"] > SPEED_BAR_MPS)
        row["flag_travel"] = bool(row["foot_travel_after1s_m"] > TRAVEL_BAR_M)
        row["flag_oscill"] = bool(rec is not None and row["load_sd_tail_bw_max"] > OSC_BAR_BW)
        row["creep"] = bool(row["flag_speed"] and row["flag_travel"] and row["flag_oscill"])
        row["any_flag"] = bool(row["flag_speed"] or row["flag_travel"] or row["flag_oscill"])
    row["wall_s"] = round(time.time() - t0, 2)
    return row, ep, metrics, (rec if keep_recorder else None)


# ── intervals ──


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) binomial interval."""
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
