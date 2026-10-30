"""Verifier's own harness for the creep scan (independent of the scan's scanlib.py / creeplib.py).

Shared with the scan only: the code under test (the committed CoM test classes' make_env, run_tracked_episode and the
test reduction, used for the env stepping and for the test's own com_speed_end_rms_mps).  Everything else is computed
here from the raw MjData:
  - foot attribution: a floor contact counts for foot k when its animal geom lies in the subtree of the root's child
    that carries foot k's site (hard-coded site names below), NOT Morphology.geom_foot;
  - body weight: the summed mass of every body in the animal's kinematic tree times |g|;
  - CoM: mass-weighted xipos over those bodies (not subtree_com);
  - end speed: RMS of |d com_xy / dt| over control steps ending in the last 2 s (forward differences);
  - foot travel after 1 s: max horizontal distance of each foot site from its position at the first control sample
    with t >= 1 s, over t >= 1 s;
  - load oscillation: per-substep per-foot normal floor load, last 2 s, linear detrend, population sd / weight;
  - per-geom floor contact over the WHOLE episode (count of substeps in contact, max normal force), root vz per
    substep, per-foot world-frame tangential floor force per substep (mean over the last 2 s and per second).
Nothing here edits the model unless a perturbation is passed explicitly.
"""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from environments.compsognathus_longipes.tests.test_com_stance_quality import (
    TestAnatomicalComStanceQuality,
    TestRobotComStanceQuality,
)
from environments.shared.tests.com_stance_helpers import run_tracked_episode
from environments.velociraptor_mongoliensis.tests.test_com_stance_quality import TestComStanceQuality as TestVelo

CLASSES = {"compy_anat": TestAnatomicalComStanceQuality, "velo": TestVelo, "robot": TestRobotComStanceQuality}
FOOT_SITES = {
    "compy_anat": ("r_foot_touch_volume", "l_foot_touch_volume"),
    "velo": ("r_foot", "l_foot"),
    "robot": ("right_foot_touch_volume", "left_foot_touch_volume"),
}
ROOT = "pelvis"
SPEED_BAR = 1e-3
TRAVEL_BAR = 5e-3
OSC_BAR = 0.01


class Layout:
    def __init__(self, m, model_name):
        self.m = m
        root = m.body(ROOT).id
        rid = int(m.body_rootid[root])
        self.animal_bodies = np.array([b for b in range(m.nbody) if int(m.body_rootid[b]) == rid])
        self.mass = float(m.body_mass[self.animal_bodies].sum())
        self.weight = self.mass * abs(float(m.opt.gravity[2]))
        self.floor = m.geom("floor").id
        self.sites = [m.site(s).id for s in FOOT_SITES[model_name]]
        # leg root of each foot: the ancestor of the site's body that is a direct child of the root body
        leg_roots = []
        for s in self.sites:
            b = int(m.site_bodyid[s])
            while int(m.body_parentid[b]) != root:
                b = int(m.body_parentid[b])
            leg_roots.append(b)
        self.leg_roots = leg_roots
        self.geom_leg = np.full(m.ngeom, -1)
        self.geom_animal = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            self.geom_animal[g] = int(m.body_rootid[b]) == rid
            x = b
            while x != 0 and x != root:
                if x in leg_roots:
                    self.geom_leg[g] = leg_roots.index(x)
                    break
                x = int(m.body_parentid[x])

    def com(self, d):
        bs = self.animal_bodies
        return (self.m.body_mass[bs, None] * d.xipos[bs]).sum(axis=0) / self.mass


class SubRec:
    """Per physics substep (after mj_step): per-foot normal and world tangential floor force, non-leg floor force,
    root vz, and per-geom floor contact counts / max force over the whole run."""

    def __init__(self, u, lay: Layout):
        self.u, self.lay = u, lay
        m = u.model
        self.f6 = np.zeros(6)
        self.t, self.fn, self.ft, self.other, self.vz = [], [], [], [], []
        self.geom_count = np.zeros(m.ngeom, int)
        self.geom_fmax = np.zeros(m.ngeom)

    def __call__(self):
        d, m, lay = self.u.data, self.u.model, self.lay
        fn = np.zeros(2)
        ft = np.zeros((2, 2))
        other = 0.0
        n = int(d.ncon)
        if n:
            gg = d.contact.geom[:n]
            idx = np.flatnonzero((gg[:, 0] == lay.floor) ^ (gg[:, 1] == lay.floor))
            for i in idx:
                g = int(gg[i, 1] if gg[i, 0] == lay.floor else gg[i, 0])
                if not lay.geom_animal[g]:
                    continue
                mujoco.mj_contactForce(m, d, int(i), self.f6)
                frame = d.contact.frame[i].reshape(3, 3)
                fw = frame.T @ self.f6[:3]
                if fw[2] < 0:
                    fw = -fw
                self.geom_count[g] += 1
                self.geom_fmax[g] = max(self.geom_fmax[g], self.f6[0])
                k = lay.geom_leg[g]
                if k >= 0:
                    fn[k] += self.f6[0]
                    ft[k] += fw[:2]
                else:
                    other += self.f6[0]
        self.t.append(float(d.time))
        self.fn.append(fn)
        self.ft.append(ft)
        self.other.append(other)
        self.vz.append(float(d.qvel[2]))


def detrended_sd(x, t):
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(np.std(x - A @ c))


def peak_hz(x, h, fmin=1.0):
    if len(x) < 16:
        return float("nan")
    y = (x - x.mean()) * np.hanning(len(x))
    sp = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(x), h)
    ok = f > fmin
    return float(f[ok][np.argmax(sp[ok])])


def run(model_name, seed, noise, steps=None, perturb=None, keep=False):
    CLS = CLASSES[model_name]
    inst = CLS()
    kw = {}
    if noise is not None:
        kw["reset_noise_scale"] = noise
    if steps is not None:
        kw["max_episode_steps"] = steps
    env = inst.make_env(**kw)
    u = env.unwrapped
    if perturb is not None:
        perturb(u)
    m = u.model
    lay = Layout(m, model_name)
    assert u._substep_probe_hook is None
    rec = SubRec(u, lay)
    u._substep_probe_hook = rec
    ctl_t, ctl_com, ctl_feet = [], [], []

    def post(env_, step, action):
        d = u.data
        ctl_t.append(float(d.time))
        ctl_com.append(lay.com(d))
        ctl_feet.append(d.site_xpos[lay.sites].copy())

    t0 = time.time()
    try:
        ep = run_tracked_episode(env, CLS.model_id, seed=seed, post_step=post)
        h = float(m.opt.timestep)
        dt = float(u.dt)
    finally:
        env.close()
    t = np.array(ctl_t)
    com = np.array(ctl_com)
    feet = np.array(ctl_feet)[:, :, :2]
    T = t[-1]
    row = {"model": model_name, "seed": seed, "noise": noise, "stood": bool(ep.stood), "steps": int(ep.steps),
           "T": T, "weight_n": lay.weight}
    # test reduction's own end speed (the scan's speed flag)
    if ep.stood:
        _, metrics = inst.reduce(ep)
        row["test_com_speed_end_rms_mps"] = float(metrics["com_speed_end_rms_mps"])
        row["test_foot_travel_max_m"] = float(metrics["foot_travel_max_m"])
        row["test_metrics"] = {k: float(v) for k, v in metrics.items()}
    v = np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1) / dt
    tv = t[1:]
    tail_c = tv > T - 2.0 + 1e-9
    row["my_end_speed_mps"] = float(np.sqrt(np.mean(v[tail_c] ** 2)))
    i1 = int(np.argmax(t >= 1.0 - 1e-9))
    trav = np.linalg.norm(feet[i1:] - feet[i1], axis=2)
    row["my_foot_travel_after1s_m"] = float(trav.max())
    iw = int(np.argmax(t >= CLS.window_start_s - 1e-9))
    row["my_foot_travel_window_m"] = float(np.linalg.norm(feet[iw:] - feet[iw], axis=2).max())
    row["my_com_travel_window_mm"] = 1e3 * float(np.linalg.norm(com[-1, :2] - com[iw, :2]))
    st = np.array(rec.t)
    fn = np.array(rec.fn)
    ft = np.array(rec.ft)
    other = np.array(rec.other)
    vz = np.array(rec.vz)
    tail = st >= T - 2.0 - 1e-9
    sds = [detrended_sd(fn[tail, k], st[tail]) for k in range(2)]
    row["my_load_sd_bw"] = [s / lay.weight for s in sds]
    row["my_load_sd_bw_max"] = max(sds) / lay.weight
    kb = int(np.argmax(sds))
    win = st >= CLS.window_start_s - 1e-9
    row["my_osc_hz"] = peak_hz(fn[win, kb], h)
    row["my_unloaded_tail_frac"] = [float(np.mean(fn[tail, k] <= 0.0)) for k in range(2)]
    row["my_total_load_tail_bw"] = float(fn[tail].sum(axis=1).mean() / lay.weight)
    row["my_other_floor_max_n_whole"] = float(other.max())
    row["my_other_floor_max_n_after1s"] = float(other[st >= 1.0].max()) if (st >= 1.0).any() else 0.0
    row["my_root_vz_tail_rms_mmps"] = 1e3 * float(np.sqrt(np.mean(vz[tail] ** 2)))
    sec = np.floor(st - 1e-12).astype(int)
    row["my_root_vz_per_s_mmps"] = [1e3 * float(np.sqrt(np.mean(vz[sec == s] ** 2))) for s in range(int(math.ceil(T)))
                                    if (sec == s).any()]
    row["my_lateral_tail_mean_n"] = [float(ft[tail, k, 1].mean()) for k in range(2)]
    row["my_lateral_per_s_n"] = [[float(ft[sec == s, k, 1].mean()) for k in range(2)] for s in range(int(math.ceil(T)))
                                 if (sec == s).any()]
    row["my_com_speed_per_s_mmps"] = [1e3 * float(np.sqrt(np.mean(v[(np.floor(tv - 1e-9).astype(int)) == s] ** 2)))
                                      for s in range(int(math.ceil(T)))]
    # whole-episode floor contact by geom (name -> (substeps in contact, max N))
    gc = {}
    for g in np.flatnonzero(rec.geom_count):
        gc[m.geom(int(g)).name] = (int(rec.geom_count[g]), float(rec.geom_fmax[g]), int(lay.geom_leg[g]))
    row["floor_geoms_whole"] = gc
    row["flag_speed_test"] = bool(row.get("test_com_speed_end_rms_mps", 0) > SPEED_BAR)
    row["flag_speed_mine"] = bool(row["my_end_speed_mps"] > SPEED_BAR)
    row["flag_travel"] = bool(row["my_foot_travel_after1s_m"] > TRAVEL_BAR)
    row["flag_oscill"] = bool(row["my_load_sd_bw_max"] > OSC_BAR)
    row["creep3"] = bool(row["flag_speed_mine"] and row["flag_travel"] and row["flag_oscill"])
    row["any_flag"] = bool(row["flag_speed_mine"] or row["flag_speed_test"] or row["flag_travel"] or row["flag_oscill"])
    row["wall_s"] = round(time.time() - t0, 2)
    if keep:
        return row, {"t": st, "fn": fn, "ft": ft, "vz": vz, "ctl_t": t, "com": com, "feet": feet}
    return row
