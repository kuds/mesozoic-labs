"""Creep lens harness (never committed): the robot's noisy spawn through the CoM test class, with an
optional in-memory plant perturbation and an optional per-physics-substep trace.

Everything here changes only the in-memory MjModel of one env instance (or the env instance's own
frame_skip / dt); no file in the worktree is written.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import mujoco
import numpy as np

from environments.compsognathus_longipes.tests.test_com_stance_quality import TestRobotComStanceQuality
from environments.shared.gait.morphology import Morphology
from environments.shared.tests import com_stance_helpers as H
from environments.shared.tests.com_stance_helpers import FAMILIES, check, run_tracked_episode

CLS = TestRobotComStanceQuality


# ── perturbations: each takes (env) and edits env.unwrapped.model (or the env's stepping) in place ──


def _leg_geoms(model):
    """Collision geoms of the two legs (contype != 0 and not the core/head/tail), and the floor."""
    floor = model.geom("floor").id
    legs = [g for g in range(model.ngeom) if g != floor and model.geom_contype[g] and model.body_rootid[model.geom_bodyid[g]] != 0]
    return floor, legs


def _scale_kv(f):
    def p(env):
        m = env.unwrapped.model
        m.actuator_biasprm[:, 2] *= f
    return p


def _scale_kp(f):
    def p(env):
        m = env.unwrapped.model
        m.actuator_gainprm[:, 0] *= f
        m.actuator_biasprm[:, 1] *= f
    return p


def _scale_forcerange(f):
    def p(env):
        env.unwrapped.model.actuator_forcerange[:] *= f
    return p


def _scale_dof(field, f):
    def p(env):
        m = env.unwrapped.model
        getattr(m, field)[6:] *= f
    return p


def _timestep(div):
    def p(env):
        u = env.unwrapped
        u.model.opt.timestep /= div
        u.frame_skip = int(u.frame_skip * div)
        # dt (control period) is unchanged: timestep * frame_skip
        assert abs(u.dt - u.model.opt.timestep * u.frame_skip) < 1e-12
    return p


def _solref(timeconst=None, dampratio=None, which="all"):
    def p(env):
        m = env.unwrapped.model
        floor, legs = _leg_geoms(m)
        gs = [floor] + legs if which == "all" else legs
        for g in gs:
            if timeconst is not None:
                m.geom_solref[g, 0] = timeconst
            if dampratio is not None:
                m.geom_solref[g, 1] = dampratio
    return p


def _solimp(vals):
    def p(env):
        m = env.unwrapped.model
        floor, legs = _leg_geoms(m)
        for g in [floor] + legs:
            m.geom_solimp[g, : len(vals)] = vals
    return p


def _opt(**kw):
    def p(env):
        o = env.unwrapped.model.opt
        for k, v in kw.items():
            setattr(o, k, v)
    return p


def _friction(f):
    def p(env):
        m = env.unwrapped.model
        floor, legs = _leg_geoms(m)
        for g in [floor] + legs:
            m.geom_friction[g, 0] *= f
    return p


def _condim(c):
    def p(env):
        m = env.unwrapped.model
        floor, legs = _leg_geoms(m)
        for g in [floor] + legs:
            m.geom_condim[g] = c
    return p


def _margin(v):
    def p(env):
        m = env.unwrapped.model
        floor, legs = _leg_geoms(m)
        for g in [floor] + legs:
            m.geom_margin[g] = v
    return p


def _no_leg_leg():
    """Stop the left leg colliding with the right leg (contype/conaffinity bits 4 and 8)."""
    def p(env):
        m = env.unwrapped.model
        floor, legs = _leg_geoms(m)
        for g in legs:
            m.geom_conaffinity[g] &= ~np.int32(12)
    return p


def _joint_armature(names, value):
    """Set the armature of the named joints (both sides) to *value* kg m2."""
    def p(env):
        m = env.unwrapped.model
        for side in ("left", "right"):
            for n in names:
                m.dof_armature[m.jnt_dofadr[m.joint(f"{side}_{n}").id]] = value
    return p


def _joint_kv(names, factor):
    """Scale the servo kv of the named joints (both sides)."""
    def p(env):
        m = env.unwrapped.model
        for side in ("left", "right"):
            for n in names:
                m.actuator_biasprm[m.actuator(f"{side}_{n}_act").id, 2] *= factor
    return p


def _compose(*ps):
    def p(env):
        for q in ps:
            q(env)
    return p


PERTURBATIONS: dict[str, Callable[[Any], None]] = {
    "base": lambda env: None,
    # servo
    "kv_x0.5": _scale_kv(0.5),
    "kv_x2": _scale_kv(2.0),
    "kv_x4": _scale_kv(4.0),
    "kp_x0.5": _scale_kp(0.5),
    "kp_x2": _scale_kp(2.0),
    "forcerange_x2": _scale_forcerange(2.0),
    "forcerange_x0.75": _scale_forcerange(0.75),
    "damping_x10": _scale_dof("dof_damping", 10.0),
    "damping_x40": _scale_dof("dof_damping", 40.0),
    "armature_x10": _scale_dof("dof_armature", 10.0),
    "frictionloss_0.01": lambda env: env.unwrapped.model.dof_frictionloss.__setitem__(slice(6, None), 0.01),
    # integration
    "ts_half": _timestep(2),
    "ts_quarter": _timestep(4),
    "euler": _opt(integrator=int(mujoco.mjtIntegrator.mjINT_EULER)),
    "implicit": _opt(integrator=int(mujoco.mjtIntegrator.mjINT_IMPLICIT)),
    "rk4": _opt(integrator=int(mujoco.mjtIntegrator.mjINT_RK4)),
    # contact
    "solref_tc0.004": _solref(timeconst=0.004),
    "solref_tc0.016": _solref(timeconst=0.016),
    "solref_dr2": _solref(dampratio=2.0),
    "solref_dr0.5": _solref(dampratio=0.5),
    "solimp_soft": _solimp([0.9, 0.95, 0.001]),
    "solimp_wide": _solimp([0.95, 0.99, 0.005]),
    "cone_pyramidal": _opt(cone=int(mujoco.mjtCone.mjCONE_PYRAMIDAL)),
    "noslip_10": _opt(noslip_iterations=10),
    "impratio_10": _opt(impratio=10.0),
    "condim_4": _condim(4),
    "condim_6": _condim(6),
    "margin_0": _margin(0.0),
    "friction_x0.1": _friction(0.1),
    "friction_x0.5": _friction(0.5),
    "friction_x2": _friction(2.0),
    "no_leg_leg": _no_leg_leg(),
    # dose-response and targeted variants of the servo-DOF fixes
    "armature_x2": _scale_dof("dof_armature", 2.0),
    "armature_x3": _scale_dof("dof_armature", 3.0),
    "armature_x5": _scale_dof("dof_armature", 5.0),
    "ankleroll_armature_3e-4": _joint_armature(["ankle_roll"], 3e-4),
    "ankleroll_armature_5e-4": _joint_armature(["ankle_roll"], 5e-4),
    "ankle_armature_5e-4": _joint_armature(["ankle", "ankle_roll"], 5e-4),
    "kv_x0.75": _scale_kv(0.75),
    "ankleroll_kv_x0.5": _joint_kv(["ankle_roll"], 0.5),
    "ankle_kv_x0.5": _joint_kv(["ankle", "ankle_roll"], 0.5),
    "tol_1e-10": _opt(tolerance=1e-10),
    "iter_1000": _opt(iterations=1000, ls_iterations=200),
}


def make_env(perturb: str = "base"):
    env = CLS().make_env(reset_noise_scale=CLS.noisy_reset_noise_scale)
    for name in perturb.split("+"):
        PERTURBATIONS[name](env)
    return env


# ── per-substep tracer ──


class SubstepTracer:
    """Records the plant at every physics substep through the env's _substep_probe_hook."""

    def __init__(self, env):
        u = env.unwrapped
        m = u.model
        self.u, self.m = u, m
        self.floor = m.geom("floor").id
        self.foot_body = [m.body("left_rolling_foot").id, m.body("right_rolling_foot").id]
        self.sole = [m.geom("left_sole").id, m.geom("right_sole").id]
        # every leg geom -> 0 left / 1 right by subtree (hip_roll_assembly = left)
        left_root = m.body("hip_roll_assembly").id
        right_root = m.body("hip_roll_assembly_2").id
        self.side = np.full(m.ngeom, -1)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b != 0:
                if b == left_root:
                    self.side[g] = 0
                    break
                if b == right_root:
                    self.side[g] = 1
                    break
                b = int(m.body_parentid[b])
        self.rows: dict[str, list] = {k: [] for k in (
            "t", "fn", "fsole", "ncon", "nsolecon", "ftan", "cop", "conx_min", "conx_max", "ctrl", "act_force",
            "qvel", "qpos", "com", "rootvel", "footvel", "leg_leg", "other_floor", "niter")}
        self._f = np.zeros(6)

    def __call__(self):
        u, m, d = self.u, self.m, self.u.data
        r = self.rows
        fn = np.zeros(2); fs = np.zeros(2); nc = np.zeros(2); nsc = np.zeros(2); ft = np.zeros((2, 2))
        copw = np.zeros((2, 2)); xmin = np.full(2, np.nan); xmax = np.full(2, np.nan)
        legleg = 0.0; other = 0.0
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            mujoco.mj_contactForce(m, d, i, self._f)
            if self.floor in (g1, g2):
                g = g2 if g1 == self.floor else g1
                s = self.side[g]
                if s < 0:
                    other += self._f[0]
                    continue
                frame = c.frame.reshape(3, 3)
                fw = frame.T @ self._f[:3]
                if fw[2] < 0:
                    fw = -fw
                fn[s] += self._f[0]
                ft[s] += fw[:2]
                nc[s] += 1
                if g == self.sole[s]:
                    fs[s] += self._f[0]
                    nsc[s] += 1
                    # contact point in the rolling foot's frame (x forward)
                    fb = self.foot_body[s]
                    local = d.xmat[fb].reshape(3, 3).T @ (c.pos - d.xpos[fb])
                    copw[s] += self._f[0] * local[:2]
                    xmin[s] = np.nanmin([xmin[s], local[0]])
                    xmax[s] = np.nanmax([xmax[s], local[0]])
            elif self.side[g1] >= 0 and self.side[g2] >= 0 and self.side[g1] != self.side[g2]:
                legleg += self._f[0]
        cop = np.where(fs[:, None] > 1e-9, copw / np.maximum(fs[:, None], 1e-12), np.nan)
        vel = np.zeros(6)
        fv = []
        for fb in self.foot_body:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb, vel, 0)
            fv.append(vel[3:].copy())
        r["t"].append(d.time); r["fn"].append(fn); r["fsole"].append(fs); r["ncon"].append(nc); r["nsolecon"].append(nsc)
        r["ftan"].append(ft); r["cop"].append(cop); r["conx_min"].append(xmin); r["conx_max"].append(xmax)
        r["ctrl"].append(d.ctrl.copy()); r["act_force"].append(d.actuator_force.copy())
        r["qvel"].append(d.qvel[6:].copy()); r["qpos"].append(d.qpos[7:].copy())
        r["com"].append(d.subtree_com[1].copy()); r["rootvel"].append(d.qvel[:6].copy())
        r["footvel"].append(np.array(fv)); r["leg_leg"].append(legleg); r["other_floor"].append(other)
        r["niter"].append(int(d.solver_niter[0]))

    def arrays(self):
        return {k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()}


def run(seed: int, perturb: str = "base", trace: bool = False, steps: int | None = None):
    """One noisy spawn: (episode, series, metrics, problems, tracer or None)."""
    inst = CLS()
    env = make_env(perturb)
    tracer = None
    try:
        morphology = Morphology.from_env(env, CLS.model_id)
        if trace:
            tracer = SubstepTracer(env)
            env.unwrapped._substep_probe_hook = tracer
        ep = run_tracked_episode(env, CLS.model_id, seed=seed, morphology=morphology, steps=steps)
    finally:
        env.close()
    problems: list[str] = []
    series = metrics = None
    if ep.stood:
        series, metrics = inst.reduce(ep)
        problems = inst.support_failures(ep, series, {k: inst.spread_bar(k) for k in inst.keys("support")})
        keys = [k for fam in FAMILIES if fam != "support" for k in inst.keys(fam)]
        problems += [v.describe() for v in check(metrics, [inst.spread_bar(k) for k in keys])]
    else:
        problems = [f"fell: {ep.termination_reason} at step {ep.steps}"]
    return ep, series, metrics, problems, tracer


CREEP_KEYS = ("com_speed_end_rms_mps", "com_speed_rms_mps", "foot_travel_max_m", "root_turn_deg",
              "lightest_foot_share_min", "support_defined_fraction", "com_margin_min_m")
STANCE_KEYS = ("com_fore_aft_mean_m", "com_lateral_mean_m", "com_height_mean_m", "root_pitch_mean_deg",
               "root_roll_mean_deg", "left_load_share_mean", "com_margin_min_m", "com_margin_lateral_min_m",
               "root_turn_from_spawn_deg", "foot_travel_from_spawn_max_m")


def summary(ep, metrics, problems):
    row: dict[str, Any] = {"stood": ep.stood, "steps": ep.steps, "n_problems": len(problems)}
    if metrics is not None:
        tr = ep.track
        t = tr.t
        w = t >= CLS.window_start_s
        i0 = int(np.argmax(w))
        row["com_disp_mm"] = 1e3 * float(np.linalg.norm(tr.com[-1, :2] - tr.com[i0, :2]))
        load = tr.foot_load.sum(axis=1)
        row["load_min_n"] = float(load[w].min()); row["load_max_n"] = float(load[w].max())
        for k in CREEP_KEYS + STANCE_KEYS:
            if k in metrics:
                row[k] = float(metrics[k])
        row["creeps"] = bool(metrics["com_speed_end_rms_mps"] > 1e-3)
    return row
