"""Shared helpers for per-species actuator force-bound tests.

The July 2026 model hardening added ``forcerange`` caps sized against
settling under the XML home-keyframe controls only; under gait-like
excitation the caps clipped a large fraction of leg torque and collapsed velociraptor stage-2 twice
(docs/investigations/STAGE2_RECOMMENDATIONS.md). Gait-critical actuators
now carry 1.5x-kp headroom on every species, and each species pins that
contract with a test built on these helpers:

* scripted gait excitation (alternating legs for bipeds, diagonal-pair
  trot for quadrupeds), and
* per-actuator clip-fraction measurement under home-control and gait regimes.

Run ``environments/shared/scripts/actuator_saturation_report.py`` for the
full per-actuator numbers on any species' current model.
"""

from typing import Literal

import mujoco
import numpy as np

CLIP_EPS = 0.999
GAIT_HZ = 2.5
GAIT_AMPLITUDE = 0.8

# Quadruped diagonal-pair trot: FR+RL in phase, FL+RR antiphase.
_QUAD_PHASES = {"fr_": 0.0, "rl_": 0.0, "fl_": np.pi, "rr_": np.pi}


def position_actuator_ids(model):
    """IDs of position servos (gaintype fixed, biastype affine) with kp > 0."""
    ids = []
    for i in range(model.nu):
        kp = model.actuator_gainprm[i, 0]
        # Motors (e.g. claw) have gainprm[0] == gear and no position bias.
        if model.actuator_biasprm[i, 1] < 0 and kp > 0:
            ids.append(i)
    return ids


class PositionActuatorConfigurationBase:
    """Shared configuration contract for bounded position servos."""

    def test_position_actuators_have_enabled_force_and_control_limits(self, model):
        ids = position_actuator_ids(model)
        assert ids, "model has no position actuators"
        for actuator_id in ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            assert model.actuator_forcelimited[actuator_id] == 1, f"{name} does not enforce its forcerange"
            assert model.actuator_ctrllimited[actuator_id] == 1, f"{name} does not enforce its ctrlrange"

    def test_home_keyframe_controls_are_finite_and_in_range(self, model):
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert home_id >= 0, "model has no named home keyframe"
        ids = position_actuator_ids(model)
        controls = model.key_ctrl[home_id, ids]
        ranges = model.actuator_ctrlrange[ids]
        assert np.all(np.isfinite(controls)), "home keyframe contains a non-finite position control"
        assert np.all(controls >= ranges[:, 0]), "home keyframe control falls below an actuator ctrlrange"
        assert np.all(controls <= ranges[:, 1]), "home keyframe control exceeds an actuator ctrlrange"


def _leg_phase(name):
    """Gait phase for a leg actuator name, or None for non-leg actuators."""
    for prefix, phase in _QUAD_PHASES.items():
        if name.startswith(prefix):
            return phase
    if name.startswith("r_"):
        return 0.0
    if name.startswith("l_"):
        return np.pi
    return None


def gait_commands(model, t, hz=GAIT_HZ, amplitude=GAIT_AMPLITUDE):
    """Scripted sinusoidal gait ctrl vector at step ``t``.

    Leg actuators (by name prefix) run full-amplitude alternating/trot
    sinusoids; everything else gets a 0.4x-amplitude wiggle.
    """
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    phase = 2 * np.pi * hz * t * model.opt.timestep
    ctrl = mid.copy()
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        leg = _leg_phase(name)
        s = np.sin(phase + leg) if leg is not None else 0.4 * np.sin(phase)
        ctrl[i] = mid[i] + amplitude * amp[i] * s
    return ctrl


def measure_clip_fractions(
    env,
    mode: Literal["home_control", "gait"],
    steps,
    hz=GAIT_HZ,
    amplitude=GAIT_AMPLITUDE,
):
    """Per-actuator fraction of ``steps`` spent at the forcerange cap.

    ``mode="home_control"`` retains the XML home-keyframe controls;
    ``mode="gait"`` drives the scripted gait.
    """
    if mode not in {"home_control", "gait"}:
        raise ValueError(f"unknown saturation mode {mode!r}")
    env.reset(seed=0)
    model, data = env.model, env.data
    fr = model.actuator_forcerange[:, 1]
    clip_hits = np.zeros(model.nu)
    for t in range(steps):
        if mode == "gait":
            data.ctrl[:] = gait_commands(model, t, hz=hz, amplitude=amplitude)
        mujoco.mj_step(model, data)
        clip_hits += np.abs(data.actuator_force) >= CLIP_EPS * np.maximum(fr, 1e-9)
    return clip_hits / steps


def clip_fraction(frac, model, name):
    """Clip fraction for a named actuator from a measure_clip_fractions result."""
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return frac[i]
