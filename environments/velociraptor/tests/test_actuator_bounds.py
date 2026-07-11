"""Plant-characterization tests for the raptor's bounded actuators.

Commit 156a933 added ``forcerange`` (~0.8x kp) to every position actuator
and switched to the implicitfast integrator, verifying only *static*
settling ("settle behavior unchanged"). Stage-2 locomotion collapsed on
the first run against the new plant (20260709_185946) while four earlier
runs with the identical training config passed on the old plant, so these
tests pin down both halves of that verification gap:

* the static no-saturation claim stays true (regression guard), and
* the dynamic regime is *documented*: under gait-like excitation the hip
  and ankle actuators saturate a large fraction of the time, meaning the
  force caps — not the reward or curriculum — bound what gaits are
  physically learnable.

Run ``environments/velociraptor/scripts/actuator_saturation_report.py``
for the full per-actuator numbers on the current model.
"""

import numpy as np
import pytest

import mujoco

from environments.velociraptor.envs.raptor_env import RaptorEnv

FORCERANGE_KP_RATIO = 0.8  # documented sizing from commit 156a933
CLIP_EPS = 0.999


@pytest.fixture(scope="module")
def model():
    env = RaptorEnv()
    try:
        yield env.model
    finally:
        env.close()


def _position_actuator_ids(model):
    """IDs of position servos (gaintype fixed, biastype affine) with kp > 0."""
    ids = []
    for i in range(model.nu):
        kp = model.actuator_gainprm[i, 0]
        # Motors (e.g. claw) have gainprm[0] == gear and no position bias.
        if model.actuator_biasprm[i, 1] < 0 and kp > 0:
            ids.append(i)
    return ids


def _sinusoidal_gait_commands(model, t):
    """Alternating left/right sinusoidal leg commands at 2.5 Hz, 0.8 amplitude."""
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    phase = 2 * np.pi * 2.5 * t * model.opt.timestep
    ctrl = mid.copy()
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith("r_"):
            s = np.sin(phase)
        elif name.startswith("l_"):
            s = np.sin(phase + np.pi)
        else:
            s = 0.4 * np.sin(phase)
        ctrl[i] = mid[i] + 0.8 * amp[i] * s
    return ctrl


class TestForceBoundsConfiguration:
    def test_integrator_is_implicitfast(self, model):
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    def test_all_position_actuators_have_forcerange(self, model):
        for i in _position_actuator_ids(model):
            fr = model.actuator_forcerange[i]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            assert fr[1] > 0, f"{name} has no forcerange upper bound"
            assert fr[0] == -fr[1], f"{name} forcerange is not symmetric"

    def test_forcerange_matches_documented_kp_ratio(self, model):
        """forcerange was sized to ~0.8x kp; catch silent re-sizing."""
        for i in _position_actuator_ids(model):
            kp = model.actuator_gainprm[i, 0]
            fr = model.actuator_forcerange[i, 1]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            ratio = fr / kp
            assert ratio == pytest.approx(FORCERANGE_KP_RATIO, abs=0.05), (
                f"{name}: forcerange/kp = {ratio:.2f}, expected ~{FORCERANGE_KP_RATIO}. "
                "If this changed intentionally, update FORCERANGE_KP_RATIO and the "
                "saturation numbers in docs/STAGE2_INVESTIGATION.md."
            )


class TestStaticSaturation:
    def test_no_saturation_during_settle(self):
        """The commit's static claim: settling from home never clips forces."""
        env = RaptorEnv()
        try:
            env.reset(seed=0)
            model, data = env.model, env.data
            pos_ids = _position_actuator_ids(model)
            fr = model.actuator_forcerange[:, 1]
            clipped = np.zeros(model.nu)
            steps = 1000
            for _ in range(steps):
                mujoco.mj_step(model, data)
                clipped += np.abs(data.actuator_force) >= CLIP_EPS * np.maximum(fr, 1e-9)
            worst = max(clipped[i] / steps for i in pos_ids)
            assert worst < 0.01, f"static settle saturates a position actuator {worst:.1%} of the time"
        finally:
            env.close()


class TestDynamicSaturation:
    def test_gait_excitation_saturates_leg_actuators(self):
        """Documents the plant limitation behind the stage-2 collapse.

        Under moderate (0.8-amplitude) alternating-leg sinusoids, the hip
        pitch and ankle servos hit their force caps a substantial fraction
        of the time. If this test starts failing because saturation
        *dropped*, the plant gained actuator headroom — re-run stage 2 and
        update docs/STAGE2_INVESTIGATION.md.
        """
        env = RaptorEnv()
        try:
            env.reset(seed=0)
            model, data = env.model, env.data
            fr = model.actuator_forcerange[:, 1]
            clip_hits = np.zeros(model.nu)
            steps = 2000
            for t in range(steps):
                data.ctrl[:] = _sinusoidal_gait_commands(model, t)
                mujoco.mj_step(model, data)
                clip_hits += np.abs(data.actuator_force) >= CLIP_EPS * np.maximum(fr, 1e-9)
            frac = clip_hits / steps

            def actuator_frac(name):
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                return frac[i]

            hip = max(actuator_frac("r_hip_pitch_act"), actuator_frac("l_hip_pitch_act"))
            ankle = max(actuator_frac("r_ankle_act"), actuator_frac("l_ankle_act"))
            # Measured on the 156a933 plant: hips ~0.34-0.40, ankles ~0.22-0.25.
            assert hip > 0.10, f"hip saturation {hip:.1%} — plant gained headroom, revisit stage-2 tuning"
            assert ankle > 0.05, f"ankle saturation {ankle:.1%} — plant gained headroom, revisit stage-2 tuning"
        finally:
            env.close()
