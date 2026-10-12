"""Plant-characterization tests for the raptor's bounded actuators.

Commit 156a933 added ``forcerange`` (~0.8x kp) to every position actuator
and switched to the implicitfast integrator, verifying only settling under
the XML home-keyframe controls. That sizing clipped 34-40% of hip-pitch and
22-25% of ankle torque during a moderate gait cycle and broke stage-2 locomotion twice
(runs 20260709_185946 and 20260711_165924, bitwise-identical collapses).
The hip-pitch and ankle caps were re-sized to 1.5x kp, which measures 0%
gait-cycle clipping and zero pelvis-z divergence from an unbounded plant
under identical commands (see docs/investigations/STAGE2_RECOMMENDATIONS.md
R2). These tests pin the resulting contract:

* the home-control no-saturation claim stays true (regression guard),
* every position actuator keeps a symmetric forcerange at its documented
  kp ratio (0.8x default, 1.5x for the gait-critical hip pitch/ankle), and
* the dynamic regime stays unclipped: gait-like excitation must not
  saturate the leg actuators, so the force caps bound impact/reset spikes
  rather than learnable gaits.

Run ``environments/velociraptor/scripts/actuator_saturation_report.py``
for the full per-actuator numbers on the current model.
"""

import mujoco
import pytest

from environments.shared.tests.actuator_bounds_helpers import (
    PositionActuatorConfigurationBase,
    clip_fraction,
    measure_clip_fractions,
    position_actuator_ids,
)
from environments.velociraptor.envs.raptor_env import RaptorEnv

# Default sizing from commit 156a933; gait-critical actuators carry extra
# headroom so the caps only bind on impact/reset spikes, not gait torques.
# The knee joined the 1.5x set after measuring 0% clip at the moderate
# 2.5 Hz/0.8-amplitude regime but 30-46% at sprint-like excitation
# (3-4 Hz, full amplitude) while still capped at 0.8x kp.
FORCERANGE_KP_RATIO = 0.8
GAIT_HEADROOM_RATIO = 1.5
GAIT_HEADROOM_ACTUATORS = frozenset(
    {
        "r_hip_pitch_act",
        "l_hip_pitch_act",
        "r_knee_act",
        "l_knee_act",
        "r_ankle_act",
        "l_ankle_act",
    }
)


@pytest.fixture(scope="module")
def model():
    env = RaptorEnv()
    try:
        yield env.model
    finally:
        env.close()


class TestForceBoundsConfiguration(PositionActuatorConfigurationBase):
    def test_integrator_is_implicitfast(self, model):
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    def test_all_position_actuators_have_forcerange(self, model):
        for i in position_actuator_ids(model):
            fr = model.actuator_forcerange[i]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            assert fr[1] > 0, f"{name} has no forcerange upper bound"
            assert fr[0] == -fr[1], f"{name} forcerange is not symmetric"

    def test_forcerange_matches_documented_kp_ratio(self, model):
        """Each actuator's forcerange matches its documented kp ratio; catch silent re-sizing."""
        for i in position_actuator_ids(model):
            kp = model.actuator_gainprm[i, 0]
            fr = model.actuator_forcerange[i, 1]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            expected = GAIT_HEADROOM_RATIO if name in GAIT_HEADROOM_ACTUATORS else FORCERANGE_KP_RATIO
            ratio = fr / kp
            assert ratio == pytest.approx(expected, abs=0.05), (
                f"{name}: forcerange/kp = {ratio:.2f}, expected ~{expected}. "
                "If this changed intentionally, update the ratios here and the "
                "saturation numbers in docs/investigations/STAGE2_RECOMMENDATIONS.md."
            )


class TestHomeControlSaturation:
    def test_no_saturation_during_home_control_settle(self):
        """Settling under XML home-keyframe controls never clips forces."""
        env = RaptorEnv()
        try:
            frac = measure_clip_fractions(env, mode="home_control", steps=1000)
            worst = max(frac[i] for i in position_actuator_ids(env.model))
            assert worst < 0.01, f"home-control settle saturates a position actuator {worst:.1%} of the time"
        finally:
            env.close()


class TestDynamicSaturation:
    def test_gait_excitation_stays_unclipped(self):
        """Guards the actuator headroom the stage-2 fix depends on.

        The 0.8x-kp caps clipped hips 34-40% and ankles 22-25% of a
        moderate (0.8-amplitude) alternating-leg gait cycle and collapsed
        stage-2 locomotion twice; at 1.5x kp the same excitation measures
        0% clipping. If this test starts failing because saturation *rose*,
        the plant lost gait headroom again — see
        docs/investigations/STAGE2_RECOMMENDATIONS.md before re-sizing.
        """
        env = RaptorEnv()
        try:
            model = env.model
            frac = measure_clip_fractions(env, mode="gait", steps=2000)
            hip = max(clip_fraction(frac, model, f"{s}_hip_pitch_act") for s in "rl")
            ankle = max(clip_fraction(frac, model, f"{s}_ankle_act") for s in "rl")
            # Measured at 1.5x kp: 0.0% on both groups (was hips 0.34-0.40,
            # ankles 0.22-0.25 at 0.8x kp). Thresholds keep margin for
            # timestep/integrator jitter while catching any real regression.
            assert hip < 0.10, f"hip saturation {hip:.1%} — plant lost gait headroom, stage 2 will clip"
            assert ankle < 0.05, f"ankle saturation {ankle:.1%} — plant lost gait headroom, stage 2 will clip"
        finally:
            env.close()

    def test_sprint_excitation_keeps_knee_unclipped(self):
        """Guards the knee's 1.5x-kp sprint headroom.

        At the 0.8x-kp cap (forcerange ±145 on kp=180) the knee measured
        0% clip at the moderate 2.5 Hz/0.8-amplitude gait but 30-46% at
        sprint-like excitation (3-4 Hz, full amplitude) — the same
        clipped-torque signature that collapsed stage 2 at the hips. At
        1.5x kp (±270) the same sprint excitation measures 0.0% on the
        knee. Hips/ankles (already 1.5x) measure ~11%/~2% at this regime,
        which is their physical envelope, so only the knee is pinned here.
        """
        env = RaptorEnv()
        try:
            model = env.model
            frac = measure_clip_fractions(env, mode="gait", steps=2000, hz=3.5, amplitude=1.0)
            knee = max(clip_fraction(frac, model, f"{s}_knee_act") for s in "rl")
            assert knee < 0.05, f"knee saturation {knee:.1%} at sprint excitation — faster-gait training will clip"
        finally:
            env.close()
