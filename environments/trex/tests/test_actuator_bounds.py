"""Plant-characterization tests for the T-Rex's bounded actuators.

The July 2026 hardening sized every position actuator's ``forcerange`` at
~0.8x kp against *static* settling only. Under gait-like excitation that
clipped 44-50% of hip-pitch, 10-14% of knee, and ~31% of ankle torque —
the same plant defect that collapsed velociraptor stage-2 twice (see
docs/investigations/STAGE2_RECOMMENDATIONS.md). Hip pitch, knee, and
ankle were re-sized to 1.5x kp, which measures <1% gait-cycle clipping
and ~0 pelvis-z divergence from an unbounded plant. These tests pin that
contract before the next trex stage-2 run trains against the plant.

Run ``environments/shared/scripts/actuator_saturation_report.py trex``
for the full per-actuator numbers on the current model.
"""

import mujoco
import pytest

from environments.shared.tests.actuator_bounds_helpers import (
    clip_fraction,
    measure_clip_fractions,
    position_actuator_ids,
)
from environments.trex.envs.trex_env import TRexEnv

FORCERANGE_KP_RATIO = 0.8  # spike-cap sizing from the July 2026 hardening
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
    env = TRexEnv()
    try:
        yield env.model
    finally:
        env.close()


class TestForceBoundsConfiguration:
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


class TestStaticSaturation:
    def test_no_saturation_during_settle(self):
        """Settling from home never clips forces (the caps are for spikes, not posture)."""
        env = TRexEnv()
        try:
            frac = measure_clip_fractions(env, gait=False, steps=1000)
            worst = max(frac[i] for i in position_actuator_ids(env.model))
            assert worst < 0.01, f"static settle saturates a position actuator {worst:.1%} of the time"
        finally:
            env.close()


class TestDynamicSaturation:
    def test_gait_excitation_stays_unclipped(self):
        """Guards the gait headroom that stage-2 locomotion depends on.

        At 0.8x kp the caps clipped hips 44-50%, knees 10-14%, and ankles
        ~31% of a moderate 2.5 Hz gait cycle; at 1.5x kp the same
        excitation measures <1%. If saturation rises past these ceilings,
        the plant lost gait headroom — see
        docs/investigations/STAGE2_RECOMMENDATIONS.md before re-sizing.
        """
        env = TRexEnv()
        try:
            model = env.model
            frac = measure_clip_fractions(env, gait=True, steps=2000)
            hip = max(clip_fraction(frac, model, f"{s}_hip_pitch_act") for s in "rl")
            knee = max(clip_fraction(frac, model, f"{s}_knee_act") for s in "rl")
            ankle = max(clip_fraction(frac, model, f"{s}_ankle_act") for s in "rl")
            assert hip < 0.10, f"hip saturation {hip:.1%} — plant lost gait headroom, stage 2 will clip"
            assert knee < 0.05, f"knee saturation {knee:.1%} — plant lost gait headroom, stage 2 will clip"
            assert ankle < 0.05, f"ankle saturation {ankle:.1%} — plant lost gait headroom, stage 2 will clip"
        finally:
            env.close()
