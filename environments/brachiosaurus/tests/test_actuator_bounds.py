"""Plant-characterization tests for the brachiosaurus's bounded actuators.

The July 2026 hardening sized position-actuator ``forcerange`` at
~0.62-0.73x kp against *static* settling only. Under a diagonal-pair
trot that clipped 20-33% of hip-pitch torque even at a slow 1.5 Hz walk
cycle — the same plant defect that collapsed velociraptor stage-2 twice
(see docs/investigations/STAGE2_RECOMMENDATIONS.md), and this species'
only stage-2 pass was a 1.12 m/s walk. The four hip-pitch actuators were
re-sized to 1.5x kp, which measures ~0% gait clipping; knees, ankles,
and toes measured 0% at their original sizing and keep it. These tests
pin that contract.

Run ``environments/shared/scripts/actuator_saturation_report.py
brachiosaurus`` for the full per-actuator numbers on the current model.
"""

import mujoco
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.tests.actuator_bounds_helpers import (
    clip_fraction,
    measure_clip_fractions,
    position_actuator_ids,
)

# Non-raised actuators keep the July 2026 spike-cap sizing, which varies
# ~0.62-0.73x kp across the model rather than a single documented ratio.
SPIKE_CAP_RATIO_RANGE = (0.55, 0.85)
GAIT_HEADROOM_RATIO = 1.5
GAIT_HEADROOM_ACTUATORS = frozenset({"fr_hip_pitch_act", "fl_hip_pitch_act", "rr_hip_pitch_act", "rl_hip_pitch_act"})
LEG_PREFIXES = ("fr_", "fl_", "rr_", "rl_")


@pytest.fixture(scope="module")
def model():
    env = BrachioEnv()
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

    def test_forcerange_matches_documented_sizing(self, model):
        """Hip pitches carry 1.5x-kp gait headroom; everything else stays a spike cap."""
        lo, hi = SPIKE_CAP_RATIO_RANGE
        for i in position_actuator_ids(model):
            kp = model.actuator_gainprm[i, 0]
            fr = model.actuator_forcerange[i, 1]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            ratio = fr / kp
            if name in GAIT_HEADROOM_ACTUATORS:
                assert ratio == pytest.approx(GAIT_HEADROOM_RATIO, abs=0.05), (
                    f"{name}: forcerange/kp = {ratio:.2f}, expected ~{GAIT_HEADROOM_RATIO}. "
                    "If this changed intentionally, update the ratios here and the "
                    "saturation numbers in docs/investigations/STAGE2_RECOMMENDATIONS.md."
                )
            else:
                assert lo <= ratio <= hi, (
                    f"{name}: forcerange/kp = {ratio:.2f}, outside the documented spike-cap band [{lo}, {hi}]."
                )


class TestStaticSaturation:
    def test_no_saturation_during_settle(self):
        """Settling from home never clips forces (the caps are for spikes, not posture)."""
        env = BrachioEnv()
        try:
            frac = measure_clip_fractions(env, gait=False, steps=1000)
            worst = max(frac[i] for i in position_actuator_ids(env.model))
            assert worst < 0.01, f"static settle saturates a position actuator {worst:.1%} of the time"
        finally:
            env.close()


class TestDynamicSaturation:
    def test_trot_excitation_stays_unclipped(self):
        """Guards the hip headroom that stage-2 locomotion depends on.

        At ~0.65x kp the hip caps clipped 20-33% of a slow 1.5 Hz walk
        and 44-51% of a 2.5 Hz trot; at 1.5x kp both measure ~0%. Knees
        keep their spike caps and may graze them (~1% at 1.5 Hz). If hip
        saturation rises past the ceiling, the plant lost gait headroom —
        see docs/investigations/STAGE2_RECOMMENDATIONS.md before
        re-sizing.
        """
        env = BrachioEnv()
        try:
            model = env.model
            frac = measure_clip_fractions(env, gait=True, steps=2000)
            hip = max(clip_fraction(frac, model, f"{p}hip_pitch_act") for p in LEG_PREFIXES)
            assert hip < 0.10, f"hip saturation {hip:.1%} — plant lost gait headroom, stage 2 will clip"
        finally:
            env.close()
