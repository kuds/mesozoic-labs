"""Plant-characterization tests for the Dibothrosuchus's bounded actuators.

Undersized ``forcerange`` caps are a documented stage-2 failure mode in this
repository: caps sized only against settling under the XML home-keyframe
controls clipped 20-40% of leg torque under gait excitation and collapsed
velociraptor and brachiosaurus locomotion (see
docs/investigations/STAGE2_RECOMMENDATIONS.md).

This plant is sized from that lesson rather than into it.  A 1.24 m animal
strides fast relative to its body length, so all three stance joints -- hip
pitch, knee, and ankle -- carry 1.5x-kp headroom on every leg, not just the
hips.  The remaining actuators keep a 0.7x-kp spike cap.  Measured on the
current model, clipping is 0.0% under home-control settling, a 1.5 Hz walk and
a 2.5 Hz trot, and 0.5% worst-case under sprint-like 4 Hz full-amplitude
excitation.

Run ``environments/shared/scripts/actuator_saturation_report.py
dibothrosuchus`` for the full per-actuator numbers on the current model.
"""

import mujoco
import pytest

from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared.tests.actuator_bounds_helpers import (
    PositionActuatorConfigurationBase,
    clip_fraction,
    measure_clip_fractions,
    position_actuator_ids,
)

SPIKE_CAP_RATIO = 0.7
GAIT_HEADROOM_RATIO = 1.5
GAIT_HEADROOM_SUFFIXES = ("hip_pitch_act", "knee_act", "ankle_act")
LEG_PREFIXES = ("fr_", "fl_", "rr_", "rl_")


@pytest.fixture(scope="module")
def model():
    env = DibothrosuchusEnv()
    try:
        yield env.model
    finally:
        env.close()


def _is_gait_critical(name: str) -> bool:
    return name.startswith(LEG_PREFIXES) and name.endswith(GAIT_HEADROOM_SUFFIXES)


class TestForceBoundsConfiguration(PositionActuatorConfigurationBase):
    def test_integrator_is_implicitfast(self, model):
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    def test_all_position_actuators_have_forcerange(self, model):
        for i in position_actuator_ids(model):
            fr = model.actuator_forcerange[i]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            assert fr[1] > 0, f"{name} has no forcerange upper bound"
            assert fr[0] == -fr[1], f"{name} forcerange is not symmetric"

    def test_forcerange_matches_documented_sizing(self, model):
        """Stance joints carry 1.5x-kp gait headroom; everything else is a 0.7x spike cap."""
        for i in position_actuator_ids(model):
            kp = model.actuator_gainprm[i, 0]
            fr = model.actuator_forcerange[i, 1]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            expected = GAIT_HEADROOM_RATIO if _is_gait_critical(name) else SPIKE_CAP_RATIO
            assert fr / kp == pytest.approx(expected, abs=0.02), (
                f"{name}: forcerange/kp = {fr / kp:.2f}, expected ~{expected}. If this changed "
                "intentionally, update the ratios here and the saturation numbers in this module's docstring."
            )

    def test_every_leg_has_all_three_gait_critical_actuators(self, model):
        found = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in position_actuator_ids(model)
            if _is_gait_critical(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        }
        expected = {f"{prefix}{suffix}" for prefix in LEG_PREFIXES for suffix in GAIT_HEADROOM_SUFFIXES}
        assert found == expected, (
            f"gait-critical actuator set drifted: missing={expected - found}, extra={found - expected}"
        )


class TestHomeControlSaturation:
    def test_no_saturation_during_home_control_settle(self):
        """Settling under XML home-keyframe controls never clips forces."""
        env = DibothrosuchusEnv()
        try:
            frac = measure_clip_fractions(env, mode="home_control", steps=1000)
            worst = max(frac[i] for i in position_actuator_ids(env.model))
            assert worst < 0.01, f"home-control settle saturates a position actuator {worst:.1%} of the time"
        finally:
            env.close()


class TestDynamicSaturation:
    def test_trot_excitation_stays_unclipped(self):
        """Guards the stance headroom that stage-2 locomotion depends on.

        A diagonal-pair trot at 2.5 Hz measures 0% clipping on every actuator
        with the committed sizing. If any stance joint rises past the ceiling
        the plant has lost gait headroom — read
        docs/investigations/STAGE2_RECOMMENDATIONS.md before re-sizing.
        """
        env = DibothrosuchusEnv()
        try:
            model = env.model
            frac = measure_clip_fractions(env, mode="gait", steps=2000)
            for suffix in GAIT_HEADROOM_SUFFIXES:
                worst = max(clip_fraction(frac, model, f"{prefix}{suffix}") for prefix in LEG_PREFIXES)
                assert worst < 0.10, f"{suffix} saturation {worst:.1%} — plant lost gait headroom, stage 2 will clip"
        finally:
            env.close()

    def test_sprint_excitation_stays_unclipped(self):
        """4 Hz full-amplitude excitation, the sprint-like regime that caught
        the velociraptor knee at its original sizing."""
        env = DibothrosuchusEnv()
        try:
            frac = measure_clip_fractions(env, mode="gait", steps=2000, hz=4.0, amplitude=1.0)
            worst = max(frac[i] for i in position_actuator_ids(env.model))
            assert worst < 0.05, f"sprint excitation saturates a position actuator {worst:.1%} of the time"
        finally:
            env.close()
