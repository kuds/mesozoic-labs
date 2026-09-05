"""Tests for the P5 freeze: the calibrated judge and the resolution producer.

Three things are pinned here, and they are the three ways P5 could rot:

* the **calibrated judge** is opt-in and single-sourced — passing no
  ``height_reference`` reproduces the pre-P5 reset-stamp panel exactly, a
  float replaces it, and the calibrated safe set has exactly one definition
  that the harnesses and the producer share;
* the **frozen decisions** survive the round trip through JSON — the
  thresholds a gate reads back are the ones §9 justifies, not whatever a
  later edit put in the file;
* the **producer fails closed** — the resolution it writes is accepted by
  ``require_gate_resolution`` at the task it was measured under, and
  refused when the file is edited or the task moves.

Panels here are deliberately tiny (two episodes on a short horizon).  The
frozen record is 40 episodes; what these tests check is mechanism, and a
40-episode roll would buy nothing but minutes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from environments.shared import recovery_evaluation
from environments.shared.curriculum.gate_resolver import (
    GateResolutionError,
    require_gate_resolution,
    thresholds_from_resolution,
)
from environments.shared.curriculum.recovery_gate import RecoveryGateThresholds
from environments.shared.harnesses import freeze_recovery_gate as producer
from environments.shared.harnesses import recovery_offdist_panel as offdist
from environments.shared.recovery_evaluation import (
    CALIBRATED_HEIGHT_REFERENCE_M,
    CALIBRATED_POSTURE_ONLY,
    DEFAULT_SAFE_SET,
    _safe_step,
    roll_recovery_panel,
    zero_action_controller,
)

#: The same short-horizon pushed task the panel-harness tests use: several
#: judged shoves per episode inside 120 steps.
_SHORT_PUSH = {
    "perturbation_capture_velocity_multiple": 1.5,
    "perturbation_interval": 0.3,
    "perturbation_jitter": 0.05,
    "perturbation_duration": 0.1,
}


def _pushed_env():
    from environments.trex.envs.trex_env import TRexEnv

    return TRexEnv(reset_noise_scale=0.0, max_episode_steps=120, **_SHORT_PUSH)


def _roll(env, *, episodes=2, seed=3042, **kwargs):
    return roll_recovery_panel(
        env,
        zero_action_controller(env.action_space.shape[0]),
        controller_id="zero_action",
        episodes=episodes,
        seed=seed,
        t_recover_steps=20,
        dwell_steps=10,
        **kwargs,
    )


class TestHeightReference:
    """``height_reference``: None is the old judge, a float is the new one."""

    def test_omitting_it_and_passing_none_are_the_same_panel(self):
        assert _roll(_pushed_env()) == _roll(_pushed_env(), height_reference=None)

    def test_a_float_reference_reproduces_none_when_it_equals_the_reset_height(self):
        """The calibrated path differs from the old one ONLY in the reference.

        With reset noise off the reset stamp is the same every episode, so
        stamping that same value as a fixed reference must reproduce the
        reset-stamp panel row for row — which is what makes the default
        byte-identical to the pre-P5 behaviour.
        """
        env = _pushed_env()
        env.reset(seed=3042)
        reset_height = float(env.data.qpos[2])
        assert _roll(_pushed_env()) == _roll(_pushed_env(), height_reference=reset_height)

    def test_a_float_reference_is_actually_judged_against(self):
        """An unreachable reference denies every shove; the default does not."""
        baseline = _roll(_pushed_env())
        assert any(shove.recovered for shove in baseline.shoves), "fixture must recover something to be a control"

        env = _pushed_env()
        shifted = _roll(env, height_reference=0.0)
        assert [shove.recovered for shove in shifted.shoves] == [False] * len(shifted.shoves)
        assert not any(record.success for record in shifted.episodes)


class TestCalibratedSafeSetIsSingleSourced:
    """One definition of "calibrated", imported everywhere it is used."""

    def test_measured_values(self):
        # First-runs record §4.1 (quiet certified stance's p99.9 x 1.5) and
        # the measured settled median pelvis height.
        assert CALIBRATED_POSTURE_ONLY == {
            "height_error_max_m": 0.0168,
            "tilt_max_rad": 0.0825,
            "planar_speed_max_mps": 0.3203,
            "min_foot_force_n": 0.0,
        }
        assert CALIBRATED_HEIGHT_REFERENCE_M == 0.9267

    def test_the_harness_and_the_producer_share_the_definition(self):
        assert offdist.CALIBRATED_POSTURE_ONLY is recovery_evaluation.CALIBRATED_POSTURE_ONLY
        assert producer.CALIBRATED_POSTURE_ONLY is recovery_evaluation.CALIBRATED_POSTURE_ONLY
        assert offdist.CALIBRATED_HEIGHT_REFERENCE_M == CALIBRATED_HEIGHT_REFERENCE_M
        assert producer.CALIBRATED_HEIGHT_REFERENCE_M == CALIBRATED_HEIGHT_REFERENCE_M

    def test_the_hand_harness_uses_the_frozen_panel_geometry(self):
        """A hand panel judged on a different clock could not be compared."""
        assert (offdist.T_RECOVER_STEPS, offdist.DWELL_STEPS) == (producer.T_RECOVER_STEPS, producer.DWELL_STEPS)
        assert (offdist.PANEL_EPISODES, offdist.PANEL_SEED) == (producer.MIN_EVAL_EPISODES, producer.PANEL_SEED_START)

    def test_the_support_clause_is_vacuous_under_the_calibrated_set(self, monkeypatch):
        """§4.2: certified stance itself reads 0.0 N, so support cannot gate.

        ``min_foot_force_n = 0.0`` expresses "posture-only" without changing
        the predicate's shape: the support clause still runs, and still
        passes, on a foot reading nothing.
        """
        env = _pushed_env()
        env.reset(seed=3042)
        height_target = float(env.data.qpos[2])
        monkeypatch.setattr(env, "_foot_contact_forces", lambda: (0.0, 0.0))
        assert _safe_step(env, dict(CALIBRATED_POSTURE_ONLY), height_target) is True
        assert _safe_step(env, dict(DEFAULT_SAFE_SET), height_target) is False


class TestNumpyForwardPass:
    """The SB3-free inference path §9's panels were produced through."""

    @staticmethod
    def _weights(scale: float) -> dict[str, np.ndarray]:
        identity = np.eye(2)
        return {
            "mlp_extractor.policy_net.0.weight": identity,
            "mlp_extractor.policy_net.0.bias": np.zeros(2),
            "mlp_extractor.policy_net.2.weight": identity,
            "mlp_extractor.policy_net.2.bias": np.zeros(2),
            "action_net.weight": scale * identity,
            "action_net.bias": np.zeros(2),
        }

    def test_two_tanh_layers_then_a_linear_head(self):
        observation = np.array([0.5, -0.25])
        action = producer.numpy_deterministic_action(
            self._weights(2.0),
            observation,
            action_low=np.array([-10.0, -10.0]),
            action_high=np.array([10.0, 10.0]),
        )
        assert action == pytest.approx(2.0 * np.tanh(np.tanh(observation)))

    def test_the_action_is_clipped_to_the_action_space(self):
        action = producer.numpy_deterministic_action(
            self._weights(50.0),
            np.array([1.0, -1.0]),
            action_low=np.array([-0.4, -0.4]),
            action_high=np.array([0.4, 0.4]),
        )
        assert action == pytest.approx(np.array([0.4, -0.4]))

    def test_observations_are_standardized_then_clipped(self):
        stats = {
            "mean": np.array([1.0, 0.0]),
            "var": np.array([4.0, 1.0]),
            "clip_obs": 2.0,
            "epsilon": 0.0,
        }
        normalized = producer.normalize_observation(np.array([3.0, 100.0]), stats)
        assert normalized == pytest.approx(np.array([1.0, 2.0]))


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One real freeze of the trex recovery stage, at a rehearsal panel size."""
    stage_dir = tmp_path_factory.mktemp("recovery_stage")
    return producer.freeze_recovery_gate(stage_dir, episodes=2, seed=3042)


class TestProducer:
    def test_the_written_resolution_is_accepted_at_the_frozen_task(self, frozen):
        loaded = require_gate_resolution(frozen.path.parent, current_task_sha256=frozen.task_fingerprint["task_sha256"])
        assert loaded == frozen.resolution
        assert loaded["resolution_sha256"].startswith("sha256:")

    def test_the_capability_spec_carries_the_p5_decisions(self, frozen):
        """The §9 thresholds survive the JSON round trip a gate reads back."""
        loaded = require_gate_resolution(frozen.path.parent, current_task_sha256=frozen.task_fingerprint["task_sha256"])
        assert thresholds_from_resolution(loaded) == RecoveryGateThresholds(
            min_recovery_success_lcb=0.30,
            t_recover_steps=100,
            dwell_steps=50,
            min_eval_episodes=40,
            min_paired_success_delta_lcb=0.20,
        )
        assert loaded["capability_spec"]["gate_kind"] == "recovery_quality/v1"
        assert loaded["decision_procedure"]["panel_seed_start"] == 3042

    def test_the_null_manifest_is_the_statue_under_the_calibrated_judge(self, frozen):
        entry = frozen.resolution["null_manifest"]["zero_action"]
        # "zero_action" is the id the resolver's paired criterion defaults to.
        assert entry["safe_set"] == dict(CALIBRATED_POSTURE_ONLY)
        assert entry["n_episodes"] == 2
        assert sorted(int(seed) for seed in entry["successes_by_seed"]) == [3042, 3043]
        assert frozen.null_evidence["zero_action"].safe_set == dict(CALIBRATED_POSTURE_ONLY)

    def test_the_report_says_loudly_what_is_missing(self, frozen):
        report = producer.summarize(frozen)
        assert "STATUE NULL ONLY" in report
        assert "brace" not in frozen.resolution["null_manifest"]
        # A rehearsal-sized panel must not be mistaken for the frozen record.
        assert "REHEARSAL" in report
        assert str(CALIBRATED_HEIGHT_REFERENCE_M) in report

    def test_a_tampered_resolution_is_refused(self, frozen, tmp_path):
        resolution = json.loads(frozen.path.read_text(encoding="utf-8"))
        resolution["capability_spec"]["min_recovery_success_lcb"] = 0.01
        (tmp_path / "gate_resolution.json").write_text(json.dumps(resolution, indent=2), encoding="utf-8")
        with pytest.raises(GateResolutionError, match="integrity hash"):
            require_gate_resolution(tmp_path, current_task_sha256=frozen.task_fingerprint["task_sha256"])

    def test_a_different_task_blocks(self, frozen):
        with pytest.raises(GateResolutionError, match="Recalibrate"):
            require_gate_resolution(frozen.path.parent, current_task_sha256="sha256:some-other-task")

    def test_the_frozen_task_is_the_stage_fingerprint(self, frozen):
        assert frozen.resolution["task_sha256"] == producer.stage_task_fingerprint("trex", "recovery")["task_sha256"]

    def test_half_a_brace_configuration_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="must be supplied together"):
            producer.freeze_recovery_gate(tmp_path, policy_zip="model.zip", vecnorm=None)

    def test_refreezing_is_guarded_and_reproducible(self, frozen):
        with pytest.raises(ValueError, match="already exists"):
            producer.freeze_recovery_gate(frozen.path.parent, episodes=2, seed=3042)
        again = producer.freeze_recovery_gate(frozen.path.parent, episodes=2, seed=3042, replace=True)
        # Deterministic panels mean a re-freeze is byte-identical: a changed
        # digest would mean the "frozen" record depends on when it was made.
        assert again.resolution == frozen.resolution
