"""Tests for the pushed-panel harness (W4 part 2) and gate resolver (W5).

Structure over outcomes: whether a statue survives a 165 N shove is a
physics question for the calibration runs; what these tests pin is that
the harness judges exactly the scheduled pushes, that identical seeds mean
identical pushes across controllers (the pairing the design rests on), and
that the resolver blocks on absence and staleness instead of skipping.
"""

from __future__ import annotations

import csv

import pytest

from environments.shared.curriculum.gate_resolver import (
    GateResolutionError,
    build_gate_resolution,
    evaluate_recovery_gate_from_resolution,
    paired_differences_from_resolution,
    require_gate_resolution,
    thresholds_from_resolution,
    write_gate_resolution,
)
from environments.shared.curriculum.recovery_gate import RecoveryGateThresholds, binomial_ucb
from environments.shared.recovery_evaluation import (
    paired_success_differences,
    roll_recovery_panel,
    write_recovery_evidence,
    zero_action_controller,
)

_SHORT_PUSH = {
    "perturbation_capture_velocity_multiple": 1.5,
    "perturbation_interval": 0.3,
    "perturbation_jitter": 0.05,
    "perturbation_duration": 0.1,
}


def _pushed_env():
    from environments.trex.envs.trex_env import TRexEnv

    return TRexEnv(reset_noise_scale=0.0, max_episode_steps=120, **_SHORT_PUSH)


def _roll(env, controller_id, episodes=2, seed=3042):
    return roll_recovery_panel(
        env,
        zero_action_controller(env.action_space.shape[0]),
        controller_id=controller_id,
        episodes=episodes,
        seed=seed,
        t_recover_steps=20,
        dwell_steps=10,
    )


class TestPanelHarness:
    def test_every_in_horizon_push_is_judged(self):
        env = _pushed_env()
        evidence = _roll(env, "policy")
        assert len(evidence.episodes) == 2
        for record in evidence.episodes:
            episode_shoves = [s for s in evidence.shoves if s.episode == record.episode]
            # Shove rows exist for exactly the in-horizon pushes, and the
            # episode row's counts reconcile with them.
            assert record.n_pushes == len(episode_shoves) > 0
            assert record.n_recovered == sum(1 for s in episode_shoves if s.recovered)
            for shove in episode_shoves:
                assert 0 <= shove.start_step < 120
                assert shove.end_step == shove.start_step + env._push_duration_steps
                assert shove.force_n == pytest.approx(env._push_force_n)
                assert shove.recovery_step == -1 or shove.recovery_step >= shove.end_step

    def test_same_seed_means_identical_evidence(self):
        a = _roll(_pushed_env(), "policy")
        b = _roll(_pushed_env(), "policy")
        assert a == b

    def test_pairing_identical_schedules_across_controllers(self):
        """The design's load-bearing property: same seeds, same pushes."""
        policy = _roll(_pushed_env(), "policy")
        null = _roll(_pushed_env(), "zero_action")

        def schedule(ev):
            return [(s.episode, s.push_index, s.start_step, s.direction_x, s.direction_y) for s in ev.shoves]

        assert schedule(policy) == schedule(null)

    def test_push_free_env_is_refused(self):
        from environments.trex.envs.trex_env import TRexEnv

        env = TRexEnv(reset_noise_scale=0.0, max_episode_steps=120)
        with pytest.raises(ValueError, match="perturbation-enabled"):
            _roll(env, "policy")

    def test_paired_differences_align_by_seed(self):
        policy = _roll(_pushed_env(), "policy")
        null = _roll(_pushed_env(), "zero_action")
        differences = paired_success_differences(policy, null)
        # Identical controllers on identical seeds: every difference is 0.
        assert differences == tuple([0.0] * len(policy.episodes))
        mismatched = _roll(_pushed_env(), "zero_action", seed=4000)
        with pytest.raises(ValueError, match="seed set"):
            paired_success_differences(policy, mismatched)

    def test_evidence_csvs_round_trip(self, tmp_path):
        evidence = _roll(_pushed_env(), "policy")
        paths = write_recovery_evidence(tmp_path, evidence)
        with paths["shoves"].open() as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == len(evidence.shoves)
        assert {"controller_id", "panel_seed", "push_index", "recovered", "recovery_step"} <= set(rows[0])


def _resolution(tmp_path, evidence, *, paired_delta=None, min_episodes=2):
    fingerprint = {"task_sha256": "sha256:task-a"}
    thresholds = RecoveryGateThresholds(
        min_recovery_success_lcb=0.01,
        t_recover_steps=20,
        dwell_steps=10,
        min_eval_episodes=min_episodes,
        min_paired_success_delta_lcb=paired_delta,
    )
    resolution = build_gate_resolution(
        task_fingerprint=fingerprint,
        thresholds=thresholds,
        null_evidence={"zero_action": evidence},
        panel_seed_start=3042,
    )
    write_gate_resolution(tmp_path, resolution)
    return resolution


class TestGateResolver:
    def test_round_trip_and_null_manifest_summary(self, tmp_path):
        evidence = _roll(_pushed_env(), "zero_action")
        resolution = _resolution(tmp_path, evidence)
        loaded = require_gate_resolution(tmp_path, current_task_sha256="sha256:task-a")
        assert loaded == resolution
        entry = loaded["null_manifest"]["zero_action"]
        assert entry["n_episodes"] == 2
        # The frozen headline is the exact bound, never the raw fraction.
        assert entry["success_ucb95"] == pytest.approx(binomial_ucb(entry["n_successes"], 2), abs=1e-9)
        assert thresholds_from_resolution(loaded).t_recover_steps == 20

    def test_missing_resolution_blocks(self, tmp_path):
        with pytest.raises(GateResolutionError, match="have not been measured"):
            require_gate_resolution(tmp_path, current_task_sha256="sha256:task-a")

    def test_stale_resolution_blocks_and_demands_recalibration(self, tmp_path):
        evidence = _roll(_pushed_env(), "zero_action")
        _resolution(tmp_path, evidence)
        with pytest.raises(GateResolutionError, match="Recalibrate"):
            require_gate_resolution(tmp_path, current_task_sha256="sha256:task-B")

    def test_paired_differences_come_from_the_frozen_panel(self, tmp_path):
        evidence = _roll(_pushed_env(), "zero_action")
        resolution = _resolution(tmp_path, evidence)
        null_seeds = {int(s) for s in resolution["null_manifest"]["zero_action"]["successes_by_seed"]}
        policy_outcomes = {seed: True for seed in null_seeds}
        differences = paired_differences_from_resolution(resolution, policy_outcomes, null_controller_id="zero_action")
        assert len(differences) == len(null_seeds)
        with pytest.raises(GateResolutionError, match="not in the frozen null manifest"):
            paired_differences_from_resolution(resolution, policy_outcomes, null_controller_id="brace")
        with pytest.raises(GateResolutionError, match="do not match"):
            paired_differences_from_resolution(resolution, {9999: True}, null_controller_id="zero_action")

    def test_end_to_end_verdict_from_the_frozen_resolution(self, tmp_path):
        evidence = _roll(_pushed_env(), "zero_action")
        resolution = _resolution(tmp_path, evidence, paired_delta=0.9)
        null_seeds = {int(s) for s in resolution["null_manifest"]["zero_action"]["successes_by_seed"]}
        null_successes = resolution["null_manifest"]["zero_action"]["successes_by_seed"]
        # A policy that succeeds wherever the null failed clears a paired
        # delta bar only if the null actually failed everywhere.
        policy_outcomes = {seed: True for seed in null_seeds}
        result = evaluate_recovery_gate_from_resolution(
            tmp_path,
            current_task_sha256="sha256:task-a",
            policy_successes_by_seed=policy_outcomes,
        )
        expected_pass = all(not v for v in null_successes.values())
        assert result.passed == expected_pass
