"""P5: the frozen recovery gate, wired into the paths that judge stages.

Before P5 the three judging paths only NAMED the resolver: a
``recovery_quality/v1`` stage failed closed everywhere, and nothing ever
called
:func:`~environments.shared.curriculum.gate_resolver.evaluate_recovery_gate_from_resolution`.
These tests pin the wiring and, more importantly, that wiring it up weakened
nothing:

* the reporting path advances a recovery stage ONLY through the frozen
  resolution — absence, tampering, staleness, and missing panel evidence are
  each a refusal that names what is wrong;
* the in-training paths (SB3 ``CurriculumManager`` and JAX
  ``check_stage_gate``) still refuse outright, because neither can obtain the
  stage directory, the current task fingerprint, or a panel pairable against
  the frozen nulls;
* unknown kinds and ``none/v1`` refuse exactly as before, and the
  ``reward_and_length/v1`` / ``stance_quality/v1`` verdicts are untouched.

The fixtures below are the MEASURED panels, not invented ones, so a passing
verdict here is the verdict the real evidence produces.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pytest

from environments.shared.curriculum import CurriculumManager
from environments.shared.curriculum.gate_resolver import (
    build_gate_resolution,
    write_gate_resolution,
)
from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.curriculum.recovery_gate import (
    RECOVERY_GATE_KIND,
    RecoveryGateThresholds,
    binomial_lcb,
    paired_difference_lcb,
)
from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND, StancePanel
from environments.shared.jax_curriculum import check_stage_gate
from environments.shared.recovery_evaluation import (
    CALIBRATED_POSTURE_ONLY,
    EpisodeRecord,
    RecoveryPanelEvidence,
)
from environments.shared.reporting import evaluate_stage_gate

# ── The values P5 freezes ────────────────────────────────────────────────
# Derivation record: docs/investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md
# §9 (measured 2026-08-28, ten 40-episode panels) and §4 (P3 calibration).
#
# ``min_recovery_success_lcb = 0.30`` is a DECISION taken by the project owner
# on those measurements (§9.5): the attainable training-schedule LCB is 0.361
# for the 3M policy and 0.338 for the 5M one, against a null UCB95 of 0.072,
# so 0.30 certifies what the stage can actually do today; the plan's
# aspirational 0.725 needs a materially stronger policy.
MIN_RECOVERY_SUCCESS_LCB = 0.30
# ``min_paired_success_delta_lcb = 0.20`` likewise (§9.1): the measured paired
# policy-minus-statue LCB is +0.365 (3M) and +0.340 (5M) at the training
# schedule, on seeds shared with the frozen null panel.
MIN_PAIRED_SUCCESS_DELTA_LCB = 0.20
#: §4.1: measured re-entry times have p90 = 84 steps, comfortably inside the
#: 100-step window; the dwell is the 1.0 s the §3/§9 panels were judged with.
T_RECOVER_STEPS = 100
DWELL_STEPS = 50
#: §3/§9: every certification panel is 40 episodes on seeds 3042-3081.
MIN_EVAL_EPISODES = 40
PANEL_SEED_START = 3042

#: §9.1, training schedule (165.5 N, 2.0 ± 0.5 s): the statue is 0/40 and the
#: 3M policy is 20/40 — "use 20/40, not 21/40" (§9.5), because the live judge
#: measures one fewer marginal episode than §4.3's offline re-judge.
STATUE_PANEL = (False,) * 40
POLICY_PANEL = (True,) * 20 + (False,) * 20

#: Stand-in for the run's derived task fingerprint.  Its VALUE is arbitrary;
#: what matters is that the same string is recorded by the stage and by the
#: resolution, which is exactly the staleness check the resolver enforces.
TASK_SHA256 = "sha256:" + "a1" * 32

FROZEN_THRESHOLDS = RecoveryGateThresholds(
    min_recovery_success_lcb=MIN_RECOVERY_SUCCESS_LCB,
    t_recover_steps=T_RECOVER_STEPS,
    dwell_steps=DWELL_STEPS,
    min_eval_episodes=MIN_EVAL_EPISODES,
    min_paired_success_delta_lcb=MIN_PAIRED_SUCCESS_DELTA_LCB,
)

#: The stage's ``[curriculum]`` block as P5 freezes it: the same numbers the
#: resolution carries, because the gate refuses when the two disagree.
RECOVERY_CURRICULUM: dict[str, Any] = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": RECOVERY_GATE_KIND,
    "min_recovery_success_lcb": MIN_RECOVERY_SUCCESS_LCB,
    "recovery_t_recover_steps": T_RECOVER_STEPS,
    "recovery_dwell_steps": DWELL_STEPS,
    "min_paired_success_delta_lcb": MIN_PAIRED_SUCCESS_DELTA_LCB,
    "min_eval_episodes": MIN_EVAL_EPISODES,
}


def _episodes(successes: Sequence[bool], controller_id: str) -> tuple[EpisodeRecord, ...]:
    """Panel rows on the registered seeds; only seed and success are load-bearing."""
    return tuple(
        EpisodeRecord(
            controller_id=controller_id,
            episode=index + 1,
            panel_seed=PANEL_SEED_START + index,
            length=1000 if success else 360,
            full_horizon=bool(success),
            n_pushes=4,
            n_recovered=4 if success else 1,
            success=bool(success),
            reward=2799.0 if success else 975.0,
        )
        for index, success in enumerate(successes)
    )


def _null_evidence(successes: Sequence[bool] = STATUE_PANEL) -> RecoveryPanelEvidence:
    return RecoveryPanelEvidence(
        controller_id="zero_action",
        episodes=_episodes(successes, "zero_action"),
        shoves=(),
        # The P3-calibrated posture-only judge, imported rather than restated
        # so this fixture cannot drift from the canonical definition (§4.1;
        # no per-step support term, §4.2).
        safe_set=dict(CALIBRATED_POSTURE_ONLY),
    )


def _freeze_resolution(
    stage_dir: Path,
    *,
    task_sha256: str = TASK_SHA256,
    null_successes: Sequence[bool] = STATUE_PANEL,
) -> dict[str, Any]:
    resolution = build_gate_resolution(
        task_fingerprint={"task_sha256": task_sha256},
        thresholds=FROZEN_THRESHOLDS,
        null_evidence={"zero_action": _null_evidence(null_successes)},
        panel_seed_start=PANEL_SEED_START,
    )
    write_gate_resolution(stage_dir, resolution)
    return resolution


def _reseal(stage_dir: Path, resolution: dict[str, Any]) -> None:
    """Rewrite a resolution with a VALID integrity hash over edited contents.

    Used to reach the checks that live past the hash: a record can be
    internally consistent and still be unusable.
    """
    payload = {key: value for key, value in resolution.items() if key != "resolution_sha256"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload["resolution_sha256"] = f"sha256:{digest}"
    write_gate_resolution(stage_dir, payload)


def _stage_dir(
    tmp_path: Path,
    *,
    resolution: bool = True,
    fingerprint: bool = True,
    stage_task_sha256: str = TASK_SHA256,
    resolution_task_sha256: str = TASK_SHA256,
) -> Path:
    """A recovery stage directory as the trainer leaves it."""
    stage_dir = tmp_path / "02_recovery"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if fingerprint:
        # What config.save_stage_config writes beside the stage config.
        (stage_dir / "task_fingerprint.json").write_text(
            json.dumps({"schema": "mesozoic.task-fingerprint/v2", "task_sha256": stage_task_sha256}, indent=2) + "\n",
            encoding="utf-8",
        )
    if resolution:
        _freeze_resolution(stage_dir, task_sha256=resolution_task_sha256)
    return stage_dir


def _successes_by_seed(successes: Sequence[bool] = POLICY_PANEL, *, seed_start: int = PANEL_SEED_START):
    return {seed_start + index: bool(success) for index, success in enumerate(successes)}


def _judge(stage_dir: "Path | None", successes, curriculum=None, stage_results=None):
    return evaluate_stage_gate(
        dict(RECOVERY_CURRICULUM if curriculum is None else curriculum),
        stage_results or {},
        stage="recovery",
        stage_dir=stage_dir,
        recovery_successes_by_seed=successes,
    )


class TestFrozenValuesAreTheMeasuredOnes:
    """The fixtures reproduce §9, so the wiring is tested on real numbers."""

    def test_the_measured_panel_clears_the_frozen_thresholds_by_the_recorded_margins(self):
        assert binomial_lcb(20, 40) == pytest.approx(0.361, abs=5e-4)
        paired = [1.0] * 20 + [0.0] * 20  # policy 20/40 against a 0/40 statue
        assert paired_difference_lcb(paired) == pytest.approx(0.365, abs=5e-4)
        # Attainable, which is the whole point of the P5 decision.
        assert binomial_lcb(20, 40) > MIN_RECOVERY_SUCCESS_LCB
        assert paired_difference_lcb(paired) > MIN_PAIRED_SUCCESS_DELTA_LCB


class TestReportingPathJudgesThroughTheFrozenResolution:
    """``reporting.gates`` is the path that can reach the frozen evidence."""

    def test_a_valid_resolution_and_passing_panel_advances(self, tmp_path):
        passed, failures = _judge(_stage_dir(tmp_path), _successes_by_seed())
        assert passed is True
        assert failures == []

    def test_a_panel_below_the_frozen_lcb_refuses_naming_the_criterion(self, tmp_path):
        # 9/40 — the LCB is 0.123, under the frozen 0.30, and the paired LCB
        # (0.112) is under 0.20 as well: both criteria must name themselves.
        weak = (True,) * 9 + (False,) * 31
        passed, failures = _judge(_stage_dir(tmp_path), _successes_by_seed(weak))
        assert passed is False
        assert any("recovery_success_lcb" in failure for failure in failures)
        assert any("paired_success_delta_lcb" in failure for failure in failures)

    def test_a_missing_resolution_refuses(self, tmp_path):
        passed, failures = _judge(_stage_dir(tmp_path, resolution=False), _successes_by_seed())
        assert passed is False
        assert any("have not been measured" in failure for failure in failures)

    def test_a_tampered_resolution_refuses(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        path = stage_dir / "gate_resolution.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        # The edit a tamperer would make: loosen the bar, keep the hash.
        record["capability_spec"]["min_recovery_success_lcb"] = 0.01
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        passed, failures = _judge(stage_dir, _successes_by_seed())
        assert passed is False
        assert any("integrity hash" in failure for failure in failures)

    def test_a_task_sha256_mismatch_refuses_and_demands_recalibration(self, tmp_path):
        stage_dir = _stage_dir(tmp_path, resolution_task_sha256="sha256:" + "b2" * 32)
        passed, failures = _judge(stage_dir, _successes_by_seed())
        assert passed is False
        assert any("Recalibrate" in failure for failure in failures)

    def test_a_stage_with_no_recorded_task_refuses(self, tmp_path):
        """An uncomparable task is treated as a stale one, never as a match."""
        stage_dir = _stage_dir(tmp_path, fingerprint=False)
        passed, failures = _judge(stage_dir, _successes_by_seed())
        assert passed is False
        assert any("records no task fingerprint" in failure for failure in failures)

    def test_the_task_may_also_be_read_from_the_stage_config_snapshot(self, tmp_path):
        """``save_stage_config`` writes it twice; either one answers."""
        stage_dir = _stage_dir(tmp_path, fingerprint=False)
        (stage_dir / "stage_config.json").write_text(
            json.dumps({"stage": "recovery", "task_fingerprint": {"task_sha256": TASK_SHA256}}) + "\n",
            encoding="utf-8",
        )
        passed, failures = _judge(stage_dir, _successes_by_seed())
        assert (passed, failures) == (True, [])

    def test_absent_panel_evidence_refuses_rather_than_counting_as_success(self, tmp_path):
        passed, failures = _judge(_stage_dir(tmp_path), None)
        assert passed is False
        assert any("no pushed-panel evidence" in failure for failure in failures)

    def test_an_empty_panel_refuses(self, tmp_path):
        passed, failures = _judge(_stage_dir(tmp_path), {})
        assert passed is False
        assert any("carried no episodes" in failure for failure in failures)

    def test_a_caller_that_passes_no_stage_dir_refuses(self, tmp_path):
        """The default call site cannot reach a resolution, so it must refuse."""
        passed, failures = evaluate_stage_gate(
            dict(RECOVERY_CURRICULUM),
            {"best_model_reward": 1e9},
            stage="recovery",
        )
        assert passed is False
        assert any("no stage_dir" in failure for failure in failures)

    def test_a_panel_on_other_seeds_cannot_be_paired_and_refuses(self, tmp_path):
        """Pairing is the estimand: unshared seeds are a defect, not a subset."""
        passed, failures = _judge(_stage_dir(tmp_path), _successes_by_seed(seed_start=9000))
        assert passed is False
        assert any("do not match the frozen null panel seeds" in failure for failure in failures)

    def test_a_short_panel_cannot_be_paired_and_refuses(self, tmp_path):
        passed, failures = _judge(_stage_dir(tmp_path), _successes_by_seed(POLICY_PANEL[:10]))
        assert passed is False
        assert any("do not match the frozen null panel seeds" in failure for failure in failures)

    def test_a_panel_under_the_frozen_panel_size_refuses(self, tmp_path):
        """Even a perfect 10/10 is refused: the bound's power needs 40."""
        stage_dir = _stage_dir(tmp_path, resolution=False)
        _freeze_resolution(stage_dir, null_successes=STATUE_PANEL[:10])
        passed, failures = _judge(stage_dir, _successes_by_seed((True,) * 10))
        assert passed is False
        assert any("min_eval_episodes 40" in failure for failure in failures)

    def test_an_unreadable_resolution_refuses_instead_of_raising(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        (stage_dir / "gate_resolution.json").write_text("{not json at all", encoding="utf-8")
        passed, failures = _judge(stage_dir, _successes_by_seed())
        assert passed is False
        assert any("could not be read" in failure for failure in failures)

    def test_a_structurally_incomplete_spec_refuses_instead_of_raising(self, tmp_path):
        """Internally consistent is not the same as usable."""
        stage_dir = _stage_dir(tmp_path, resolution=False)
        resolution = _freeze_resolution(stage_dir)
        del resolution["capability_spec"]["min_eval_episodes"]
        _reseal(stage_dir, resolution)
        curriculum = {key: value for key, value in RECOVERY_CURRICULUM.items() if key != "min_eval_episodes"}
        passed, failures = _judge(stage_dir, _successes_by_seed(), curriculum=curriculum)
        assert passed is False
        assert any("could not be read" in failure for failure in failures)

    def test_a_config_edited_away_from_the_frozen_spec_refuses(self, tmp_path):
        """Judging by the frozen record must not silently ignore the config."""
        curriculum = dict(RECOVERY_CURRICULUM, min_recovery_success_lcb=0.725)
        passed, failures = _judge(_stage_dir(tmp_path), _successes_by_seed(), curriculum=curriculum)
        assert passed is False
        assert any("Re-resolve the gate" in failure for failure in failures)

    def test_a_declared_reward_rail_is_enforced_as_an_extra_conjunct(self, tmp_path):
        """The frozen spec has no rail, so a declared one is checked here."""
        curriculum = dict(RECOVERY_CURRICULUM, min_avg_reward=1000.0)
        stage_dir = _stage_dir(tmp_path)
        passed, failures = _judge(stage_dir, _successes_by_seed(), curriculum=curriculum, stage_results={})
        assert passed is False
        assert any("no reward measurement" in failure for failure in failures)

        passed, failures = _judge(
            stage_dir,
            _successes_by_seed(),
            curriculum=curriculum,
            stage_results={"best_model_reward": 500.0},
        )
        assert passed is False
        assert any("recovery rail" in failure for failure in failures)

        passed, failures = _judge(
            stage_dir,
            _successes_by_seed(),
            curriculum=curriculum,
            stage_results={"best_model_reward": 2900.0},
        )
        assert (passed, failures) == (True, [])

    def test_a_rail_cannot_rescue_a_failing_recovery_verdict(self, tmp_path):
        """The rail only ever refuses; it never advances anything."""
        curriculum = dict(RECOVERY_CURRICULUM, min_avg_reward=1000.0)
        passed, _ = _judge(
            _stage_dir(tmp_path),
            _successes_by_seed(STATUE_PANEL),
            curriculum=curriculum,
            stage_results={"best_model_reward": 1e9},
        )
        assert passed is False


class TestReportingPathStillFailsClosedElsewhere:
    """Wiring the resolver in must not soften any other branch."""

    def test_an_unknown_kind_still_refuses_even_with_full_evidence(self, tmp_path):
        passed, failures = _judge(
            _stage_dir(tmp_path),
            _successes_by_seed(),
            curriculum={"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "made_up/v9"},
        )
        assert passed is False
        assert any("unknown gate_kind" in failure for failure in failures)

    def test_none_v1_still_refuses_even_with_full_evidence(self, tmp_path):
        passed, failures = _judge(
            _stage_dir(tmp_path),
            _successes_by_seed(),
            curriculum={"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "none/v1"},
        )
        assert passed is False
        assert any("non-advancing pilot" in failure for failure in failures)

    def test_an_undeclared_gate_still_refuses(self, tmp_path):
        passed, failures = _judge(_stage_dir(tmp_path), _successes_by_seed(), curriculum={})
        assert passed is False
        assert any("declares no gate_kind" in failure for failure in failures)

    def test_reward_and_length_is_unaffected_by_the_new_arguments(self, tmp_path):
        curriculum = {
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "gate_kind": "reward_and_length/v1",
            "min_avg_reward": 1950.0,
        }
        without = evaluate_stage_gate(curriculum, {"best_model_reward": 2000.0}, stage=1)
        with_new = evaluate_stage_gate(
            curriculum,
            {"best_model_reward": 2000.0},
            stage=1,
            stage_dir=_stage_dir(tmp_path),
            recovery_successes_by_seed=_successes_by_seed(),
        )
        assert without == (True, [])
        assert with_new == without

        below = evaluate_stage_gate(
            curriculum,
            {"best_model_reward": 1000.0},
            stage=1,
            stage_dir=_stage_dir(tmp_path),
            recovery_successes_by_seed=_successes_by_seed(),
        )
        assert below[0] is False

    def test_stance_quality_is_unaffected_by_the_new_arguments(self, tmp_path):
        curriculum = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": STANCE_GATE_KIND}
        report = {"gate_kind": STANCE_GATE_KIND, "passed": True}
        without = evaluate_stage_gate(curriculum, {}, stage=1, stance_report=report)
        with_new = evaluate_stage_gate(
            curriculum,
            {},
            stage=1,
            stance_report=report,
            stage_dir=_stage_dir(tmp_path),
            recovery_successes_by_seed=_successes_by_seed(),
        )
        assert without == (True, [])
        assert with_new == without

        # And a stance stage with no panel still refuses, resolution or not.
        missing = evaluate_stage_gate(
            curriculum,
            {},
            stage=1,
            stage_dir=_stage_dir(tmp_path),
            recovery_successes_by_seed=_successes_by_seed(),
        )
        assert missing[0] is False
        assert any("no stance panel was measured" in failure for failure in missing[1])


def _manager(gate_kind: str, **threshold_fields: Any) -> CurriculumManager:
    return CurriculumManager(
        species="velociraptor",
        stage_thresholds={
            1: dict(
                {"gate_kind": gate_kind, "min_eval_episodes": 1, "required_consecutive": 1},
                **threshold_fields,
            )
        },
    )


class TestInTrainingManagerRefusesRecovery:
    """The SB3 curriculum callback cannot reach the frozen evidence."""

    def test_it_refuses_even_when_a_valid_resolution_exists_on_disk(self, tmp_path):
        """The manager knows a species and a stage number, not a directory."""
        _stage_dir(tmp_path)  # a perfectly good frozen resolution, unreachable
        manager = _manager(RECOVERY_GATE_KIND)
        for _ in range(5):
            assert not manager.should_advance([1e9] * 10, [1000.0] * 10)
        assert manager.summary()["consecutive_passes"][1] == 0

    def test_the_refusal_names_every_input_the_path_lacks(self, caplog):
        manager = _manager(RECOVERY_GATE_KIND)
        with caplog.at_level(logging.ERROR, logger="environments.shared.curriculum.manager"):
            assert not manager.should_advance([1e9] * 10, [1000.0] * 10)
        assert RECOVERY_GATE_KIND in caplog.text
        assert "cannot evaluate" in caplog.text
        assert "gate_resolution.json" in caplog.text
        assert "task_sha256" in caplog.text
        assert "3042-3081" in caplog.text

    def test_an_unknown_kind_still_refuses(self):
        manager = _manager("made_up/v9")
        for _ in range(5):
            assert not manager.should_advance([1e9] * 10, [1000.0] * 10)

    def test_none_v1_still_refuses(self):
        manager = _manager("none/v1")
        for _ in range(5):
            assert not manager.should_advance([1e9] * 10, [1000.0] * 10)

    def test_reward_and_length_still_advances(self):
        manager = _manager("reward_and_length/v1", min_avg_reward=100.0)
        assert manager.should_advance([150.0] * 10, [1000.0] * 10)

    def test_stance_quality_still_advances(self):
        manager = _manager(
            STANCE_GATE_KIND,
            min_full_horizon_fraction=0.9,
            max_unsupported_duty=0.05,
            max_unsupported_duty_ucb=0.08,
            min_eval_episodes=10,
        )
        panel = StancePanel(
            n_episodes=10,
            full_horizon_fraction=1.0,
            mean_reward=3000.0,
            n_duty_episodes=10,
            mean_unsupported_duty=0.01,
            unsupported_duty_ucb=0.02,
        )
        assert manager.should_advance([3000.0] * 10, [1000.0] * 10, None, None, panel)


class TestJaxCurriculumRefusesRecovery:
    """The MJX path has no pushed-panel roller and no stage directory."""

    def test_recovery_is_refused_with_a_message_naming_the_missing_inputs(self):
        config = {"stage": "recovery", "curriculum_kwargs": dict(RECOVERY_CURRICULUM)}
        with pytest.raises(GateSchemaError) as excinfo:
            check_stage_gate({"mean_episode_return": 1e9, "mean_episode_length": 1000.0}, config)
        message = str(excinfo.value)
        assert "cannot evaluate" in message
        assert "gate_resolution.json" in message
        assert "task_sha256" in message

    def test_a_declared_reward_rail_does_not_turn_it_into_a_reward_gate(self):
        config = {"stage": "recovery", "curriculum_kwargs": dict(RECOVERY_CURRICULUM, min_avg_reward=100.0)}
        with pytest.raises(GateSchemaError, match="cannot evaluate"):
            check_stage_gate({"mean_episode_return": 1e9}, config)

    def test_an_unknown_kind_and_none_v1_still_refuse(self):
        unknown = {"curriculum_kwargs": {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "made_up/v9"}}
        with pytest.raises(GateSchemaError, match="unknown gate_kind"):
            check_stage_gate({"mean_episode_return": 1e9}, unknown)
        pilot = {"curriculum_kwargs": {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "none/v1"}}
        with pytest.raises(GateSchemaError, match="non-advancing pilot"):
            check_stage_gate({"mean_episode_return": 1e9}, pilot)

    def test_reward_and_length_is_unaffected(self):
        config = {
            "stage": 1,
            "curriculum_kwargs": {
                "gate_schema_version": GATE_SCHEMA_VERSION,
                "gate_kind": "reward_and_length/v1",
                "min_avg_reward": 1950.0,
            },
        }
        assert check_stage_gate({"mean_episode_return": 2000.0}, config) is True
        assert check_stage_gate({"mean_episode_return": 1900.0}, config) is False

    def test_stance_quality_is_unaffected(self):
        config = {
            "stage": 1,
            "curriculum_kwargs": {
                "gate_schema_version": GATE_SCHEMA_VERSION,
                "gate_kind": STANCE_GATE_KIND,
                "min_full_horizon_fraction": 0.9,
                "max_unsupported_duty": 0.05,
                "max_unsupported_duty_ucb": 0.08,
                "min_eval_episodes": 10,
            },
        }
        metrics = {
            "n_eval_episodes": 10.0,
            "full_horizon_fraction": 1.0,
            "n_duty_episodes": 10.0,
            "mean_unsupported_duty": 0.01,
            "unsupported_duty_ucb": 0.02,
        }
        assert check_stage_gate(metrics, config) is True
        assert check_stage_gate(dict(metrics, unsupported_duty_ucb=0.5), config) is False
