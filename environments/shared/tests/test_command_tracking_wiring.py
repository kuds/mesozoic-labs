"""``command_tracking/v1`` wired into the paths that judge, record and publish a stage (decision D-D61).

* **The judge**: ``reporting.gates.evaluate_stage_gate`` dispatches the kind
  to the command-tracking report, admits it only when it is bound to the
  stage (schema, gate, frozen normalization, the certification seed, the
  handoff pair, the handoff choice, task, horizon, control step, command and
  the reference's parent) and re-derives the verdict from its records;
  ``evaluate_recorded_gate`` reads ``None``; the in-training manager refuses
  the kind.
* **The handoff**: chosen by the gated statistic -- ``handoff_selection_key``
  ranks a passing panel above a failing one and then by the profile's bound,
  never by reward; ``choose_command_handoff`` records the winner, and
  ``select_handoff_checkpoint`` honors a valid record and ignores a stale one.
* **The evidence**: the record and the panel CSV round-trip exactly, and
  publication re-derives a pass from the CSV alone and refuses each binding
  it does not hold.
* **The readers**: the catalog renders the gate and its headline, the
  results CSV, the summary stage row and the stage summary carry the
  statistics.
* **End to end**: a temporary live compsognathus walk stage trains a few
  hundred PPO steps, chooses its handoff, and is judged through the real
  post-stage path (``generate_stage_artifacts``, and the CLI curriculum's
  post-training judge) on a tiny panel; both paths draw the command figures
  from the certified chart episodes, and training records the tracking
  fraction.

Everything but the end-to-end class is pure (no env): the panels are the
gate tests' synthetic records on the seeds the stage's command plans.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from environments.shared.command_tracking_panel import CommandPanelPlan, plan_command_panel
from environments.shared.config import env_kwargs_record
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum import CurriculumManager
from environments.shared.curriculum.checkpoints import (
    HANDOFF_CHOICE_FILENAME,
    HANDOFF_CHOICE_SCHEMA,
    handoff_choice_candidates_problem,
    recorded_handoff_choice,
    select_handoff_checkpoint,
    write_handoff_choice,
)
from environments.shared.curriculum.command_tracking_evidence import (
    COMMAND_TRACKING_PANEL_CSV,
    COMMAND_TRACKING_PANEL_FIELDNAMES,
    COMMAND_TRACKING_REPORT_JSON,
    read_command_tracking_panel,
)
from environments.shared.curriculum.command_tracking_gate import (
    COMMAND_TRACKING_GATE_KIND,
    CRUISE_FAMILY,
    HANDOFF_SELECTION_RULES,
    PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    PROVISIONAL_FOLLOW_THRESHOLDS,
    PROVISIONAL_WALK_THRESHOLDS,
    STAND_FAMILY,
    CommandTrackingEpisode,
    CommandTrackingResult,
    FamilyResult,
    PanelMeasurement,
    handoff_selection_key,
    panel_measurement,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, gate_config_sha256, gate_config_view
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.model_id import ModelId
from environments.shared.reporting import command_tracking_report as report_module
from environments.shared.reporting.command_tracking_report import (
    COMMAND_TRACKING_STAGE_RESULT_KEYS,
    assemble_command_tracking_report,
    choose_command_handoff,
    command_tracking_stage_result,
    handoff_candidates,
    write_command_tracking_report,
)
from environments.shared.reporting.csv_output import build_results_csv_rows
from environments.shared.reporting.gates import (
    command_tracking_report_refusals,
    evaluate_recorded_gate,
    evaluate_stage_gate,
)
from environments.shared.reporting.stage_artifacts import _apply_stage_gate
from environments.shared.reporting.summaries import _canonical_stage_summary
from environments.shared.reporting.text_summaries import write_stage_summary
from environments.shared.result_bundle import ResultBundleError
from environments.shared.result_bundle.evidence import (
    _validate_command_tracking_panel_evidence,
    compare_summary_to_csv,
)
from environments.shared.result_bundle.gate_verdict import _PERSISTED_STAGE_RESULT_KEYS
from environments.shared.result_bundle.hashing import sha256_file
from environments.shared.tests.stage_toml_helpers import replace_commands_table

COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
TASK_SHA256 = "sha256:" + "c3" * 32
PARENT_SHA256 = "sha256:" + "d4" * 32

#: The design's walk command (as test_command_tracking_gate's).
WALK_CONFIG = DirectionCommandConfig(
    cruise_speed=0.5,
    speed_range=(0.25, 0.5),
    stop_probability=0.15,
    turn_increment_max=0.0,
    switch_interval_s=100.0,
    initial_heading_offset_max=0.26,
)
#: A biped walk's gait metrics that clear every provisional biped bar.
WALKING_GAIT: tuple[tuple[str, float], ...] = (
    ("limbs", 2.0),
    ("flight_fraction", 0.0),
    ("hop_flight_fraction", 0.0),
    ("min_limb_duty", 0.6),
    ("alternation_index", 1.0),
    ("pair_alternation_index", 1.0),
    ("simultaneous_fraction", 0.0),
    ("lead_swaps_per_stride", 2.0),
    ("foot_contact_fraction", 0.0),
    ("phantom_support_fraction", 0.0),
)
#: A small walk panel (4 cruise, 2 stand) whose bounds 4/4 clears: LCB95(4, 4) = 0.473.
WALK_CURRICULUM: dict[str, Any] = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": COMMAND_TRACKING_GATE_KIND,
    "timesteps": 10_000,
    **PROVISIONAL_WALK_THRESHOLDS,
    **PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    "min_cruise_episodes": 4,
    "min_stand_episodes": 2,
    "min_cruise_survival_lcb": 0.3,
    "min_cruise_success_lcb": 0.3,
}
WALK = thresholds_from_curriculum(WALK_CURRICULUM)
MEASUREMENT = panel_measurement(WALK_CONFIG, WALK)
PLAN = plan_command_panel(WALK, WALK_CONFIG)
HORIZON = 1000
FRAME_SKIP = 5  # dt 0.01 at the 0.002 s physics step


def _record(seed: int, index: int, family: str, **overrides: Any) -> CommandTrackingEpisode:
    """A passing record of *family* on *seed* (a cruise that tracks, or a stand that holds the stop)."""
    values: dict[str, Any] = dict(
        episode=index,
        seed=seed,
        family=family,
        length=HORIZON,
        horizon=HORIZON,
        dt=0.01,
        full_horizon=True,
        fall=False,
        termination_reason=None,
        reward=120.0,
        desired_speed=0.4,
        planned_event_count=1,
        initial_heading_error_rad=0.12,
        first_event_settled=True,
        tracking_fraction=0.9,
        tracking_fraction_after_settle=0.95,
        mean_speed_along_heading_mps=0.38,
        mean_desired_speed_mps=0.4,
        stand_hold_fraction=0.93,
        eligible_events=1,
        settled_events=1,
        eligible_stops=0,
        settled_stops=0,
        eligible_turns=0,
        settled_turns=0,
        max_course_progress_m=3.6,
        measurement=MEASUREMENT,
        gait_metrics=WALKING_GAIT,
    )
    if family == STAND_FAMILY:
        values.update(
            desired_speed=0.0,
            initial_heading_error_rad=0.0,
            tracking_fraction=0.98,
            tracking_fraction_after_settle=1.0,
            mean_speed_along_heading_mps=math.nan,
            mean_desired_speed_mps=math.nan,
            stand_hold_fraction=1.0,
            eligible_stops=1,
            settled_stops=1,
            max_course_progress_m=0.01,
        )
    values.update(overrides)
    return CommandTrackingEpisode(**values)


def _panel(plan: CommandPanelPlan = PLAN, **overrides_by_seed: Any) -> list[CommandTrackingEpisode]:
    """The plan's episodes, all passing, with ``seed_<n>={field: value}`` overrides."""
    return [
        _record(seed, index, family, **overrides_by_seed.get(f"seed_{seed}", {}))
        for index, (seed, family) in enumerate(zip(plan.seeds, plan.families))
    ]


def _pair(directory: Path, name: str, payload: bytes) -> None:
    (directory / f"{name}.zip").write_bytes(b"zip:" + payload)
    (directory / f"{name}_vecnorm.pkl").write_bytes(b"pkl:" + payload)


def _stage_dir(tmp_path: Path, curriculum: dict[str, Any] = WALK_CURRICULUM, *, command: Any = WALK_CONFIG) -> Path:
    """A stage directory: three candidate pairs, the stage record (task, [env] with the command, run edge)."""
    stage_dir = tmp_path / "stage2_locomotion"
    models = stage_dir / "models"
    models.mkdir(parents=True)
    for name in ("robust_best_model", "best_model", "stage2_final"):
        _pair(models, name, name.encode())
    env = {"max_episode_steps": HORIZON, "frame_skip": FRAME_SKIP, "command_mode": "heading_and_speed"}
    if command is not None:
        env["command_config"] = env_kwargs_record({"command_config": command})["command_config"]
    else:
        env["command_mode"] = "none"
    record = {
        **COMPSOGNATHUS_LONGIPES.record(),
        "stage": 2,
        "reward_weights": env,
        "curriculum": curriculum,
        "run": {"load_path": "parent.zip", "parent_checkpoint_sha256": PARENT_SHA256},
        "task_fingerprint": {"task_sha256": TASK_SHA256},
    }
    (stage_dir / "stage_config.json").write_text(json.dumps(record), encoding="utf-8")
    return stage_dir


def _candidate(models: Path, name: str) -> dict[str, str]:
    return {
        "name": name,
        "checkpoint_sha256": sha256_file(models / f"{name}.zip"),
        "normalization_sha256": sha256_file(models / f"{name}_vecnorm.pkl"),
    }


def _choose(
    stage_dir: Path,
    name: str = "robust_best_model",
    curriculum: dict[str, Any] = WALK_CURRICULUM,
    *,
    seed_start: int = report_module.HANDOFF_SELECTION_SEED_START,
) -> dict:
    """Record *name* as the stage's handoff choice among every pair in it, under *curriculum*'s gate and the task,
    made on the selection panel planned from *seed_start* (as ``choose_command_handoff`` records it)."""
    models = stage_dir / "models"
    ranked = sorted(path.name[: -len("_vecnorm.pkl")] for path in models.glob("*_vecnorm.pkl"))
    selection = plan_command_panel(thresholds_from_curriculum(curriculum), WALK_CONFIG, seed_start=seed_start)
    record = {
        "schema": HANDOFF_CHOICE_SCHEMA,
        "gate_kind": COMMAND_TRACKING_GATE_KIND,
        "gate_sha256": gate_config_sha256(gate_config_view(curriculum)),
        "selection_seed_start": seed_start,
        "plan": selection.as_dict(),
        "task_sha256": TASK_SHA256,
        "selected": name,
        "checkpoint_sha256": sha256_file(models / f"{name}.zip"),
        "normalization_sha256": sha256_file(models / f"{name}_vecnorm.pkl"),
        "candidates": [_candidate(models, ranked_name) for ranked_name in ranked],
    }
    write_handoff_choice(models, record)
    return record


def _report(
    stage_dir: Path,
    episodes: list[CommandTrackingEpisode],
    *,
    curriculum: dict[str, Any] = WALK_CURRICULUM,
    plan: CommandPanelPlan = PLAN,
    scored: str = "robust_best_model",
    reference: list[CommandTrackingEpisode] | None = None,
    reference_digest: str = PARENT_SHA256,
) -> dict[str, Any]:
    """The report the builder writes for *episodes*, rolled on the stage's *scored* pair."""
    models = stage_dir / "models"
    return assemble_command_tracking_report(
        COMPSOGNATHUS_LONGIPES,
        2,
        curriculum=curriculum,
        command_config=WALK_CONFIG,
        plan=plan,
        episodes=episodes,
        measurement=MEASUREMENT,
        horizon=HORIZON,
        control_dt=0.01,
        policy=f"{scored}.zip — VecNormalize stats applied (clip 10)",
        normalization={"mode": "vecnormalize-frozen", "clip_obs": 10.0, "epsilon": 1e-8},
        handoff={
            "checkpoint": f"{scored}.zip",
            "checkpoint_sha256": sha256_file(models / f"{scored}.zip"),
            "normalization_sha256": sha256_file(models / f"{scored}_vecnorm.pkl"),
        },
        task_sha256=TASK_SHA256,
        plant_identity={"plant": "test"},
        checkpoint_plant_validated=True,
        reference=(
            None
            if reference is None
            else {
                "policy": "parent.zip",
                "normalization": {"mode": "vecnormalize-frozen"},
                "handoff": {"checkpoint": "parent.zip", "checkpoint_sha256": reference_digest},
            }
        ),
        reference_episodes=reference,
    )


def _judge(stage_dir: Path, report: Any, curriculum: dict[str, Any] = WALK_CURRICULUM) -> tuple[bool, list[str]]:
    return evaluate_stage_gate(curriculum, {}, stage=2, stage_dir=stage_dir, command_tracking_report=report)


@pytest.fixture
def stage_dir(tmp_path: Path) -> Path:
    directory = _stage_dir(tmp_path)
    _choose(directory)
    return directory


# ── the judge ────────────────────────────────────────────────────────────────


class TestJudge:
    def test_the_plan_is_four_cruise_and_two_stand_episodes_from_the_certification_seed(self):
        assert PLAN.family_counts() == {CRUISE_FAMILY: 4, STAND_FAMILY: 2}
        assert PLAN.seed_start == PUBLICATION_SEED_START

    def test_an_admitted_passing_report_passes(self, stage_dir):
        report = _report(stage_dir, _panel())
        assert report["passed"] is True
        assert command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=stage_dir) == []
        assert _judge(stage_dir, report) == (True, [])

    def test_a_failing_panel_fails_on_the_re_derived_criteria(self, stage_dir):
        stand_seed = PLAN.seeds[PLAN.families.index(STAND_FAMILY)]
        fallen = {"full_horizon": False, "fall": True, "length": 400, "termination_reason": "fall"}
        report = _report(stage_dir, _panel(**{f"seed_{stand_seed}": fallen}))
        passed, failures = _judge(stage_dir, report)
        assert passed is False
        assert failures and all(failure.startswith("stage 2 ") for failure in failures)
        assert any("stand family" in failure and "max_stand_falls" in failure for failure in failures), failures

    def test_no_report_and_no_stage_dir_fail_closed(self, stage_dir):
        passed, failures = _judge(stage_dir, None)
        assert passed is False and COMMAND_TRACKING_REPORT_JSON in failures[0]
        report = _report(stage_dir, _panel())
        passed, failures = evaluate_stage_gate(WALK_CURRICULUM, {}, stage=2, command_tracking_report=report)
        assert passed is False and "no stage_dir was given" in failures[0]

    def test_the_reward_never_certifies_the_kind(self, stage_dir):
        """A stage result with a huge reward and no report fails: the arm never reaches the reward conjunction."""
        results = {"best_model_reward": 1e9, "best_model_length": 1e9, "best_model_fwd_vel": 1e9}
        passed, failures = evaluate_stage_gate(WALK_CURRICULUM, results, stage=2, stage_dir=stage_dir)
        assert passed is False and "no command-tracking report" in failures[0]
        assert not any("has no evaluator" in failure for failure in failures)

    @pytest.mark.parametrize(
        ("edit", "expected"),
        [
            (lambda r: r.update(schema="mesozoic.stance-gate-report/v5"), "has schema"),
            (lambda r: r.update(scored_gate_kind="stance_quality/v2"), "was scored by"),
            (lambda r: r.update(protocol="command_panel/v0"), "protocol"),
            (lambda r: r.update(policy_deterministic=False), "policy_deterministic"),
            (lambda r: r.update(normalization={"mode": "none"}), "frozen VecNormalize"),
            (lambda r: r.update(checkpoint_plant_validated=False), "checkpoint_plant_validated"),
            (lambda r: r.update(seed_start=2042), "not the certification seed"),
            (lambda r: r.update(task_sha256="sha256:" + "00" * 32), "rolled task"),
            (lambda r: r.update(horizon=999), "step horizon"),
            (lambda r: r.update(control_dt=0.02), "control step"),
            (lambda r: r["command_config"].update(stop_probability=0.5), "command config"),
            (lambda r: r.update(gate_sha256="sha256:" + "11" * 32), "gate_sha256"),
            (lambda r: r["handoff"].update(checkpoint_sha256="sha256:" + "22" * 32), "scored checkpoint"),
            (lambda r: r.update(passed="yes"), "boolean verdict"),
        ],
    )
    def test_each_binding_is_a_refusal_by_name(self, stage_dir, edit, expected):
        report = _report(stage_dir, _panel())
        edit(report)
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=stage_dir)
        assert any(expected in refusal for refusal in refusals), refusals
        assert _judge(stage_dir, report)[0] is False

    def test_a_report_scored_under_other_thresholds_is_refused(self, stage_dir):
        report = _report(stage_dir, _panel())
        edited = {**WALK_CURRICULUM, "min_cruise_success_lcb": 0.4}
        refusals = command_tracking_report_refusals(edited, report, stage=2, stage_dir=stage_dir)
        assert any("other thresholds" in refusal for refusal in refusals), refusals

    def test_a_handoff_not_chosen_by_the_gated_statistic_is_refused(self, tmp_path):
        """No choice record: select_handoff_checkpoint falls back to reward, and the judge refuses that pair."""
        directory = _stage_dir(tmp_path)
        report = _report(directory, _panel())
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory)
        assert any("no valid handoff choice" in refusal for refusal in refusals), refusals

    def test_a_choice_made_under_another_gate_is_refused(self, tmp_path):
        directory = _stage_dir(tmp_path)
        _choose(directory, curriculum={**WALK_CURRICULUM, "min_cruise_success_lcb": 0.4})
        report = _report(directory, _panel())
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory)
        assert any("was chosen under gate" in refusal for refusal in refusals), refusals

    def test_a_choice_made_on_the_certification_seeds_is_refused(self, tmp_path):
        """The draws that chose the handoff must not certify it: a selection panel sharing a seed with the
        certification panel is refused by name, and so is a choice whose selection plan cannot be read."""
        directory = _stage_dir(tmp_path)
        _choose(directory, seed_start=PUBLICATION_SEED_START + 1)
        report = _report(directory, _panel())
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory)
        assert any("sharing" in refusal and "certification panel" in refusal for refusal in refusals), refusals
        assert _judge(directory, report)[0] is False
        record = _choose(directory)
        assert command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory) == []
        write_handoff_choice(directory / "models", {key: value for key, value in record.items() if key != "plan"})
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory)
        assert any("no readable selection plan" in refusal for refusal in refusals), refusals

    def test_a_report_of_the_reward_chosen_pair_is_refused_when_the_choice_is_another(self, tmp_path):
        directory = _stage_dir(tmp_path)
        _choose(directory, "best_model")
        assert select_handoff_checkpoint(directory / "models")[0] == "best_model"
        report = _report(directory, _panel(), scored="robust_best_model")
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory)
        assert any("not the handoff best_model" in refusal for refusal in refusals), refusals
        assert _judge(directory, _report(directory, _panel(), scored="best_model")) == (True, [])

    def test_a_stage_that_ran_no_live_command_is_refused(self, tmp_path):
        directory = _stage_dir(tmp_path, command=None)
        _choose(directory)
        refusals = command_tracking_report_refusals(
            WALK_CURRICULUM, _report(directory, _panel()), stage=2, stage_dir=directory
        )
        assert any("no live command_config" in refusal for refusal in refusals), refusals

    def test_records_off_the_plan_or_edited_are_refused(self, stage_dir):
        shuffled = _panel()
        shuffled[0], shuffled[1] = (
            replace(shuffled[0], seed=shuffled[1].seed),
            replace(shuffled[1], seed=shuffled[0].seed),
        )
        report = _report(stage_dir, shuffled)
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=stage_dir)
        assert any("cannot be re-derived" in refusal for refusal in refusals), refusals
        flipped = _report(stage_dir, _panel())
        flipped["episode_evidence"][0]["success"] = False
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, flipped, stage=2, stage_dir=stage_dir)
        assert any("success flags differ" in refusal for refusal in refusals), refusals
        lied = _report(stage_dir, _panel())
        lied["episode_evidence"][0]["record"]["fall"] = True
        lied["episode_evidence"][0]["record"]["full_horizon"] = False
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, lied, stage=2, stage_dir=stage_dir)
        assert any("disagrees with the one re-derived" in refusal for refusal in refusals), refusals
        broken = _report(stage_dir, _panel())
        del broken["episode_evidence"][0]["record"]["gait_metrics"]
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, broken, stage=2, stage_dir=stage_dir)
        assert any("cannot be re-derived" in refusal for refusal in refusals), refusals

    def test_a_reference_panel_is_bound_to_the_parent_the_stage_entered_on(self, stage_dir):
        report = _report(stage_dir, _panel(), reference=_panel(), reference_digest="sha256:" + "ee" * 32)
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=stage_dir)
        assert any("reference panel scored checkpoint" in refusal for refusal in refusals), refusals
        bound = _report(stage_dir, _panel(), reference=_panel())
        assert command_tracking_report_refusals(WALK_CURRICULUM, bound, stage=2, stage_dir=stage_dir) == []

    def test_a_recorded_history_reads_none(self):
        """No in-training scalar can stand for the panel: never the generic reward arm."""
        history = [{"mean_reward": 1e6, "mean_episode_length": 1e6, "n_episodes": 100}] * 5
        assert evaluate_recorded_gate(WALK_CURRICULUM, history) is None
        assert evaluate_recorded_gate({**WALK_CURRICULUM, "min_avg_reward": 1.0}, history) is None

    def test_the_in_training_manager_refuses_the_kind(self, caplog):
        manager = CurriculumManager(
            COMPSOGNATHUS_LONGIPES,
            stage_thresholds={1: {"gate_kind": COMMAND_TRACKING_GATE_KIND, "required_consecutive": 1}},
            total_stages=1,
        )
        with caplog.at_level(logging.ERROR):
            assert manager.should_advance([1e6] * 20, [1000.0] * 20) is False
        assert any(
            "command_tracking/v1" in record.getMessage() and "after the" in record.getMessage()
            for record in caplog.records
        )

    def test_the_manager_copies_none_of_the_kinds_threshold_keys(self):
        from environments.shared.curriculum.manager import thresholds_from_configs

        thresholds = thresholds_from_configs({2: {"curriculum_kwargs": dict(WALK_CURRICULUM)}})
        assert thresholds[2] == {"gate_kind": COMMAND_TRACKING_GATE_KIND}

    def test_apply_stage_gate_records_the_re_derived_numbers(self, stage_dir):
        report = _report(stage_dir, _panel())
        results: dict[str, Any] = {}
        _apply_stage_gate(
            stage=2,
            stage_config={"curriculum_kwargs": WALK_CURRICULUM},
            stage_results=results,
            stance_report=None,
            stage_dir=None,
            model_id=COMPSOGNATHUS_LONGIPES,
            command_tracking_report=report,
        )
        assert results["gate_passed"] is False, "no stage_dir: refused, and no numbers copied"
        assert "command_cruise_success_lcb" not in results
        results = {}
        _apply_stage_gate(
            stage=2,
            stage_config={"curriculum_kwargs": WALK_CURRICULUM},
            stage_results=results,
            stance_report=None,
            stage_dir=stage_dir,
            model_id=COMPSOGNATHUS_LONGIPES,
            command_tracking_report=report,
        )
        assert results["gate_passed"] is True and results["gate_failures"] == []
        assert results["command_profile"] == "walk"
        assert results["command_cruise_successes"] == 4 and results["command_stand_successes"] == 2
        assert results["command_cruise_success_lcb"] == pytest.approx(0.05**0.25)
        assert results["command_stand_success_fraction"] == 1.0 and results["command_stand_falls"] == 0
        assert results["command_handoff_selected"] == "robust_best_model"
        verdict = json.loads((stage_dir / "gate_verdict.json").read_text())
        assert verdict["passed"] is True
        assert verdict["stage_result"]["command_cruise_success_lcb"] == pytest.approx(0.05**0.25)

    def test_the_cli_curriculum_judges_the_kind_after_the_budget(self):
        """By AST: under a ``gate_kind == COMMAND_TRACKING_GATE_KIND`` test of an uninterrupted node, ``passed``
        comes from ``_post_training_command_verdict`` and ``judged_by`` is the kind's own name."""
        import ast
        import inspect

        from environments.shared import train_base

        tree = ast.parse(inspect.getsource(train_base.train_curriculum))
        branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and {"COMMAND_TRACKING_GATE_KIND", "interrupted"}
            <= {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        ]
        assert len(branches) == 1
        calls = {c.func.id for c in ast.walk(branches[0]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "_post_training_command_verdict" in calls
        assigned = {
            target.id: node.value
            for node in ast.walk(branches[0])
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        judged_by = assigned["verdict_judged_by"]
        assert isinstance(judged_by, ast.Name) and judged_by.id == "COMMAND_TRACKING_POST_TRAINING_JUDGED_BY"
        assert COMMAND_TRACKING_GATE_KIND in train_base.POST_TRAINING_JUDGED_GATE_KINDS
        assert train_base.COMMAND_TRACKING_POST_TRAINING_JUDGED_BY not in {
            train_base.CURRICULUM_MANAGER_JUDGED_BY,
            train_base.STANCE_V2_POST_TRAINING_JUDGED_BY,
            train_base.STANCE_HEADING_POST_TRAINING_JUDGED_BY,
        }

    def test_every_stage_result_key_is_persisted_by_the_verdict(self):
        assert set(COMMAND_TRACKING_STAGE_RESULT_KEYS) <= set(_PERSISTED_STAGE_RESULT_KEYS)


# ── the handoff chosen by the gated statistic ────────────────────────────────


def _result(passed: bool, *, cruise_lcb: float, stand: float, survival: float = 0.9) -> CommandTrackingResult:
    return CommandTrackingResult(
        passed=passed,
        failures=() if passed else ("cruise family: ...",),
        profile="walk",
        n_episodes=25,
        families=(
            FamilyResult("cruise", 20, 20, 18, 0.9, survival_lcb=survival, success_lcb=cruise_lcb),
            FamilyResult("stand", 5, 5, 4, stand),
        ),
    )


class TestHandoffChoice:
    def test_the_key_ranks_a_pass_first_then_the_bound_then_the_stand_fraction(self):
        failing_high = handoff_selection_key(_result(False, cruise_lcb=0.95, stand=1.0))
        passing_low = handoff_selection_key(_result(True, cruise_lcb=0.61, stand=0.6))
        passing_high = handoff_selection_key(_result(True, cruise_lcb=0.70, stand=0.6))
        passing_high_stand = handoff_selection_key(_result(True, cruise_lcb=0.70, stand=0.8))
        assert failing_high < passing_low < passing_high < passing_high_stand
        unmeasured = handoff_selection_key(_result(False, cruise_lcb=math.nan, stand=math.nan))
        assert unmeasured < failing_high and unmeasured[1] == -math.inf
        assert "never" not in HANDOFF_SELECTION_RULES["walk"] and "reward" not in HANDOFF_SELECTION_RULES["walk"]

    def test_the_follow_key_ranks_by_the_follow_bound(self):
        def follow(passed: bool, lcb: float) -> CommandTrackingResult:
            return CommandTrackingResult(
                passed=passed,
                failures=(),
                profile="follow",
                n_episodes=20,
                families=(FamilyResult("follow", 20, 20, 17, 0.85, survival_lcb=0.9, success_lcb=lcb),),
            )

        assert handoff_selection_key(follow(True, 0.6)) < handoff_selection_key(follow(True, 0.7))
        assert handoff_selection_key(follow(False, 0.9)) < handoff_selection_key(follow(True, 0.6))

    def test_the_candidates_are_the_two_handoff_names_then_the_final_pair(self, tmp_path):
        directory = _stage_dir(tmp_path)
        models = directory / "models"
        assert [name for name, _, _ in handoff_candidates(models, models / "stage2_final")] == [
            "robust_best_model",
            "best_model",
            "stage2_final",
        ]
        (models / "best_model_vecnorm.pkl").unlink()
        assert [name for name, _, _ in handoff_candidates(models)] == ["robust_best_model"]

    def test_select_honors_a_valid_choice_and_ignores_a_stale_one(self, tmp_path, caplog):
        directory = _stage_dir(tmp_path)
        models = directory / "models"
        assert select_handoff_checkpoint(models)[0] == "robust_best_model", "no record: the reward order"
        _choose(directory, "stage2_final")
        assert select_handoff_checkpoint(models) == (
            "stage2_final",
            str(models / "stage2_final"),
            str(models / "stage2_final_vecnorm.pkl"),
        )
        (models / "stage2_final.zip").write_bytes(b"rewritten after the choice")
        record, problem = recorded_handoff_choice(models)
        assert record is None and "no longer hashes" in problem
        with caplog.at_level(logging.WARNING):
            assert select_handoff_checkpoint(models)[0] == "robust_best_model"
        assert any(HANDOFF_CHOICE_FILENAME in record.getMessage() for record in caplog.records)
        for bad in ({"schema": "other"}, {"schema": HANDOFF_CHOICE_SCHEMA, "selected": "../x"}, []):
            (models / HANDOFF_CHOICE_FILENAME).write_text(json.dumps(bad))
            assert recorded_handoff_choice(models)[0] is None
            assert select_handoff_checkpoint(models)[0] == "robust_best_model"

    def test_a_choice_whose_digests_are_not_its_pair_is_never_written(self, tmp_path):
        directory = _stage_dir(tmp_path)
        record = _choose(directory)
        with pytest.raises(ValueError, match="digests"):
            write_handoff_choice(directory / "models", {**record, "checkpoint_sha256": "sha256:" + "00" * 32})
        with pytest.raises(ValueError, match="schema"):
            write_handoff_choice(directory / "models", {**record, "schema": "other"})

    def test_a_choice_stands_only_while_the_pairs_it_ranked_are_the_stages(self, tmp_path):
        """A ranked pair rewritten, or a pair it never ranked, makes the choice stale: the judge refuses it."""
        directory = _stage_dir(tmp_path)
        models = directory / "models"
        record = _choose(directory, "best_model")
        assert handoff_choice_candidates_problem(record, models) is None
        report = _report(directory, _panel(), scored="best_model")
        assert _judge(directory, report) == (True, [])
        # The final pair rewritten after the choice (training continued in the stage directory): the selected
        # pair is untouched, so select_handoff_checkpoint still honors the record, but it is not a choice among
        # the stage's pairs any more.
        _pair(models, "stage2_final", b"continued")
        assert select_handoff_checkpoint(models)[0] == "best_model"
        assert "stage2_final no longer hashes" in handoff_choice_candidates_problem(record, models)
        refusals = command_tracking_report_refusals(WALK_CURRICULUM, report, stage=2, stage_dir=directory)
        assert any("no longer a choice among the stage's checkpoints" in refusal for refusal in refusals), refusals
        # A complete pair the choice never ranked.
        fresh = _stage_dir(tmp_path / "fresh")
        (fresh / "models" / "best_model_vecnorm.pkl").unlink()
        record = _choose(fresh, "robust_best_model")
        _pair(fresh / "models", "best_model", b"late")
        assert "never ranked (best_model)" in handoff_choice_candidates_problem(record, fresh / "models")
        # A record without candidates, or whose selected pair is not among them, is never written.
        models = fresh / "models"
        with pytest.raises(ValueError, match="no candidates"):
            write_handoff_choice(models, {**record, "candidates": []})
        with pytest.raises(ValueError, match="not among its candidates"):
            write_handoff_choice(models, {**record, "candidates": [_candidate(models, "best_model")]})
        with pytest.raises(ValueError, match="never ranked"):
            write_handoff_choice(models, {**record, "candidates": [_candidate(models, "robust_best_model")]})

    def test_a_stale_choice_is_made_again_and_a_current_one_is_kept(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        directory = _stage_dir(tmp_path)
        models = directory / "models"
        _choose(directory, "best_model")
        calls: list[str] = []

        def fake_choose(model_id: Any, stage: Any, *, stage_config: Any, model_dir: Any, final_stem: Any) -> dict:
            calls.append(Path(final_stem).name)
            return _choose(directory, "stage2_final")

        monkeypatch.setattr(report_module, "choose_command_handoff", fake_choose)
        import environments.shared.task_fingerprint as task_fingerprint

        monkeypatch.setattr(task_fingerprint, "stage_task_fingerprint", lambda *a, **k: {"task_sha256": TASK_SHA256})
        stage_config = {"curriculum_kwargs": WALK_CURRICULUM, "env_kwargs": {"command_config": WALK_CONFIG}}

        def ensure() -> Any:
            return stage_artifacts.ensure_command_handoff_choice(
                model_id=COMPSOGNATHUS_LONGIPES,
                stage=2,
                stage_config=stage_config,
                model_dir=models,
                final_stem=models / "stage2_final",
            )

        assert ensure() is None and calls == [], "a current choice under this gate and task is kept"
        _pair(models, "stage2_final", b"continued")
        assert ensure()["selected"] == "stage2_final" and calls == ["stage2_final"]
        edited = {**stage_config, "curriculum_kwargs": {**WALK_CURRICULUM, "min_cruise_success_lcb": 0.4}}
        stage_config = edited
        assert ensure() is not None and len(calls) == 2, "a choice made under another gate is made again"
        other = {"curriculum_kwargs": {"gate_kind": "stance_quality/v2"}, "env_kwargs": {}}
        assert (
            stage_artifacts.ensure_command_handoff_choice(
                model_id=COMPSOGNATHUS_LONGIPES, stage=2, stage_config=other, model_dir=models
            )
            is None
            and len(calls) == 2
        ), "every other kind is a no-op"

    def test_no_choice_is_recorded_when_no_candidate_can_be_rolled(self, tmp_path, monkeypatch):
        directory = _stage_dir(tmp_path)
        models = directory / "models"

        def broken(*args: Any, **kwargs: Any) -> dict:
            raise RuntimeError("the env would not build")

        monkeypatch.setattr(report_module, "build_command_tracking_report", broken)
        import environments.shared.task_fingerprint as task_fingerprint

        monkeypatch.setattr(task_fingerprint, "stage_task_fingerprint", lambda *a, **k: {"task_sha256": TASK_SHA256})
        stage_config = {"curriculum_kwargs": WALK_CURRICULUM, "env_kwargs": {"command_config": WALK_CONFIG}}
        with pytest.raises(report_module.CommandTrackingReportError, match="no handoff is chosen"):
            choose_command_handoff(
                COMPSOGNATHUS_LONGIPES,
                2,
                stage_config=stage_config,
                model_dir=models,
                final_stem=models / "stage2_final",
            )
        assert not (models / HANDOFF_CHOICE_FILENAME).exists()
        assert select_handoff_checkpoint(models)[0] == "robust_best_model"

    @pytest.mark.parametrize(
        ("config", "stands", "seed_start"),
        [
            (WALK_CONFIG, 2, PUBLICATION_SEED_START),
            # A rare stand family: at a stop probability of 0.002 the block 2042-3041 draws five stand episodes, so a
            # selection panel of six runs into the certification block and keeps its first stand seed (3739).
            (replace(WALK_CONFIG, stop_probability=0.002), 6, report_module.HANDOFF_SELECTION_SEED_START),
        ],
        ids=["the_certification_block", "a_rare_stand_family"],
    )
    def test_no_choice_is_made_on_a_selection_panel_that_meets_the_certification_panel(
        self, tmp_path, monkeypatch, config, stands, seed_start
    ):
        curriculum = {**WALK_CURRICULUM, "min_stand_episodes": stands}
        thresholds = thresholds_from_curriculum(curriculum)
        directory = _stage_dir(tmp_path, curriculum, command=config)
        models = directory / "models"
        selection = plan_command_panel(thresholds, config, seed_start=seed_start)
        certification = plan_command_panel(thresholds, config, seed_start=PUBLICATION_SEED_START)
        assert report_module.shared_panel_seeds(selection.seeds, certification.seeds)
        rolled: list[str] = []
        monkeypatch.setattr(report_module, "build_command_tracking_report", lambda *a, **k: rolled.append("x"))
        stage_config = {"curriculum_kwargs": curriculum, "env_kwargs": {"command_config": config}}
        with pytest.raises(report_module.CommandTrackingReportError, match="shares .* certification panel"):
            choose_command_handoff(
                COMPSOGNATHUS_LONGIPES,
                2,
                stage_config=stage_config,
                model_dir=models,
                final_stem=models / "stage2_final",
                seed_start=seed_start,
            )
        assert rolled == [] and not (models / HANDOFF_CHOICE_FILENAME).exists()

    def test_the_default_selection_panel_never_meets_the_certification_panel(self):
        """At the design's stand rate (0.15) and the provisional 20 + 5 panel the two blocks stay far apart."""
        design = replace(WALK_CONFIG, stop_probability=0.15)
        provisional = thresholds_from_curriculum(
            {
                **PROVISIONAL_WALK_THRESHOLDS,
                **PROVISIONAL_BIPED_GAIT_THRESHOLDS,
                "gate_kind": COMMAND_TRACKING_GATE_KIND,
            }
        )
        selection = plan_command_panel(provisional, design, seed_start=report_module.HANDOFF_SELECTION_SEED_START)
        certification = plan_command_panel(provisional, design, seed_start=PUBLICATION_SEED_START)
        assert selection.family_counts() == certification.family_counts() == {CRUISE_FAMILY: 20, STAND_FAMILY: 5}
        assert max(selection.seeds) < min(certification.seeds)

    def test_the_choice_is_the_gated_statistic_not_the_reward(self, tmp_path, monkeypatch):
        """robust_best_model has the higher reward but fails; best_model passes; the final pair passes lower."""
        directory = _stage_dir(tmp_path)
        models = directory / "models"
        selection_plan = plan_command_panel(WALK, WALK_CONFIG, seed_start=report_module.HANDOFF_SELECTION_SEED_START)
        assert not set(selection_plan.seeds) & set(PLAN.seeds), "the selection draws never certify"
        cruise_seeds = [
            seed for seed, family in zip(selection_plan.seeds, selection_plan.families) if family == "cruise"
        ]
        unsettled = {"first_event_settled": False, "reward": 900.0}
        panels = {
            "robust_best_model": _panel(selection_plan, **{f"seed_{seed}": unsettled for seed in cruise_seeds}),
            "best_model": _panel(selection_plan),
            "stage2_final": _panel(selection_plan, **{f"seed_{cruise_seeds[0]}": {"first_event_settled": False}}),
        }
        calls: list[tuple[str, int]] = []

        def fake_build(
            model_id: Any, stage: Any, *, stage_config: Any, model_path: str, vecnorm_path: str, seed_start: int
        ):
            name = Path(model_path).stem
            calls.append((name, seed_start))
            return _report(directory, panels[name], plan=selection_plan, scored=name)

        monkeypatch.setattr(report_module, "build_command_tracking_report", fake_build)
        import environments.shared.task_fingerprint as task_fingerprint

        monkeypatch.setattr(task_fingerprint, "stage_task_fingerprint", lambda *a, **k: {"task_sha256": TASK_SHA256})
        stage_config = {"curriculum_kwargs": WALK_CURRICULUM, "env_kwargs": {"command_config": WALK_CONFIG}}
        record = choose_command_handoff(
            COMPSOGNATHUS_LONGIPES, 2, stage_config=stage_config, model_dir=models, final_stem=models / "stage2_final"
        )
        assert calls == [(name, 2042) for name in ("robust_best_model", "best_model", "stage2_final")]
        assert record is not None and record["selected"] == "best_model"
        assert [entry["passed"] for entry in record["candidates"]] == [False, True, False]
        assert record["rule"] == HANDOFF_SELECTION_RULES["walk"] and record["selection_seed_start"] == 2042
        assert record["gate_sha256"] == gate_config_sha256(gate_config_view(WALK_CURRICULUM))
        assert select_handoff_checkpoint(models)[0] == "best_model"
        on_disk = json.loads((models / HANDOFF_CHOICE_FILENAME).read_text())
        assert on_disk["selected"] == "best_model" and on_disk["checkpoint_sha256"] == sha256_file(
            models / "best_model.zip"
        )
        # A tie keeps the candidate order: every candidate identical, the first wins.
        panels.update({name: _panel(selection_plan) for name in panels})
        record = choose_command_handoff(
            COMPSOGNATHUS_LONGIPES, 2, stage_config=stage_config, model_dir=models, final_stem=models / "stage2_final"
        )
        assert record["selected"] == "robust_best_model"


# ── the evidence: records and the panel CSV ─────────────────────────────────


class TestEvidence:
    def test_a_record_round_trips_exactly_through_json(self):
        episode = _record(3043, 1, STAND_FAMILY, termination_reason=None)
        payload = json.loads(json.dumps(episode.as_dict()))
        restored = CommandTrackingEpisode.from_dict(payload)
        assert restored.as_dict() == episode.as_dict()
        assert math.isnan(restored.mean_speed_along_heading_mps) and restored.gait_metrics == episode.gait_metrics
        # Key order of the gait metrics does not matter (a sorted JSON writer reorders them).  Records are
        # compared through as_dict(): a dataclass with a NaN field never equals another under ==.
        reordered = {**payload, "gait_metrics": dict(sorted(payload["gait_metrics"].items()))}
        assert CommandTrackingEpisode.from_dict(reordered).as_dict() == restored.as_dict()
        assert CommandTrackingEpisode.from_dict(reordered).gait_metrics == episode.gait_metrics
        assert PanelMeasurement.from_dict(MEASUREMENT.as_dict()) == MEASUREMENT

    @pytest.mark.parametrize(
        "edit",
        [
            lambda p: p.pop("seed"),
            lambda p: p.update(extra=1),
            lambda p: p.update(seed=1.5),
            lambda p: p.update(fall="no"),
            lambda p: p.update(dt=0.0),
            lambda p: p.update(reward="1"),
            lambda p: p.update(termination_reason=3),
            lambda p: p["measurement"].pop("protocol"),
            lambda p: p["measurement"]["tolerances"].pop("velocity_tolerance"),
            lambda p: p.update(gait_metrics=[1, 2]),
        ],
    )
    def test_a_record_is_read_strictly(self, edit):
        payload = json.loads(json.dumps(_record(3042, 0, CRUISE_FAMILY).as_dict()))
        edit(payload)
        with pytest.raises(ValueError):
            CommandTrackingEpisode.from_dict(payload)

    def test_the_panel_csv_round_trips_every_record(self, stage_dir):
        report = _report(stage_dir, _panel(), reference=_panel())
        written = write_command_tracking_report(stage_dir, report)
        assert {path.name for path in written.values()} == {
            "command_tracking_report.txt",
            "command_tracking_report.json",
            COMMAND_TRACKING_PANEL_CSV,
        }
        with written["command_tracking_panel_csv"].open(newline="") as source:
            assert next(csv.reader(source)) == list(COMMAND_TRACKING_PANEL_FIELDNAMES)
        evidence = read_command_tracking_panel(written["command_tracking_panel_csv"])
        expected = [episode.as_dict() for episode in _panel()]
        assert [episode.as_dict() for episode in evidence.policy] == expected
        assert [episode.as_dict() for episode in evidence.reference] == expected
        assert evidence.policy_success == (True,) * 6 and evidence.measurement == MEASUREMENT
        assert evidence.checkpoint_sha256 == sha256_file(stage_dir / "models" / "robust_best_model.zip")
        assert evidence.reference_checkpoint_sha256 == PARENT_SHA256
        assert (evidence.task_sha256, evidence.seed_start, evidence.horizon) == (TASK_SHA256, 3042, HORIZON)
        assert evidence.gate_sha256 == gate_config_sha256(gate_config_view(WALK_CURRICULUM))
        # The JSON round-trips too, and the judge admits what was written.
        loaded = json.loads(written["command_tracking_report_json"].read_text())
        assert _judge(stage_dir, loaded) == (True, [])
        text = written["command_tracking_report_txt"].read_text()
        assert "GATE: PASS" in text and "cruise" in text and "stand" in text

    def test_the_panel_csv_reader_is_strict(self, stage_dir: Path, tmp_path: Path) -> None:
        path = write_command_tracking_report(stage_dir, _report(stage_dir, _panel()))["command_tracking_panel_csv"]
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))

        def rewritten(edit: Any) -> Path:
            target = tmp_path / "edited.csv"
            edited = [dict(row) for row in rows]
            edit(edited)
            with target.open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=list(edited[0]))
                writer.writeheader()
                writer.writerows(edited)
            return target

        for edit, message in (
            (lambda r: r[0].update(role="statue"), "role"),
            (lambda r: r[0].update(length="ten"), "unparsable length"),
            (lambda r: r[1].update(task_sha256="sha256:" + "00" * 32), "more than one task_sha256"),
            (lambda r: [row.pop("success") for row in r], "lacks the column"),
            (lambda r: r.clear() or r.append({key: "" for key in rows[0]}), "unparsable"),
        ):
            with pytest.raises(ValueError, match=message):
                read_command_tracking_panel(rewritten(edit))

    def _validate(self, stage_dir: Path, **overrides: Any) -> None:
        record = json.loads((stage_dir / "stage_config.json").read_text())
        kwargs: dict[str, Any] = dict(
            env_kwargs=record["reward_weights"],
            stage=2,
            panel_seed_start=PUBLICATION_SEED_START,
            certified_hash=sha256_file(stage_dir / "models" / "robust_best_model.zip"),
            certified_normalization=sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
            task_sha256=TASK_SHA256,
            parent_checkpoint_sha256=PARENT_SHA256,
        )
        kwargs.update(overrides)
        curriculum = kwargs.pop("curriculum", WALK_CURRICULUM)
        _validate_command_tracking_panel_evidence(stage_dir / COMMAND_TRACKING_PANEL_CSV, curriculum, **kwargs)

    def test_publication_re_derives_a_pass_from_the_csv_alone(self, stage_dir):
        write_command_tracking_report(stage_dir, _report(stage_dir, _panel(), reference=_panel()))
        (stage_dir / COMMAND_TRACKING_REPORT_JSON).unlink()
        self._validate(stage_dir)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"certified_hash": "sha256:" + "01" * 32}, "not the certified selected checkpoint"),
            ({"certified_normalization": "sha256:" + "02" * 32}, "VecNormalize statistics"),
            ({"task_sha256": "sha256:" + "03" * 32}, "under task"),
            ({"task_sha256": None}, "under task"),
            ({"curriculum": {**WALK_CURRICULUM, "min_stand_hold_fraction": 0.7}}, "scored under gate"),
            ({"panel_seed_start": 4042}, "certification_panel seed"),
            ({"parent_checkpoint_sha256": "sha256:" + "04" * 32}, "reference rows"),
            ({"env_kwargs": {"max_episode_steps": HORIZON, "frame_skip": FRAME_SKIP}}, "no live command_config"),
            ({"env_kwargs": {"max_episode_steps": 500}}, "the stage ran 500 steps"),
            ({"env_kwargs": {"frame_skip": 10}}, "at 0.02 s"),
        ],
    )
    def test_publication_refuses_each_binding(self, stage_dir, overrides, message):
        write_command_tracking_report(stage_dir, _report(stage_dir, _panel(), reference=_panel()))
        env = json.loads((stage_dir / "stage_config.json").read_text())["reward_weights"]
        if "env_kwargs" in overrides and "command_config" not in overrides["env_kwargs"]:
            edited = {**env, **overrides["env_kwargs"]}
            if message == "no live command_config":
                edited.pop("command_config")
            overrides = {"env_kwargs": edited}
        with pytest.raises(ResultBundleError, match=message):
            self._validate(stage_dir, **overrides)

    def test_publication_refuses_a_flipped_flag_a_failing_panel_and_a_missing_file(self, stage_dir):
        with pytest.raises(ResultBundleError, match="is missing"):
            self._validate(stage_dir)
        report = _report(stage_dir, _panel())
        report["episode_evidence"][2]["success"] = False
        write_command_tracking_report(stage_dir, report)
        with pytest.raises(ResultBundleError, match="re-derives success=True"):
            self._validate(stage_dir)
        cruise_seed = PLAN.seeds[PLAN.families.index(CRUISE_FAMILY)]
        failing = _report(stage_dir, _panel(**{f"seed_{cruise_seed}": {"first_event_settled": False}}))
        assert failing["passed"] is False
        write_command_tracking_report(stage_dir, failing)
        with pytest.raises(ResultBundleError, match="publication gate fails command_tracking/v1"):
            self._validate(stage_dir)
        off_plan = _report(stage_dir, _panel()[1:])
        write_command_tracking_report(stage_dir, off_plan)
        with pytest.raises(ResultBundleError, match="not the panel the stage's thresholds and command plan"):
            self._validate(stage_dir)

    def test_the_stage_result_numbers_come_from_the_re_derivation(self, stage_dir):
        report = _report(stage_dir, _panel())
        assert command_tracking_stage_result(None) == {}
        assert command_tracking_stage_result({"scored_gate_kind": "stance_quality/v2"}) == {}
        unjudged = command_tracking_stage_result(report)
        assert unjudged["command_profile"] == "walk" and unjudged["command_cruise_success_lcb"] is None
        report["result"]["families"][0]["success_lcb"] = 0.99  # an edited report block is never what is copied
        from environments.shared.reporting.gates import _admit_command_tracking_report

        refusals, rederived = _admit_command_tracking_report(WALK_CURRICULUM, report, stage=2, stage_dir=stage_dir)
        assert refusals == [] and rederived is not None
        values = command_tracking_stage_result(report, rederived=rederived)
        assert values["command_cruise_success_lcb"] == pytest.approx(0.05**0.25)
        assert set(values) == set(COMMAND_TRACKING_STAGE_RESULT_KEYS)


# ── the follow profile ───────────────────────────────────────────────────────

#: A follow command (turns, speed changes and stops at the switches), as test_command_tracking_gate's.
FOLLOW_CONFIG = DirectionCommandConfig(
    speed_range=(0.5, 1.05), stop_probability=0.15, turn_increment_max=math.pi / 6, switch_interval_s=6.0
)
#: The provisional follow bars with the biped gait clause and the paired reference against the parent.
FOLLOW_CURRICULUM: dict[str, Any] = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": COMMAND_TRACKING_GATE_KIND,
    "timesteps": 10_000,
    **PROVISIONAL_FOLLOW_THRESHOLDS,
    **PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    "min_follow_reference_delta_lcb": -0.1,
}
FOLLOW = thresholds_from_curriculum(FOLLOW_CURRICULUM)
FOLLOW_MEASUREMENT = panel_measurement(FOLLOW_CONFIG, FOLLOW)
FOLLOW_PLAN = plan_command_panel(FOLLOW, FOLLOW_CONFIG)
FOLLOW_HORIZON = 2500  # 25 s at 0.01 s, past min_follow_horizon_s


def _follow_panel(**overrides_by_seed: Any) -> list[CommandTrackingEpisode]:
    """The follow plan's episodes, all passing: 25 s, four events (a stop and two turns), each settled."""
    follow = dict(
        horizon=FOLLOW_HORIZON,
        length=FOLLOW_HORIZON,
        planned_event_count=4,
        tracking_fraction=0.8,
        eligible_events=4,
        settled_events=4,
        eligible_stops=1,
        settled_stops=1,
        eligible_turns=2,
        settled_turns=2,
        measurement=FOLLOW_MEASUREMENT,
    )
    return [
        _record(seed, index, family, **{**follow, **overrides_by_seed.get(f"seed_{seed}", {})})
        for index, (seed, family) in enumerate(zip(FOLLOW_PLAN.seeds, FOLLOW_PLAN.families))
    ]


def _follow_stage(tmp_path: Path) -> Path:
    directory = _stage_dir(tmp_path, FOLLOW_CURRICULUM, command=FOLLOW_CONFIG)
    record = json.loads((directory / "stage_config.json").read_text())
    record["reward_weights"]["max_episode_steps"] = FOLLOW_HORIZON
    (directory / "stage_config.json").write_text(json.dumps(record), encoding="utf-8")
    _choose(directory, curriculum=FOLLOW_CURRICULUM)
    return directory


def _follow_report(stage_dir: Path, episodes: list[CommandTrackingEpisode], reference: list[CommandTrackingEpisode]):
    models = stage_dir / "models"
    return assemble_command_tracking_report(
        COMPSOGNATHUS_LONGIPES,
        2,
        curriculum=FOLLOW_CURRICULUM,
        command_config=FOLLOW_CONFIG,
        plan=FOLLOW_PLAN,
        episodes=episodes,
        measurement=FOLLOW_MEASUREMENT,
        horizon=FOLLOW_HORIZON,
        control_dt=0.01,
        policy="robust_best_model.zip — VecNormalize stats applied (clip 10)",
        normalization={"mode": "vecnormalize-frozen", "clip_obs": 10.0, "epsilon": 1e-8},
        handoff={
            "checkpoint": "robust_best_model.zip",
            "checkpoint_sha256": sha256_file(models / "robust_best_model.zip"),
            "normalization_sha256": sha256_file(models / "robust_best_model_vecnorm.pkl"),
        },
        task_sha256=TASK_SHA256,
        plant_identity={"plant": "test"},
        checkpoint_plant_validated=True,
        reference={
            "policy": "parent.zip",
            "normalization": {"mode": "vecnormalize-frozen"},
            "handoff": {"checkpoint": "parent.zip", "checkpoint_sha256": PARENT_SHA256},
        },
        reference_episodes=reference,
    )


class TestFollowProfile:
    def test_a_follow_report_with_its_reference_is_judged_on_the_paired_bound(self, tmp_path):
        stage_dir = _follow_stage(tmp_path)
        report = _follow_report(stage_dir, _follow_panel(), _follow_panel())
        assert report["profile"] == "follow" and report["passed"] is True
        assert evaluate_stage_gate(
            FOLLOW_CURRICULUM, {}, stage=2, stage_dir=stage_dir, command_tracking_report=report
        ) == (True, [])
        from environments.shared.reporting.stage_artifacts import admitted_command_tracking_stage_result

        values = admitted_command_tracking_stage_result(FOLLOW_CURRICULUM, report, stage=2, stage_dir=stage_dir)
        assert values["command_profile"] == "follow" and values["command_follow_episodes"] == 20
        assert values["command_paired_delta_lcb"] == 0.0 and values["command_cruise_episodes"] is None
        # The policy forgot what the reference does: three seeds fail that the reference passes.
        forgetful = _follow_panel(**{f"seed_{seed}": {"tracking_fraction": 0.1} for seed in FOLLOW_PLAN.seeds[:3]})
        failing = _follow_report(stage_dir, forgetful, _follow_panel())
        passed, failures = evaluate_stage_gate(
            FOLLOW_CURRICULUM, {}, stage=2, stage_dir=stage_dir, command_tracking_report=failing
        )
        assert not passed and any(failure.startswith("stage 2 paired_delta_lcb") for failure in failures), failures
        # A declared comparison without the reference panel fails closed.
        alone = _follow_report(stage_dir, _follow_panel(), _follow_panel())
        alone["reference"] = None
        refusals = command_tracking_report_refusals(FOLLOW_CURRICULUM, alone, stage=2, stage_dir=stage_dir)
        assert any("disagrees with the one re-derived" in refusal for refusal in refusals), refusals

    def test_publication_re_derives_the_follow_pass_with_its_reference_rows(self, tmp_path):
        stage_dir = _follow_stage(tmp_path)
        write_command_tracking_report(stage_dir, _follow_report(stage_dir, _follow_panel(), _follow_panel()))
        evidence = read_command_tracking_panel(stage_dir / COMMAND_TRACKING_PANEL_CSV)
        assert evidence.profile == "follow" and len(evidence.reference) == len(evidence.policy) == 20
        record = json.loads((stage_dir / "stage_config.json").read_text())
        kwargs: dict[str, Any] = dict(
            env_kwargs=record["reward_weights"],
            stage=2,
            panel_seed_start=PUBLICATION_SEED_START,
            certified_hash=sha256_file(stage_dir / "models" / "robust_best_model.zip"),
            certified_normalization=sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
            task_sha256=TASK_SHA256,
            parent_checkpoint_sha256=PARENT_SHA256,
        )
        _validate_command_tracking_panel_evidence(stage_dir / COMMAND_TRACKING_PANEL_CSV, FOLLOW_CURRICULUM, **kwargs)
        with pytest.raises(ResultBundleError, match="reference rows"):
            _validate_command_tracking_panel_evidence(
                stage_dir / COMMAND_TRACKING_PANEL_CSV,
                FOLLOW_CURRICULUM,
                **{**kwargs, "parent_checkpoint_sha256": "sha256:" + "05" * 32},
            )

    def test_a_reference_on_the_planned_seeds_but_other_families_is_refused(self, tmp_path):
        """The judge and publication check the reference rows' families against the plan, as they do the policy's."""
        stage_dir = _follow_stage(tmp_path)
        assert {"cruise", "stand"} <= set(FOLLOW_PLAN.families) or len(set(FOLLOW_PLAN.families)) == 1
        swapped = [
            replace(episode, family="stand" if episode.family != "stand" else "cruise") for episode in _follow_panel()
        ]
        report = _follow_report(stage_dir, _follow_panel(), swapped)
        assert report["passed"] is False
        passed, failures = evaluate_stage_gate(
            FOLLOW_CURRICULUM, {}, stage=2, stage_dir=stage_dir, command_tracking_report=report
        )
        assert not passed
        assert any("reference panel's families are not the planned panel's" in failure for failure in failures), (
            failures
        )
        write_command_tracking_report(stage_dir, report)
        record = json.loads((stage_dir / "stage_config.json").read_text())
        with pytest.raises(ResultBundleError, match="reference rows are not the planned panel"):
            _validate_command_tracking_panel_evidence(
                stage_dir / COMMAND_TRACKING_PANEL_CSV,
                FOLLOW_CURRICULUM,
                env_kwargs=record["reward_weights"],
                stage=2,
                panel_seed_start=PUBLICATION_SEED_START,
                certified_hash=sha256_file(stage_dir / "models" / "robust_best_model.zip"),
                certified_normalization=sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
                task_sha256=TASK_SHA256,
                parent_checkpoint_sha256=PARENT_SHA256,
            )

    def test_the_results_csv_carries_the_follow_columns(self):
        results = {
            "stage": 2,
            "name": "Follow",
            "description": "follow the command",
            "timesteps": 1000,
            "duration_seconds": 10.0,
            "publication_gate_passed": True,
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            **dict.fromkeys(COMMAND_TRACKING_STAGE_RESULT_KEYS),
            "command_profile": "follow",
            "command_n_episodes": 20,
            "command_follow_episodes": 20,
            "command_follow_successes": 20,
            "command_follow_success_lcb": 0.8609,
            "command_follow_survival_lcb": 0.8609,
            "command_paired_delta_lcb": 0.0,
            "command_handoff_selected": "robust_best_model",
        }
        row = build_results_csv_rows(
            [results], {2: {"curriculum_kwargs": FOLLOW_CURRICULUM}}, COMPSOGNATHUS_LONGIPES, "ppo", 3
        )[0]
        assert row["command_profile"] == "follow" and row["command_follow_success_lcb"] == pytest.approx(0.8609)
        assert row["follow_success_lcb_threshold"] == 0.6 and row["follow_survival_lcb_threshold"] == 0.8
        assert row["command_paired_delta_lcb"] == 0.0
        assert not any(key.startswith(("cruise_", "stand_", "command_cruise_", "command_stand_")) for key in row)


# ── the readers: catalog, CSV, summaries ─────────────────────────────────────


class TestReaders:
    def _gate(self, curriculum: dict[str, Any]) -> dict[str, Any]:
        from environments.shared.species_catalog import _advancement_gate
        from environments.shared.stage_manifest import load_stage_manifest

        return _advancement_gate(load_stage_manifest(COMPSOGNATHUS_LONGIPES).by_id("locomotion"), curriculum)

    def test_the_catalog_exports_and_renders_the_walk_gate(self):
        from environments.shared.species_catalog import _format_advancement_gate

        gate = self._gate(dict(WALK_CURRICULUM))
        assert gate["profile"] == "walk" and gate["min_cruise_success_lcb"] == 0.3 and gate["gait_limbs"] == 2
        assert gate["min_follow_success_lcb"] is None, "every key of the kind, declared or null"
        text = _format_advancement_gate(gate)
        assert "cruise episodes (a non-zero reset command) success LCB95 ≥ 0.3" in text
        assert "stand episodes (an exactly zero command) over ≥ 2" in text and "≤ 0 fall" in text
        assert "min_alternation_index ≥ 0.75" in text and "max_hop_flight_fraction ≤ 0.15" in text
        assert "handoff chosen by the gated statistic" in text and "consecutive passes" not in text

    def test_the_catalog_renders_the_follow_gate(self):
        from environments.shared.species_catalog import _format_advancement_gate

        follow = {
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            **PROVISIONAL_FOLLOW_THRESHOLDS,
            **PROVISIONAL_BIPED_GAIT_THRESHOLDS,
            "min_follow_reference_delta_lcb": 0.0,
        }
        text = _format_advancement_gate(self._gate(follow))
        assert "command-following episodes success LCB95 ≥ 0.6" in text
        assert "paired Δ success vs the parent policy LCB95 ≥ 0" in text and "cruise" not in text

    def test_a_non_command_stage_exports_exactly_what_it_did(self):
        """The stance (its kind is not command_tracking/v1) exports none of the kind's keys; the walk, which declares
        the kind since decision D-D62, exports its profile and gait clause."""
        from environments.shared.config import load_stage_config

        curriculum = load_stage_config(COMPSOGNATHUS_LONGIPES, "stance")["curriculum_kwargs"]
        assert "profile" not in self._gate(curriculum) and "gait_limbs" not in self._gate(curriculum)
        walk = self._gate(load_stage_config(COMPSOGNATHUS_LONGIPES, "locomotion")["curriculum_kwargs"])
        assert walk["profile"] == "walk" and walk["gait_limbs"] == 2

    def test_the_headline_reads_the_summary_row(self):
        from environments.shared.species_catalog import _deliverable_headline

        row = {"command_cruise_success_lcb": 0.473, "command_stand_success_fraction": 1.0}
        walk = _deliverable_headline(COMMAND_TRACKING_GATE_KIND, row, self._gate(dict(WALK_CURRICULUM)))
        assert [(entry["key"], entry["value"]) for entry in walk] == [
            ("command_cruise_success_lcb", 0.473),
            ("command_stand_success_fraction", 1.0),
        ]
        follow = _deliverable_headline(
            COMMAND_TRACKING_GATE_KIND, {"command_follow_success_lcb": 0.7}, {"profile": "follow"}
        )
        assert [(entry["key"], entry["value"]) for entry in follow] == [("command_follow_success_lcb", 0.7)]

    def _results(self) -> dict[str, Any]:
        return {
            "stage": 2,
            "name": "Locomotion",
            "description": "walk",
            "timesteps": 1000,
            "duration_seconds": 10.0,
            "publication_gate_passed": True,
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            "command_profile": "walk",
            "command_n_episodes": 6,
            "command_cruise_episodes": 4,
            "command_cruise_successes": 4,
            "command_cruise_success_lcb": 0.4728708045,
            "command_cruise_survival_lcb": 0.4728708045,
            "command_stand_episodes": 2,
            "command_stand_successes": 2,
            "command_stand_success_fraction": 1.0,
            "command_stand_falls": 0,
            "command_follow_episodes": None,
            "command_follow_successes": None,
            "command_follow_success_lcb": None,
            "command_follow_survival_lcb": None,
            "command_paired_delta_lcb": None,
            "command_handoff_selected": "best_model",
        }

    def test_the_results_csv_carries_the_statistics_on_a_command_row_only(self):
        results = self._results()
        rows = build_results_csv_rows(
            [results], {2: {"curriculum_kwargs": WALK_CURRICULUM}}, COMPSOGNATHUS_LONGIPES, "ppo", 3
        )
        row = rows[0]
        assert row["command_profile"] == "walk" and row["command_n_episodes"] == 6
        assert row["command_cruise_success_lcb"] == pytest.approx(0.4729)
        assert row["cruise_success_lcb_threshold"] == 0.3 and row["stand_success_fraction_threshold"] == 0.6
        assert row["command_stand_falls"] == 0 and row["command_handoff_selected"] == "best_model"
        other = build_results_csv_rows(
            [{**results, "gate_kind": "reward_and_length/v1"}],
            {2: {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 1.0}}},
            COMPSOGNATHUS_LONGIPES,
            "ppo",
            3,
        )[0]
        assert not any(key.startswith(("command_", "cruise_", "stand_")) for key in other)

    def test_the_summary_row_carries_them_and_the_csv_agrees(self, tmp_path):
        results = self._results()
        summary_row = _canonical_stage_summary(results)
        assert summary_row["command_profile"] == "walk"
        assert summary_row["command_cruise_success_lcb"] == pytest.approx(0.4729)
        assert summary_row["command_stand_successes"] == 2 and summary_row["command_follow_success_lcb"] is None
        assert "command_handoff_selected" not in summary_row
        plain = _canonical_stage_summary({**results, "gate_kind": "reward_and_length/v1"})
        assert not any(key.startswith("command_") for key in plain)
        row = build_results_csv_rows(
            [results], {2: {"curriculum_kwargs": WALK_CURRICULUM}}, COMPSOGNATHUS_LONGIPES, "ppo", 3
        )[0]
        path = tmp_path / "collected_results.csv"
        with path.open("w", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        summary = {"stages": {"2": summary_row}}
        problems = compare_summary_to_csv(summary, path)
        assert not any("command_" in problem for problem in problems), problems
        tampered = {"stages": {"2": {**summary_row, "command_cruise_success_lcb": 0.9}}}
        assert any(
            "command_cruise_success_lcb differs" in problem for problem in compare_summary_to_csv(tampered, path)
        )

    def test_the_stage_summary_states_the_families(self, tmp_path):
        results = {
            **self._results(),
            "gate_passed": True,
            "gate_failures": [],
            "mean_reward": 1.0,
            "std_reward": 0.0,
            "mean_episode_length": 1000.0,
            "std_episode_length": 0.0,
            "sim_dt": 0.02,
        }
        path = write_stage_summary(tmp_path, results, COMPSOGNATHUS_LONGIPES, "ppo", evaluation_episodes=30)
        text = path.read_text()
        assert "Profile:      walk (6 panel episodes)" in text
        assert "Cruise:       4/4 successful; success LCB95 0.4729" in text
        assert "Stand:        2/2 holding the stop (1.000); falls 0" in text
        assert "Handoff:      best_model (chosen by the gated statistic)" in text


# ── end to end: a short PPO run judged through the real post-stage path ─────


#: The design's walk command, as a stage TOML declares it.
WALK_COMMANDS = """
[commands]
mode = "heading_and_speed"
cruise_speed = 0.5
speed_range = [0.25, 0.5]
stop_probability = 0.15
turn_increment_max = 0.0
switch_interval_s = 100.0
initial_heading_offset_max = 0.26
"""


@pytest.fixture(scope="module")
def trained_stage(tmp_path_factory):
    """A temporary live compsognathus walk stage under a tiny command panel, trained 256 PPO steps."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from environments.shared import train_base
    from environments.shared.config import load_all_stages, load_stage_config
    from environments.shared.curriculum.gate_schema import NON_THRESHOLD_CURRICULUM_KEYS
    from environments.shared.paths import REPOSITORY_ROOT
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import load_stage_manifest

    root = tmp_path_factory.mktemp("command_walk")
    directory = root / "configs" / COMPSOGNATHUS_LONGIPES.species / COMPSOGNATHUS_LONGIPES.embodiment
    shutil.copytree(COMPSOGNATHUS_LONGIPES.path(REPOSITORY_ROOT / "configs"), directory)
    entry = load_stage_manifest(COMPSOGNATHUS_LONGIPES).by_id("locomotion")
    path = directory / entry.config_file
    replace_commands_table(path, WALK_COMMANDS)
    config = load_stage_config(COMPSOGNATHUS_LONGIPES, "locomotion", config_path=str(path))
    config["env_kwargs"]["max_episode_steps"] = 75  # 1.5 s at the compsognathus's 0.02 s step
    config["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1)
    curriculum = {
        key: value for key, value in config["curriculum_kwargs"].items() if key in NON_THRESHOLD_CURRICULUM_KEYS
    }
    curriculum.update(
        {
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            **PROVISIONAL_WALK_THRESHOLDS,
            **PROVISIONAL_BIPED_GAIT_THRESHOLDS,
            "min_cruise_episodes": 2,
            "min_stand_episodes": 1,
        }
    )
    config["curriculum_kwargs"] = curriculum
    configs = load_all_stages(COMPSOGNATHUS_LONGIPES)
    configs[entry.reference] = config
    species = get_species_config(COMPSOGNATHUS_LONGIPES)
    train_base.train(
        species,
        configs,
        entry.reference,
        total_timesteps=256,
        n_envs=1,
        seed=3,
        eval_freq=64,
        save_freq=1_000,
        verbose=0,
        algorithm="ppo",
        output_dir=str(root / "run"),
        use_tensorboard=False,
        post_eval_episodes=0,
        report_metrics=False,
        save_on_interrupt=False,
    )
    (stage_dir,) = [record.parent for record in (root / "run").rglob("stage_config.json")]
    return species, config, entry.reference, stage_dir


class TestEndToEnd:
    def test_training_records_a_handoff_chosen_by_the_gated_statistic(self, trained_stage):
        _species, config, _key, stage_dir = trained_stage
        models = stage_dir / "models"
        choice, problem = recorded_handoff_choice(models)
        assert problem is None and choice is not None
        names = [entry["name"] for entry in choice["candidates"]]
        assert names[:2] == ["robust_best_model", "best_model"] and names[-1].endswith("_final")
        assert choice["selection_seed_start"] == 2042 and choice["plan"]["seed_start"] == 2042
        assert choice["gate_sha256"] == gate_config_sha256(gate_config_view(config["curriculum_kwargs"]))
        assert all(entry["key"] is not None for entry in choice["candidates"])
        assert select_handoff_checkpoint(models)[0] == choice["selected"]

    def test_the_post_stage_path_judges_it_from_its_report(self, trained_stage):
        from environments.shared.reporting import generate_stage_artifacts

        species, config, key, stage_dir = trained_stage
        results = generate_stage_artifacts(
            species, config, key, "ppo", stage_dir, 3, record_videos=False, generate_graphs=False
        )
        report = json.loads((stage_dir / COMMAND_TRACKING_REPORT_JSON).read_text())
        curriculum = config["curriculum_kwargs"]
        assert command_tracking_report_refusals(curriculum, report, stage=key, stage_dir=stage_dir) == []
        assert report["plan"]["seeds"] == list(
            plan_command_panel(thresholds_from_curriculum(curriculum), config["env_kwargs"]["command_config"]).seeds
        )
        verdict = json.loads((stage_dir / "gate_verdict.json").read_text())
        assert verdict["passed"] is report["passed"] is results["gate_passed"]
        assert verdict["failures"] == results["gate_failures"]
        assert all(not failure.startswith(f"stage {key} report") for failure in verdict["failures"]), "re-derived"
        assert results["command_profile"] == "walk" and results["command_cruise_episodes"] == 2
        assert results["command_stand_episodes"] == 1
        assert results["command_handoff_selected"] == select_handoff_checkpoint(stage_dir / "models")[0]
        assert verdict["stage_result"]["command_cruise_episodes"] == 2
        assert "Profile:      walk" in (stage_dir / "stage_summary.txt").read_text()
        evidence = read_command_tracking_panel(stage_dir / COMMAND_TRACKING_PANEL_CSV)
        assert [episode.seed for episode in evidence.policy] == report["plan"]["seeds"]
        # Publication reaches the re-derivation: the binding holds, and the verdict is the judge's.
        recorded = json.loads((stage_dir / "stage_config.json").read_text())
        handoff = select_handoff_checkpoint(stage_dir / "models")
        kwargs = dict(
            env_kwargs=recorded["reward_weights"],
            stage=key,
            panel_seed_start=PUBLICATION_SEED_START,
            certified_hash=sha256_file(handoff[1] + ".zip"),
            certified_normalization=sha256_file(handoff[2]),
            task_sha256=recorded["task_fingerprint"]["task_sha256"],
            parent_checkpoint_sha256=None,
        )
        if report["passed"]:
            _validate_command_tracking_panel_evidence(stage_dir / COMMAND_TRACKING_PANEL_CSV, curriculum, **kwargs)
        else:
            with pytest.raises(ResultBundleError, match="publication gate fails command_tracking/v1"):
                _validate_command_tracking_panel_evidence(stage_dir / COMMAND_TRACKING_PANEL_CSV, curriculum, **kwargs)
        # An edited report is refused, never re-judged as a pass.
        report["passed"] = not report["passed"]
        refusals = command_tracking_report_refusals(curriculum, report, stage=key, stage_dir=stage_dir)
        assert any("disagrees with the one re-derived" in refusal for refusal in refusals)

    def test_the_judge_branch_makes_a_missing_choice_before_reading_the_handoff(self, trained_stage, tmp_path):
        """The notebook's JUDGE branch (evaluate_stage_checkpoints) on a copy whose choice record was lost."""
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.reporting.stage_artifacts import evaluate_stage_checkpoints

        species, config, key, trained_dir = trained_stage
        stage_dir = tmp_path / trained_dir.name
        shutil.copytree(trained_dir, stage_dir)
        models = stage_dir / "models"
        expected = json.loads((models / HANDOFF_CHOICE_FILENAME).read_text())
        (models / HANDOFF_CHOICE_FILENAME).unlink()
        assert select_handoff_checkpoint(models)[0] == "robust_best_model", "without the record: the reward order"
        (final_zip,) = models.glob("*_final.zip")
        final_stem = models / final_zip.stem
        _model, handoff_stem, _final, handoff_vecnorm, results = evaluate_stage_checkpoints(
            species,
            config,
            key,
            "ppo",
            stage_dir,
            final_path=final_stem,
            final_vecnorm_path=f"{final_stem}_vecnorm.pkl",
            timesteps=256,
            duration_seconds=0.0,
            plant_identity=current_plant_identity(COMPSOGNATHUS_LONGIPES),
            evaluation_seed=1003,
        )
        choice, problem = recorded_handoff_choice(models)
        assert problem is None and choice is not None
        assert handoff_choice_candidates_problem(choice, models) is None
        # The same seeds, the same deterministic policies: the same choice as at the end of training.
        assert choice["selected"] == expected["selected"]
        assert [entry["key"] for entry in choice["candidates"]] == [entry["key"] for entry in expected["candidates"]]
        assert Path(handoff_stem).name == choice["selected"] == Path(results["model_path"]).name
        assert handoff_vecnorm == str(models / f"{choice['selected']}_vecnorm.pkl")

    def test_the_post_stage_path_draws_the_command_figures(self, trained_stage, tmp_path):
        """With graphs on (the notebook's call): the trace of the certified chart episodes and both figures."""
        pytest.importorskip("matplotlib")
        import numpy as np

        from environments.shared.reporting import generate_stage_artifacts, stage_layout
        from environments.shared.reporting.command_tracking_trace import (
            COMMAND_TRACKING_TRACE_CSV,
            chart_plan,
            read_command_tracking_trace,
            trace_episodes,
        )

        species, config, key, trained_dir = trained_stage
        stage_dir = tmp_path / trained_dir.name
        shutil.copytree(trained_dir, stage_dir)
        verdict_before = (
            json.loads((stage_dir / "gate_verdict.json").read_text())
            if (stage_dir / "gate_verdict.json").is_file()
            else None
        )
        generate_stage_artifacts(species, config, key, "ppo", stage_dir, 3, record_videos=False, generate_graphs=True)
        figures = stage_layout.figures_dir(stage_dir)
        assert (figures / "command_tracking.png").stat().st_size > 0
        assert (figures / "command_gate_progress.png").stat().st_size > 0
        assert (figures / "gate_progress.png").is_file(), "the generic figure is drawn as before"
        report = json.loads((stage_dir / COMMAND_TRACKING_REPORT_JSON).read_text())
        rows = read_command_tracking_trace(stage_dir / COMMAND_TRACKING_TRACE_CSV)
        assert rows is not None
        plan = chart_plan(
            CommandPanelPlan(
                **{
                    **report["plan"],
                    "seeds": tuple(report["plan"]["seeds"]),
                    "families": tuple(report["plan"]["families"]),
                }
            )
        )
        grouped = trace_episodes(rows)
        assert [(seed, family) for seed, family, _ in grouped] == list(zip(plan.seeds, plan.families))
        assert all(row["record_matches_report"] for row in rows), "the chart episodes are the certified ones"
        # The live stage's training recorded its tracking fraction, on the diagnostics clock.
        with np.load(stage_dir / "diagnostics.npz") as data:
            assert data["tracking_in_tolerance"].shape == data["timesteps"].shape
            assert np.isfinite(data["tracking_in_tolerance"]).all()
        if verdict_before is not None:
            verdict = json.loads((stage_dir / "gate_verdict.json").read_text())
            assert verdict["passed"] is verdict_before["passed"], "the figures change no verdict"

    @pytest.mark.parametrize("generate_graphs", [True, False])
    def test_a_rejudgement_without_a_report_leaves_no_published_command_figure(
        self, trained_stage, tmp_path, monkeypatch, generate_graphs
    ):
        """The notebook path draws into a staging directory that is only copied over the stage directory, so the
        published figures of an earlier judgement must be removed by the stage-directory sweep, with or without
        graphs: else a PASS-titled figure could stand beside a FAIL gate_verdict.json."""
        pytest.importorskip("matplotlib")
        from environments.shared.reporting import generate_stage_artifacts, stage_artifacts, stage_layout
        from environments.shared.reporting.command_tracking_trace import COMMAND_TRACKING_TRACE_CSV

        species, config, key, trained_dir = trained_stage
        stage_dir = tmp_path / trained_dir.name
        shutil.copytree(trained_dir, stage_dir)
        figures = stage_layout.figures_dir(stage_dir)
        generate_stage_artifacts(species, config, key, "ppo", stage_dir, 3, record_videos=False, generate_graphs=True)
        assert (figures / "command_tracking.png").is_file() and (figures / "command_gate_progress.png").is_file()
        assert (stage_dir / COMMAND_TRACKING_TRACE_CSV).is_file()
        monkeypatch.setattr(stage_artifacts, "_write_command_tracking_report", lambda *args, **kwargs: None)
        results = generate_stage_artifacts(
            species, config, key, "ppo", stage_dir, 3, record_videos=False, generate_graphs=generate_graphs
        )
        assert results["gate_passed"] is False
        assert json.loads((stage_dir / "gate_verdict.json").read_text())["passed"] is False
        assert not (figures / "command_tracking.png").exists()
        assert not (figures / "command_gate_progress.png").exists()
        assert not (stage_dir / COMMAND_TRACKING_TRACE_CSV).exists()

    def test_the_cli_curriculums_post_training_judge_agrees(self, trained_stage):
        from environments.shared.reporting import stage_layout
        from environments.shared.reporting.command_tracking_trace import COMMAND_TRACKING_TRACE_CSV
        from environments.shared.train_base import _post_training_command_verdict

        _species, config, key, stage_dir = trained_stage
        passed, failures, stage_result = _post_training_command_verdict(
            COMPSOGNATHUS_LONGIPES, key, config, stage_dir=stage_dir, model_dir=stage_dir / "models"
        )
        # The report the CLI judge rolled and wrote itself (no other test's run of the shared stage is assumed).
        report = json.loads((stage_dir / COMMAND_TRACKING_REPORT_JSON).read_text())
        assert (
            command_tracking_report_refusals(config["curriculum_kwargs"], report, stage=key, stage_dir=stage_dir) == []
        )
        assert passed is report["passed"] and failures == [f"stage {key} {failure}" for failure in report["failures"]]
        assert stage_result["gate_kind"] == COMMAND_TRACKING_GATE_KIND and stage_result["command_profile"] == "walk"
        assert stage_result["command_n_episodes"] == 3
        # After the verdict, the CLI draws what the notebook draws.
        if importlib.util.find_spec("matplotlib") is not None:
            figures = stage_layout.figures_dir(stage_dir)
            assert (figures / "command_tracking.png").is_file() and (figures / "command_gate_progress.png").is_file()
        assert (stage_dir / COMMAND_TRACKING_TRACE_CSV).is_file()
