"""The ``stance_quality/v2`` report end to end on a real plant: roll, score, write, judge, publish.

A short floor-truth panel on compsognathus (the cheapest plant: 0.02 s
steps, ~1.5 s per three 150-step episodes) under a v2 block at the floor map
§5.4 compsognathus bars, through ``build_stance_gate_report`` exactly as the
post-stage pipeline calls it.  The "policy" is scripted -- the checkpoint
loader is stubbed to return a fixed command, and the handoff pair is a pair
of byte files the report digests -- so the panel is real physics, the
recorder is the real recorder, and the verdict is the real gate:

* the zero-action statue (as a policy) classifies clean on every episode and
  passes, with its statue panel rolled separately on the same seeds;
* three scripted hacks do not: a neck held at its limit (only the saturation
  criterion catches it -- the animal keeps standing), a hip rolled out (one
  foot lifted: support, chatter, the settle hop and impact) and every
  actuator saturated (it falls);
* the statue-relative ratios of a checkpoint are taken against a statue
  panel rolled separately, never the checkpoint's own panel;
* the judge (``reporting.gates.evaluate_stage_gate``) certifies the written
  report, and refuses each substitute: no report, a v1-scored report
  labelled v2 (or a v2-scored one labelled v1), other thresholds or settle
  window, another checkpoint, the statue's own report, a checkpoint scored
  with the plant contract waived, another measurement (constants, species,
  schema, registry, step), a statue block that is not a separate zero-action
  panel, a probe, a panel rolled under another task than the stage
  recorded (another ``[env]`` block, same horizon), a stage that records no
  task or horizon, rows on other seeds, a tampered statue reference, and a
  recorded verdict, clean count or clean flag its own rows contradict;
* publication re-derives the verdict from the written panel CSV and binds it
  to the certified pair, the stage's recorded task and this checkout's
  measurement definition, and refuses a recorded reached-horizon or clean
  flag the rows contradict;
* a probe on a v2 stage keeps the v1-shaped report with no recorder, no
  statue panel and no CSV, and the recorder is detached even when the
  rollout raises.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.curriculum.gate_schema import gate_config_view
from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND, STANCE_V2_REPORT_SCHEMA
from environments.shared.gait.constants import measurement_constants, measurement_sha256
from environments.shared.gait.stance_metrics import StanceEpisodeMetrics
from environments.shared.plant_contract import current_plant_identity
from environments.shared.reporting import evaluate_stage_gate, stance_report
from environments.shared.result_bundle import ResultBundleError, sha256_file
from environments.shared.task_fingerprint import stage_task_fingerprint

SPECIES = "compsognathus"
HORIZON = 150
EPISODES = 3

#: The floor map §5.4 compsognathus criteria, on a 3-episode panel (3/3 bounds at 0.368).
CURRICULUM: dict[str, Any] = {
    "gate_schema_version": 1,
    "gate_kind": STANCE_GATE_V2_KIND,
    "min_eval_episodes": EPISODES,
    "min_clean_stance_lcb": 0.3,
    "settle_steps": 50,
    "min_all_feet_support": 0.98,
    "max_touchdown_rate": 0.25,
    "max_window_displacement_m": 0.10,
    "min_foot_load_share": 0.30,
    "max_actuator_saturation_fraction": 0.10,
    "max_settle_airborne_substeps": 0,
    "max_settle_peak_floor_force_bw": 1.5,
    "max_foot_contact_fraction": 0.02,
    "max_phantom_support_fraction": 0.05,
    "max_sole_corner_lift_m": 0.003,
    "min_support_geom_duty": 0.90,
    "min_foot_load_share_statue_ratio": 0.8,
    "min_avg_reward_statue_ratio": 0.6,
}


def _stage_config(**curriculum: Any) -> dict[str, Any]:
    config = copy.deepcopy(load_stage_config(SPECIES, 1))
    config["env_kwargs"]["max_episode_steps"] = HORIZON
    config["curriculum_kwargs"] = {**CURRICULUM, **curriculum}
    return config


def _record_stage(stage_dir: Path, config: dict[str, Any] | None = None) -> str:
    """Write the ``stage_config.json`` / ``task_fingerprint.json`` training writes for *config*; its task digest."""
    config = _stage_config() if config is None else config
    identity = current_plant_identity(SPECIES)
    fingerprint = stage_task_fingerprint(SPECIES, 1, stage_config=config, plant_identity=identity)
    save_stage_config(
        stage_dir,
        1,
        config,
        "PPO",
        env_class=stance_report.SPECIES_FACTORIES[SPECIES]().env_class,
        species=SPECIES,
        plant_identity=identity,
        task_fingerprint=fingerprint,
    )
    return str(fingerprint["task_sha256"])


def _handoff(stage_dir: Path) -> tuple[Path, Path]:
    models = stage_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    model, vecnorm = models / "robust_best_model.zip", models / "robust_best_model_vecnorm.pkl"
    model.write_bytes(b"scripted policy weights")
    vecnorm.write_bytes(b"scripted observation statistics")
    return model, vecnorm


def _scripted(command: np.ndarray, *, raise_at: int | None = None) -> Any:
    calls = {"n": 0}

    def predict(_obs: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        if raise_at is not None and calls["n"] >= raise_at:
            raise RuntimeError("the scripted policy failed mid-episode")
        return command

    return predict


def _report(monkeypatch: Any, stage_dir: Path, command: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Score *command* as the checkpoint in *stage_dir*'s handoff pair."""
    model, vecnorm = _handoff(stage_dir)
    monkeypatch.setattr(stance_report, "_load_policy", lambda *a, **k: (_scripted(command), "scripted command"))
    return stance_report.build_stance_gate_report(
        SPECIES,
        1,
        stage_config=kwargs.pop("stage_config", _stage_config()),
        model_path=str(model),
        vecnorm_path=str(vecnorm),
        **kwargs,
    )


class _PostSettleHop:
    """Zero action through the settle, then both legs in phase (knees +a; ankles and hip pitch -a), 10 Hz square wave.

    The two-foot hop the D-D26 review certified under the step-level window
    bars: on this light plant its flights last 1-3 of a step's ten substeps,
    so every leg reads down on every step.  Stateful: the report calls
    ``reset`` before each episode.
    """

    SIGNS = {"r_knee_act": 1, "l_knee_act": 1, "r_ankle_act": -1, "l_ankle_act": -1, "r_hip_pitch_act": -1,
             "l_hip_pitch_act": -1}  # fmt: skip

    def __init__(self, nu: int, *, settle: int, amplitude: float = 0.15, period: int = 5) -> None:
        env = stance_report.SPECIES_FACTORIES[SPECIES]().env_class(**_stage_config()["env_kwargs"])
        try:
            self.pattern = np.zeros(nu)
            for name, sign in self.SIGNS.items():
                self.pattern[env.model.actuator(name).id] = sign * amplitude
        finally:
            env.close()
        self.settle, self.period, self.step = settle, period, 0

    def reset(self) -> None:
        self.step = 0

    def __call__(self, _obs: np.ndarray) -> np.ndarray:
        step, self.step = self.step, self.step + 1
        if step < self.settle:
            return np.zeros_like(self.pattern)
        return self.pattern if (step - self.settle) % self.period < self.period / 2 else -self.pattern


@pytest.fixture(scope="module")
def nu() -> int:
    env = stance_report.SPECIES_FACTORIES[SPECIES]().env_class(**_stage_config()["env_kwargs"])
    try:
        return int(env.action_space.shape[0])
    finally:
        env.close()


@pytest.fixture(scope="module")
def statue(tmp_path_factory: Any, nu: int) -> tuple[Path, dict[str, Any]]:
    """The zero-action statue scored as a policy, written beside its handoff pair as the pipeline writes it."""
    stage_dir = tmp_path_factory.mktemp("stance_v2_stage")
    _record_stage(stage_dir)
    with pytest.MonkeyPatch.context() as monkeypatch:
        report = _report(monkeypatch, stage_dir, np.zeros(nu))
    stance_report.write_stance_gate_report(stage_dir, report)
    return stage_dir, report


def _written(stage_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((stage_dir / "stance_gate_report.json").read_text(encoding="utf-8"))
    return report


def _judge(stage_dir: Path, report: Any, **curriculum: Any) -> tuple[bool, list[str]]:
    return evaluate_stage_gate({**CURRICULUM, **curriculum}, {}, stage=1, stance_report=report, stage_dir=stage_dir)


def _keys(row: dict[str, Any]) -> set[str]:
    return {reason.split(":", 1)[0] for reason in row["reasons"]}


def _rehashed(edit: Any) -> Any:
    """*edit* applied to a report's measurement manifest, its digest recomputed -- a forgery, not a slip."""

    def apply(report: dict[str, Any]) -> None:
        edit(report["measurement"])
        report["measurement_sha256"] = measurement_sha256(report["measurement"])

    return apply


# ── the statue, and the scripted hacks ───────────────────────────────────────


class TestThePanel:
    def test_the_statue_classifies_clean_on_every_episode_and_passes(self, statue):
        _, report = statue
        assert report["passed"] is True and report["failures"] == []
        assert report["result"]["n_clean"] == EPISODES == report["result"]["n_episodes"]
        assert [row["clean"] for row in report["episode_evidence"]] == [True] * EPISODES
        assert [row["seed"] for row in report["episode_evidence"]] == [3042, 3043, 3044]
        # Its statue panel was rolled separately (the policy is a checkpoint here) and agrees with it.
        assert report["statue"]["reused_policy_panel"] is False
        assert report["statue"]["reference"]["n_full_horizon"] == EPISODES
        assert report["statue"]["reference"]["foot_load_share"] == pytest.approx([0.5, 0.5], abs=0.01)
        assert report["result"]["statue_mean_reward"] == pytest.approx(report["metrics"]["reward_mean"])

    def test_the_report_records_what_scored_it(self, statue):
        stage_dir, report = statue
        assert report["schema"] == STANCE_V2_REPORT_SCHEMA == "mesozoic.stance-gate-report/v3"
        assert report["gate_kind"] == report["scored_gate_kind"] == STANCE_GATE_V2_KIND
        # The task the panel rolled is the one training records for this stage config.
        assert report["task_sha256"] == json.loads((stage_dir / "task_fingerprint.json").read_text())["task_sha256"]
        assert report["checkpoint_plant_validated"] is True
        assert report["thresholds"] == gate_config_view(CURRICULUM)["thresholds"]
        assert report["thresholds"]["settle_steps"] == 50  # unlike the v1 report
        assert report["measurement"]["constants"] == measurement_constants()
        assert report["measurement"]["species"] == SPECIES and report["measurement"]["frame_skip"] == 10
        assert report["handoff"] == {
            "checkpoint": "robust_best_model.zip",
            "checkpoint_sha256": sha256_file(stage_dir / "models" / "robust_best_model.zip"),
            "normalization_sha256": sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
        }
        text = (stage_dir / "stance_gate_report.txt").read_text(encoding="utf-8")
        assert "GATE: PASS" in text and "NOTE" not in text
        assert "max_actuator_saturation_fraction" in text and "statue reference" in text

    def test_the_written_report_round_trips_every_metric_exactly(self, statue):
        """JSON writes NaN as null and from_row reads null back as NaN; every other value is bit-exact."""
        stage_dir, report = statue

        def same(a: Any, b: Any) -> bool:
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                return True
            return bool(a == b)

        for live, written in zip(report["episode_evidence"], _written(stage_dir)["episode_evidence"]):
            expected = StanceEpisodeMetrics.from_row(live).as_row()
            actual = StanceEpisodeMetrics.from_row(written).as_row()
            assert expected.keys() == actual.keys()
            assert all(same(expected[key], actual[key]) for key in expected), [
                key for key in expected if not same(expected[key], actual[key])
            ]

    @pytest.mark.parametrize(
        ("hack", "expected"),
        [
            ("neck_at_limit", {"max_actuator_saturation_fraction"}),
            ("hip_rolled_out", {"min_all_feet_support", "max_touchdown_rate", "max_settle_peak_floor_force_bw"}),
            ("all_saturated", {"horizon", "max_actuator_saturation_fraction"}),
        ],
    )
    def test_a_scripted_hack_is_not_clean(self, tmp_path, monkeypatch, nu, statue, hack, expected):
        """...and its statue-relative ratios are taken against the STATUE's panel, rolled separately.

        Both panels go through the recorder, and the statue reference is the
        statue fixture's to the bit (same env, seeds and recorder): were the
        hack's own panel reused as its statue, every ratio would be 1 and the
        reference the hack's.
        """
        command = np.zeros(nu)
        if hack == "neck_at_limit":
            command[0] = 1.0  # neck_pitch_act: the animal keeps standing; only the actuation gives it away
        elif hack == "hip_rolled_out":
            command[10] = 1.0  # l_hip_roll_act: one foot swung off the floor
        else:
            command[:] = 1.0
        panels: list[bool] = []
        original = stance_report.run_panel

        def counted(*args: Any, **kwargs: Any) -> Any:
            panels.append(bool(kwargs.get("floor_truth")))
            return original(*args, **kwargs)

        monkeypatch.setattr(stance_report, "run_panel", counted)
        report = _report(monkeypatch, tmp_path, command)
        assert report["passed"] is False and report["result"]["n_clean"] == 0
        for row in report["episode_evidence"]:
            assert row["clean"] is False
            assert expected <= _keys(row), row["reasons"]
        if hack == "neck_at_limit":
            assert all(_keys(row) == expected for row in report["episode_evidence"])
        assert panels == [True, True]
        assert report["statue"]["reused_policy_panel"] is False
        assert report["statue"]["reference"] == statue[1]["statue"]["reference"]
        assert report["result"]["statue_mean_reward"] != report["result"]["mean_reward"]

    def test_a_post_settle_hop_reads_a_statue_step_by_step_and_fails_the_window_hop_pair(
        self, tmp_path, monkeypatch, nu, statue
    ):
        """Real physics, recorder, gate and judge: the window pair sees what the step-level bars cannot.

        Every leg is down on every window step, with no touchdown and no
        flight, and the settle is the statue's; but both feet leave the floor
        on dozens of window substeps and land at several body weights.
        """
        window = {"max_window_airborne_substeps": 10, "max_window_peak_floor_force_bw": 2.0}
        config = _stage_config(**window)
        _record_stage(tmp_path, config)
        hop = _PostSettleHop(nu, settle=CURRICULUM["settle_steps"])
        model, vecnorm = _handoff(tmp_path)
        monkeypatch.setattr(stance_report, "_load_policy", lambda *a, **k: (hop, "scripted post-settle hop"))
        report = stance_report.build_stance_gate_report(
            SPECIES, 1, stage_config=config, model_path=str(model), vecnorm_path=str(vecnorm)
        )
        step_level = {"min_all_feet_support", "max_touchdown_rate", "max_settle_airborne_substeps",
                      "max_settle_peak_floor_force_bw"}  # fmt: skip
        for row in report["episode_evidence"]:
            assert row["length"] == HORIZON
            assert (row["all_feet_support"], row["touchdown_rate"], row["flight_fraction"]) == (1.0, 0.0, 0.0)
            assert row["settle_airborne_substeps"] == 0.0
            assert row["window_airborne_substeps"] > 3 * window["max_window_airborne_substeps"]
            assert row["window_peak_floor_force_bw"] > 1.5 * window["max_window_peak_floor_force_bw"]
            assert set(window) <= _keys(row) and not step_level & _keys(row), row["reasons"]
        assert report["passed"] is False and report["result"]["n_clean"] == 0
        # The statue the same report rolled meets the pair with room: no flight, its own weight on the floor.
        for row in report["statue"]["episode_evidence"]:
            assert row["window_airborne_substeps"] == 0.0 and row["window_peak_floor_force_bw"] < 1.1
        # The judge re-derives the refusal from the written report.
        stance_report.write_stance_gate_report(tmp_path, report)
        passed, reasons = _judge(tmp_path, _written(tmp_path), **window)
        assert passed is False and reasons
        # Undeclared, the pair is no criterion: the same episodes fail only what the rest of the block reads.
        assert all(not set(window) & _keys(row) for row in _reclassified(report, CURRICULUM))
        # The zero-action statue scored as the checkpoint under the same block passes, and the judge certifies it.
        statue_dir = tmp_path / "statue"
        _record_stage(statue_dir, config)
        model, vecnorm = _handoff(statue_dir)
        monkeypatch.setattr(stance_report, "_load_policy", lambda *a, **k: (_scripted(np.zeros(nu)), "zero"))
        quiet = stance_report.build_stance_gate_report(
            SPECIES, 1, stage_config=config, model_path=str(model), vecnorm_path=str(vecnorm)
        )
        assert quiet["passed"] is True and quiet["result"]["n_clean"] == EPISODES
        stance_report.write_stance_gate_report(statue_dir, quiet)
        assert _judge(statue_dir, _written(statue_dir), **window) == (True, [])


def _reclassified(report: dict[str, Any], curriculum: dict[str, Any]) -> list[dict[str, Any]]:
    """*report*'s episode rows with their reasons re-derived under *curriculum*."""
    from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds, classify_stance_episode

    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    rows = []
    for row in report["episode_evidence"]:
        episode = StanceEpisodeMetrics.from_row(row)
        rows.append({**row, "reasons": list(classify_stance_episode(episode, thresholds, horizon=HORIZON))})
    return rows


# ── the judge ─────────────────────────────────────────────────────────────────


class TestTheJudge:
    def test_it_certifies_the_written_report_by_re_deriving_it(self, statue):
        stage_dir, _ = statue
        assert _judge(stage_dir, _written(stage_dir)) == (True, [])

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            (lambda r: r.update(scored_gate_kind="stance_quality/v1"), "was scored by 'stance_quality/v1'"),
            (lambda r: r.update(gate_kind="stance_quality/v1"), "records declared gate_kind 'stance_quality/v1'"),
            (lambda r: r.update(schema="mesozoic.stance-gate-report/v2"), "has schema"),
            (lambda r: r.update(filter_actions_hz=5.0), "is a probe"),
            (lambda r: r.update(spawn_yaw={"kind": "spawn_yaw/v1", "yaw_deg": 90.0}), "is a probe (spawn_yaw)"),
            (lambda r: r.update(checkpoint_plant_validated=False), "checkpoint_plant_validated=False"),
            (lambda r: r["measurement"]["constants"].update(contact_threshold_n=0.5), "measured under constants"),
            (lambda r: r["measurement"].update(body_weight_n=1.0), "is not the digest of its own manifest"),
            (_rehashed(lambda m: m.update(species="trex")), "was measured on 'trex', not 'compsognathus'"),
            (_rehashed(lambda m: m.update(schema="mesozoic.floor-truth-measurement/v0")), "measurement manifest is"),
            (lambda r: r["measurement"].update(registry=None), "support registry entry"),
            (_rehashed(lambda m: m.update(dt=0.01, frame_skip=5)), "was measured at dt 0.01 / frame_skip 5"),
            (lambda r: r["statue"].update(reused_policy_panel=True), "is not a separately rolled one"),
            (
                lambda r: r["statue"].update(policy="robust_best_model.zip"),
                "statue panel scored 'robust_best_model.zip'",
            ),
            (lambda r: r.update(task_sha256="sha256:" + "0" * 64), "but the stage ran task"),
            (lambda r: r.update(passed="yes"), "no boolean verdict"),
            (lambda r: r.update(passed=False), "passed False recorded, True re-derived"),
            (lambda r: r["result"].update(n_clean=2), "2 clean episodes recorded, 3 re-derived"),
            (lambda r: r["handoff"].update(checkpoint_sha256=None), "scored checkpoint None"),
        ],
        ids=[
            "v1-scored",
            "v1-declared",
            "schema",
            "probe",
            "heading-probe",
            "legacy-plant",
            "constants",
            "manifest-digest",
            "species",
            "manifest-schema",
            "registry",
            "step",
            "statue-reused",
            "statue-policy",
            "task",
            "verdict",
            "passed-flipped",
            "n-clean-edited",
            "statue-report",
        ],
    )
    def test_a_report_that_does_not_describe_this_gate_is_refused(self, statue, edit, message):
        stage_dir, _ = statue
        report = _written(stage_dir)
        edit(report)
        passed, failures = _judge(stage_dir, report)
        assert passed is False
        assert any(message in failure for failure in failures), failures

    def test_no_report_is_refused_not_routed_to_the_reward_rail(self, statue):
        stage_dir, _ = statue
        passed, failures = evaluate_stage_gate(
            {**CURRICULUM, "min_avg_reward": 1.0}, {"best_model_reward": 1e9}, stage=1, stage_dir=stage_dir
        )
        assert passed is False and any("no stance gate report was produced" in f for f in failures)

    @pytest.mark.parametrize(
        "change", [{"min_all_feet_support": 0.99}, {"settle_steps": 60}, {"max_sole_tilt_deg": 2.0}]
    )
    def test_a_report_scored_under_other_thresholds_is_refused(self, statue, change):
        stage_dir, _ = statue
        passed, failures = _judge(stage_dir, _written(stage_dir), **change)
        assert passed is False
        assert any("scored under other thresholds" in failure and next(iter(change)) in failure for failure in failures)

    def test_a_report_for_another_checkpoint_or_no_stage_dir_is_refused(self, statue, tmp_path):
        stage_dir, _ = statue
        other = tmp_path / "other"
        shutil.copytree(stage_dir, other)
        (other / "models" / "robust_best_model.zip").write_bytes(b"retrained weights")
        passed, failures = _judge(other, _written(stage_dir))
        assert passed is False and any("not the handoff robust_best_model's" in f for f in failures)
        passed, failures = _judge(None, _written(stage_dir))  # type: ignore[arg-type]
        assert passed is False and any("no stage_dir was given" in f for f in failures)

    def test_a_horizon_other_than_the_stage_ran_is_refused(self, statue, tmp_path):
        stage_dir, _ = statue
        other = tmp_path / "other"
        shutil.copytree(stage_dir, other)
        record = json.loads((other / "stage_config.json").read_text(encoding="utf-8"))
        record["reward_weights"]["max_episode_steps"] = 1000
        (other / "stage_config.json").write_text(json.dumps(record), encoding="utf-8")
        passed, failures = _judge(other, _written(stage_dir))
        assert passed is False and any("the stage ran 1000 steps" in f for f in failures)

    def test_a_stage_that_records_no_task_or_horizon_is_refused_not_skipped(self, statue, tmp_path):
        """The env binding fails closed: an absent record is a refusal, never a check that was not run."""
        stage_dir, _ = statue
        bare = tmp_path / "bare"
        shutil.copytree(stage_dir, bare)
        (bare / "stage_config.json").unlink()
        (bare / "task_fingerprint.json").unlink()
        passed, failures = _judge(bare, _written(stage_dir))
        assert passed is False
        assert any("records no task fingerprint" in f for f in failures), failures
        assert any("stage_config.json is absent or unreadable" in f for f in failures), failures
        # A horizon in env_kwargs is read even when reward_weights omits the key.
        record = json.loads((stage_dir / "stage_config.json").read_text(encoding="utf-8"))
        del record["reward_weights"]["max_episode_steps"]
        record["env_kwargs"] = {"max_episode_steps": 2000}
        (bare / "stage_config.json").write_text(json.dumps(record), encoding="utf-8")
        passed, failures = _judge(bare, _written(stage_dir))
        assert passed is False and any("the stage ran 2000 steps" in f for f in failures), failures

    def test_a_panel_rolled_under_another_env_block_is_refused(self, statue, tmp_path, monkeypatch, nu):
        """The same handoff pair and horizon, rolled with alive_bonus x3 and no reset noise: another task.

        Its rewards are another number (an absolute or statue-relative rail
        then reads another scale), and only the horizon used to be compared.
        """
        stage_dir, _ = statue
        config = _stage_config()
        config["env_kwargs"]["alive_bonus"] = 3.0 * float(config["env_kwargs"]["alive_bonus"])
        config["env_kwargs"]["reset_noise_scale"] = 0.0
        other = tmp_path / "other"
        shutil.copytree(stage_dir, other)
        report = _report(monkeypatch, other, np.zeros(nu), stage_config=config, episodes=1)
        passed, failures = _judge(other, report, min_eval_episodes=1)
        assert passed is False
        assert any("but the stage ran task" in f for f in failures), failures

    def test_a_recorded_verdict_its_own_rows_contradict_is_refused(self, statue):
        """Re-derivation: an edited row turns 3/3 into 2/3 (LCB 0.14 < 0.3) while the report still says PASS."""
        stage_dir, _ = statue
        report = _written(stage_dir)
        report["episode_evidence"][1]["settle_airborne_substeps"] = 4.0
        passed, failures = _judge(stage_dir, report)
        assert passed is False and any("disagrees with the one re-derived" in f for f in failures)
        flags = _written(stage_dir)
        flags["episode_evidence"][0]["clean"] = False
        passed, failures = _judge(stage_dir, flags)
        assert passed is False and any("per-episode clean flags differ" in f for f in failures)

    def test_rows_off_the_certification_panel_or_a_tampered_statue_are_refused(self, statue):
        stage_dir, _ = statue
        seeds = _written(stage_dir)
        seeds["episode_evidence"][2]["seed"] = 5000
        passed, failures = _judge(stage_dir, seeds)
        assert passed is False and any("not the certification panel seed 3044" in f for f in failures)
        tampered = _written(stage_dir)
        tampered["statue"]["reference"]["mean_reward"] *= 2
        passed, failures = _judge(stage_dir, tampered)
        assert passed is False and any("is not the one its own rows reduce to" in f for f in failures)
        no_statue = _written(stage_dir)
        no_statue["statue"] = None
        passed, failures = _judge(stage_dir, no_statue)
        assert passed is False and any("rolled no statue panel" in f for f in failures)

    def test_a_failing_report_fails_with_its_re_derived_reasons(self, tmp_path, monkeypatch, nu):
        command = np.zeros(nu)
        command[0] = 1.0
        _record_stage(tmp_path)
        report = _report(monkeypatch, tmp_path, command)
        stance_report.write_stance_gate_report(tmp_path, report)
        passed, failures = _judge(tmp_path, _written(tmp_path))
        assert passed is False
        assert failures == [f"stage 1 {failure}" for failure in report["failures"]]


# ── publication ──────────────────────────────────────────────────────────────


class TestPublication:
    def _validate(
        self,
        stage_dir: Path,
        *,
        certified_hash: Any = None,
        normalization: Any = None,
        species: str = SPECIES,
        task: Any = None,
        **curriculum: Any,
    ):
        from environments.shared.result_bundle.evidence import _validate_stance_v2_panel_evidence

        _validate_stance_v2_panel_evidence(
            stage_dir / "stance_panel_selected.csv",
            {**CURRICULUM, **curriculum},
            env_kwargs=_stage_config()["env_kwargs"],
            stage=1,
            panel_seed_start=3042,
            certified_hash=certified_hash or sha256_file(stage_dir / "models" / "robust_best_model.zip"),
            certified_normalization=normalization
            or sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
            species=species,
            task_sha256=json.loads((stage_dir / "task_fingerprint.json").read_text())["task_sha256"]
            if task is None
            else task,
        )

    def test_the_written_panel_re_derives_the_pass(self, statue):
        stage_dir, _ = statue
        self._validate(stage_dir)

    def test_the_panel_is_bound_to_the_certified_pair_and_the_declared_gate(self, statue):
        stage_dir, _ = statue
        with pytest.raises(ResultBundleError, match="not the certified selected checkpoint"):
            self._validate(stage_dir, certified_hash="sha256:" + "0" * 64)
        with pytest.raises(ResultBundleError, match="not the certified selected statistics"):
            self._validate(stage_dir, normalization="sha256:" + "0" * 64)
        with pytest.raises(ResultBundleError, match="publication gate fails stance_quality/v2"):
            self._validate(stage_dir, min_clean_stance_lcb=0.5)

    def test_the_panel_is_bound_to_the_task_the_stage_recorded(self, statue):
        stage_dir, _ = statue
        with pytest.raises(ResultBundleError, match="not the task the stage ran"):
            self._validate(stage_dir, task="sha256:" + "0" * 64)
        with pytest.raises(ResultBundleError, match="records no task fingerprint"):
            self._validate(stage_dir, task="")

    def test_the_panel_is_bound_to_this_checkouts_measurement_definition(self, statue):
        """A registry or constant edit with MEASUREMENT_VERSION unbumped: the definition digest still moves."""
        stage_dir, _ = statue
        with pytest.raises(ResultBundleError, match="not this checkout's sha256:[0-9a-f]+ for trex"):
            self._validate(stage_dir, species="trex")

    def test_a_tampered_row_is_refused(self, statue, tmp_path):
        stage_dir, _ = statue
        other = tmp_path / "other"
        shutil.copytree(stage_dir, other)
        path = other / "stance_panel_selected.csv"
        text = path.read_text(encoding="utf-8").replace("floor-truth/v1", "floor-truth/v0")
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ResultBundleError, match="measured under 'floor-truth/v0'"):
            self._validate(other)

    @pytest.mark.parametrize(
        ("column", "message"),
        [
            ("reached_horizon", "episode 0 claims reached_horizon=False but its length 150"),
            ("clean", "episode 0 records clean=False but re-derives clean=True"),
            ("measurement_definition_sha256", "measured under definition sha256:0+, not this checkout's"),
        ],
    )
    def test_a_recorded_flag_or_stamp_the_rows_contradict_is_refused(self, statue, tmp_path, column, message):
        from .result_bundle_helpers import _rewrite_csv_cell

        stage_dir, _ = statue
        other = tmp_path / "other"
        shutil.copytree(stage_dir, other)
        path = other / "stance_panel_selected.csv"
        if column == "measurement_definition_sha256":
            # A panel-wide stamp: every row must carry the same value.
            with path.open(newline="", encoding="utf-8") as source:
                stamp = next(csv.DictReader(source))[column]
            path.write_text(path.read_text(encoding="utf-8").replace(stamp, "sha256:" + "0" * 64), encoding="utf-8")
        else:
            _rewrite_csv_cell(path, field=column, value="False")
        with pytest.raises(ResultBundleError, match=message):
            self._validate(other)


# ── probes and hook hygiene ──────────────────────────────────────────────────


class TestProbesAndHygiene:
    def test_a_probe_on_a_v2_stage_rolls_no_recorder_no_statue_and_writes_no_csv(self, tmp_path, monkeypatch, nu):
        panels: list[bool] = []
        original = stance_report.run_panel

        def counted(*args: Any, **kwargs: Any) -> Any:
            panels.append(bool(kwargs.get("floor_truth")))
            return original(*args, **kwargs)

        monkeypatch.setattr(stance_report, "run_panel", counted)
        report = _report(monkeypatch, tmp_path, np.zeros(nu), episodes=1, filter_actions_hz=5.0)
        assert panels == [False]  # one panel, unrecorded: no statue roll
        assert report["schema"] == "mesozoic.stance-gate-report/v2"
        assert report["scored_gate_kind"] == "stance_quality/v1" and report["gate_kind"] == STANCE_GATE_V2_KIND
        assert "statue" not in report and "measurement" not in report
        written = stance_report.write_stance_gate_report(tmp_path, report)
        assert set(written) == {"stance_gate_report_txt", "stance_gate_report_json"}
        assert not (tmp_path / "stance_panel_selected.csv").exists()
        text = written["stance_gate_report_txt"].read_text(encoding="utf-8")
        assert "PROBE" in text and "scored the 'stance_quality/v1' criteria" in text
        passed, failures = _judge(tmp_path, report)
        assert passed is False

    def test_the_recorder_is_detached_and_the_env_closed_when_the_rollout_raises(self, tmp_path, monkeypatch, nu):
        from environments.shared.gait.recorder import SubstepContactRecorder

        detached: list[bool] = []
        original_detach = SubstepContactRecorder.detach

        def detach(self: SubstepContactRecorder) -> None:
            original_detach(self)
            detached.append(self.attached)

        monkeypatch.setattr(SubstepContactRecorder, "detach", detach)
        model, vecnorm = _handoff(tmp_path)
        monkeypatch.setattr(
            stance_report, "_load_policy", lambda *a, **k: (_scripted(np.zeros(nu), raise_at=20), "scripted")
        )
        with pytest.raises(RuntimeError, match="failed mid-episode"):
            stance_report.build_stance_gate_report(
                SPECIES, 1, stage_config=_stage_config(), model_path=str(model), vecnorm_path=str(vecnorm)
            )
        assert detached == [False]

    def test_a_block_the_gate_cannot_judge_is_refused_before_anything_is_rolled(self, monkeypatch):
        monkeypatch.setattr(stance_report, "run_panel", lambda *a, **k: pytest.fail("rolled a panel"))
        incomplete = _stage_config()
        del incomplete["curriculum_kwargs"]["min_all_feet_support"]
        with pytest.raises(ValueError, match="min_all_feet_support is missing"):
            stance_report.build_stance_gate_report(SPECIES, 1, stage_config=incomplete, zero_action=True)
        with pytest.raises(ValueError, match="spawn grace"):
            stance_report.build_stance_gate_report(
                SPECIES, 1, stage_config=_stage_config(settle_steps=5), zero_action=True
            )

    def test_a_zero_action_report_is_its_own_statue_and_certifies_no_checkpoint(self, tmp_path):
        report = stance_report.build_stance_gate_report(SPECIES, 1, stage_config=_stage_config(), zero_action=True)
        assert report["passed"] is True
        assert report["statue"]["reused_policy_panel"] is True
        assert report["handoff"] == {"checkpoint": None, "checkpoint_sha256": None, "normalization_sha256": None}
        _handoff(tmp_path)
        _record_stage(tmp_path)
        passed, failures = _judge(tmp_path, report)
        assert passed is False and any("scored checkpoint None" in failure for failure in failures)
