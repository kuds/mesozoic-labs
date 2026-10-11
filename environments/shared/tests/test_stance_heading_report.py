"""The six-heading stance certificate on a real plant: roll, write, read back, judge (decision D-D32).

Short grids -- two seeds at each of the six headings, 150-step episodes --
through ``build_stance_heading_report`` exactly as the post-stage pipeline
calls it.  The "policy" is scripted (the checkpoint loader is stubbed; the
handoff pair is two byte files the certificate digests), so the physics,
the whole-scene turn, the floor-truth recorder and the verdict are real:

* every heading is rolled with its own statue panel, and the turned statue
  repeats the unturned one to solver tolerance on both certified species
  (every plant declares the elliptic friction cone);
* the 0 degree cell is the ``stance_quality/v2`` panel bit for bit;
* the certificate records everything the judge binds, round-trips through
  its JSON and its grid CSV exactly, and the judge and publication admit it;
* a rollout that raises at one heading leaves no certificate and no
  yaw-zero report, and the stage fails;
* a stage whose task the turn does not rotate (commands, pushes, terrain)
  is refused before anything is rolled;
* a non-probe v1/v2 build of a heading-kind stage is refused, and the turn's
  docstring and probe text give the elliptic cone as the reason a turn is
  physically neutral and say which plants observe no world heading (D-D31).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND, StatueReference
from environments.shared.curriculum.stance_heading_gate import (
    FROZEN_VECNORMALIZE_MODE,
    HEADING_GRID_DEG,
    HEADING_PROTOCOL,
    STANCE_HEADING_GATE_KIND,
    STANCE_HEADING_REPORT_SCHEMA,
    read_stance_heading_panel,
    statue_references,
)
from environments.shared.gait.stance_metrics import StanceEpisodeMetrics
from environments.shared.plant_contract import current_plant_identity
from environments.shared.reporting import evaluate_stage_gate, stance_report
from environments.shared.reporting.gates import stance_heading_report_refusals
from environments.shared.result_bundle import sha256_file
from environments.shared.task_fingerprint import stage_task_fingerprint

SPECIES = "compsognathus"
HORIZON = 150
SEEDS = 2

#: The compsognathus floor map §5.4 criteria on a 2-seed grid (2/2 bounds at 0.2236), with both statue ratios.
CURRICULUM: dict[str, Any] = {
    "gate_schema_version": 1,
    "gate_kind": STANCE_HEADING_GATE_KIND,
    "min_eval_episodes": SEEDS,
    "min_clean_stance_lcb": 0.2,
    "settle_steps": 50,
    "min_all_feet_support": 0.98,
    "max_touchdown_rate": 0.25,
    "max_window_displacement_m": 0.10,
    "min_foot_load_share": 0.30,
    "max_actuator_saturation_fraction": 0.10,
    "max_settle_airborne_substeps": 0,
    "max_settle_peak_floor_force_bw": 1.5,
    "max_sole_corner_lift_m": 0.003,
    "min_foot_load_share_statue_ratio": 0.8,
    "min_avg_reward_statue_ratio": 0.6,
}


def _stage_config(species: str = SPECIES, **curriculum: Any) -> dict[str, Any]:
    config = copy.deepcopy(load_stage_config(species, 1))
    config["env_kwargs"]["max_episode_steps"] = HORIZON
    config["curriculum_kwargs"] = {**CURRICULUM, **curriculum}
    return config


def _record_stage(stage_dir: Path, config: dict[str, Any], *, species: str = SPECIES) -> str:
    """Write the ``stage_config.json`` / ``task_fingerprint.json`` training writes for *config*; its task digest."""
    identity = current_plant_identity(species)
    fingerprint = stage_task_fingerprint(species, 1, stage_config=config, plant_identity=identity)
    save_stage_config(
        stage_dir,
        1,
        config,
        "PPO",
        env_class=stance_report.SPECIES_FACTORIES[species]().env_class,
        species=species,
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
    """A checkpoint that commands *command*, normalised as _load_policy records a frozen VecNormalize policy."""
    calls = {"n": 0}

    def predict(_obs: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        if raise_at is not None and calls["n"] >= raise_at:
            raise RuntimeError("the scripted policy failed mid-episode")
        return command

    predict.normalization = {"mode": FROZEN_VECNORMALIZE_MODE, "clip_obs": 10.0, "epsilon": 1e-8}  # type: ignore[attr-defined]
    return predict


def _nu(species: str = SPECIES) -> int:
    env = stance_report.SPECIES_FACTORIES[species]().env_class(**_stage_config(species)["env_kwargs"])
    try:
        return int(env.action_space.shape[0])
    finally:
        env.close()


def _certificate(monkeypatch: Any, stage_dir: Path, predict: Any, *, config: dict[str, Any] | None = None):
    model, vecnorm = _handoff(stage_dir)
    monkeypatch.setattr(stance_report, "_load_policy", lambda *a, **k: (predict, "scripted command"))
    return stance_report.build_stance_heading_report(
        SPECIES,
        1,
        stage_config=_stage_config() if config is None else config,
        model_path=str(model),
        vecnorm_path=str(vecnorm),
    )


@pytest.fixture(scope="module")
def statue_as_policy(tmp_path_factory: Any) -> tuple[Path, dict[str, Any]]:
    """The zero command scored as a checkpoint: six policy panels and six separately rolled statue panels."""
    monkeypatch = pytest.MonkeyPatch()
    stage_dir = tmp_path_factory.mktemp("certificate")
    try:
        report = _certificate(monkeypatch, stage_dir, _scripted(np.zeros(_nu(), dtype=np.float32)))
    finally:
        monkeypatch.undo()
    _record_stage(stage_dir, _stage_config())
    return stage_dir, report


def _rows_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two evidence rows equal key for key, NaN equal to NaN."""

    def same(x: Any, y: Any) -> bool:
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return True
        return bool(x == y)

    return set(a) == set(b) and all(same(a[key], b[key]) for key in a)


def test_the_statue_certificate_rolls_every_heading_with_its_own_statue_panel(statue_as_policy):
    _, report = statue_as_policy
    assert [heading["yaw_deg"] for heading in report["headings"]] == list(HEADING_GRID_DEG)
    for heading in report["headings"]:
        statue = heading["statue"]
        assert statue["reused_policy_panel"] is False and statue["policy"] == stance_report.STATUE_POLICY
        assert [row["seed"] for row in statue["episode_evidence"]] == [3042, 3043]
        assert {row["yaw_deg"] for row in statue["episode_evidence"]} == {heading["yaw_deg"]}
        assert [row["seed"] for row in heading["episode_evidence"]] == [3042, 3043]
        assert heading["turn"] == {"kind": "spawn_yaw/v1", "yaw_deg": heading["yaw_deg"], "whole_scene": True}
        assert heading["result"]["n_clean"] == SEEDS
    assert report["passed"] is True and report["result"]["n_clean_seeds"] == SEEDS


@pytest.mark.parametrize("species", ["trex", "compsognathus"])
def test_the_turned_statue_repeats_the_unturned_one_to_solver_tolerance(species):
    """Every certified plant declares the elliptic cone, so a whole-scene turn changes nothing physical."""
    config = _stage_config(species)
    if species == "trex":
        config["curriculum_kwargs"]["max_sole_corner_lift_m"] = 0.006
    report = stance_report.build_stance_heading_report(species, 1, stage_config=config, zero_action=True)
    zero = next(heading for heading in report["headings"] if heading["yaw_deg"] == 0.0)
    for heading in report["headings"]:
        assert heading["statue"]["reused_policy_panel"] is True  # the statue IS the policy here
        for row, reference in zip(heading["episode_evidence"], zero["episode_evidence"]):
            assert row["length"] == reference["length"]
            assert row["reward"] == pytest.approx(reference["reward"], rel=0, abs=1e-8)
            for name in (
                "all_feet_support",
                "min_foot_load_share",
                "settle_peak_floor_force_bw",
                "window_displacement_m",
            ):
                assert row[name] == pytest.approx(reference[name], rel=0, abs=1e-8), (heading["yaw_deg"], name)
        # The turn itself is not drift: the episode's yaw change is the statue's own.
        for row, reference in zip(heading["episode_evidence"], zero["episode_evidence"]):
            assert row["episode_yaw_change_deg"] == pytest.approx(reference["episode_yaw_change_deg"], abs=1e-6)


def test_the_zero_cell_is_the_v2_panel_bit_for_bit(statue_as_policy, monkeypatch, tmp_path):
    """The 0 deg turn is the identity: the certificate's cell repeats the v2 report of the same block exactly."""
    _, report = statue_as_policy
    model, vecnorm = _handoff(tmp_path)
    monkeypatch.setattr(
        stance_report,
        "_load_policy",
        lambda *a, **k: (_scripted(np.zeros(_nu(), dtype=np.float32)), "scripted command"),
    )
    v2 = stance_report.build_stance_gate_report(
        SPECIES,
        1,
        stage_config=_stage_config(gate_kind=STANCE_GATE_V2_KIND),
        model_path=str(model),
        vecnorm_path=str(vecnorm),
    )
    zero = stance_report.heading_zero_panel(report)
    assert len(zero["episode_evidence"]) == len(v2["episode_evidence"]) == SEEDS
    for rows, references in (
        (zero["episode_evidence"], v2["episode_evidence"]),
        (zero["statue"]["episode_evidence"], v2["statue"]["episode_evidence"]),
    ):
        for row, reference in zip(rows, references):
            assert row["yaw_deg"] == 0.0
            assert _rows_equal({key: value for key, value in row.items() if key != "yaw_deg"}, reference)
    assert zero["statue"]["reference"] == v2["statue"]["reference"]
    assert zero["measurement"] == v2["measurement"] and zero["task_sha256"] == v2["task_sha256"]
    # The cell is rendered and probed as v2, but it can never be written or admitted as a report.
    assert zero["heading_cell_of"] == STANCE_HEADING_REPORT_SCHEMA and zero["gate_kind"] == STANCE_HEADING_GATE_KIND


def test_the_report_records_everything_the_judge_binds(statue_as_policy):
    stage_dir, report = statue_as_policy
    assert report["schema"] == STANCE_HEADING_REPORT_SCHEMA
    assert report["gate_kind"] == report["scored_gate_kind"] == STANCE_HEADING_GATE_KIND
    assert report["protocol"] == HEADING_PROTOCOL and report["heading_grid_deg"] == list(HEADING_GRID_DEG)
    assert report["seeds"] == [3042, 3043] and report["seed"] == 3042 and report["episodes"] == SEEDS
    assert report["policy_deterministic"] is True
    assert report["normalization"]["mode"] == FROZEN_VECNORMALIZE_MODE
    assert (report["settle_steps"], report["horizon"], report["control_dt"]) == (50, HORIZON, pytest.approx(0.02))
    view = gate_config_view(_stage_config()["curriculum_kwargs"])
    assert report["thresholds"] == view["thresholds"] and report["gate_sha256"] == gate_config_sha256(view)
    assert report["measurement"]["species"] == SPECIES and report["measurement_sha256"].startswith("sha256:")
    assert report["handoff"]["checkpoint_sha256"] == sha256_file(stage_dir / "models" / "robust_best_model.zip")
    assert report["handoff"]["normalization_sha256"] == sha256_file(
        stage_dir / "models" / "robust_best_model_vecnorm.pkl"
    )
    assert (
        report["task_sha256"]
        == stage_task_fingerprint(
            SPECIES, 1, stage_config=_stage_config(), plant_identity=current_plant_identity(SPECIES)
        )["task_sha256"]
    )
    assert report["plant_identity"] == current_plant_identity(SPECIES).to_dict()
    assert report["checkpoint_plant_validated"] is True
    assert all(report[key] is None for key in stance_report._PROBE_MARKERS)
    assert stance_report.probe_stem(report) is None
    # The judge admits it and re-derives the PASS; its numbers are what the verdict records.
    assert (
        stance_heading_report_refusals(_stage_config()["curriculum_kwargs"], report, stage=1, stage_dir=stage_dir) == []
    )
    assert evaluate_stage_gate(
        _stage_config()["curriculum_kwargs"], {}, stage=1, stance_report=report, stage_dir=stage_dir
    ) == (
        True,
        [],
    )


def test_the_written_report_round_trips_every_metric_exactly(statue_as_policy, tmp_path):
    stage_dir, report = statue_as_policy
    written = stance_report.write_stance_heading_report(tmp_path, report)
    assert {path.name for path in written.values()} == {
        "stance_heading_report.txt",
        "stance_heading_report.json",
        "stance_heading_panel_selected.csv",
    }
    assert not (tmp_path / "stance_gate_report.json").exists() and not (tmp_path / "stance_panel_selected.csv").exists()
    text = written["stance_heading_report_txt"].read_text(encoding="utf-8")
    assert (
        "seeds clean at every heading" in text and "NOT the statistic" in text and text.rstrip().endswith("GATE: PASS")
    )
    assert "the 0 deg cell in detail" in text and "NOTE:" not in text
    from_json = json.loads(written["stance_heading_report_json"].read_text(encoding="utf-8"))
    assert evaluate_stage_gate(
        _stage_config()["curriculum_kwargs"], {}, stage=1, stance_report=from_json, stage_dir=stage_dir
    ) == (True, [])
    evidence = read_stance_heading_panel(written["stance_heading_panel_csv"])
    by_cell = {(row.seed, row.yaw_deg): row.metrics for row in evidence.policy}
    for heading in report["headings"]:
        for row in heading["episode_evidence"]:
            expected = StanceEpisodeMetrics.from_row(row).as_row()
            assert _rows_equal(by_cell[(row["seed"], heading["yaw_deg"])].as_row(), expected)
    reduced = statue_references(evidence.statue, horizon=HORIZON)
    for heading in report["headings"]:
        recorded = StatueReference.from_dict(heading["statue"]["reference"])
        assert reduced[heading["yaw_deg"]] == recorded == evidence.statue_by_heading[heading["yaw_deg"]]
    assert evidence.gate_sha256 == report["gate_sha256"] and evidence.task_sha256 == report["task_sha256"]
    assert evidence.heading_grid_deg == HEADING_GRID_DEG and (evidence.seed_start, evidence.n_seeds) == (3042, SEEDS)
    assert evidence.policy_clean == (True,) * (SEEDS * len(HEADING_GRID_DEG))
    with pytest.raises(ValueError, match="write_stance_heading_report"):
        stance_report.write_stance_gate_report(tmp_path / "v2", report)


def test_a_rollout_that_raises_at_one_heading_writes_no_certificate_and_the_stage_fails(monkeypatch, tmp_path, caplog):
    """A failure partway through the grid gives no certificate -- never the cells that were rolled, never yaw zero."""
    from environments.shared.reporting import stage_artifacts

    config = _stage_config()
    _record_stage(tmp_path, config)
    # Two seeds x 150 steps per panel, policy and statue at each heading: the policy fails in the third heading.
    predict = _scripted(np.zeros(_nu(), dtype=np.float32), raise_at=2 * SEEDS * HORIZON + 10)
    with pytest.raises(stance_report.StanceGateReportError, match=r"rollout at \+0 deg raised RuntimeError"):
        _certificate(monkeypatch, tmp_path, predict, config=config)
    predict = _scripted(np.zeros(_nu(), dtype=np.float32), raise_at=2 * SEEDS * HORIZON + 10)
    monkeypatch.setattr(stance_report, "_load_policy", lambda *a, **k: (predict, "scripted command"))
    with caplog.at_level("WARNING"):
        report = stage_artifacts._write_stance_gate_report(
            species=SPECIES, stage=1, stage_config=config, stage_dir=tmp_path, model_dir=tmp_path / "models"
        )
    assert report is None and "no yaw-zero fallback" in caplog.text
    assert not any(tmp_path.glob("stance_heading_*")) and not any(tmp_path.glob("stance_gate_report*"))
    assert not (tmp_path / "stance_panel_selected.csv").exists()
    passed, failures = evaluate_stage_gate(
        config["curriculum_kwargs"], {"best_model_reward": 1e9}, stage=1, stance_report=report, stage_dir=tmp_path
    )
    assert passed is False and "never falls back to the yaw-zero stance_quality/v2 panel" in failures[0]


def test_a_stage_with_commands_pushes_or_terrain_is_refused_before_anything_is_rolled(monkeypatch):
    monkeypatch.setattr(stance_report, "run_panel", lambda *a, **k: pytest.fail("a panel was rolled"))
    assert stance_report.heading_protocol_refusals({}, SimpleNamespace(nhfield=0)) == []
    (command,) = stance_report.heading_protocol_refusals({"command_mode": "heading"})
    assert "does not rotate the command controller's world heading" in command
    (push,) = stance_report.heading_protocol_refusals({"perturbation_capture_velocity_multiple": 1.5})
    assert "does not rotate the world push schedule" in push
    (terrain,) = stance_report.heading_protocol_refusals({}, SimpleNamespace(nhfield=1))
    assert "does not rotate the terrain" in terrain
    for env in ({"command_mode": "heading_and_speed"}, {"perturbation_capture_velocity_multiple": 1.0}):
        config = _stage_config()
        config["env_kwargs"].update(env)
        with pytest.raises(stance_report.StanceGateReportError, match="cannot be certified under whole_scene_yaw/v1"):
            stance_report.build_stance_heading_report(SPECIES, 1, stage_config=config, zero_action=True)


def test_a_non_probe_v1_or_v2_build_of_a_heading_stage_is_refused(monkeypatch):
    config = _stage_config()
    with pytest.raises(ValueError, match="build it with build_stance_heading_report"):
        stance_report.build_stance_gate_report(SPECIES, 1, stage_config=config, zero_action=True)
    with pytest.raises(ValueError, match="not 'stance_quality_heading/v1'"):
        stance_report.build_stance_heading_report(
            SPECIES, 1, stage_config=_stage_config(gate_kind=STANCE_GATE_V2_KIND), zero_action=True
        )
    # A probe stays the v1-shaped report it always was: it annotates, it never certifies.
    probe = stance_report.build_stance_gate_report(
        SPECIES, 1, stage_config=config, zero_action=True, episodes=1, spawn_yaw=stance_report.SpawnYaw(90.0, True)
    )
    assert probe["scored_gate_kind"] == "stance_quality/v1" and stance_report.probe_stem(probe) is not None


def test_the_turn_text_cites_the_elliptic_cone():
    """The reason a turn is physically neutral is the elliptic friction cone every plant declares (D-D29, D-D30)."""
    doc = " ".join((stance_report.SpawnYaw.__doc__ or "").split())
    assert 'cone="elliptic"' in doc and "D-D29" in doc and "D-D30" in doc
    assert "changes nothing physical -- the floor is level and isotropic" not in doc
    assert "stance_quality_heading/v1" in doc and "makes this probe a gate check" not in doc
    from .test_stance_heading_probe import STANDS, _sweeps

    # The observation half (D-D31): the trex from r14 and the anatomical compsognathus from r4 observe no world
    # heading, so a whole-scene turn changes nothing they see; the velociraptor and the robot still observe it.
    assert "D-D31" in doc and "policy interface r14" in doc and "anatomical compsognathus from r4" in doc
    assert "The velociraptor and the compsognathus robot still observe those world-frame channels" in doc
    assert "only the world-frame observation differs" not in doc
    text, payload = stance_report.render_heading_probe(*_sweeps(STANDS), probe_episodes=8)
    assert "elliptic friction cone" in payload["note"] and "stance_quality_heading/v1" in payload["note"]
    assert "the six-heading certificate, D-D32" in text and "not a gate result" in text
    assert "observe no world heading (D-D31)" in payload["note"] and "only the world-frame" not in payload["note"]
    assert "the trex r14 and anatomical compsognathus r4 observe none" in text
