"""Tests for environments.shared.replication — replicate discovery over LOG_BASE siblings (plan §4.5, D-B16).

A replicate is a sibling run of the SAME recipe: a passed, reusable verdict
for the node with equal ``task_sha256`` and ``gate_sha256``, a
``stage_config.json`` recording the same plant and the same
``hyperparameters_sha256`` (recorded, else derived), and a different
training seed.  The sibling directories are built with
``test_ancestors.build_trunk_run`` — the same judged-run shape reuse rule 7
reads — one per skip reason, so every branch of the definition is pinned
in one place.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from environments.shared.config import hyperparameters_sha256, recorded_hyperparameters_sha256
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.replication import REPLICATE_RUN_FIELDS, discover_replicates, discover_replicates_for_run
from environments.shared.stage_manifest import load_stage_manifest

from .reporting_helpers import make_plant_identity
from .test_ancestors import OTHER_TASK, STANCE_CURRICULUM, STANCE_TASK, build_trunk_run, strip_gate_record, trunk_plant

#: The recipe every same-recipe sibling records (D-A21 shape or pre-D-A21 shape).
RECIPE = {"learning_rate": 3e-4, "n_steps": 2048}
RECIPE_DIGEST = hyperparameters_sha256({"ppo_kwargs": RECIPE, "curriculum_kwargs": STANCE_CURRICULUM}, "PPO")
STANCE_GATE = gate_config_sha256(gate_config_view(STANCE_CURRICULUM))
STANCE_ENTRY = load_stage_manifest("trex").by_id("stance")


def _run(log_base: Path, name: str, **overrides: Any) -> Path:
    """A sibling run under ``LOG_BASE/trex/ppo/<name>`` with the same-recipe defaults."""
    run = log_base / name
    run.mkdir(parents=True)
    kwargs: dict[str, Any] = dict(algorithm="PPO", hyperparameters=RECIPE, record_recipe_digest=True)
    kwargs.update(overrides)
    build_trunk_run(run, **kwargs)
    return run


@pytest.fixture
def log_base(tmp_path: Path) -> Path:
    base = tmp_path / "logs" / "trex" / "ppo"
    base.mkdir(parents=True)
    return base


def _discover(run: Path, **overrides: Any) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = dict(
        species="trex",
        entry=STANCE_ENTRY,
        task_sha256=STANCE_TASK,
        plant_identity=trunk_plant(),
        gate_sha256=STANCE_GATE,
        hyperparameters_sha256=RECIPE_DIGEST,
        training_seed=42,
    )
    kwargs.update(overrides)
    return discover_replicates(run, **kwargs)


def test_replicate_run_fields_are_the_schema_record_fields() -> None:
    assert REPLICATE_RUN_FIELDS == ("run_id", "training_seed")


def test_only_passed_same_recipe_distinct_seed_siblings_are_replicates(log_base: Path, caplog: Any) -> None:
    """Seeds 42 (this run), 43 PASS, 44 FAIL, 45 pre-D-A22, 46 another recipe, 47 another plant, 48 pre-D-A21."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    _run(log_base, "20260902_120000", seed=43)
    _run(log_base, "20260903_120000", seed=44, passed=False)
    strip_gate_record(_run(log_base, "20260904_120000", seed=45) / "01_stance")
    _run(log_base, "20260905_120000", seed=46, hyperparameters={**RECIPE, "learning_rate": 1e-4})
    other_plant = make_plant_identity(
        species="trex", model_path="environments/trex/assets/trex.xml", physics_sha256="sha256:" + "9" * 64
    )
    _run(log_base, "20260906_120000", seed=47, plant=other_plant)
    # Today's certified runs: a run block without hyperparameters_sha256,
    # but the recorded blocks digest to the same recipe (D-B16: derived,
    # never skipped for the missing field alone).
    _run(log_base, "20260907_120000", seed=48, record_recipe_digest=False)
    _run(log_base, "20260908_120000", seed=49, task_sha256=OTHER_TASK)
    _run(log_base, "20260909_120000", seed=42)  # this run's own seed, re-run

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        replicates = _discover(this_run)

    assert replicates == [
        {"run_id": "20260902_120000", "training_seed": 43},
        {"run_id": "20260907_120000", "training_seed": 48},
    ]
    skipped = "\n".join(record.getMessage() for record in caplog.records if "skipping" in record.getMessage())
    assert "skipping 20260903_120000: its gate verdict is FAILED" in skipped
    assert "skipping 20260904_120000: its verdict records no gate_sha256" in skipped
    assert "backfill_gate_verdict.py --force" in skipped
    assert "skipping 20260905_120000: recipe digest" in skipped and "another recipe" in skipped
    assert "skipping 20260906_120000: its recorded plant identity differs" in skipped
    assert "skipping 20260908_120000: judged under task" in skipped
    assert "skipping 20260909_120000: it trained this run's own seed 42" in skipped
    assert "skipping 20260901_120000" not in skipped, "the run itself is never a candidate"


def test_replicates_are_sorted_by_run_id_and_one_per_run_id(log_base: Path) -> None:
    """A sibling that carries another sibling's provenance run_id (a copied directory) is counted once."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    later = _run(log_base, "20260930_120000", seed=44)
    earlier = _run(log_base, "20260910_120000", seed=43)
    copy = _run(log_base, "20260920_120000", seed=44)
    for run in (later, copy):
        (run / "provenance.json").write_text(json.dumps({"run_id": "trex-sb3-ppo-seed44"}) + "\n", encoding="utf-8")
    (earlier / "provenance.json").write_text(json.dumps({"run_id": "trex-sb3-ppo-seed43"}) + "\n", encoding="utf-8")

    assert _discover(this_run) == [
        {"run_id": "trex-sb3-ppo-seed43", "training_seed": 43},
        {"run_id": "trex-sb3-ppo-seed44", "training_seed": 44},
    ]


def test_a_sibling_carrying_this_runs_own_id_is_skipped(log_base: Path) -> None:
    this_run = _run(log_base, "20260901_120000", seed=42)
    (this_run / "provenance.json").write_text(json.dumps({"run_id": "trex-own"}) + "\n", encoding="utf-8")
    copy = _run(log_base, "20260902_120000", seed=43)
    (copy / "provenance.json").write_text(json.dumps({"run_id": "trex-own"}) + "\n", encoding="utf-8")

    assert _discover(this_run) == []


def test_two_siblings_of_one_seed_count_once(log_base: Path, caplog: Any) -> None:
    """Distinct seeds are the record's invariant, so the second run of a seed is not a second replicate."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    _run(log_base, "20260902_120000", seed=43)
    _run(log_base, "20260903_120000", seed=43)

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        replicates = _discover(this_run)

    assert replicates == [{"run_id": "20260902_120000", "training_seed": 43}]
    assert any("repeats training seed 43" in record.getMessage() for record in caplog.records)


def test_a_malformed_sibling_is_skipped_with_a_logged_reason_never_raised(log_base: Path, caplog: Any) -> None:
    this_run = _run(log_base, "20260901_120000", seed=42)
    broken = _run(log_base, "20260902_120000", seed=43)
    (broken / "01_stance" / "gate_verdict.json").write_text("{not json", encoding="utf-8")
    unreadable_config = _run(log_base, "20260903_120000", seed=44)
    (unreadable_config / "01_stance" / "stage_config.json").write_text("[]", encoding="utf-8")
    no_seed = _run(log_base, "20260904_120000", seed=45)
    config_path = no_seed / "01_stance" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["run"]["seed"]
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    (log_base / "20260905_120000").mkdir()  # no stage directory at all
    (log_base / "notes.txt").write_text("not a run\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert _discover(this_run) == []

    messages = [record.getMessage() for record in caplog.records]
    assert any("skipping 20260902_120000: unreadable gate_verdict.json" in message for message in messages)
    assert any("skipping 20260903_120000: no readable stage_config.json" in message for message in messages)
    assert any(
        "skipping 20260904_120000: its stage_config.json run block records no training seed" in m for m in messages
    )
    assert any("skipping 20260905_120000: no stage directory for 'stance'" in message for message in messages)


@pytest.mark.parametrize("plant_value", ["trex", ["a", "b"], 7, None], ids=["string", "list", "int", "null"])
def test_a_sibling_recording_a_non_object_plant_identity_is_skipped_never_raised(
    log_base: Path, caplog: Any, plant_value: Any
) -> None:
    """The plant normaliser reads a mapping: a sibling whose ``plant_identity`` is anything else is skipped with
    a logged reason (not an AttributeError out of ``PlantIdentity.from_mapping``)."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    sibling = _run(log_base, "20260902_120000", seed=43)
    config_path = sibling / "01_stance" / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["plant_identity"] = plant_value
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert _discover(this_run) == []
    assert any(
        "skipping 20260902_120000: its recorded plant identity is not an object" in record.getMessage()
        for record in caplog.records
    )


def test_a_sibling_judged_under_another_gate_is_another_recipe(log_base: Path, caplog: Any) -> None:
    this_run = _run(log_base, "20260901_120000", seed=42)
    _run(log_base, "20260902_120000", seed=43, judged_under={**STANCE_CURRICULUM, "min_avg_reward": 1940.0})

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert _discover(this_run) == []
    assert any("judged under gate" in record.getMessage() for record in caplog.records)


def test_recorded_hyperparameters_sha256_prefers_the_run_block_and_derives_otherwise() -> None:
    recorded = {"algorithm": "PPO", "hyperparameters": RECIPE, "curriculum": STANCE_CURRICULUM, "run": {"seed": 1}}
    assert recorded_hyperparameters_sha256(recorded) == RECIPE_DIGEST
    stated = {**recorded, "run": {"seed": 1, "hyperparameters_sha256": "sha256:" + "a" * 64}}
    assert recorded_hyperparameters_sha256(stated) == "sha256:" + "a" * 64
    assert recorded_hyperparameters_sha256({"hyperparameters": RECIPE, "curriculum": STANCE_CURRICULUM}) is None


def test_discover_replicates_for_run_keys_by_stage_key(log_base: Path) -> None:
    """Every deliverable with a certifiable verdict is keyed by its stage key; the rest contribute no key."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    _run(log_base, "20260902_120000", seed=43)
    _run(log_base, "20260903_120000", seed=44, record_recipe_digest=False)

    replicates = discover_replicates_for_run(this_run, species="trex", plant_identity=trunk_plant())

    assert replicates == {
        "1": [
            {"run_id": "20260902_120000", "training_seed": 43},
            {"run_id": "20260903_120000", "training_seed": 44},
        ]
    }
    # The plant may be given as the dict the notebook passes to the writer.
    assert discover_replicates_for_run(this_run, species="trex", plant_identity=trunk_plant().to_dict()) == replicates


def test_discover_replicates_for_run_skips_a_node_whose_own_verdict_lacks_gate_sha256(
    log_base: Path, caplog: Any
) -> None:
    """A pre-D-A22 node cannot match a gate it did not record: it counts this run alone until re-judged."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    strip_gate_record(this_run / "01_stance")
    _run(log_base, "20260902_120000", seed=43)

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert discover_replicates_for_run(this_run, species="trex", plant_identity=trunk_plant()) == {}
    assert any("records no gate_sha256" in record.getMessage() for record in caplog.records)


def test_discover_replicates_for_run_logs_a_node_without_a_readable_stage_config(log_base: Path, caplog: Any) -> None:
    """The counting run's own skip is logged too, so it is not mistaken for 'no replicates found'."""
    this_run = _run(log_base, "20260901_120000", seed=42)
    (this_run / "01_stance" / "stage_config.json").write_text("[]", encoding="utf-8")
    _run(log_base, "20260902_120000", seed=43)

    with caplog.at_level(logging.INFO, logger="environments.shared.replication"):
        assert discover_replicates_for_run(this_run, species="trex", plant_identity=trunk_plant()) == {}
    assert any(
        "'stance' has no readable stage_config.json, counting this run alone" in record.getMessage()
        for record in caplog.records
    )


def test_discover_replicates_for_run_skips_a_failed_or_unjudged_node(log_base: Path) -> None:
    failed = _run(log_base, "20260901_120000", seed=42, passed=False)
    _run(log_base, "20260902_120000", seed=43)
    assert discover_replicates_for_run(failed, species="trex", plant_identity=trunk_plant()) == {}

    unjudged = _run(log_base, "20260903_120000", seed=44, verdict=False)
    assert discover_replicates_for_run(unjudged, species="trex", plant_identity=trunk_plant()) == {}


def test_a_pre_d_a21_node_derives_its_own_recipe_digest(log_base: Path) -> None:
    """The counting run itself may predate D-A21: its recipe is derived from its recorded blocks too."""
    this_run = _run(log_base, "20260901_120000", seed=42, record_recipe_digest=False)
    _run(log_base, "20260902_120000", seed=43)

    assert discover_replicates_for_run(this_run, species="trex", plant_identity=trunk_plant()) == {
        "1": [{"run_id": "20260902_120000", "training_seed": 43}]
    }
