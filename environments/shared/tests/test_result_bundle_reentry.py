"""``environments.shared.result_bundle.reentry``: what a notebook session re-entering a run is refused.

Consolidation PR-14a. A run whose ``artifact_manifest.json`` records ``complete`` is immutable
(docs/RESULT_BUNDLES.md), so a session that would judge or train a node into it is refused before anything
is trained or written, and a reuse-only session passes; a root widened on the command line keeps its parent's
seed (decision D-C14) and is judged before any trunk may stand in for it (decision D-C13). Every check reads
the run directory only, so the cases below also assert the directory is byte-identical afterwards. No SB3,
torch or IPython: these run in the shared test matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared.config import WIDEN_LINEAGE_KEYS
from environments.shared.result_bundle import (
    ANCESTOR_RECORD_NAME,
    ANCESTORS_DIRNAME,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_PROVENANCE_NAME,
    GATE_VERDICT_FILENAME,
    ResultBundleError,
    read_bundle_status,
    refuse_complete_run_session,
    refuse_trunk_over_unjudged_widened_root,
    refuse_widened_seed_mismatch,
    refuse_write_into_complete_run,
)
from environments.shared.result_bundle.reentry import WIDENED_FROM_RUN_ID, complete_run_writes
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname

from .result_bundle_helpers import _complete_bundle, _snapshot_files

MANIFEST = load_stage_manifest("trex")
STANCE, RECOVERY, LOCOMOTION, BEHAVIOR = (
    MANIFEST.by_id(node) for node in ("stance", "recovery", "locomotion", "behavior")
)
WALK = (STANCE, LOCOMOTION)
FRESH = "20260924_120000"


def _marker(run_dir: Path, status: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / DEFAULT_MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "run_id": run_dir.name, "status": status})
    )


def _judged(run_dir: Path, *nodes) -> None:
    for node in nodes:
        stage_dir = run_dir / stage_dirname("trex", node.reference)
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / GATE_VERDICT_FILENAME).write_text("{}")


def _recorded(run_dir: Path, node) -> None:
    record = run_dir / ANCESTORS_DIRNAME / node.id / ANCESTOR_RECORD_NAME
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}")


def _session(run_dir: Path, *, chain=WALK, target=LOCOMOTION, retrain_from=None, trunk_dir=None) -> None:
    refuse_complete_run_session(
        run_dir,
        species="trex",
        chain=chain,
        target=target,
        retrain_from=retrain_from,
        trunk_dir=trunk_dir,
        fresh_run_id=FRESH,
    )


class TestReadBundleStatus:
    def test_no_manifest_reads_none(self, tmp_path):
        assert read_bundle_status(tmp_path / "missing") is None
        assert read_bundle_status(tmp_path) is None

    @pytest.mark.parametrize("status", ["partial", "failed", "complete"])
    def test_each_status_reads_back_without_touching_the_run(self, tmp_path, status):
        _marker(tmp_path, status)
        before = _snapshot_files(tmp_path)
        assert read_bundle_status(tmp_path) == status
        assert _snapshot_files(tmp_path) == before

    @pytest.mark.parametrize(
        "data",
        [b"{not json", b"\xff\xfe\x00garbage", json.dumps({"status": "sealed"}).encode(), b'["complete"]'],
    )
    def test_an_unreadable_or_unknown_marker_refuses(self, tmp_path, data):
        """Malformed JSON and undecodable bytes included: never a bare codec error."""
        (tmp_path / DEFAULT_MANIFEST_NAME).write_bytes(data)
        with pytest.raises(ResultBundleError, match="artifact manifest"):
            read_bundle_status(tmp_path)

    def test_a_real_complete_bundle_reads_complete(self, tmp_path, stable_provenance):
        run_dir = tmp_path / "run"
        _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
        assert read_bundle_status(run_dir) == "complete"


class TestCompleteRunSession:
    def test_a_run_without_a_complete_marker_is_continued_in_place(self, tmp_path):
        _session(tmp_path / "fresh")  # nothing on disk at all
        for status in ("partial", "failed"):
            run_dir = tmp_path / status
            _marker(run_dir, status)
            _judged(run_dir, STANCE)
            _session(run_dir)  # locomotion trains into it: the writer rebuilds over this marker

    def test_a_reuse_only_session_passes(self, tmp_path):
        _marker(tmp_path, "complete")
        _judged(tmp_path, STANCE, LOCOMOTION)
        before = _snapshot_files(tmp_path)
        _session(tmp_path)
        _session(tmp_path, chain=(STANCE,), target=STANCE)
        assert complete_run_writes(tmp_path, species="trex", chain=WALK, target=LOCOMOTION, trunk_dir=None) == ()
        assert _snapshot_files(tmp_path) == before

    def test_a_new_node_is_refused_with_the_fresh_run_remedy(self, tmp_path):
        run_dir = tmp_path / "20260920_010912"
        _marker(run_dir, "complete")
        _judged(run_dir, STANCE, RECOVERY)
        before = _snapshot_files(run_dir)
        with pytest.raises(ResultBundleError) as excinfo:
            _session(run_dir)
        message = str(excinfo.value)
        assert "whose result bundle is complete" in message
        assert "would judge or train 'locomotion' here" in message and "'stance'" not in message
        assert f'RUN_ID = "{FRESH}"' in message and 'TRUNK_FROM = "20260920_010912"' in message
        assert "Nothing has been trained or written" in message
        assert _snapshot_files(run_dir) == before

    def test_retrain_from_is_refused_on_a_complete_run(self, tmp_path):
        _marker(tmp_path, "complete")
        _judged(tmp_path, STANCE, LOCOMOTION)
        with pytest.raises(ResultBundleError, match="RETRAIN_FROM = 'locomotion'.*keep RETRAIN_FROM in the fresh run"):
            _session(tmp_path, retrain_from=LOCOMOTION)

    @pytest.mark.parametrize("content", ["final pair", "periodic checkpoints only", "empty directory"])
    def test_a_directory_without_a_verdict_would_be_judged_or_resumed(self, tmp_path, content):
        _marker(tmp_path, "complete")
        _judged(tmp_path, STANCE)
        models = tmp_path / stage_dirname("trex", LOCOMOTION.reference) / "models"
        models.mkdir(parents=True)
        if content == "final pair":
            (models / "stage2_final.zip").write_bytes(b"zip")
            (models / "stage2_final_vecnorm.pkl").write_bytes(b"pkl")
        elif content == "periodic checkpoints only":
            (models / "stage2_100000_steps.zip").write_bytes(b"zip")
        assert complete_run_writes(tmp_path, species="trex", chain=WALK, target=LOCOMOTION, trunk_dir=None) == (
            "locomotion",
        )

    def test_a_verdict_outside_the_loop_directory_does_not_count(self, tmp_path):
        """The loop judges and trains only stage_dirname's directory: a legacy stage1/ verdict is refused."""
        _marker(tmp_path, "complete")
        (tmp_path / "stage1").mkdir()
        (tmp_path / "stage1" / GATE_VERDICT_FILENAME).write_text("{}")
        _judged(tmp_path, LOCOMOTION)
        assert complete_run_writes(tmp_path, species="trex", chain=WALK, target=LOCOMOTION, trunk_dir=None) == (
            "stance",
        )

    def test_an_ancestor_record_stands_in_only_for_a_non_target_node_with_a_trunk(self, tmp_path):
        _marker(tmp_path, "complete")
        _recorded(tmp_path, STANCE)
        _judged(tmp_path, LOCOMOTION)
        trunk = tmp_path.parent / "trunk"
        assert complete_run_writes(tmp_path, species="trex", chain=WALK, target=LOCOMOTION, trunk_dir=trunk) == ()
        assert complete_run_writes(tmp_path, species="trex", chain=WALK, target=LOCOMOTION, trunk_dir=None) == (
            "stance",
        )
        # The target is looked for in this run only: a record never stands in for it.
        _recorded(tmp_path, BEHAVIOR)
        chain = (STANCE, LOCOMOTION, BEHAVIOR)
        assert complete_run_writes(tmp_path, species="trex", chain=chain, target=BEHAVIOR, trunk_dir=trunk) == (
            "behavior",
        )

    def test_a_record_with_a_stage_directory_beside_it_does_not_stand_in(self, tmp_path):
        """A stage directory without a verdict is what the loop judges or trains into when the trunk does not cover
        the node, so the record beside it no longer counts as writing nothing."""
        _marker(tmp_path, "complete")
        _recorded(tmp_path, STANCE)
        _judged(tmp_path, LOCOMOTION)
        (tmp_path / stage_dirname("trex", STANCE.reference) / "models").mkdir(parents=True)
        trunk = tmp_path.parent / "trunk"
        assert complete_run_writes(tmp_path, species="trex", chain=WALK, target=LOCOMOTION, trunk_dir=trunk) == (
            "stance",
        )

    def test_an_unreadable_marker_refuses(self, tmp_path):
        (tmp_path / DEFAULT_MANIFEST_NAME).write_text("{truncated")
        with pytest.raises(ResultBundleError):
            _session(tmp_path)


class TestWriteIntoCompleteRun:
    def test_only_a_complete_run_refuses(self, tmp_path):
        refuse_write_into_complete_run(tmp_path, what="Resuming 2")
        _marker(tmp_path, "partial")
        refuse_write_into_complete_run(tmp_path, what="Resuming 2")
        _marker(tmp_path, "complete")
        before = _snapshot_files(tmp_path)
        with pytest.raises(ResultBundleError) as excinfo:
            refuse_write_into_complete_run(tmp_path, what="Resuming 2", fresh_run_id=FRESH)
        message = str(excinfo.value)
        assert "Resuming 2 would write into run" in message and "whose result bundle is complete" in message
        assert f'RUN_ID = "{FRESH}"' in message and f'TRUNK_FROM = "{tmp_path.name}"' in message
        assert _snapshot_files(tmp_path) == before


def _widened_root(run_dir: Path, *, seed=44, judged=False) -> Path:
    stage_dir = run_dir / stage_dirname("trex", STANCE.reference)
    stage_dir.mkdir(parents=True, exist_ok=True)
    run = {"n_envs": 4, "widened_from_run_id": "20260815_205206"}
    if seed is not None:
        run["seed"] = seed
    (stage_dir / "stage_config.json").write_text(json.dumps({"run": run}))
    if judged:
        (stage_dir / GATE_VERDICT_FILENAME).write_text("{}")
    return stage_dir


class TestWidenedRoot:
    def test_the_seed_must_be_the_parents(self, tmp_path):
        refuse_widened_seed_mismatch(tmp_path / "not-yet-minted", seed=42)
        _widened_root(tmp_path)
        before = _snapshot_files(tmp_path)
        with pytest.raises(ResultBundleError, match=r"SEED = 42.*'20260815_205206'.*seed 44.*set SEED = 44"):
            refuse_widened_seed_mismatch(tmp_path, seed=42)
        refuse_widened_seed_mismatch(tmp_path, seed=44)
        assert _snapshot_files(tmp_path) == before

    def test_only_a_widened_run_block_is_checked(self, tmp_path):
        trained = tmp_path / stage_dirname("trex", LOCOMOTION.reference)
        trained.mkdir()
        (trained / "stage_config.json").write_text(json.dumps({"run": {"seed": 7}}))
        (tmp_path / "02_recovery").mkdir()
        (tmp_path / "02_recovery" / "stage_config.json").write_text("{unreadable")
        refuse_widened_seed_mismatch(tmp_path, seed=42)

    def test_the_key_is_the_one_the_widen_tool_writes(self):
        assert WIDENED_FROM_RUN_ID in WIDEN_LINEAGE_KEYS, "config.WIDEN_LINEAGE_KEYS renamed: the guards go blind"

    def test_a_provenance_minted_under_another_seed_names_the_remedy_that_works(self, tmp_path):
        """The storage cell may have minted the run before the root was widened into it: no SEED fixes that run."""
        _widened_root(tmp_path)
        (tmp_path / DEFAULT_PROVENANCE_NAME).write_text(json.dumps({"training_seed": 42}))
        before = _snapshot_files(tmp_path)
        for seed in (42, 44):
            with pytest.raises(ResultBundleError) as excinfo:
                refuse_widened_seed_mismatch(tmp_path, seed=seed)
            message = str(excinfo.value)
            assert "records training_seed 42" in message and "keeps that run's seed 44" in message
            assert "no SEED fixes this run" in message
            assert "--to-stage-dir <LOG_BASE>/<species>/<algo>/<new run id>/01_stance" in message
        assert _snapshot_files(tmp_path) == before
        # A provenance minted under the parent's seed is the normal re-entry: only SEED is checked.
        (tmp_path / DEFAULT_PROVENANCE_NAME).write_text(json.dumps({"training_seed": 44}))
        refuse_widened_seed_mismatch(tmp_path, seed=44)
        with pytest.raises(ResultBundleError, match="SEED = 42.*set SEED = 44"):
            refuse_widened_seed_mismatch(tmp_path, seed=42)

    def test_a_widened_root_without_a_seed_refuses(self, tmp_path):
        _widened_root(tmp_path, seed=None)
        with pytest.raises(ResultBundleError, match="without an integer run seed"):
            refuse_widened_seed_mismatch(tmp_path, seed=42)

    def test_a_trunk_is_refused_until_the_widened_root_holds_a_verdict(self, tmp_path):
        trunk = tmp_path.parent / "trunk"
        kwargs = dict(species="trex", chain=WALK, target=LOCOMOTION, retrain_from=None)
        refuse_trunk_over_unjudged_widened_root(tmp_path, trunk_dir=trunk, **kwargs)  # nothing widened
        stage_dir = _widened_root(tmp_path)
        refuse_trunk_over_unjudged_widened_root(tmp_path, trunk_dir=None, **kwargs)
        with pytest.raises(ResultBundleError, match="never judge the widened root") as excinfo:
            refuse_trunk_over_unjudged_widened_root(tmp_path, trunk_dir=trunk, **kwargs)
        # The storage cell has minted the run's provenance by the time this refuses: say so, and what may change.
        message = str(excinfo.value)
        assert "Nothing has been trained or judged" in message and "Nothing has been trained or written" not in message
        assert "fixes SEED, N_ENVS and the plant" in message and 'set TRUNK_FROM = ""' in message
        # The root is the target (never reused across runs) or RETRAIN_FROM names it: no trunk stands in.
        refuse_trunk_over_unjudged_widened_root(
            tmp_path, species="trex", chain=(STANCE,), target=STANCE, retrain_from=None, trunk_dir=trunk
        )
        refuse_trunk_over_unjudged_widened_root(
            tmp_path, species="trex", chain=WALK, target=LOCOMOTION, retrain_from=STANCE, trunk_dir=trunk
        )
        (stage_dir / GATE_VERDICT_FILENAME).write_text("{}")
        refuse_trunk_over_unjudged_widened_root(tmp_path, trunk_dir=trunk, **kwargs)
