"""``environments.shared.result_bundle.reentry``: what a notebook session re-entering a run is refused.

Consolidation PR-14a. A run whose ``artifact_manifest.json`` records ``complete`` is immutable
(docs/RESULT_BUNDLES.md), so a session that would judge or train a node into it is refused before anything
is trained or written, and a reuse-only session passes; a root widened on the command line keeps its parent's
seed (decision D-C14) and is judged before any trunk may stand in for it (decision D-C13; the trunk refusal the
resolve cell made for it stays exported and tested here, although since cleanup ROW-4/6 the chain loop judges
such a root first and the notebook no longer calls it). Cleanup ROW-4/6 adds the resume check against the trunk
``trunk_run.json`` records (decision 4 (a)) and the two helpers the chain loop calls to judge a node this run
trained but never judged before any trunk (decision 6 (b)): its loop directory, and rule 4 of the reuse rule
applied to it. Every check reads the run directory only, so the cases below also assert the directory
is byte-identical afterwards. No SB3, torch or IPython: these run in the shared test matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared import config
from environments.shared.config import WIDEN_LINEAGE_KEYS, save_stage_config
from environments.shared.result_bundle import (
    ANCESTOR_RECORD_NAME,
    ANCESTORS_DIRNAME,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_PROVENANCE_NAME,
    GATE_VERDICT_FILENAME,
    TRUNK_RECORD_NAME,
    TRUNK_RECORD_SCHEMA,
    ResultBundleError,
    read_bundle_status,
    refuse_complete_run_session,
    refuse_judging_off_the_resolved_parent,
    refuse_trunk_other_than_recorded,
    refuse_trunk_over_unjudged_widened_root,
    refuse_widened_seed_mismatch,
    refuse_write_into_complete_run,
    sha256_file,
    unjudged_stage_dir,
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


# ---------------------------------------------------------------------------
# Cleanup ROW-4/6, decision 4 (a): a resume under the trunk trunk_run.json records.
# ---------------------------------------------------------------------------


def _trunk_record(run_dir: Path, trunk_from: str) -> None:
    (run_dir / TRUNK_RECORD_NAME).write_text(json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": trunk_from}))


class TestTrunkOtherThanRecorded:
    """The RESUME cell's check: ``what`` is its ``f"Resuming {stage!r}"`` and ``resumed`` its node."""

    @staticmethod
    def _run(tmp_path: Path) -> "tuple[Path, Path]":
        log_dir = tmp_path / "logs" / "trex" / "ppo"
        run_dir = log_dir / "20261003_000000"
        run_dir.mkdir(parents=True)
        return run_dir, log_dir

    @staticmethod
    def _check(run_dir: Path, log_dir: Path, trunk_dir, *, resumed=LOCOMOTION) -> None:
        refuse_trunk_other_than_recorded(
            run_dir, trunk_dir=trunk_dir, log_dir=log_dir, what="Resuming 2", resumed=resumed
        )

    def test_a_run_without_ancestor_records_is_consistent_with_any_trunk(self, tmp_path):
        run_dir, log_dir = self._run(tmp_path)
        _judged(run_dir, STANCE)
        for recorded in (None, "A", "{"):
            if recorded == "{":
                (run_dir / TRUNK_RECORD_NAME).write_text("{")
            elif recorded is not None:
                _trunk_record(run_dir, recorded)
            before = _snapshot_files(run_dir)
            for trunk_dir in (None, log_dir / "A", log_dir / "B", tmp_path / "elsewhere"):
                self._check(run_dir, log_dir, trunk_dir)
            assert _snapshot_files(run_dir) == before

    def test_a_run_opened_before_row46_has_no_record_and_passes(self, tmp_path):
        """Records but no trunk_run.json: the resolve cell wrote none (no guess); the recipe's manual route."""
        run_dir, log_dir = self._run(tmp_path)
        _recorded(run_dir, STANCE)
        before = _snapshot_files(run_dir)
        for trunk_dir in (None, log_dir / "A", tmp_path / "elsewhere"):
            self._check(run_dir, log_dir, trunk_dir)
        assert _snapshot_files(run_dir) == before

    def test_the_recorded_trunk_passes(self, tmp_path):
        """By id (however the session spelled the directory), by absolute directory, and as no trunk at all."""
        run_dir, log_dir = self._run(tmp_path)
        _recorded(run_dir, STANCE)
        elsewhere = tmp_path / "elsewhere" / "A"
        for recorded, sessions in (
            ("A", (log_dir / "A", str(log_dir / "A"), log_dir / "x" / ".." / "A")),
            (str(elsewhere.resolve()), (elsewhere, str(elsewhere))),
            ("", (None,)),
        ):
            _trunk_record(run_dir, recorded)
            before = _snapshot_files(run_dir)
            for trunk_dir in sessions:
                self._check(run_dir, log_dir, trunk_dir)
            assert _snapshot_files(run_dir) == before

    @pytest.mark.parametrize("session", ["", "B", "elsewhere"])
    def test_another_trunk_is_refused_before_anything_is_trained(self, tmp_path, session):
        run_dir, log_dir = self._run(tmp_path)
        _recorded(run_dir, STANCE)
        _recorded(run_dir, RECOVERY)
        _trunk_record(run_dir, "A")
        trunk_dir = {"": None, "B": log_dir / "B", "elsewhere": tmp_path / "elsewhere" / "A"}[session]
        before = _snapshot_files(run_dir)
        with pytest.raises(ResultBundleError) as excinfo:
            self._check(run_dir, log_dir, trunk_dir)
        message = str(excinfo.value)
        shown = str((tmp_path / "elsewhere" / "A").resolve()) if session == "elsewhere" else session
        assert f'Resuming 2 under TRUNK_FROM = "{shown}" is refused before anything is trained' in message
        assert "this run's ancestor records ('recovery', 'stance') were reused through TRUNK_FROM = \"A\"" in message
        assert 'Set TRUNK_FROM = "A" in the configuration cell and re-run sections 2-3, then this cell.' in message
        assert _snapshot_files(run_dir) == before

    def test_an_absolute_record_is_compared_as_the_directory_it_names(self, tmp_path):
        """A value the writer would have spelt otherwise (a hand edit, or a run copied to another ``LOG_BASE``) passes
        a session on the directory it names, and the TRUNK_FROM the refusal names for another session passes."""
        run_dir, log_dir = self._run(tmp_path)
        _recorded(run_dir, STANCE)
        in_tree = str((log_dir / "A").resolve())  # the writer records "A"
        out_of_tree = str((tmp_path / "elsewhere" / "B").resolve()) + "/"  # the writer records it without the slash
        for recorded, sessions in ((in_tree, (log_dir / "A", in_tree)), (out_of_tree, (tmp_path / "elsewhere" / "B",))):
            _trunk_record(run_dir, recorded)
            before = _snapshot_files(run_dir)
            for trunk_dir in sessions:
                self._check(run_dir, log_dir, trunk_dir)
            with pytest.raises(ResultBundleError, match=f'Set TRUNK_FROM = "{recorded}" in the configuration cell'):
                self._check(run_dir, log_dir, log_dir / "C")
            self._check(run_dir, log_dir, Path(recorded))
            assert _snapshot_files(run_dir) == before

    def test_an_unreadable_record_in_a_run_with_records_is_refused(self, tmp_path):
        run_dir, log_dir = self._run(tmp_path)
        _recorded(run_dir, STANCE)
        (run_dir / TRUNK_RECORD_NAME).write_text(json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": "a/b"}))
        before = _snapshot_files(run_dir)
        with pytest.raises(ResultBundleError) as excinfo:
            self._check(run_dir, log_dir, None)
        message = str(excinfo.value)
        assert message.startswith("Resuming 2 is refused before anything is trained: invalid trunk record")
        assert "Remove that file and pin TRUNK_FROM by the recipe's manual route (section 5, step 1)" in message
        assert _snapshot_files(run_dir) == before

    def test_a_node_the_run_holds_as_an_ancestor_record_is_never_resumed(self, tmp_path):
        """The record stays the node in the run (the chain loop takes it while a trunk certifies the node, and a bundle
        refuses a node both reused and trained here), so the resume would never be judged into the run's bundle."""
        run_dir, log_dir = self._run(tmp_path)
        _recorded(run_dir, STANCE)
        _recorded(run_dir, LOCOMOTION)
        before = _snapshot_files(run_dir)
        for recorded in (None, "A"):
            if recorded is not None:
                _trunk_record(run_dir, recorded)
                before = _snapshot_files(run_dir)
            with pytest.raises(ResultBundleError) as excinfo:
                self._check(run_dir, log_dir, log_dir / "A")
            message = str(excinfo.value)
            assert (
                "Resuming 2 is refused before anything is trained: this run holds 'locomotion' as a reused" in message
            )
            assert "that record stays the node in this run" in message and "would never be judged" in message
            assert "Train 'locomotion' again in a fresh RUN_ID, with TRUNK_FROM naming this run" in message
            assert _snapshot_files(run_dir) == before
        # Another node of the same run, or no node (a session-level check), is not that refusal.
        self._check(run_dir, log_dir, log_dir / "A", resumed=BEHAVIOR)
        self._check(run_dir, log_dir, log_dir / "A", resumed=None)

    def test_a_complete_run_is_left_to_the_complete_run_guard(self, tmp_path):
        run_dir, log_dir = self._run(tmp_path)
        _marker(run_dir, "complete")
        _recorded(run_dir, LOCOMOTION)
        _trunk_record(run_dir, "A")
        before = _snapshot_files(run_dir)
        self._check(run_dir, log_dir, None)
        assert _snapshot_files(run_dir) == before


# ---------------------------------------------------------------------------
# Cleanup ROW-4/6, decision 6 (b): a node this run trained but never judged.
# ---------------------------------------------------------------------------

SHA_A = "sha256:" + "a" * 64
SHA_C = "sha256:" + "c" * 64


def _checkpoints(run_dir: Path, *names: str, dirname: "str | None" = None) -> Path:
    stage_dir = run_dir / (dirname or stage_dirname("trex", LOCOMOTION.reference))
    models = stage_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name in names:
        (models / name).write_bytes(name.encode())
    return stage_dir


class TestUnjudgedStageDir:
    @pytest.mark.parametrize(
        "names",
        [
            ("stage2_final.zip", "stage2_final_vecnorm.pkl"),
            ("stage2_100000_steps.zip", "stage2_vecnormalize_100000_steps.pkl"),
            ("stage2_final.zip",),
        ],
        ids=["intact-final-pair", "periodic-checkpoints-only", "final-zip-without-its-sidecar"],
    )
    def test_any_checkpoint_without_a_verdict_is_unjudged(self, tmp_path, names):
        """An intact pair goes to JUDGE, anything else to the interrupted-node refusal: either way not the trunk."""
        stage_dir = _checkpoints(tmp_path, *names)
        before = _snapshot_files(tmp_path)
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) == stage_dir
        assert _snapshot_files(tmp_path) == before

    def test_no_loop_directory_or_no_checkpoint_is_not(self, tmp_path):
        assert unjudged_stage_dir(tmp_path / "missing", species="trex", entry=LOCOMOTION) is None
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is None
        _checkpoints(tmp_path, "stage2_final_vecnorm.pkl")  # models/ without a zip
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is None

    def test_a_checkpoint_outside_the_loop_directory_is_not(self, tmp_path):
        """The loop judges and trains stage_dirname's directory only, as complete_run_writes reads it, and counts
        the zips directly in its ``models/``, as the loop's interrupted-node branch does (``model_dir.glob``)."""
        _checkpoints(tmp_path, "stage2_final.zip", "stage2_final_vecnorm.pkl", dirname="stage2")
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is None
        stage_dir = _checkpoints(tmp_path, "stage2_final_vecnorm.pkl")
        (stage_dir / "stage2_final.zip").write_bytes(b"zip")
        (stage_dir / "models" / "old").mkdir()
        (stage_dir / "models" / "old" / "stage2_final.zip").write_bytes(b"zip")
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is None

    @pytest.mark.parametrize(
        "verdict",
        ["{}", json.dumps({"passed": False, "failures": ["x"]}), "{truncated"],
        ids=["empty", "failed", "bad"],
    )
    def test_any_verdict_file_is_not(self, tmp_path, verdict):
        """Tested by presence: a malformed verdict still reaches the reuse rule and the loop's verdict branch."""
        stage_dir = _checkpoints(tmp_path, "stage2_final.zip", "stage2_final_vecnorm.pkl")
        (stage_dir / GATE_VERDICT_FILENAME).write_text(verdict)
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is None

    def test_a_node_the_run_holds_as_an_ancestor_record_is_not(self, tmp_path):
        """A run that took the trunk's copy over its own before ROW-4/6: the record is the node here, and judging the
        directory too would make the bundle write refuse ("recorded both as trained in this run and as reused
        ancestors")."""
        _checkpoints(tmp_path, "stage2_final.zip", "stage2_final_vecnorm.pkl")
        _recorded(tmp_path, LOCOMOTION)
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is None
        _recorded(tmp_path, STANCE)  # another node's record is not this one's
        (tmp_path / ANCESTORS_DIRNAME / LOCOMOTION.id / ANCESTOR_RECORD_NAME).unlink()
        assert unjudged_stage_dir(tmp_path, species="trex", entry=LOCOMOTION) is not None

    def test_a_root_widened_beside_its_ancestor_record_is_refused(self, tmp_path):
        """``widen_checkpoint`` into a run that already took the root from a trunk: the record would stand in for the
        widened root, which would never be judged (what D-C13 refused before ROW-4/6), and judging it beside the
        record would make the bundle refuse it as both trained and reused. Refused, naming a new run id."""
        stage_dir = _widened_root(tmp_path, seed=42)
        for name in ("stage1_final.zip", "stage1_final_vecnorm.pkl"):
            (stage_dir / "models").mkdir(exist_ok=True)
            (stage_dir / "models" / name).write_bytes(name.encode())
        assert unjudged_stage_dir(tmp_path, species="trex", entry=STANCE) == stage_dir
        _recorded(tmp_path, STANCE)
        before = _snapshot_files(tmp_path)
        with pytest.raises(
            ResultBundleError,
            match=r"widened from run '20260815_205206' .*already holds 'stance' as a reused ancestor.*new run id",
        ):
            unjudged_stage_dir(tmp_path, species="trex", entry=STANCE)
        assert _snapshot_files(tmp_path) == before


def _run_block(stage_dir: Path, run: dict) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage_config.json").write_text(json.dumps({"run": run}))


class TestJudgingOffTheResolvedParent:
    """Rule 4 of the reuse rule (ancestors._check_chain) on the directory the loop would judge ahead of the trunk."""

    def _passes(self, stage_dir: Path, entry, parent_model_sha256) -> None:
        before = _snapshot_files(stage_dir.parent)
        refuse_judging_off_the_resolved_parent(stage_dir, entry=entry, parent_model_sha256=parent_model_sha256)
        assert _snapshot_files(stage_dir.parent) == before

    def _refused(self, stage_dir: Path, entry, parent_model_sha256) -> str:
        before = _snapshot_files(stage_dir.parent)
        with pytest.raises(ResultBundleError) as excinfo:
            refuse_judging_off_the_resolved_parent(stage_dir, entry=entry, parent_model_sha256=parent_model_sha256)
        assert _snapshot_files(stage_dir.parent) == before
        message = str(excinfo.value)
        assert f"{entry.id!r} in {stage_dir} was trained in this run but never judged" in message
        assert "does not chain onto the parent this session resolved" in message
        assert f"Train {entry.id!r} in a fresh RUN_ID" in message
        assert "gate_verdict.json" not in message, "re-judging is no remedy here: the parent itself is in question"
        return message

    def test_a_root_without_a_parent_passes(self, tmp_path):
        stance = tmp_path / stage_dirname("trex", STANCE.reference)
        stance.mkdir()
        self._passes(stance, STANCE, None)  # no stage_config.json at all
        _run_block(stance, {"seed": 42, "n_envs": 4})
        self._passes(stance, STANCE, None)
        _run_block(stance, {"load_path": "x.zip", "load_mode": "resume_same_stage", "parent_checkpoint_sha256": SHA_A})
        self._passes(stance, STANCE, None)

    def test_a_widened_root_passes(self, tmp_path):
        """A widened root records its parent under WIDEN_LINEAGE_KEYS, never as a load lineage."""
        stance = tmp_path / stage_dirname("trex", STANCE.reference)
        _run_block(stance, {"seed": 44, "n_envs": 4, **{key: "x" for key in WIDEN_LINEAGE_KEYS}})
        self._passes(stance, STANCE, None)

    def test_a_root_that_entered_from_a_parent_is_refused(self, tmp_path):
        stance = tmp_path / stage_dirname("trex", STANCE.reference)
        _run_block(
            stance, {"load_path": "x.zip", "load_mode": "initialize_next_stage", "parent_checkpoint_sha256": SHA_A}
        )
        assert "is a root node in the current manifest" in self._refused(stance, STANCE, None)

    def test_a_non_root_on_the_resolved_parent_passes_and_on_another_is_refused(self, tmp_path):
        locomotion = tmp_path / stage_dirname("trex", LOCOMOTION.reference)
        _run_block(
            locomotion, {"load_path": "x.zip", "load_mode": "initialize_next_stage", "parent_checkpoint_sha256": SHA_A}
        )
        self._passes(locomotion, LOCOMOTION, SHA_A)
        message = self._refused(locomotion, LOCOMOTION, SHA_C)
        assert f"descends from 'stance' checkpoint {SHA_A}, not the {SHA_C} resolved" in message

    def test_a_non_root_without_a_recorded_edge_is_refused(self, tmp_path):
        locomotion = tmp_path / stage_dirname("trex", LOCOMOTION.reference)
        locomotion.mkdir()
        assert "records no initialize_next_stage load" in self._refused(locomotion, LOCOMOTION, SHA_A)
        _run_block(
            locomotion, {"load_path": "x.zip", "load_mode": "resume_same_stage", "parent_checkpoint_sha256": SHA_A}
        )
        assert "records no initialize_next_stage load" in self._refused(locomotion, LOCOMOTION, SHA_A)

    def test_a_resumed_non_root_keeps_its_edge(self, tmp_path, monkeypatch):
        """The real save_stage_config: a same-stage resume keeps the initialize_next_stage edge (and records the
        periodic checkpoint under RESUME_LINEAGE_KEYS), so a resumed node is judged on the parent it entered from."""
        monkeypatch.setattr(config, "_detect_gpu_info", lambda: {})
        parent = tmp_path / "stance_final.zip"
        parent.write_bytes(b"parent")
        periodic = tmp_path / "stage2_100000_steps.zip"
        periodic.write_bytes(b"periodic")
        locomotion = tmp_path / "run" / stage_dirname("trex", LOCOMOTION.reference)
        stage_config = {"name": "locomotion", "curriculum_kwargs": {}, "ppo_kwargs": {}}
        save_stage_config(locomotion, 2, stage_config, "ppo", load_path=str(parent), load_mode="initialize_next_stage")
        save_stage_config(locomotion, 2, stage_config, "ppo", load_path=str(periodic), load_mode="resume_same_stage")
        run = json.loads((locomotion / "stage_config.json").read_text(encoding="utf-8"))["run"]
        assert run["load_mode"] == "initialize_next_stage" and run["resume_load_path"] == str(periodic)
        self._passes(locomotion, LOCOMOTION, sha256_file(parent))
        self._refused(locomotion, LOCOMOTION, sha256_file(periodic))
