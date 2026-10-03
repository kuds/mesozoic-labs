"""``environments.shared.result_bundle.trunk_record``: the run's ``trunk_run.json`` (cleanup ROW-4/6, decision 4 (a)).

The file names the trunk run a notebook session resolved for a run, as the ``TRUNK_FROM`` value that reproduces it.
Its writer never writes into a ``complete`` bundle, rewrites it freely while the run holds no ``ancestors/`` record,
and never once the run holds one; its reader refuses a file it cannot read back exactly.  The file is a plain run-tree
artifact: the next bundle write declares and hashes it.  No SB3, torch or IPython: these run in the shared test matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared.reporting import save_result_bundle
from environments.shared.result_bundle import (
    ANCESTOR_RECORD_NAME,
    ANCESTORS_DIRNAME,
    DEFAULT_MANIFEST_NAME,
    TRUNK_RECORD_NAME,
    TRUNK_RECORD_SCHEMA,
    ResultBundleError,
    build_artifact_manifest,
    initialize_result_bundle,
    read_trunk_record,
    record_trunk_run,
    refuse_trunk_other_than_recorded,
    trunk_from_value,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from environments.shared.result_bundle.trunk_record import ancestor_record_ids

from .result_bundle_helpers import _complete_bundle_inputs, _plant_identity, _snapshot_files

RUN_ID = "20261003_000000"


def _log_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "velociraptor" / "ppo"


def _run_dir(tmp_path: Path) -> Path:
    run_dir = _log_dir(tmp_path) / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_record(run_dir: Path, trunk_from: str) -> None:
    (run_dir / TRUNK_RECORD_NAME).write_text(json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": trunk_from}))


def _ancestor_record(run_dir: Path, node: str = "stance") -> None:
    record = run_dir / ANCESTORS_DIRNAME / node / ANCESTOR_RECORD_NAME
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}")


def _marker(run_dir: Path, status: str) -> None:
    (run_dir / DEFAULT_MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "run_id": run_dir.name, "status": status})
    )


def _expected_bytes(trunk_from: str) -> bytes:
    return (
        json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": trunk_from}, indent=2, sort_keys=True).encode() + b"\n"
    )


class TestTrunkFromValue:
    def test_no_trunk_is_the_empty_knob(self, tmp_path):
        assert trunk_from_value(None, log_dir=_log_dir(tmp_path)) == ""

    def test_a_trunk_under_the_log_directory_is_its_run_id(self, tmp_path, monkeypatch):
        """Where the storage cell resolves a TRUNK_FROM id; the trunk need not exist (nothing is read)."""
        log_dir = _log_dir(tmp_path)
        assert not (log_dir / "20260921_000000").exists()
        assert trunk_from_value(log_dir / "20260921_000000", log_dir=log_dir) == "20260921_000000"
        assert trunk_from_value(str(log_dir / "20260921_000000"), log_dir=str(log_dir)) == "20260921_000000"
        # Spelled relative to the working directory on either side, or through a "..", it is still that run.
        monkeypatch.chdir(tmp_path)
        assert trunk_from_value(Path("logs/velociraptor/ppo/A"), log_dir=log_dir) == "A"
        assert trunk_from_value(log_dir / "A", log_dir=Path("logs/velociraptor/ppo")) == "A"
        assert trunk_from_value(log_dir / ".." / "ppo" / "A", log_dir=log_dir) == "A"
        # A run linked into the log directory is its link's name, the id the storage cell resolves to that link.
        (tmp_path / "elsewhere" / "B").mkdir(parents=True)
        log_dir.mkdir(parents=True)
        (log_dir / "L").symlink_to(tmp_path / "elsewhere" / "B", target_is_directory=True)
        assert trunk_from_value(log_dir / "L", log_dir=log_dir) == "L"

    def test_a_trunk_anywhere_else_is_its_absolute_directory(self, tmp_path, monkeypatch):
        """TRUNK_FROM's absolute-path form: typed back as an id it would resolve under the log directory instead."""
        log_dir = _log_dir(tmp_path)
        elsewhere = tmp_path / "elsewhere" / "20260921_000000"
        assert trunk_from_value(elsewhere, log_dir=log_dir) == str(elsewhere.resolve())
        nested = log_dir / "group" / "20260921_000000"
        assert trunk_from_value(nested, log_dir=log_dir) == str(nested.resolve())
        monkeypatch.chdir(tmp_path)
        assert trunk_from_value(Path("elsewhere/B"), log_dir=log_dir) == str((tmp_path / "elsewhere" / "B").resolve())

    def test_a_name_that_is_no_run_id_is_recorded_as_its_absolute_directory(self, tmp_path):
        log_dir = _log_dir(tmp_path)
        assert trunk_from_value(log_dir / "..", log_dir=log_dir) == str(log_dir.parent.resolve())
        root = Path(log_dir.anchor)
        assert trunk_from_value(root, log_dir=root) == str(root.resolve())

    def test_a_run_named_auto_is_recorded_as_its_absolute_directory(self, tmp_path):
        """``TRUNK_FROM = "auto"`` selects a trunk (D-A25) instead of naming the run of that name, so the only value
        that names it is its absolute directory."""
        log_dir = _log_dir(tmp_path)
        assert trunk_from_value(log_dir / "auto", log_dir=log_dir) == str((log_dir / "auto").resolve())

    @pytest.mark.parametrize("spelling", ["dotdot", "symlink"])
    def test_a_spelling_that_resolves_to_an_in_tree_run_is_its_run_id(self, tmp_path, spelling):
        """A ``..`` spelling of an in-tree run, or a link to one from outside the log directory: its run id, so every
        spelling of one trunk is one value. A session on that trunk then writes nothing, and once the run holds an
        ancestor record it is neither warned nor refused a resume."""
        log_dir = _log_dir(tmp_path)
        (log_dir / "A" / "x").mkdir(parents=True)
        if spelling == "dotdot":
            trunk_dir = log_dir / "A" / "x" / ".."
        else:
            trunk_dir = tmp_path / "elsewhere" / "link"
            trunk_dir.parent.mkdir(parents=True)
            trunk_dir.symlink_to(log_dir / "A", target_is_directory=True)
        assert trunk_from_value(trunk_dir, log_dir=log_dir) == "A"
        run_dir = _run_dir(tmp_path)
        path = run_dir / TRUNK_RECORD_NAME
        assert record_trunk_run(run_dir, trunk_dir=trunk_dir, log_dir=log_dir) == (
            f'Trunk record:  wrote TRUNK_FROM = "A" to {path}.'
        )
        first, stamp = path.read_bytes(), path.stat().st_mtime_ns
        same = f'Trunk record:  {path} names TRUNK_FROM = "A", this session\'s trunk.'
        assert record_trunk_run(run_dir, trunk_dir=trunk_dir, log_dir=log_dir) == same
        assert path.read_bytes() == first and path.stat().st_mtime_ns == stamp
        _ancestor_record(run_dir)
        before = _snapshot_files(run_dir)
        assert record_trunk_run(run_dir, trunk_dir=trunk_dir, log_dir=log_dir) == same
        refuse_trunk_other_than_recorded(run_dir, trunk_dir=trunk_dir, log_dir=log_dir, what="Resuming 2")
        assert _snapshot_files(run_dir) == before

    @pytest.mark.parametrize(
        "trunk",
        [
            None,
            "logs/velociraptor/ppo/A",
            "elsewhere/B",
            "logs/velociraptor/ppo/..",
            "logs/velociraptor/ppo/x y",
            "logs/velociraptor/ppo/A/x/..",
            "logs/velociraptor/ppo/auto",
        ],
    )
    def test_every_value_it_writes_reads_back(self, tmp_path, trunk):
        run_dir = _run_dir(tmp_path)
        trunk_dir = None if trunk is None else tmp_path / trunk
        record_trunk_run(run_dir, trunk_dir=trunk_dir, log_dir=_log_dir(tmp_path))
        assert read_trunk_record(run_dir) == trunk_from_value(trunk_dir, log_dir=_log_dir(tmp_path))


class TestReadTrunkRecord:
    def test_an_absent_file_reads_none(self, tmp_path):
        assert read_trunk_record(tmp_path) is None
        assert read_trunk_record(tmp_path / "missing") is None

    @pytest.mark.parametrize("value", ["", "20260921_000000", "/content/drive/MyDrive/mesozoic-labs/other/run"])
    def test_a_well_formed_record_reads_its_value(self, tmp_path, value):
        _write_record(tmp_path, value)
        before = _snapshot_files(tmp_path)
        assert read_trunk_record(tmp_path) == value
        assert _snapshot_files(tmp_path) == before

    @pytest.mark.parametrize(
        "data",
        [
            b"{",
            b"[]",
            json.dumps({"schema": TRUNK_RECORD_SCHEMA}).encode(),
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": "", "recorded_at": "2026-10-03"}).encode(),
            json.dumps({"schema": "mesozoic.trunk-run/v2", "trunk_from": ""}).encode(),
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": None}).encode(),
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": "a/b"}).encode(),
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": "."}).encode(),
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": ".."}).encode(),
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": "auto"}).encode(),
            b"\xff\xfe\x00garbage",
        ],
        ids=[
            "truncated",
            "not-an-object",
            "missing-key",
            "extra-key",
            "wrong-schema",
            "non-string",
            "a-path-not-an-id",
            "dot",
            "dot-dot",
            "auto",
            "not-utf8",
        ],
    )
    def test_a_record_it_cannot_read_back_exactly_refuses(self, tmp_path, data):
        """Never a bare codec or key error: the trunk the file recorded is unknown."""
        (tmp_path / TRUNK_RECORD_NAME).write_bytes(data)
        with pytest.raises(ResultBundleError, match="trunk record"):
            read_trunk_record(tmp_path)

    def test_a_directory_in_its_place_refuses(self, tmp_path):
        (tmp_path / TRUNK_RECORD_NAME).mkdir()
        with pytest.raises(ResultBundleError, match="cannot read the trunk record"):
            read_trunk_record(tmp_path)


def test_ancestor_record_ids_counts_only_record_files(tmp_path):
    assert ancestor_record_ids(tmp_path) == ()
    _ancestor_record(tmp_path, "stance")
    _ancestor_record(tmp_path, "locomotion")
    (tmp_path / ANCESTORS_DIRNAME / "recovery").mkdir()
    (tmp_path / ANCESTORS_DIRNAME / "behavior" / ANCESTOR_RECORD_NAME).mkdir(parents=True)
    assert ancestor_record_ids(tmp_path) == ("locomotion", "stance")


#: The writer's whole table: this session's trunk "T" against a recorded one that is absent, "T", another ("S") or
#: unreadable, each with and without an ancestors/ record, under each bundle status. Nothing is written when the
#: file already names "T" or the bundle is complete; "T" is written when the run holds no record; a run that holds
#: one keeps its file (or its lack of one) and is told why.
_STATUSES = (None, "partial", "failed", "complete")
_RECORDED = ("absent", "same", "other", "unreadable")


def _expected_row(status, recorded, records):
    """(the trunk_from the file holds afterwards, or None for byte-identical; the message key)."""
    if recorded == "same":
        return None, 'names TRUNK_FROM = "T", this session\'s trunk.'
    if status == "complete":
        return None, "nothing is written into a complete bundle."
    if not records:
        key = {
            "absent": 'Trunk record:  wrote TRUNK_FROM = "T" to ',
            "other": 'Trunk record:  TRUNK_FROM = "S" -> "T" in ',
            "unreadable": "Trunk record:  replaced the unreadable ",
        }[recorded]
        return "T", key
    key = {
        "absent": "Trunk record:  none; this run holds ancestor records ('stance') but no record of the trunk",
        "other": "WARNING: this run's ancestor records ('stance') were reused through TRUNK_FROM = \"S\"",
        "unreadable": "WARNING: invalid trunk record ",
    }[recorded]
    return None, key


@pytest.mark.parametrize("records", [False, True], ids=["no-records", "records"])
@pytest.mark.parametrize("recorded", _RECORDED)
@pytest.mark.parametrize("status", _STATUSES, ids=["no-manifest", "partial", "failed", "complete"])
def test_the_writer_table(tmp_path, status, recorded, records):
    log_dir = _log_dir(tmp_path)
    run_dir = _run_dir(tmp_path)
    if status is not None:
        _marker(run_dir, status)
    if records:
        _ancestor_record(run_dir)
    if recorded in ("same", "other"):
        (run_dir / TRUNK_RECORD_NAME).write_bytes(_expected_bytes("T" if recorded == "same" else "S"))
    elif recorded == "unreadable":
        (run_dir / TRUNK_RECORD_NAME).write_text(json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": "a/b"}))
    before = _snapshot_files(run_dir)

    message = record_trunk_run(run_dir, trunk_dir=log_dir / "T", log_dir=log_dir)

    written, key = _expected_row(status, recorded, records)
    assert key in message
    after = _snapshot_files(run_dir)
    if written is None:
        assert after == before
    else:
        assert after == {**before, Path(TRUNK_RECORD_NAME): _expected_bytes(written)}
    assert not list(run_dir.glob(".*.tmp"))


def test_the_warnings_name_the_trunk_the_records_came_through_and_the_resume_refusal(tmp_path):
    log_dir = _log_dir(tmp_path)
    run_dir = _run_dir(tmp_path)
    _ancestor_record(run_dir, "stance")
    _ancestor_record(run_dir, "locomotion")
    _write_record(run_dir, "A")
    message = record_trunk_run(run_dir, trunk_dir=None, log_dir=log_dir)
    assert message.startswith("WARNING: this run's ancestor records ('locomotion', 'stance') were reused through")
    assert 'this session resolved TRUNK_FROM = ""' in message
    assert 'Set TRUNK_FROM = "A" in the configuration cell and re-run sections 2-3.' in message
    (run_dir / TRUNK_RECORD_NAME).write_text("{")
    message = record_trunk_run(run_dir, trunk_dir=None, log_dir=log_dir)
    assert message.startswith("WARNING: cannot read the trunk record")
    assert "the RESUME cell refuses a resume until the file is removed" in message
    assert "section 5's manual route" in message
    (run_dir / TRUNK_RECORD_NAME).unlink()
    message = record_trunk_run(run_dir, trunk_dir=None, log_dir=log_dir)
    assert "section 5's manual route" in message and "WARNING" not in message
    assert not (run_dir / TRUNK_RECORD_NAME).exists(), "never recorded with a guess"


def test_an_absolute_record_is_compared_as_the_directory_it_names(tmp_path):
    """A value the writer would have spelt otherwise (a hand edit, or a run copied to another ``LOG_BASE``): a
    session on the directory it names writes nothing and is not warned, and the TRUNK_FROM the warning names for
    another session is one a session can match."""
    log_dir = _log_dir(tmp_path)
    run_dir = _run_dir(tmp_path)
    path = run_dir / TRUNK_RECORD_NAME
    _ancestor_record(run_dir)
    in_tree = str((log_dir / "A").resolve())  # the writer records "A"
    out_of_tree = str((tmp_path / "elsewhere" / "B").resolve()) + "/"  # the writer records it without the slash
    for recorded, sessions in ((in_tree, (log_dir / "A", in_tree)), (out_of_tree, (tmp_path / "elsewhere" / "B",))):
        _write_record(run_dir, recorded)
        before = _snapshot_files(run_dir)
        for trunk_dir in sessions:
            message = record_trunk_run(run_dir, trunk_dir=trunk_dir, log_dir=log_dir)
            assert message == f'Trunk record:  {path} names TRUNK_FROM = "{recorded}", this session\'s trunk.'
        warning = record_trunk_run(run_dir, trunk_dir=log_dir / "C", log_dir=log_dir)
        assert f'Set TRUNK_FROM = "{recorded}" in the configuration cell' in warning
        assert "WARNING" not in record_trunk_run(run_dir, trunk_dir=Path(recorded), log_dir=log_dir)
        assert _snapshot_files(run_dir) == before


def test_a_complete_bundle_message_says_what_the_file_holds(tmp_path):
    log_dir = _log_dir(tmp_path)
    run_dir = _run_dir(tmp_path)
    _marker(run_dir, "complete")
    assert " is absent; nothing is written" in record_trunk_run(run_dir, trunk_dir=log_dir / "T", log_dir=log_dir)
    _write_record(run_dir, "S")
    assert 'names TRUNK_FROM = "S"; nothing is written' in record_trunk_run(
        run_dir, trunk_dir=log_dir / "T", log_dir=log_dir
    )
    (run_dir / TRUNK_RECORD_NAME).write_text("[]")
    assert "is unreadable (invalid trunk record" in record_trunk_run(run_dir, trunk_dir=log_dir / "T", log_dir=log_dir)


def test_an_unreadable_artifact_manifest_raises(tmp_path):
    """Its status is unknown, so the run is not treated as writable (manifest.read_bundle_status)."""
    run_dir = _run_dir(tmp_path)
    (run_dir / DEFAULT_MANIFEST_NAME).write_text("{truncated")
    with pytest.raises(ResultBundleError, match="artifact manifest"):
        record_trunk_run(run_dir, trunk_dir=None, log_dir=_log_dir(tmp_path))
    assert not (run_dir / TRUNK_RECORD_NAME).exists()


class TestDeterminism:
    def test_the_bytes_are_fixed(self, tmp_path):
        run_dir = _run_dir(tmp_path)
        record_trunk_run(run_dir, trunk_dir=None, log_dir=_log_dir(tmp_path))
        assert (run_dir / TRUNK_RECORD_NAME).read_bytes() == (
            b'{\n  "schema": "mesozoic.trunk-run/v1",\n  "trunk_from": ""\n}\n'
        )

    def test_a_session_on_the_same_trunk_writes_nothing(self, tmp_path):
        log_dir = _log_dir(tmp_path)
        run_dir = _run_dir(tmp_path)
        path = run_dir / TRUNK_RECORD_NAME
        assert "wrote" in record_trunk_run(run_dir, trunk_dir=log_dir / "A", log_dir=log_dir)
        first, stamp = path.read_bytes(), path.stat().st_mtime_ns
        # The same trunk named another way ("auto" selecting it, or its absolute path) is the same value.
        for trunk_dir in (log_dir / "A", str(log_dir / "A"), log_dir / "x" / ".." / "A"):
            message = record_trunk_run(run_dir, trunk_dir=trunk_dir, log_dir=log_dir)
            assert message == f'Trunk record:  {path} names TRUNK_FROM = "A", this session\'s trunk.'
        assert path.read_bytes() == first and path.stat().st_mtime_ns == stamp


def test_an_interrupted_write_keeps_the_old_record_and_the_next_bundle_write_drops_its_temporary(
    tmp_path, stable_provenance, monkeypatch
):
    """hashing._write_json writes ``.trunk_run.json.tmp`` and renames it: a crash before the rename leaves the old
    file whole, and build_artifact_manifest discards the stranded temporary before hashing."""
    log_dir = _log_dir(tmp_path)
    run_dir = _run_dir(tmp_path)
    initialize_result_bundle(run_dir, species="velociraptor", algorithm="PPO", seed=42, run_id=RUN_ID)
    record_trunk_run(run_dir, trunk_dir=log_dir / "A", log_dir=log_dir)
    old = (run_dir / TRUNK_RECORD_NAME).read_bytes()
    temporary = run_dir / f".{TRUNK_RECORD_NAME}.tmp"
    original_replace = Path.replace

    def reclaimed(self, target):
        if Path(self) == temporary:
            raise OSError("runtime reclaimed")
        return original_replace(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "replace", reclaimed)
        with pytest.raises(OSError, match="runtime reclaimed"):
            record_trunk_run(run_dir, trunk_dir=log_dir / "B", log_dir=log_dir)
    assert temporary.is_file() and (run_dir / TRUNK_RECORD_NAME).read_bytes() == old
    assert read_trunk_record(run_dir) == "A"
    manifest = build_artifact_manifest(run_dir, status="partial")
    assert not temporary.exists()
    assert TRUNK_RECORD_NAME in {entry["path"] for entry in manifest["files"]}
    assert not any(entry["path"].endswith(".tmp") for entry in manifest["files"])


class TestManifest:
    def _initialized(self, tmp_path):
        run_dir = _run_dir(tmp_path)
        initialize_result_bundle(run_dir, species="velociraptor", algorithm="PPO", seed=42, run_id=RUN_ID)
        return run_dir, _log_dir(tmp_path)

    def test_declared_by_the_next_bundle_write_and_kept_verifying_by_a_same_trunk_session(
        self, tmp_path, stable_provenance
    ):
        run_dir, log_dir = self._initialized(tmp_path)
        record_trunk_run(run_dir, trunk_dir=log_dir / "A", log_dir=log_dir)
        manifest = json.loads(write_artifact_manifest(run_dir, status="partial").read_text(encoding="utf-8"))
        assert TRUNK_RECORD_NAME in {entry["path"] for entry in manifest["files"]}
        verify_artifact_manifest(run_dir)
        record_trunk_run(run_dir, trunk_dir=log_dir / "A", log_dir=log_dir)
        verify_artifact_manifest(run_dir)

    def test_a_rewrite_fails_verification_until_the_next_bundle_write(self, tmp_path, stable_provenance):
        """A trunk change in a partial run without ancestor records: the file is a run-tree artifact like any other,
        so the manifest disagrees with it until the next bundle write rebuilds the manifest."""
        run_dir, log_dir = self._initialized(tmp_path)
        record_trunk_run(run_dir, trunk_dir=log_dir / "A", log_dir=log_dir)
        write_artifact_manifest(run_dir, status="partial")
        record_trunk_run(run_dir, trunk_dir=log_dir / "B", log_dir=log_dir)
        with pytest.raises(ResultBundleError, match=f"mismatch: {TRUNK_RECORD_NAME}"):
            verify_artifact_manifest(run_dir)
        write_artifact_manifest(run_dir, status="partial")
        verify_artifact_manifest(run_dir)
        assert read_trunk_record(run_dir) == "B"

    def test_a_record_written_after_a_bundle_write_is_undeclared_until_the_next(self, tmp_path, stable_provenance):
        run_dir, log_dir = self._initialized(tmp_path)
        write_artifact_manifest(run_dir, status="partial")
        record_trunk_run(run_dir, trunk_dir=None, log_dir=log_dir)
        with pytest.raises(ResultBundleError, match=f"undeclared bundle artifacts: \\['{TRUNK_RECORD_NAME}'\\]"):
            verify_artifact_manifest(run_dir)
        write_artifact_manifest(run_dir, status="partial")
        verify_artifact_manifest(run_dir)

    @pytest.mark.parametrize("held", ["A", None], ids=["recorded", "absent"])
    def test_a_complete_bundle_is_never_written(self, tmp_path, stable_provenance, held):
        run_dir, log_dir = self._initialized(tmp_path)
        if held is not None:
            record_trunk_run(run_dir, trunk_dir=log_dir / held, log_dir=log_dir)
        write_artifact_manifest(run_dir, status="complete")
        verify_artifact_manifest(run_dir)
        before = _snapshot_files(run_dir)
        for trunk_dir in (log_dir / "B", None, tmp_path / "elsewhere"):
            assert "nothing is written into a complete bundle" in record_trunk_run(
                run_dir, trunk_dir=trunk_dir, log_dir=log_dir
            )
        assert _snapshot_files(run_dir) == before
        verify_artifact_manifest(run_dir)

    def test_a_complete_export_certifies_it_and_refuses_it_changed_after_publication(self, tmp_path, stable_provenance):
        """The export does not regenerate the record (it is not in ``reporting.bundles._REGENERATED_ARTIFACTS``): a
        complete bundle that holds it verifies, and a record changed after publication is refused by the next export,
        never re-certified under a new hash, and the bundle is left exactly as found."""
        log_dir = _log_dir(tmp_path)
        run_dir = _run_dir(tmp_path)
        stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
        record_trunk_run(run_dir, trunk_dir=log_dir / "A", log_dir=log_dir)

        def export() -> None:
            save_result_bundle(
                stage_results,
                stage_configs,
                "velociraptor",
                "PPO",
                42,
                run_dir,
                backend="stable-baselines3",
                backend_version="test-backend-1.0",
                parallel_envs=4,
                evaluation_episodes=3,
                evaluation_seeds=[101, 102, 103],
                plant_identity=_plant_identity(),
                run_id="velociraptor-stable-baselines3-ppo-test",
            )

        export()
        assert json.loads((run_dir / DEFAULT_MANIFEST_NAME).read_text(encoding="utf-8"))["status"] == "complete"
        verify_artifact_manifest(run_dir)
        pristine = _snapshot_files(run_dir)
        edited = b'{\n  "schema": "mesozoic.trunk-run/v1",\n  "trunk_from": "B"\n}\n'
        (run_dir / TRUNK_RECORD_NAME).write_bytes(edited)
        with pytest.raises(
            ResultBundleError, match=r"certified artifact\(s\) changed after publication: \['trunk_run\.json'\]"
        ):
            export()
        assert _snapshot_files(run_dir) == {**pristine, Path(TRUNK_RECORD_NAME): edited}
