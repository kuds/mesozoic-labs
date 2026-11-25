"""Screens, pilots and experiments live outside the certification tree (``environments/shared/run_tree.py``).

Before ``RUN_KIND`` the SB3 notebook wrote branch screens, walking pilots and
certification runs into one tree, ``LOG_BASE/<species>/<embodiment>/<algo>/``, and only
``QUICK_TEST`` had a tree of its own. Replicate discovery (``replication.py``) reads the
task, gate, plant and recipe digests, never the commit, branch or ``RUN_LABEL``, and
trunk reuse has no recipe rule, so a screen whose digests equal a landed recipe's could
count as a certification run's seed replicate or become its ``TRUNK_FROM = "auto"``
trunk; the hand-kept list in ``run_exclusions.py`` was the only guard. ``RUN_KIND =
"screen"`` now writes such a run under ``<algo>_screen/``, which neither scan reaches;
a certification run refuses a pinned trunk from a screen or a quick-test tree, and
every session a ``RUN_ID`` that names a run only in another tree (a screen resumed in
a fresh runtime, ``RUN_KIND`` back at its default), before anything is written; and
the Drive summary labels every run with the kind its tree (or the ledger) gives it and
refuses to promote a screen.

The storage-cell tests run the cell as ``test_sb3_notebook_pins.py``'s quick-test tests
do, with its namespace builder; the summary tests run the Drive summary's cells as
``test_drive_summary_notebook.py`` does.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from environments.shared.model_id import ModelId
from environments.shared.replication import discover_replicates_for_run
from environments.shared.run_exclusions import EXCLUDED_COMMITS
from environments.shared.run_tree import (
    CERTIFY,
    QUICK_TEST,
    RUN_KINDS,
    SCREEN,
    refuse_run_id_of_another_tree,
    refuse_uncertified_trunk,
    run_kind_of,
    run_tree_name,
    run_tree_names,
    trunk_id_hint,
)
from environments.shared.stage_manifest import stage_dirname

from .ancestors_helpers import build_trunk_run, trunk_plant
from .notebook_cells import cell_index, code_cell, code_cell_sources
from .result_bundle_helpers import _complete_bundle
from .test_drive_summary_notebook import DRIVE_SUMMARY_PATH, READER_CELL_MARKERS, _legacy_run, _legacy_sweep
from .test_sb3_notebook_pins import (
    CONFIG_CELL_MARKER,
    NOTEBOOK_PATH,
    STORAGE_CELL_MARKER,
    _bound_names,
    _loaded_names,
    _storage_namespace,
    _top_level_assigns,
)

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
#: The certification tree a TRUNK_FROM id resolves under, as every cell spells it.
CERTIFICATION_TREE = "MODEL_ID.path(LOG_BASE) / ALGORITHM.lower()"


def _model_dir(tmp_path: Path, model_id: ModelId = TYRANNOSAURUS_REX) -> Path:
    return model_id.path(tmp_path / "logs")


def _provenance_run(run_dir: Path) -> Path:
    """A run directory holding the provenance fields the storage cell's TRUNK_FROM identity check reads."""
    run_dir.mkdir(parents=True)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                **TYRANNOSAURUS_REX.record(),
                "algorithm": "PPO",
                "backend": "stable-baselines3",
                "run_id": run_dir.name,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------------------------- the library
def test_the_tree_is_the_algorithm_then_the_kinds_suffix_and_a_quick_test_wins():
    assert RUN_KINDS == (CERTIFY, SCREEN) == ("certify", "screen")
    assert run_tree_name("PPO", run_kind=CERTIFY, quick_test=False) == "ppo"
    assert run_tree_name("ppo", run_kind=SCREEN, quick_test=False) == "ppo_screen"
    assert run_tree_name("SAC", run_kind=SCREEN, quick_test=False) == "sac_screen"
    for kind in RUN_KINDS:
        assert run_tree_name("PPO", run_kind=kind, quick_test=True) == "ppo_quick_test"
    assert run_tree_names("PPO") == ("ppo", "ppo_screen", "ppo_quick_test")
    # Read back from a run directory in each tree, the kind is the one the tree was built for.
    for kind, quick_test, expected in ((CERTIFY, False, CERTIFY), (SCREEN, False, SCREEN), (SCREEN, True, QUICK_TEST)):
        tree = run_tree_name("PPO", run_kind=kind, quick_test=quick_test)
        assert run_kind_of(Path("/logs/tyrannosaurus_rex/anatomical") / tree / "20260101_000000") == expected


@pytest.mark.parametrize("run_kind", ["Screen", "pilot", "certification", "quick_test", ""])
def test_an_unknown_run_kind_is_refused_under_quick_test_too(run_kind):
    for quick_test in (False, True):
        with pytest.raises(
            ValueError, match=r"is not a run kind: \"certify\" .* or \"screen\" .*A quick test is QUICK_TEST = True"
        ):
            run_tree_name("PPO", run_kind=run_kind, quick_test=quick_test)


@pytest.mark.parametrize(
    ("run_dir", "kind"),
    [
        ("/logs/tyrannosaurus_rex/anatomical/ppo/20260101_000000", CERTIFY),
        ("/logs/tyrannosaurus_rex/anatomical/ppo/stance7_r10LTSBpc_s42", CERTIFY),
        ("/logs/tyrannosaurus_rex/anatomical/ppo_screen/pilot3_walk_s42", SCREEN),
        ("/logs/tyrannosaurus_rex/anatomical/sac_quick_test/20260101_000000", QUICK_TEST),
        ("ppo_screen/20260101_000000", SCREEN),  # a model-relative run path, as the Drive summary passes
        ("ppo/20260101_000000", CERTIFY),
        ("ppo_20260301_120000", CERTIFY),  # the flat layout before 2026-03-31
    ],
)
def test_the_kind_is_read_off_the_directory_the_run_sits_in(run_dir, kind):
    assert run_kind_of(run_dir) == kind


@pytest.mark.parametrize("spelling", ["relative", "absolute", "link"])
@pytest.mark.parametrize("tree", ["ppo_screen", "ppo_quick_test"])
def test_a_certification_run_refuses_a_trunk_from_a_screen_or_a_quick_test(tmp_path, tree, spelling):
    model_dir = _model_dir(tmp_path)
    trunk = model_dir / tree / "20260101_000000"
    trunk.mkdir(parents=True)
    log_dir = model_dir / "ppo"
    log_dir.mkdir()
    if spelling == "relative":
        trunk_from = f"../{tree}/{trunk.name}"
    elif spelling == "absolute":
        trunk_from = str(trunk)
    else:  # a run id in the certification tree that is a link into the other tree
        os.symlink(trunk, log_dir / "linked")
        trunk_from = "linked"
    label = "screen" if tree == "ppo_screen" else "quick test"
    with pytest.raises(
        ValueError, match=rf"resolves to .*, a {label} run .*certification run .*Nothing has been written"
    ):
        refuse_uncertified_trunk(trunk_from, log_dir=log_dir, run_dir=log_dir / "20260924_120000")
    # A screen or a quick test may pin it.
    for session_tree in ("ppo_screen", "ppo_quick_test"):
        refuse_uncertified_trunk(trunk_from, log_dir=log_dir, run_dir=model_dir / session_tree / "20260924_120000")


def test_an_existing_certification_run_on_a_screen_trunk_is_told_to_train_the_node_in_a_new_run(tmp_path):
    """A certification run that recorded a screen or quick-test trunk before the refusal existed is refused as well:
    it is never continued as a certification run, and the refusal says the remedy is a new ``RUN_ID``."""
    model_dir = _model_dir(tmp_path)
    (model_dir / "ppo_quick_test" / "20260101_000000").mkdir(parents=True)
    run_dir = model_dir / "ppo" / "20260924_120000"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match=r"This run already exists: .* so train the node again in a new RUN_ID\. "):
        refuse_uncertified_trunk("../ppo_quick_test/20260101_000000", log_dir=model_dir / "ppo", run_dir=run_dir)


def test_a_certification_run_may_pin_a_certification_run_or_no_trunk(tmp_path):
    model_dir = _model_dir(tmp_path)
    log_dir = model_dir / "ppo"
    run_dir = log_dir / "20260924_120000"
    for trunk_from in ("", "auto", "20260101_000000", str(log_dir / "20260101_000000"), str(tmp_path / "copy" / "r")):
        refuse_uncertified_trunk(trunk_from, log_dir=log_dir, run_dir=run_dir)  # nothing need exist, nothing raises
    assert not tmp_path.joinpath("logs").exists(), "the refusal reads and writes nothing"


@pytest.mark.parametrize(
    ("other_tree", "session_tree", "knobs"),
    [
        ("ppo_screen", "ppo", 'RUN_KIND = "screen" and QUICK_TEST = False'),
        ("ppo_quick_test", "ppo", "QUICK_TEST = True"),
        ("ppo", "ppo_screen", 'RUN_KIND = "certify" and QUICK_TEST = False'),
    ],
)
def test_a_run_id_of_another_tree_is_refused_naming_the_knobs_that_reach_it(tmp_path, other_tree, session_tree, knobs):
    model_dir = _model_dir(tmp_path)
    (model_dir / other_tree / "pilot4_walk_s42").mkdir(parents=True)
    run_dir = model_dir / session_tree / "pilot4_walk_s42"
    with pytest.raises(
        ValueError,
        match=(
            rf"RUN_ID='pilot4_walk_s42' names no run under .*{session_tree}, the tree RUN_KIND and QUICK_TEST select, "
            rf"but it names .*{other_tree}.pilot4_walk_s42 \(reached with {re.escape(knobs)}\)\. A fresh runtime "
            r"resets RUN_KIND and QUICK_TEST .*Nothing has been written\."
        ),
    ):
        refuse_run_id_of_another_tree("pilot4_walk_s42", algorithm="PPO", run_dir=run_dir)
    assert not run_dir.exists()


def test_a_run_id_new_to_the_model_or_held_in_this_tree_passes(tmp_path):
    """``""`` (the memo or a fresh timestamp), an id no tree of the model holds, an id this session's tree holds (even
    with a namesake from before the refusal in another tree) and an id of another algorithm's tree all pass."""
    model_dir = _model_dir(tmp_path)
    for tree in ("ppo", "ppo_screen"):
        (model_dir / tree / "both").mkdir(parents=True)
    (model_dir / "sac_screen" / "a_sac_screen").mkdir(parents=True)
    for run_id, tree in (
        ("", "ppo"),
        ("new_id", "ppo"),
        ("both", "ppo"),
        ("both", "ppo_screen"),
        ("a_sac_screen", "ppo"),
    ):
        refuse_run_id_of_another_tree(run_id, algorithm="ppo", run_dir=model_dir / tree / (run_id or "20260101_000000"))


def test_a_bare_trunk_id_that_only_a_screen_holds_gets_a_hint_naming_how_to_pin_it(tmp_path):
    """A ``TRUNK_FROM`` id resolves in ``<algo>/``; when the id is a screen's (or a quick test's), the storage cell's
    refusal says so and, for a screen or a quick test, how to pin it. Anything else adds nothing."""
    model_dir = _model_dir(tmp_path)
    log_dir = model_dir / "ppo"
    screen = model_dir / "ppo_screen" / "pilot3_walk_s42"
    screen.mkdir(parents=True)
    hint = trunk_id_hint(screen.name, log_dir=log_dir, run_dir=model_dir / "ppo_screen" / "20260924_120000")
    assert hint == (
        f". A TRUNK_FROM run id resolves under {log_dir} whatever RUN_KIND says; {screen} has that id, a screen run: "
        "pin it as TRUNK_FROM = '../ppo_screen/pilot3_walk_s42' or by its absolute path"
    )
    certifying = trunk_id_hint(screen.name, log_dir=log_dir, run_dir=log_dir / "20260924_120000")
    assert certifying.endswith(f"{screen} has that id, a screen run, which a certification run never pins")
    for trunk_from in ("", "auto", "no_such_run", f"../ppo_screen/{screen.name}", str(screen)):
        assert trunk_id_hint(trunk_from, log_dir=log_dir, run_dir=model_dir / "ppo_screen" / "20260924_120000") == ""


# ---------------------------------------------------------------------------------------------- the storage cell
class TestStorageCell:
    def test_a_screen_lives_under_the_screen_tree_and_still_pins_a_certified_trunk(self, tmp_path, monkeypatch):
        """``RUN_KIND = "screen"`` opens ``<algo>_screen/<run_id>/``, outside the ``<algo>/`` tree trunk selection
        and replicate discovery scan, while a pinned run id still resolves in ``<algo>/``; a quick test of a screen
        is a quick test."""
        model_dir = _model_dir(tmp_path)
        real = _provenance_run(model_dir / "ppo" / "20260101_000000")
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(
            tmp_path, monkeypatch, RUN_KIND="screen", TRUNK_FROM=real.name
        )
        exec(compile(src, "sb3_storage_screen", "exec"), namespace)
        screen = namespace["RUN_DIR"]
        assert screen == model_dir / "ppo_screen" / "20260101_000001" and screen.is_dir()
        assert provenance_calls[-1][0] == screen and provenance_calls[-1][1]["run_id"] == screen.name
        assert namespace["TRUNK_DIR"] == real, "a pinned id resolves under the certification tree"
        assert sorted(path.name for path in (model_dir / "ppo").iterdir()) == [real.name]
        # QUICK_TEST wins over RUN_KIND.
        namespace.update(QUICK_TEST=True, RUN_ID="")
        exec(compile(src, "sb3_storage_screen_quick", "exec"), namespace)
        assert namespace["RUN_DIR"] == model_dir / "ppo_quick_test" / "20260101_000002"
        # The refusal of a bad RUN_ID names the tree RUN_KIND selects.
        namespace.update(QUICK_TEST=False, RUN_ID="../x")
        with pytest.raises(ValueError, match=r"names a run directory under .*ppo_screen"):
            exec(compile(src, "sb3_storage_bad_screen", "exec"), namespace)

    def test_the_memo_counts_only_in_the_tree_run_kind_selects(self, tmp_path, monkeypatch, capsys):
        """A screen first and then a certification session (or back) in one runtime gives each its own timestamp,
        never the other tree's run id; a rerun in one tree keeps its run."""
        model_dir = _model_dir(tmp_path)
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch, RUN_KIND="screen")
        exec(compile(src, "sb3_storage_screen", "exec"), namespace)
        assert namespace["RUN_DIR"] == model_dir / "ppo_screen" / "20260101_000001"
        namespace.update(RUN_KIND="certify", RUN_ID="")
        exec(compile(src, "sb3_storage_certify", "exec"), namespace)
        certify = namespace["RUN_DIR"]
        assert certify == model_dir / "ppo" / "20260101_000002" and "(new run)" in capsys.readouterr().out
        exec(compile(src, "sb3_storage_certify_rerun", "exec"), namespace)
        assert namespace["RUN_DIR"] == certify, "a rerun in the same tree keeps the run"
        namespace.update(RUN_KIND="screen")
        exec(compile(src, "sb3_storage_screen_again", "exec"), namespace)
        assert namespace["RUN_DIR"] == model_dir / "ppo_screen" / "20260101_000003"
        assert [call[1]["run_id"] for call in provenance_calls] == [
            "20260101_000001",
            "20260101_000002",
            "20260101_000002",
            "20260101_000003",
        ]

    @pytest.mark.parametrize("run_kind", ["screens", "Certify", "pilot"])
    def test_a_misspelt_run_kind_is_refused_before_anything_is_written(self, tmp_path, monkeypatch, run_kind):
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch, RUN_KIND=run_kind)
        with pytest.raises(ValueError, match=rf"RUN_KIND={run_kind!r} is not a run kind"):
            exec(compile(src, "sb3_storage_bad_kind", "exec"), namespace)
        assert provenance_calls == [] and "RUN_DIR" not in namespace
        assert not _model_dir(tmp_path).exists()

    @pytest.mark.parametrize("spelling", ["relative", "absolute"])
    def test_a_certification_run_refuses_a_screen_trunk_before_anything_is_written(
        self, tmp_path, monkeypatch, spelling
    ):
        """A pinned ``TRUNK_FROM`` that resolves into ``<algo>_screen/`` is refused when the run would land in the
        certification tree: before the run directory, the provenance or the memo move. A screen may pin it, and so
        may a quick test."""
        model_dir = _model_dir(tmp_path)
        screen = _provenance_run(model_dir / "ppo_screen" / "pilot3_walk_s42")
        trunk_from = str(screen) if spelling == "absolute" else f"../ppo_screen/{screen.name}"
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch)
        exec(compile(src, "sb3_storage_earlier_run", "exec"), namespace)  # this runtime's earlier run
        earlier = namespace["RUN_DIR"]
        namespace.update(RUN_ID="20260924_120000", TRUNK_FROM=trunk_from)
        with pytest.raises(ValueError, match=r"a screen run .*a certification run .*Nothing has been written"):
            exec(compile(src, "sb3_storage_screen_trunk", "exec"), namespace)
        assert len(provenance_calls) == 1 and not (model_dir / "ppo" / "20260924_120000").exists()
        assert namespace["RUN_DIR"] == earlier and namespace["_ACTIVE_RUN_ID"] == earlier.name
        for run_id, knobs, tree in (
            ("20260924_120001", {"RUN_KIND": "screen"}, "ppo_screen"),
            ("20260924_120002", {"QUICK_TEST": True}, "ppo_quick_test"),
        ):
            namespace.update({"RUN_KIND": "certify", "QUICK_TEST": False, "RUN_ID": run_id, **knobs})
            exec(compile(src, f"sb3_storage_{tree}_pins_a_screen", "exec"), namespace)
            assert namespace["RUN_DIR"] == model_dir / tree / run_id
            assert namespace["TRUNK_DIR"].resolve() == screen

    def test_a_screen_resumed_in_a_fresh_runtime_with_the_default_kind_is_refused(self, tmp_path, monkeypatch):
        """A fresh runtime resets RUN_KIND to "certify". An explicit RUN_ID that names a screen and no certification run
        is refused before anything is written, naming the knob that reaches the screen, instead of opening
        ``<algo>/<the screen's id>`` (and training the screen's recipe there); with the knob set again the cell re-enters
        the screen. The same holds for a certification run's id under RUN_KIND = "screen" and a quick test's id."""
        model_dir = _model_dir(tmp_path)
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        first, _, _ = _storage_namespace(tmp_path, monkeypatch, RUN_KIND="screen", RUN_ID="pilot4_walk_s42")
        exec(compile(src, "sb3_storage_screen", "exec"), first)
        screen = first["RUN_DIR"]
        assert screen == model_dir / "ppo_screen" / "pilot4_walk_s42"
        # A fresh runtime: the configuration cell's defaults and the screen's id, as section 5 sets it.
        fresh, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch, RUN_ID=screen.name)
        with pytest.raises(
            ValueError,
            match=(
                r"RUN_ID='pilot4_walk_s42' names no run under .*ppo, .* names .*ppo_screen.pilot4_walk_s42 "
                r"\(reached with RUN_KIND = \"screen\" and QUICK_TEST = False\).*Nothing has been written"
            ),
        ):
            exec(compile(src, "sb3_storage_resume_default_kind", "exec"), fresh)
        assert provenance_calls == [] and "RUN_DIR" not in fresh
        assert not (model_dir / "ppo").exists(), "no certification run under the screen's id"
        fresh["RUN_KIND"] = "screen"
        exec(compile(src, "sb3_storage_resume_screen", "exec"), fresh)
        assert fresh["RUN_DIR"] == screen and provenance_calls[-1][0] == screen
        # A certification run's id under RUN_KIND = "screen", and a quick test's id without QUICK_TEST.
        fresh.update(RUN_KIND="certify", RUN_ID="cert_s42")
        exec(compile(src, "sb3_storage_certify", "exec"), fresh)
        fresh.update(RUN_KIND="screen")
        with pytest.raises(ValueError, match=r"reached with RUN_KIND = \"certify\" and QUICK_TEST = False"):
            exec(compile(src, "sb3_storage_certify_as_screen", "exec"), fresh)
        fresh.update(RUN_KIND="certify", QUICK_TEST=True, RUN_ID="pipeline_check")
        exec(compile(src, "sb3_storage_quick", "exec"), fresh)
        fresh.update(QUICK_TEST=False)
        with pytest.raises(ValueError, match=r"reached with QUICK_TEST = True"):
            exec(compile(src, "sb3_storage_quick_as_certify", "exec"), fresh)
        assert sorted(path.relative_to(model_dir).as_posix() for path in model_dir.glob("*/*")) == [
            "ppo/cert_s42",
            "ppo_quick_test/pipeline_check",
            "ppo_screen/pilot4_walk_s42",
        ]
        assert fresh["RUN_DIR"] == model_dir / "ppo_quick_test" / "pipeline_check", "a refusal never moves the run"

    def test_a_screen_pinning_a_screen_by_its_bare_id_is_told_how_to_name_it(self, tmp_path, monkeypatch):
        """A bare TRUNK_FROM id resolves in ``<algo>/`` whatever RUN_KIND says, so a screen that names a screen by its
        id gets the provenance refusal, which now says how to pin it; ``../<algo>_screen/<id>`` pins it, read lexically,
        so it works while the model has no ``<algo>/`` directory (here it has none)."""
        model_dir = _model_dir(tmp_path)
        trunk = _provenance_run(model_dir / "ppo_screen" / "pilot3_walk_s42")
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        namespace, _, _ = _storage_namespace(tmp_path, monkeypatch, RUN_KIND="screen", TRUNK_FROM=trunk.name)
        with pytest.raises(
            RuntimeError,
            match=(
                r"(?s)is not a run with a provenance\.json.*\. A TRUNK_FROM run id resolves under .*ppo whatever "
                r"RUN_KIND says; .*pilot3_walk_s42 has that id, a screen run: pin it as "
                r"TRUNK_FROM = '\.\./ppo_screen/pilot3_walk_s42' or by its absolute path"
            ),
        ):
            exec(compile(src, "sb3_storage_bare_screen_trunk", "exec"), namespace)
        namespace["TRUNK_FROM"] = f"../ppo_screen/{trunk.name}"
        assert not (model_dir / "ppo").exists()
        exec(compile(src, "sb3_storage_screen_trunk_path", "exec"), namespace)
        assert namespace["TRUNK_DIR"] == trunk

    def test_the_refusal_comes_before_the_first_write(self):
        """In the cell's order: the tree is chosen from RUN_KIND, then the run-id and trunk refusals run on
        ``_run_dir``, before the seed check, the directory and the provenance."""
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        tree = ast.parse(src)
        lines = {
            name: min(
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) in (name, f"_run_dir.{name}")
            )
            for name in (
                "run_tree_name",
                "refuse_run_id_of_another_tree",
                "refuse_uncertified_trunk",
                "refuse_widened_seed_mismatch",
                "mkdir",
            )
        }
        assert (
            lines["run_tree_name"]
            < lines["refuse_run_id_of_another_tree"]
            < lines["refuse_uncertified_trunk"]
            < lines["refuse_widened_seed_mismatch"]
            < lines["mkdir"]
        )
        calls = {
            ast.unparse(node.func): ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func).startswith("refuse_")
        }
        assert calls["refuse_run_id_of_another_tree"] == (
            "refuse_run_id_of_another_tree(RUN_ID, algorithm=ALGORITHM, run_dir=_run_dir)"
        ), "the knob itself: only an explicit RUN_ID is checked"
        assert calls["refuse_uncertified_trunk"] == (
            f"refuse_uncertified_trunk(TRUNK_FROM, log_dir={CERTIFICATION_TREE}, run_dir=_run_dir)"
        )


# ---------------------------------------------------------------------------------------------- every cell
#: The trunk helpers whose ``log_dir`` is the certification tree a ``TRUNK_FROM`` id resolves under: the storage
#: cell's refusal and hint, the resolve cell's trunk record and the RESUME cell's trunk check.
TRUNK_HELPERS = ("refuse_uncertified_trunk", "trunk_id_hint", "record_trunk_run", "refuse_trunk_other_than_recorded")


def _model_trees(src: str) -> list[str]:
    """Every outermost ``MODEL_ID.path(LOG_BASE) / ...`` expression in *src* that names an algorithm's tree (it reads
    ``ALGORITHM``), unparsed; another directory of the model (a baseline's, say) is no run tree."""
    tree = ast.parse(src)
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and _leftmost(node) == "MODEL_ID.path(LOG_BASE)"
    ]
    inner = {id(node.left) for node in hits}
    return [ast.unparse(node) for node in hits if id(node) not in inner and "ALGORITHM" in ast.unparse(node)]


def _path_strings(tree: ast.AST) -> set[str]:
    """The string constants *tree* builds paths from: in a ``/`` expression or a ``Path(...)`` / ``.joinpath(...)``."""
    roots = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div))
        or (isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in ("Path", "joinpath"))
    ]
    return {
        sub.value
        for root in roots
        for sub in ast.walk(root)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }


def _leftmost(node: ast.expr) -> str:
    while isinstance(node, ast.BinOp):
        node = node.left
    return ast.unparse(node)


def test_every_cell_reads_the_run_tree_the_storage_cell_resolved():
    """Only the storage cell builds a run's tree from ``RUN_KIND`` and ``QUICK_TEST`` (once, as ``_runs_root``) and
    binds ``RUN_DIR``; every other algorithm tree a cell derives from the model's log directory is the certification
    tree a ``TRUNK_FROM`` id resolves under and ``select_trunk`` scans: the storage cell's id resolution, refusal and
    hint, the resolve cell's selection and trunk record, the RESUME cell's trunk check and the chain loop's selection
    check. No cell builds a path from a screen or quick-test tree's name."""
    cells = code_cell_sources(NOTEBOOK_PATH)
    config_at = cell_index(cells, CONFIG_CELL_MARKER)
    storage_at = cell_index(cells, STORAGE_CELL_MARKER)
    runs_root = "MODEL_ID.path(LOG_BASE) / run_tree_name(ALGORITHM, run_kind=RUN_KIND, quick_test=QUICK_TEST)"
    trees: dict[int, list[str]] = {}
    for index, src in enumerate(cells):
        tree = ast.parse(src)
        if index != storage_at:
            assert not {"RUN_DIR", "_runs_root", "_run_dir"} & _bound_names(tree), f"code cell {index} binds the run"
        if index not in (config_at, storage_at):
            assert "RUN_KIND" not in _loaded_names(tree), f"code cell {index} re-derives the tree from RUN_KIND"
        for value in _path_strings(tree):
            assert "_screen" not in value and "_quick_test" not in value, (index, value)
        if _model_trees(src):
            trees[index] = _model_trees(src)
    assert ast.unparse(_top_level_assigns(cells[storage_at])["_runs_root"]) == runs_root
    every = [expression for found in trees.values() for expression in found]
    assert every.count(runs_root) == 1 and runs_root in trees[storage_at]
    others = [expression for expression in every if expression != runs_root]
    assert set(others) == {CERTIFICATION_TREE, f"{CERTIFICATION_TREE} / TRUNK_FROM"}, sorted(set(others))
    # Each log_dir= a trunk helper takes is that tree (the chain loop compares the selection's own log_dir with it).
    log_dirs: dict[str, set[str]] = {}
    for src in cells:
        for node in ast.walk(ast.parse(src)):
            name = ast.unparse(node.func).split(".")[-1] if isinstance(node, ast.Call) else ""
            if name in TRUNK_HELPERS:
                spelt = {ast.unparse(keyword.value) for keyword in node.keywords if keyword.arg == "log_dir"}
                log_dirs.setdefault(name, set()).update(spelt)
    assert log_dirs == {name: {CERTIFICATION_TREE} for name in TRUNK_HELPERS}, log_dirs


# ---------------------------------------------------------------------------------------------- replication
#: One recipe for every run below: the landed recipe a screen may repeat byte for byte.
RECIPE = {"learning_rate": 3e-4, "n_steps": 2048}


def _judged_run(run_dir: Path, seed: int) -> Path:
    run_dir.mkdir(parents=True)
    build_trunk_run(run_dir, seed=seed, algorithm="PPO", hyperparameters=RECIPE, record_recipe_digest=True)
    return run_dir


def test_a_screen_of_the_landed_recipe_is_never_a_replicate_of_a_certification_run(tmp_path):
    """Replicate discovery scans a run's own siblings, so a screen in ``<algo>_screen/`` with the same task, gate,
    plant and recipe digests and another seed is no replicate of a run in ``<algo>/`` (a sibling there is); screens
    may count one another, as quick tests do."""
    model_dir = _model_dir(tmp_path)
    certify = _judged_run(model_dir / run_tree_name("ppo", run_kind=CERTIFY, quick_test=False) / "cert_s42", 42)
    screen_tree = model_dir / run_tree_name("ppo", run_kind=SCREEN, quick_test=False)
    screen = _judged_run(screen_tree / "screen_s60", 60)
    _judged_run(screen_tree / "screen_s80", 80)

    def found(run: Path) -> dict[str, Any]:
        return discover_replicates_for_run(run, model_id=TYRANNOSAURUS_REX, plant_identity=trunk_plant())

    assert found(certify) == {"1": []}, "the screens are no siblings: this run alone"
    _judged_run(certify.parent / "cert_s43", 43)
    assert found(certify) == {"1": [{"run_id": "cert_s43", "training_seed": 43}]}
    assert found(screen) == {"1": [{"run_id": "screen_s80", "training_seed": 80}]}


# ---------------------------------------------------------------------------------------------- the Drive summary
def test_the_drive_summary_finds_screen_runs_and_labels_each_runs_kind(tmp_path, monkeypatch, stable_provenance):
    """``discover_runs`` walks into ``<algo>_screen/`` as into ``<algo>_quick_test/``; every training row carries
    the ``run_kind`` its tree gives it, ``screen`` for a run of ``<algo>/`` the ledger names (the r10 stance screens
    trained there before RUN_KIND), and sweep rows none. The identity table, the pass/fail pivot, the detailed and
    settings tables and ``runs_summary.csv`` all show it."""
    pytest.importorskip("pandas")
    from environments.shared import reporting

    monkeypatch.setattr(reporting, "OFFLINE_JUDGEMENTS_PATH", tmp_path / "no_ledger.csv")

    def _no_install(command, **kwargs):
        raise AssertionError(f"a reader cell ran {command}")

    monkeypatch.setattr(subprocess, "check_call", _no_install)
    logs = tmp_path / "logs"
    model_dir = VELOCIRAPTOR_MONGOLIENSIS.path(logs)
    certify, screen, quick = "ppo/20261001_000000", "ppo_screen/20261002_000000", "ppo_quick_test/20261003_000000"
    for run in (certify, screen):
        _complete_bundle(
            model_dir / run,
            algorithm="PPO",
            backend="stable-baselines3",
            dirname=lambda stage: stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, stage),
        )
    _legacy_run(model_dir / quick)
    # A pre-knob screen under <algo>/: a legacy run whose stage config records a ledger commit.
    ledger = "ppo/stance7_r10LTSBpc_s42"
    _legacy_run(model_dir / ledger)
    stage_config = model_dir / ledger / "stage1" / "stage_config.json"
    recorded = {**json.loads(stage_config.read_text(encoding="utf-8")), "git_commit": sorted(EXCLUDED_COMMITS)[0]}
    stage_config.write_text(json.dumps(recorded), encoding="utf-8")
    sweep = "sweeps/stage1_ppo_20260315_101500"
    _legacy_sweep(model_dir / sweep)

    sources = code_cell_sources(DRIVE_SUMMARY_PATH)
    displayed: list = []
    namespace = {"LOGS_DIR": logs, "Path": Path, "display": displayed.append}
    for marker in READER_CELL_MARKERS:
        index = cell_index(sources, marker)
        exec(compile(sources[index], f"google_drive_summary code cell {index}", "exec"), namespace)
    kinds = {certify: "certify", screen: "screen", quick: "quick_test", ledger: "screen", sweep: ""}
    df = namespace["df"]
    assert set(df["run_path"]) == set(kinds)
    assert dict(zip(df["run_path"], df["run_kind"])) == kinds
    identity = displayed[-1]
    assert dict(zip(identity["run_path"], identity["run_kind"])) == kinds
    displayed.clear()
    for marker in ("Training runs pivot", "styled =", "Settings vs Outcome", "Combined CSV across all models"):
        exec(compile(sources[cell_index(sources, marker)], f"google_drive_summary {marker}", "exec"), namespace)
    pivot = namespace["df_training_pivot"]
    assert sorted(pivot["run_kind"]) == ["certify", "quick_test", "screen", "screen"]
    assert list(pivot.columns[:4]) == ["species", "embodiment", "algorithm", "run_kind"]
    details = [shown.data for shown in displayed if hasattr(shown, "data")]
    settings = [shown for shown in displayed if not hasattr(shown, "data") and "stage_passed" in shown.columns]
    for table in (*details, *settings):
        assert sorted(set(table["run_kind"])) == ["", "certify", "quick_test", "screen"], table.columns
    assert len(details) == 1 and len(settings) == 1
    with (logs / "runs_summary.csv").open(encoding="utf-8", newline="") as handle:
        exported = {(row["run_path"], row["run_kind"]) for row in csv.DictReader(handle)}
    assert exported == set(kinds.items())


@pytest.mark.parametrize(
    ("run_path", "refused"),
    [
        ("velociraptor_mongoliensis/anatomical/ppo_screen/20261002_000000", "screen"),
        ("velociraptor_mongoliensis/anatomical/ppo_quick_test/20261003_000000", "quick test"),
        ("velociraptor_mongoliensis/anatomical/ppo/20261001_000000", None),
    ],
)
def test_the_strict_promotion_cell_refuses_a_screen_or_a_quick_test(tmp_path, run_path, refused):
    """The tree is a run's only record of its kind and a promoted copy loses it, so section 8 refuses a run under
    ``<algo>_screen/`` or ``<algo>_quick_test/`` before it validates anything; a certification run is validated."""
    from environments.shared.result_bundle import ResultBundleError

    source = code_cell(DRIVE_SUMMARY_PATH, "STRICT_PROMOTION_VALIDATION = False")
    assert source.count("STRICT_PROMOTION_VALIDATION = False") == 1 and source.count('PROMOTION_RUN_DIR = ""') == 1
    source = source.replace("STRICT_PROMOTION_VALIDATION = False", "STRICT_PROMOTION_VALIDATION = True").replace(
        'PROMOTION_RUN_DIR = ""', f"PROMOTION_RUN_DIR = {run_path!r}"
    )
    validated: list = []

    def validate(path, **kwargs):
        validated.append((path, kwargs))
        return {"status": "canonical-valid"}

    namespace = {
        "LOGS_DIR": tmp_path / "logs",
        "Path": Path,
        "json": json,
        "run_kind_of": run_kind_of,
        "validate_result_bundle": validate,
        "ResultBundleError": ResultBundleError,
    }
    if refused is None:
        exec(compile(source, "google_drive_summary promotion", "exec"), namespace)
        assert validated == [(tmp_path / "logs" / run_path, {"require_complete": True})]
        return
    with pytest.raises(RuntimeError, match=rf"not ready for promotion: .* is a {refused} run, never a certification"):
        exec(compile(source, "google_drive_summary promotion", "exec"), namespace)
    assert validated == []
