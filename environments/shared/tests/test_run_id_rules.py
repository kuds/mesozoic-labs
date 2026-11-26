"""The SB3 notebook storage cell's rules for an explicit ``RUN_ID`` (the maintainer's decisions of 2026-11-26).

``SPECIES`` and ``EMBODIMENT`` select the model, never ``RUN_ID``. A fresh runtime opens on the Velociraptor, and on
2026-10-27 a T. rex ``RUN_ID`` typed there opened ``velociraptor_mongoliensis/anatomical/ppo/stance6_r10LTSB_s60``, a
new Velociraptor run under the T. rex run's id. The storage cell now:

* refuses, before anything is written, an explicit ``RUN_ID`` that names no run of this model in its trees for
  ``ALGORITHM`` but a run of another manifest model in that model's trees for ``ALGORITHM`` under ``LOG_BASE``, naming
  every run of the id (either algorithm) and the knobs that reach it;
* refuses a new explicit ``RUN_ID`` with characters other than letters, digits, ``_`` and ``-``, naming any run of the
  id under the other algorithm and an id of allowed characters, while a run directory the selected tree already holds
  is re-entered whatever its characters (Drive has a run named ``stance screen LTSB s52``), with a note;
* prints a banner for every explicit ``RUN_ID`` that names the model display name, its species and embodiment, the
  algorithm, the run tree, whether the run is new or re-entered, and every other run of the id under ``LOG_BASE``, of
  any manifest model and either algorithm (so re-entering the 2026-10-27 Velociraptor run says the T. rex run has its
  id). A run of the id under the other algorithm alone is named, not refused.

The rules sit in the cell, not in ``run_tree.py``, so a ``REPO_REF`` at any commit that has ``run_tree.py``
(2026-11-24) still runs the cell. The cell runs as ``test_screen_run_tree.py`` and ``test_write_guards.py`` run it:
executed in a namespace that supplies every name it reads (pinned below), with ``initialize_result_bundle`` recorded
instead of run.
"""

from __future__ import annotations

import ast
import builtins
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from environments.shared.model_id import ModelId
from environments.shared.species_names import model_display_name, model_ids

from .notebook_cells import cell_index, cell_sources, code_cell
from .test_sb3_notebook_pins import CONFIG_CELL_MARKER, NOTEBOOK_PATH, STORAGE_CELL_MARKER, _bound_names

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
COMPSOGNATHUS_LONGIPES_ANATOMICAL = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
#: The run id of the accidental Velociraptor run of 2026-10-27: a T. rex run's id.
TYRANNOSAURUS_REX_RUN_ID = "stance6_r10LTSB_s60"
#: A run on Drive named before the character rule, with spaces in its id.
LEGACY_RUN_ID = "stance screen LTSB s52"
#: The names run_tree.py had when it landed (29cd50405, 2026-11-24): the storage cell imports only these, so a
#: REPO_REF at any commit with run_tree.py still runs it.
RUN_TREE_NAMES_OF_2026_11_24 = frozenset(
    {
        "CERTIFY",
        "SCREEN",
        "RUN_KINDS",
        "QUICK_TEST",
        "TREE_SUFFIXES",
        "TREE_KNOBS",
        "run_tree_name",
        "run_tree_names",
        "run_kind_of",
        "refuse_run_id_of_another_tree",
        "refuse_uncertified_trunk",
        "trunk_id_hint",
    }
)
BAR = "!" * 72
RULE = "a new RUN_ID may use only letters (A-Z, a-z), digits (0-9), '_' and '-'"
#: The trees of the sessions below (ALGORITHM = "ppo"), as the refusals name them.
PPO_TREES = "ppo/, ppo_screen/ and ppo_quick_test/"
CERTIFY_KNOBS = 'RUN_KIND = "certify" and QUICK_TEST = False'
SCREEN_KNOBS = 'RUN_KIND = "screen" and QUICK_TEST = False'


def _reach(model_id: ModelId, algorithm: str = "ppo", knobs: str = CERTIFY_KNOBS) -> str:
    """The knobs the cell names for a run of *model_id* in *algorithm*'s tree that *knobs* select."""
    return f"SPECIES = {model_id.species!r}, EMBODIMENT = {model_id.embodiment!r}, ALGORITHM = {algorithm!r}, {knobs}"


def _storage_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: ModelId = VELOCIRAPTOR_MONGOLIENSIS, **knobs: Any
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    """Every name the storage cell reads, for *model_id*, with its provenance write recorded instead of run."""
    from environments.shared import plant_contract, result_bundle

    monkeypatch.setattr(
        plant_contract, "current_plant_identity", lambda model: SimpleNamespace(to_dict=lambda: model.record())
    )
    calls: list[tuple[Path, dict[str, Any]]] = []

    def initialize(directory: Path, **kwargs: Any) -> Path:
        calls.append((directory, kwargs))
        return directory / "provenance.json"

    monkeypatch.setattr(result_bundle, "initialize_result_bundle", initialize)
    namespace: dict[str, Any] = {
        "Path": Path,
        "datetime": datetime,
        "repo_root": tmp_path,
        "IN_COLAB": False,
        "USE_GOOGLE_DRIVE": False,
        "REPO_REF": "main",
        "SPECIES": model_id.species,
        "EMBODIMENT": model_id.embodiment,
        "MODEL_ID": model_id,
        "ALGORITHM": "ppo",
        "SEED": 42,
        "N_ENVS": 1,
        "QUICK_TEST": False,
        "RUN_KIND": "certify",
        "TRUNK_FROM": "",
        "RUN_ID": "",
        **knobs,
    }
    return namespace, calls


def _run_storage(namespace: dict[str, Any], name: str = "sb3_storage") -> None:
    exec(compile(code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER), name, "exec"), namespace)


def _use_model(namespace: dict[str, Any], model_id: ModelId) -> None:
    """Set SPECIES and EMBODIMENT as the configuration cell does, and the MODEL_ID it resolves."""
    namespace.update(SPECIES=model_id.species, EMBODIMENT=model_id.embodiment, MODEL_ID=model_id)


def _run(tmp_path: Path, model_id: ModelId, tree: str, run_id: str) -> Path:
    run_dir = model_id.path(tmp_path / "logs") / tree / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def _written(tmp_path: Path) -> list[str]:
    return sorted(path.relative_to(tmp_path / "logs").as_posix() for path in (tmp_path / "logs").glob("*/*/*/*"))


# ---------------------------------------------------------------------------------------------- the harness
def test_the_namespace_supplies_every_name_the_storage_cell_reads(tmp_path, monkeypatch):
    """Every name the cell loads and does not bind, and every name it reads through ``globals().get`` and never binds
    (``REPO_REF``, which only the import refusal reads), is in the namespace these tests execute it in."""
    tree = ast.parse(code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER))
    loaded = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    through_globals = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "globals().get"
        and isinstance(node.args[0], ast.Constant)
    }
    bound = _bound_names(tree)
    namespace, _ = _storage_namespace(tmp_path, monkeypatch)
    free = (loaded - bound - set(dir(builtins))) | (through_globals - bound)
    assert "REPO_REF" in free and "RUN_KIND" in free
    assert free <= set(namespace), sorted(free - set(namespace))


def test_the_storage_cell_imports_only_what_run_tree_had_on_2026_11_24():
    """The rules are written in the cell, so a REPO_REF at a commit from 2026-11-24 on (run_tree.py's first) still
    runs it; a name run_tree.py gained later would make such a REPO_REF fail at the cell's import refusal."""
    tree = ast.parse(code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "environments.shared.run_tree"
        for alias in node.names
    }
    assert {"TREE_KNOBS", "run_kind_of", "run_tree_names"} <= imported <= RUN_TREE_NAMES_OF_2026_11_24


# ---------------------------------------------------------------------------------------------- another model's id
class TestARunIdOfAnotherModel:
    def test_a_tyrannosaurus_rex_id_on_the_velociraptor_mongoliensis_is_refused_before_any_write(
        self, tmp_path, monkeypatch
    ):
        """The accident of 2026-10-27: a T. rex run's id with SPECIES still on the Velociraptor opened a new
        Velociraptor run. It is refused, naming the model that holds it and the knobs that reach it, before the run
        directory, the provenance or the memo; with SPECIES set to match, the cell re-enters the T. rex run."""
        tyrannosaurus_rex_run = _run(tmp_path, TYRANNOSAURUS_REX, "ppo", TYRANNOSAURUS_REX_RUN_ID)
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, RUN_ID=TYRANNOSAURUS_REX_RUN_ID)
        with pytest.raises(
            ValueError,
            match=(
                rf"RUN_ID='{TYRANNOSAURUS_REX_RUN_ID}' names no run of Velociraptor mongoliensis \(Anatomical\) in its "
                rf"{re.escape(PPO_TREES)} trees under .*velociraptor_mongoliensis.anatomical, but it names "
                rf".*tyrannosaurus_rex.anatomical.ppo.{TYRANNOSAURUS_REX_RUN_ID}, "
                r"a run of Tyrannosaurus rex \(Anatomical\) \(reached with SPECIES = 'tyrannosaurus_rex', "
                r"EMBODIMENT = 'anatomical', ALGORITHM = 'ppo', RUN_KIND = \"certify\" and QUICK_TEST = False\)\. "
                r"SPECIES and EMBODIMENT select the model, never RUN_ID: .*Nothing has been written\."
            ),
        ):
            _run_storage(namespace)
        assert provenance_calls == [] and "RUN_DIR" not in namespace and "_ACTIVE_RUN_ID" not in namespace
        assert not VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs").exists()
        _use_model(namespace, TYRANNOSAURUS_REX)
        _run_storage(namespace, "sb3_storage_tyrannosaurus_rex")
        assert namespace["RUN_DIR"] == tyrannosaurus_rex_run and provenance_calls[-1][0] == tyrannosaurus_rex_run
        assert _written(tmp_path) == [f"tyrannosaurus_rex/anatomical/ppo/{TYRANNOSAURUS_REX_RUN_ID}"]

    @pytest.mark.parametrize("session", model_ids(), ids=str)
    def test_every_model_refuses_the_run_id_of_every_other_model(self, tmp_path, monkeypatch, session):
        """Every manifest model, anatomical or robot, refuses an id that only another model's run holds, whether the
        two share a species (the Compsognathus pair) or not."""
        for other in model_ids():
            if other == session:
                continue
            base = tmp_path / f"{other.species}-{other.embodiment}"
            _run(base, other, "ppo", "shared_name_s42")
            namespace, provenance_calls = _storage_namespace(base, monkeypatch, session, RUN_ID="shared_name_s42")
            with pytest.raises(ValueError, match=r"names no run of .* but it names") as refusal:
                _run_storage(namespace, f"sb3_storage_{session.species}_{other.species}")
            message = str(refusal.value)
            assert f"names no run of {model_display_name(session)} in its {PPO_TREES} trees under" in message, message
            assert f"a run of {model_display_name(other)} (reached with {_reach(other)})" in message, message
            assert provenance_calls == [] and not session.path(base / "logs").exists()

    @pytest.mark.parametrize(
        ("tree", "knobs"),
        [
            ("ppo_screen", 'RUN_KIND = "screen" and QUICK_TEST = False'),
            ("ppo_quick_test", "QUICK_TEST = True"),
        ],
    )
    def test_another_models_screen_or_quick_test_is_named_with_the_knobs_that_reach_it(
        self, tmp_path, monkeypatch, tree, knobs
    ):
        _run(tmp_path, COMPSOGNATHUS_LONGIPES_ROBOT, tree, "pilot4_walk_s42")
        namespace, provenance_calls = _storage_namespace(
            tmp_path, monkeypatch, COMPSOGNATHUS_LONGIPES_ANATOMICAL, RUN_ID="pilot4_walk_s42"
        )
        with pytest.raises(
            ValueError,
            match=(
                rf"names .*compsognathus_longipes.robot.{tree}.pilot4_walk_s42, a run of Compsognathus longipes "
                rf"\(Robot\) \(reached with SPECIES = 'compsognathus_longipes', EMBODIMENT = 'robot', ALGORITHM = 'ppo', "
                rf"{re.escape(knobs)}\)"
            ),
        ):
            _run_storage(namespace)
        assert provenance_calls == [] and _written(tmp_path) == [f"compsognathus_longipes/robot/{tree}/pilot4_walk_s42"]

    def test_every_model_holding_the_id_is_named(self, tmp_path, monkeypatch):
        for model_id in (TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES_ROBOT):
            _run(tmp_path, model_id, "ppo", TYRANNOSAURUS_REX_RUN_ID)
        namespace, _ = _storage_namespace(tmp_path, monkeypatch, RUN_ID=TYRANNOSAURUS_REX_RUN_ID)
        with pytest.raises(ValueError, match=r"Tyrannosaurus rex \(Anatomical\).*; .*Compsognathus longipes \(Robot\)"):
            _run_storage(namespace)

    def test_an_id_this_model_holds_re_enters_its_run_whatever_another_model_holds(self, tmp_path, monkeypatch, capsys):
        """A run of this model wins: the id is this model's run (re-entered in its own tree; another tree's is the tree
        refusal's), and a namesake of another model refuses nothing. The banner names it: retyping the T. rex id on the
        Velociraptor re-enters the accidental run of 2026-10-27, and says the T. rex run has that id too."""
        tyrannosaurus_rex_run = _run(tmp_path, TYRANNOSAURUS_REX, "ppo", TYRANNOSAURUS_REX_RUN_ID)
        own = _run(tmp_path, VELOCIRAPTOR_MONGOLIENSIS, "ppo", TYRANNOSAURUS_REX_RUN_ID)
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, RUN_ID=TYRANNOSAURUS_REX_RUN_ID)
        _run_storage(namespace)
        assert namespace["RUN_DIR"] == own and provenance_calls[-1][0] == own
        banner = TestTheExplicitRunIdBanner._banner(capsys.readouterr().out)
        assert banner[0].startswith(
            f"RUN_ID {TYRANNOSAURUS_REX_RUN_ID!r} RE-ENTERS the existing run of Velociraptor mongoliensis (Anatomical):"
        )
        assert banner[1] == (
            f"Note: {tyrannosaurus_rex_run}, a run of Tyrannosaurus rex (Anatomical), has this id too (reached with "
            f"{_reach(TYRANNOSAURUS_REX)})."
        )
        assert len(banner) == 3 and str(own) not in "\n".join(banner)
        namespace["RUN_KIND"] = "screen"
        with pytest.raises(
            ValueError, match=r"names no run under .*ppo_screen, the tree RUN_KIND and QUICK_TEST select"
        ):
            _run_storage(namespace, "sb3_storage_own_other_tree")

    def test_the_memo_a_new_id_and_another_algorithms_run_pass(self, tmp_path, monkeypatch):
        """Only an explicit RUN_ID is checked, and only in the trees of this session's ALGORITHM: ``""`` mints a fresh
        timestamp, an id no model holds starts a new run, and a T. rex SAC run's id is no PPO run of any model."""
        _run(tmp_path, TYRANNOSAURUS_REX, "sac", "sac_only_s42")
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch)
        _run_storage(namespace)
        assert namespace["RUN_DIR"].parent == VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "ppo"
        for run_id in ("velociraptor_mongoliensis_new_s42", "sac_only_s42"):
            namespace["RUN_ID"] = run_id
            _run_storage(namespace, f"sb3_storage_{run_id}")
            assert namespace["RUN_DIR"] == VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "ppo" / run_id
        assert len(provenance_calls) == 3

    def test_a_run_of_the_id_under_the_other_algorithm_is_named_in_the_banner_not_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """Only the trees of this session's ALGORITHM refuse: a T. rex PPO run's id on a Velociraptor SAC session, or
        this model's own SAC run's id on a PPO session, starts a new run, and the banner names that run and the knobs
        that reach it, ALGORITHM among them."""
        tyrannosaurus_rex_run = _run(tmp_path, TYRANNOSAURUS_REX, "ppo", "tyrannosaurus_rex_stance_s44")
        namespace, provenance_calls = _storage_namespace(
            tmp_path, monkeypatch, RUN_ID="tyrannosaurus_rex_stance_s44", ALGORITHM="sac"
        )
        _run_storage(namespace)
        new_run = VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "sac" / "tyrannosaurus_rex_stance_s44"
        assert namespace["RUN_DIR"] == new_run and provenance_calls[-1][0] == new_run
        banner = TestTheExplicitRunIdBanner._banner(capsys.readouterr().out)
        assert banner[0].startswith(
            "RUN_ID 'tyrannosaurus_rex_stance_s44' starts a NEW run of Velociraptor mongoliensis"
        )
        assert banner[1] == (
            f"Note: {tyrannosaurus_rex_run}, a run of Tyrannosaurus rex (Anatomical), has this id too (reached with "
            f"{_reach(TYRANNOSAURUS_REX)})."
        )
        own_sac_run = _run(tmp_path, VELOCIRAPTOR_MONGOLIENSIS, "sac_screen", "sac_pilot_s42")
        namespace.update(RUN_ID="sac_pilot_s42", ALGORITHM="ppo")
        _run_storage(namespace, "sb3_storage_own_sac")
        assert namespace["RUN_DIR"] == VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "ppo" / "sac_pilot_s42"
        assert TestTheExplicitRunIdBanner._banner(capsys.readouterr().out)[1] == (
            f"Note: {own_sac_run}, a run of Velociraptor mongoliensis (Anatomical), has this id too (reached with "
            f"{_reach(VELOCIRAPTOR_MONGOLIENSIS, 'sac', SCREEN_KNOBS)})."
        )

    def test_the_refusal_names_this_models_run_under_the_other_algorithm(self, tmp_path, monkeypatch):
        """The Velociraptor holds the id under sac/ and the T. rex under ppo/: a PPO Velociraptor session is refused,
        and the refusal names both runs, so the remedy (ALGORITHM, or SPECIES) is in it."""
        own_sac_run = _run(tmp_path, VELOCIRAPTOR_MONGOLIENSIS, "sac", "x_s42")
        tyrannosaurus_rex_run = _run(tmp_path, TYRANNOSAURUS_REX, "ppo", "x_s42")
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, RUN_ID="x_s42")
        with pytest.raises(ValueError) as refusal:
            _run_storage(namespace)
        message = str(refusal.value)
        assert message.startswith(
            f"RUN_ID='x_s42' names no run of Velociraptor mongoliensis (Anatomical) in its {PPO_TREES} trees under "
        ), message
        assert (
            f"but it names {own_sac_run}, a run of Velociraptor mongoliensis (Anatomical) (reached with "
            f"{_reach(VELOCIRAPTOR_MONGOLIENSIS, 'sac')}); {tyrannosaurus_rex_run}, a run of Tyrannosaurus rex "
            f"(Anatomical) (reached with {_reach(TYRANNOSAURUS_REX)})."
        ) in message
        assert provenance_calls == [] and not (VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "ppo").exists()


# ---------------------------------------------------------------------------------------------- the characters
class TestTheCharactersOfANewRunId:
    @pytest.mark.parametrize(
        ("run_id", "suggestion"),
        [
            (LEGACY_RUN_ID, "stance_screen_LTSB_s52"),
            ("stance5.LTSB", "stance5_LTSB"),
            ("run+1", "run_1"),
            ("run#2", "run_2"),
            ("café_s42", "caf_s42"),
            ("a:b", "a_b"),
            ("tab\tinside", "tab_inside"),
            ("50%", "50"),
            ("%%", ""),
        ],
    )
    def test_a_new_run_id_with_other_characters_is_refused_before_anything_is_written(
        self, tmp_path, monkeypatch, run_id, suggestion
    ):
        """The refusal states the rule and the next step: an id of only those characters, offered from the one typed
        (none when nothing of it is left), or the knobs that reach the run meant."""
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, RUN_ID=run_id)
        with pytest.raises(ValueError) as refusal:
            _run_storage(namespace)
        message = str(refusal.value)
        assert message.startswith(f"RUN_ID={run_id!r} names no run of this model in its {PPO_TREES} trees, and {RULE}")
        assert "the maintainer's decision of 2026-11-26" in message and message.endswith(
            "Choose a new RUN_ID of only those characters"
            + (f" (such as {suggestion!r})" if suggestion else "")
            + ", or set the knobs that reach the run you mean. Nothing has been written."
        ), message
        assert "It names" not in message
        assert str(VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "ppo") in message
        assert provenance_calls == [] and "RUN_DIR" not in namespace
        assert not VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs").exists()

    @pytest.mark.parametrize("run_id", ["stance6_r10LTSB_s60", "pilot-4", "20260924_120000", "A1"])
    def test_letters_digits_underscores_and_hyphens_start_a_new_run(self, tmp_path, monkeypatch, run_id):
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, RUN_ID=run_id)
        _run_storage(namespace)
        assert namespace["RUN_DIR"] == VELOCIRAPTOR_MONGOLIENSIS.path(tmp_path / "logs") / "ppo" / run_id
        assert provenance_calls[-1][1]["run_id"] == run_id

    @pytest.mark.parametrize(("run_kind", "tree"), [("certify", "ppo"), ("screen", "ppo_screen")])
    def test_a_run_the_tree_already_holds_is_re_entered_whatever_its_characters_with_a_note(
        self, tmp_path, monkeypatch, capsys, run_kind, tree
    ):
        legacy = _run(tmp_path, TYRANNOSAURUS_REX, tree, LEGACY_RUN_ID)
        namespace, provenance_calls = _storage_namespace(
            tmp_path, monkeypatch, TYRANNOSAURUS_REX, RUN_ID=LEGACY_RUN_ID, RUN_KIND=run_kind
        )
        _run_storage(namespace)
        assert namespace["RUN_DIR"] == legacy and provenance_calls[-1][1]["run_id"] == LEGACY_RUN_ID
        out = capsys.readouterr().out
        assert (
            "Note: this id has characters a new RUN_ID may not use (only letters, digits, '_' and '-'); it is "
            "re-entered because this tree already holds that run."
        ) in out
        assert f"RUN_ID {LEGACY_RUN_ID!r} RE-ENTERS the existing run of Tyrannosaurus rex (Anatomical)" in out

    def test_the_refusal_names_a_run_of_the_id_under_the_other_algorithm(self, tmp_path, monkeypatch):
        """A legacy id held only under sac/, typed in a PPO session (a fresh runtime resets ALGORITHM), is no new id of
        allowed characters: the refusal names the SAC run and ALGORITHM = 'sac', which reaches it."""
        sac_run = _run(tmp_path, TYRANNOSAURUS_REX, "sac", LEGACY_RUN_ID)
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, TYRANNOSAURUS_REX, RUN_ID=LEGACY_RUN_ID)
        with pytest.raises(ValueError) as refusal:
            _run_storage(namespace)
        message = str(refusal.value)
        assert (
            RULE in message
            and (
                f"It names {sac_run}, a run of Tyrannosaurus rex (Anatomical) (reached with "
                f"{_reach(TYRANNOSAURUS_REX, 'sac')}). Choose a new RUN_ID"
            )
            in message
        ), message
        assert provenance_calls == [] and not (TYRANNOSAURUS_REX.path(tmp_path / "logs") / "ppo").exists()

    def test_a_legacy_id_held_in_another_tree_gets_the_tree_refusal_not_the_character_rule(self, tmp_path, monkeypatch):
        """The tree refusal comes first and names the knobs that reach the run, the remedy for a legacy id typed with
        RUN_KIND left at its default."""
        _run(tmp_path, TYRANNOSAURUS_REX, "ppo_screen", LEGACY_RUN_ID)
        namespace, provenance_calls = _storage_namespace(tmp_path, monkeypatch, TYRANNOSAURUS_REX, RUN_ID=LEGACY_RUN_ID)
        with pytest.raises(ValueError, match=r"reached with RUN_KIND = \"screen\" and QUICK_TEST = False") as refusal:
            _run_storage(namespace)
        assert RULE not in str(refusal.value) and provenance_calls == []
        assert not (TYRANNOSAURUS_REX.path(tmp_path / "logs") / "ppo").exists()


# ---------------------------------------------------------------------------------------------- the banner
class TestTheExplicitRunIdBanner:
    @staticmethod
    def _banner(out: str) -> list[str]:
        lines = out.splitlines()
        start = lines.index(BAR)
        end = lines.index(BAR, start + 1)
        return lines[start + 1 : end]

    def test_a_new_and_a_re_entered_run_each_name_the_model_algorithm_and_tree(self, tmp_path, monkeypatch, capsys):
        namespace, _ = _storage_namespace(tmp_path, monkeypatch, TYRANNOSAURUS_REX, RUN_ID=TYRANNOSAURUS_REX_RUN_ID)
        _run_storage(namespace)
        assert self._banner(capsys.readouterr().out) == [
            f"RUN_ID {TYRANNOSAURUS_REX_RUN_ID!r} starts a NEW run of Tyrannosaurus rex (Anatomical): species "
            "tyrannosaurus_rex, embodiment anatomical, algorithm PPO, run tree ppo/ (certification).",
            "SPECIES and EMBODIMENT select the model, never RUN_ID: stop here if this is not the run the id is meant "
            "for.",
        ]
        _run_storage(namespace, "sb3_storage_again")
        [line, _] = self._banner(capsys.readouterr().out)
        assert line.startswith(
            f"RUN_ID {TYRANNOSAURUS_REX_RUN_ID!r} RE-ENTERS the existing run of Tyrannosaurus rex (Anatomical):"
        )

    @pytest.mark.parametrize(
        ("knobs", "tree"),
        [
            ({"RUN_KIND": "screen"}, "sac_screen/ (screen)"),
            ({"QUICK_TEST": True}, "sac_quick_test/ (quick test)"),
        ],
    )
    def test_the_robot_screens_and_quick_tests_name_their_tree(self, tmp_path, monkeypatch, capsys, knobs, tree):
        namespace, _ = _storage_namespace(
            tmp_path, monkeypatch, COMPSOGNATHUS_LONGIPES_ROBOT, RUN_ID="robot_pilot_s42", ALGORITHM="sac", **knobs
        )
        _run_storage(namespace)
        [line, _] = self._banner(capsys.readouterr().out)
        assert line == (
            "RUN_ID 'robot_pilot_s42' starts a NEW run of Compsognathus longipes (Robot): species "
            f"compsognathus_longipes, embodiment robot, algorithm SAC, run tree {tree}."
        )

    def test_the_memo_and_a_fresh_timestamp_print_no_banner(self, tmp_path, monkeypatch, capsys):
        namespace, _ = _storage_namespace(tmp_path, monkeypatch)
        _run_storage(namespace)
        _run_storage(namespace, "sb3_storage_memo")
        out = capsys.readouterr().out
        assert BAR not in out and "RUN_ID " not in out and out.count("Run directory: ") == 2

    def test_the_banner_comes_after_every_refusal_and_before_the_run_directory(self):
        """It reports the run the cell is about to open: after the run-id, trunk and seed refusals, before the
        ``Run directory:`` line and the first write."""
        src = code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER)
        tree = ast.parse(src)
        [banner] = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "print" and "_banner" in ast.unparse(node)
        ]
        [run_directory] = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "print"
            and "Run directory:" in ast.unparse(node)
        ]
        lines = {
            name: min(
                node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == name
            )
            for name in (
                "refuse_run_id_of_another_tree",
                "refuse_uncertified_trunk",
                "refuse_widened_seed_mismatch",
                "_run_dir.mkdir",
                "initialize_result_bundle",
            )
        }
        refusals = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise) and "RUN_ID={RUN_ID!r} names no run of" in ast.unparse(node)
        ]
        assert len(refusals) == 2
        assert (
            lines["refuse_run_id_of_another_tree"] < min(refusals) < max(refusals) < lines["refuse_uncertified_trunk"]
        )
        assert lines["refuse_widened_seed_mismatch"] < banner < run_directory < lines["_run_dir.mkdir"]
        assert lines["_run_dir.mkdir"] < lines["initialize_result_bundle"]


# ---------------------------------------------------------------------------------------------- the prose
def test_the_run_id_knob_and_its_configuration_bullet_state_the_rules():
    sources = cell_sources(NOTEBOOK_PATH)
    config_at = cell_index(sources, CONFIG_CELL_MARKER)
    [knob] = [line for line in sources[config_at].splitlines() if line.startswith("RUN_ID = ")]
    assert (
        "a new id starts a fresh run (letters, digits, _ and - only; refused if, in this ALGORITHM's trees, it names "
        "only another model's run: SPECIES and EMBODIMENT pick the model)"
    ) in knob
    [bullet] = [line for line in sources[config_at - 1].splitlines() if line.startswith("- `RUN_ID`")]
    for phrase in (
        "`SPECIES` and `EMBODIMENT` select the model, never the id",
        "an id that names no run of this model but a run of another model under `LOG_BASE`, in the trees of this "
        "`ALGORITHM` (it names every run of that id)",
        "a new id with characters other than letters, digits, `_` and `-` (a run the tree already holds is "
        "re-entered whatever its characters)",
        "an explicit id prints a banner naming the model, the algorithm, the run tree and whether the run is new, and "
        "every other run of that id under `LOG_BASE`, of any model and either algorithm",
        "the maintainer's decisions of 2026-11-26",
    ):
        assert phrase in bullet, phrase
