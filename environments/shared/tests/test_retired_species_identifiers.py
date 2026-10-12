"""No living code, config or notebook names a model by a retired identifier (decision D-D33).

Every species is named by its binomial and every model is the pair (species,
embodiment): ``velociraptor`` became ``velociraptor_mongoliensis/anatomical``,
``trex`` ``tyrannosaurus_rex/anatomical``, ``brachiosaurus``
``brachiosaurus_altithorax/anatomical``, ``dibothrosuchus``
``dibothrosuchus_elaphros/anatomical``, ``compsognathus``
``compsognathus_longipes/anatomical`` and ``compsognathus_robot``
``compsognathus_longipes/robot``; the ``biological`` embodiment is
``anatomical``.  Nothing resolves a retired ID, an alias, a display name or a
case or separator variant, and nothing names one:

* the identifiers, attribute names, def and class names, arguments,
  keywords, import paths and string constants (docstrings aside) of every
  ``environments/**/*.py`` file and of every notebook code cell;
* the values of every TOML file under ``configs/``;
* the keys and values of the recovery calibrations.

Exempt: docstrings and comments (prose may name an animal, "the T-Rex"),
``fixtures/``, this module (which holds the refusal cases below), the
retired schema name ``mesozoic.trex-command-terrain/v1`` in the test that
proves it refused, a hyphenated binomial (``compsognathus-longipes-sensors/v1``:
a genus followed by ``-`` and its own epithet), and recorded history: each
calibration's ``restamp_history`` (the identities and reasons it recorded) and
a test string quoting one of those reasons verbatim.

The boundary pattern is the plan's guard prototype, verbatim: a plain ``\\b``
would let ``test_trex_bite`` pass, because ``_`` is a word character.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from environments.shared.gait.morphology import Morphology, support_entry
from environments.shared.model_id import ModelId
from environments.shared.species_names import require_model_id, species_display_name
from environments.shared.species_registry import env_class, get_species_config
from environments.shared.train_behaviors import read_recipe

from .notebook_cells import code_cells, strip_magics

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = REPOSITORY_ROOT / "environments"
CONFIGS = REPOSITORY_ROOT / "configs"

RETIRED = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:compsognathus_robot|velociraptor|trex|brachiosaurus|dibothrosuchus|compsognathus)(?![a-z0-9]|_(?:mongoliensis|rex|altithorax|elaphros|longipes))"
    r"|raptor(?=_|\b|[A-Z])|t-rex|t_rex|brachio(?!saurus)|dibo(?!thro)|compso(?!gnathus)|biological"
    r")"
    r"|(?:Raptor|TRex|Brachio(?!saurus)|Dibo(?!thro)|Compso(?!gnathus)|Biological)(?=[A-Z_]|Env|-v\d|\b)",
)
#: Each retired genus ID's epithet: the genus followed by ``-<epithet>`` is a hyphenated binomial, not the old ID.
EPITHETS = {
    "velociraptor": "mongoliensis",
    "brachiosaurus": "altithorax",
    "dibothrosuchus": "elaphros",
    "compsognathus": "longipes",
}
#: Retired names kept where they are: the schema a refusal test proves is refused (consolidation PR-7, D-D9).
KEPT = {("environments/shared/tests/test_behavior_checkpoint.py", "mesozoic.trex-command-terrain/v1")}

#: Every retired species ID, alias, display name and case or separator variant; none resolves.
ALIASES = (
    "trex",
    "t-rex",
    "T-Rex",
    "t_rex",
    "TRex",
    "raptor",
    "velociraptor",
    "brachio",
    "brachiosaurus",
    "dibo",
    "dibothrosuchus",
    "compso",
    "compso-robot",
    "compsognathus",
    "compsognathus_robot",
    "Tyrannosaurus Rex",
    "Velociraptor Mongoliensis",
    "Compsognathus Longipes (Robot)",
    "Tyrannosaurus_Rex",
    "TYRANNOSAURUS_REX",
    "tyrannosaurus-rex",
    "tyrannosaurus rex",
    " tyrannosaurus_rex",
    "tyrannosaurus_rex ",
)
#: The aliases ``ModelId`` itself refuses, as no binomial; the binomial-shaped rest are refused by the manifest.
NON_BINOMIAL_ALIASES = tuple(alias for alias in ALIASES if alias not in ("t_rex", "compsognathus_robot"))
#: Retired and variant embodiment names.
EMBODIMENT_ALIASES = ("biological", "Biological", "Anatomical", "robot_rev_b", "")


# ── the scan ─────────────────────────────────────────────────────────────────


def retired_names(text: str) -> list[str]:
    """Each retired identifier in *text*, a hyphenated binomial's genus excepted."""
    names = []
    for match in RETIRED.finditer(text):
        epithet = EPITHETS.get(match.group(0))
        if epithet is not None and text.startswith(f"-{epithet}", match.end()):
            continue
        names.append(match.group(0))
    return names


def _python_names(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, text)`` of each identifier, import path and non-docstring string constant of *tree*."""
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield line, node.value
        elif isinstance(node, ast.Name):
            yield line, node.id
        elif isinstance(node, ast.Attribute):
            yield line, node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield line, node.name
        elif isinstance(node, ast.arg):
            yield line, node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield line, node.arg
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                yield line, node.module
            for alias in node.names:
                yield line, alias.name
                if alias.asname:
                    yield line, alias.asname


def _calibration_paths() -> list[Path]:
    return sorted(CONFIGS.glob("*/*/recovery_calibration.json"))


def _recorded_reasons() -> set[str]:
    """Every reason a committed calibration's ``restamp_history`` records."""
    return {
        str(entry["reason"])
        for path in _calibration_paths()
        for entry in json.loads(path.read_text(encoding="utf-8")).get("restamp_history", [])
        if "reason" in entry
    }


def _python_files() -> list[Path]:
    this_file = Path(__file__).resolve()
    return sorted(
        path
        for path in ENVIRONMENTS.rglob("*.py")
        if path.resolve() != this_file and "fixtures" not in path.relative_to(ENVIRONMENTS).parts
    )


def _source_hits(shown: str, tree: ast.AST, recorded: set[str], *, offset: str = "") -> list[str]:
    return [
        f"{shown}{offset}:{line}: {name!r} in {text[:100]!r}"
        for line, text in _python_names(tree)
        if (shown, text) not in KEPT and text not in recorded
        for name in retired_names(text)
    ]


def _toml_values(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _toml_values(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _toml_values(item, (*path, index))
    elif isinstance(value, str):
        yield path, value


def _calibration_texts(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], str]]:
    """Each key and string value of a calibration, its recorded history (``restamp_history[*]``'s previous identity
    and reason) aside."""
    if len(path) == 3 and path[0] == "restamp_history" and path[2] in ("previous_identity", "reason"):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield (*path, key), str(key)
            yield from _calibration_texts(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _calibration_texts(item, (*path, index))
    elif isinstance(value, str):
        yield path, value


def test_no_python_source_names_a_retired_identifier() -> None:
    recorded = _recorded_reasons()
    hits = [
        hit
        for path in _python_files()
        for hit in _source_hits(
            path.relative_to(REPOSITORY_ROOT).as_posix(), ast.parse(path.read_text(encoding="utf-8")), recorded
        )
    ]
    assert not hits, (
        "these name a model by a retired identifier (decision D-D33): use the binomial species and the "
        "embodiment (ModelId), or a display name in a docstring or comment\n" + "\n".join(hits)
    )


def test_no_notebook_code_cell_names_a_retired_identifier() -> None:
    notebooks = sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    hits = [
        hit
        for notebook in notebooks
        for index, source in code_cells(notebook)
        for hit in _source_hits(
            notebook.relative_to(REPOSITORY_ROOT).as_posix(),
            ast.parse(strip_magics(source)),
            set(),
            offset=f" cell {index}",
        )
    ]
    assert not hits, "these notebook code cells name a model by a retired identifier (D-D33)\n" + "\n".join(hits)


def test_no_config_value_names_a_retired_identifier() -> None:
    paths = sorted(CONFIGS.rglob("*.toml"))
    assert CONFIGS / "tyrannosaurus_rex" / "anatomical" / "stance.toml" in paths
    hits = [
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}: {'.'.join(map(str, key))} = {value!r}"
        for path in paths
        for key, value in _toml_values(tomllib.loads(path.read_text(encoding="utf-8")))
        if retired_names(value)
    ]
    assert not hits, "these config values name a model by a retired identifier (D-D33)\n" + "\n".join(hits)


def test_no_recovery_calibration_names_a_retired_identifier_outside_its_recorded_history() -> None:
    paths = _calibration_paths()
    assert [path.relative_to(CONFIGS).parent.as_posix() for path in paths] == [
        "compsognathus_longipes/anatomical",
        "compsognathus_longipes/robot",
    ]
    hits = [
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}: {'.'.join(map(str, key))}: {text!r}"
        for path in paths
        for key, text in _calibration_texts(json.loads(path.read_text(encoding="utf-8")))
        if retired_names(text)
    ]
    assert not hits, "these calibration entries name a model by a retired identifier (D-D33)\n" + "\n".join(hits)
    # The recorded history is what is exempt, and it does record the old identities.
    histories = [json.loads(path.read_text(encoding="utf-8"))["restamp_history"] for path in paths]
    assert all(any(retired_names(json.dumps(entry)) for entry in history) for history in histories)


def test_the_pattern_catches_a_retired_identifier_at_any_boundary_and_spares_the_binomials() -> None:
    for retired in (
        "trex",
        "test_trex_bite",
        "TRexEnv",
        "RaptorEnv",
        "raptor_env",
        "MesozoicLabs/Raptor-v0",
        "configs/velociraptor/stance.toml",
        "BrachioEnv",
        "brachio",
        "dibo",
        "DiboEnv",
        "compso",
        "compsognathus_robot",
        "CompsognathusBiologicalEnv",
        "biological",
        "t_rex",
        "t-rex",
        "compsognathus",
        "compsognathus_anatomical",
    ):
        assert retired_names(retired), retired
    for current in (
        "tyrannosaurus_rex",
        "velociraptor_mongoliensis",
        "brachiosaurus_altithorax",
        "dibothrosuchus_elaphros",
        "compsognathus_longipes",
        "TyrannosaurusRexAnatomicalEnv",
        "VelociraptorMongoliensisEnv",
        "MesozoicLabs/CompsognathusLongipes-Robot-v0",
        "Velociraptor",
        "T-Rex",
        "the T-Rex's stance",
        "compsognathus-longipes-sensors/v1",
        "mesozoic.compsognathus-longipes-preflight/v3",
        "trexler",
        "anatomical",
    ):
        assert not retired_names(current), current


def test_the_scan_reaches_its_files_and_exempts_only_what_it_names() -> None:
    names = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in _python_files()}
    assert {
        "environments/shared/train.py",
        "environments/shared/tests/test_model_id.py",
        "environments/compsognathus_longipes/robot/env.py",
    } <= names
    assert Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix() not in names
    assert not any("/fixtures/" in name for name in names)
    tree = ast.parse('"""A docstring naming the trex."""\n# a comment naming the raptor\nX = "velociraptor"\n')
    assert [text for _, text in _python_names(tree) if retired_names(text)] == ["velociraptor"]
    # A kept name is kept only in its own file.
    kept = ast.parse('SCHEMA = "mesozoic.trex-command-terrain/v1"\n')
    assert _source_hits("environments/shared/tests/test_behavior_checkpoint.py", kept, set()) == []
    assert _source_hits("environments/shared/other.py", kept, set())


# ── nothing resolves a retired identifier ───────────────────────────────────


@pytest.mark.parametrize("alias", ALIASES)
def test_require_model_id_refuses_every_alias(alias: str) -> None:
    with pytest.raises(ValueError, match="Unknown species"):
        require_model_id(alias, "anatomical")
    with pytest.raises(ValueError, match="Unknown species"):
        species_display_name(alias)


@pytest.mark.parametrize("embodiment", EMBODIMENT_ALIASES)
def test_require_model_id_refuses_every_embodiment_alias(embodiment: str) -> None:
    with pytest.raises(ValueError, match="has no .* embodiment"):
        require_model_id("compsognathus_longipes", embodiment)
    with pytest.raises(ValueError, match="embodiment must be one of anatomical, robot"):
        ModelId("compsognathus_longipes", embodiment)


@pytest.mark.parametrize("alias", NON_BINOMIAL_ALIASES)
def test_a_model_id_and_a_record_refuse_every_non_binomial_alias(alias: str) -> None:
    with pytest.raises(ValueError, match="binomial"):
        ModelId(alias, "anatomical")
    with pytest.raises(ValueError, match="binomial"):
        ModelId.from_record({"species": alias, "embodiment": "anatomical"})


@pytest.mark.parametrize("alias", ALIASES)
def test_the_registry_resolves_no_alias(alias: str) -> None:
    with pytest.raises(ValueError, match="binomial|Unknown species"):
        get_species_config(ModelId(alias, "anatomical"))
    with pytest.raises(ValueError, match="binomial|Unknown species"):
        env_class(ModelId(alias, "anatomical"))


@pytest.mark.parametrize("alias", ALIASES)
def test_the_training_command_line_refuses_every_alias(alias: str, monkeypatch, capsys) -> None:
    from environments.shared import train as train_entry

    monkeypatch.setattr(sys, "argv", ["train.py", "--species", alias, "--embodiment", "anatomical", "train"])
    with pytest.raises(SystemExit) as excinfo:
        train_entry._main()
    assert excinfo.value.code == 2
    assert f"argument --species: invalid choice: {alias!r}" in capsys.readouterr().err


@pytest.mark.parametrize("embodiment", ["biological", "Anatomical"])
def test_the_training_command_line_refuses_every_embodiment_alias(embodiment: str, monkeypatch, capsys) -> None:
    from environments.shared import train as train_entry

    argv = ["train.py", "--species", "compsognathus_longipes", "--embodiment", embodiment, "train"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        train_entry._main()
    assert excinfo.value.code == 2
    assert f"argument --embodiment: invalid choice: {embodiment!r}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("species", "embodiment"),
    [*((alias, "anatomical") for alias in ALIASES), ("tyrannosaurus_rex", "biological")],
)
def test_a_behavior_recipe_naming_an_alias_is_refused(species: str, embodiment: str, tmp_path) -> None:
    recipe = tmp_path / "walk.toml"
    recipe.write_text(
        "[behavior]\n"
        f"species = {json.dumps(species)}\n"
        f"embodiment = {json.dumps(embodiment)}\n"
        'name = "walk"\nparent = "locomotion"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_recipe(recipe, ModelId("tyrannosaurus_rex", "anatomical"))


def test_the_foot_support_registry_has_no_entry_for_an_alias() -> None:
    from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

    env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
    try:
        with pytest.raises((AttributeError, ValueError)):
            Morphology.from_env(env, "raptor")  # type: ignore[arg-type]
        for alias in ("t_rex", "compsognathus_robot"):
            with pytest.raises(ValueError, match="Unknown species"):
                support_entry(ModelId(alias, "anatomical"))
    finally:
        env.close()
