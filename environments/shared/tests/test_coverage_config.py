"""Coverage counts the certification code; what it omits is named (cleanup CU-9).

Until CU-9, ``[tool.coverage.run] omit`` in pyproject.toml held the blanket
``*/scripts/*`` and ``environments/shared/harnesses/*``, which kept the
recovery gate's freeze producer, ``widen_checkpoint.py`` and
``backfill_gate_verdict.py`` out of the union that ``fail_under`` gates on.
These tests keep the list explicit:

* besides ``*/tests/*``, every entry names ``.py`` files, with a ``*`` only
  for the species directory, and matches a file that exists, so a file
  added under ``scripts/`` or ``harnesses/`` counts until an entry names it;
* the certification code, and the statue baselines that measure the
  stage-1 gate constants, count;
* ``harnesses/digest_snapshot.py`` stays omitted by name (its own CI step
  checks it);
* no omitted file is imported by a test, so tested code is never hidden;
  ``digest_snapshot.py`` is the one named exception;
* the omit list is the only way out of the union: no report-time ``omit``,
  no ``include``, ``source`` is the whole package, and no other coverage
  config file overrides pyproject.toml.

An omit entry is a glob relative to the repository root, where CI runs
coverage, and coverage's ``*`` does not cross ``/``; ``Path.glob`` resolves
these entries the same way.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
TESTS_GLOB = "*/tests/*"
DIGEST_SNAPSHOT = "environments/shared/harnesses/digest_snapshot.py"
#: The certification code the CU-9 row names, and the rest of the recovery
#: gate's chain that lives outside the library: each must be measured.
CERTIFICATION_CODE = (
    "environments/shared/harnesses/freeze_recovery_gate.py",
    "environments/shared/scripts/widen_checkpoint.py",
    "environments/shared/scripts/backfill_gate_verdict.py",
    "environments/shared/scripts/restamp_recovery_calibration.py",
    "environments/compsognathus/scripts/calibrate_recovery.py",
)
#: The statue baselines: re-running them is the only evidence for the
#: statue-derived stage-1 gate constants (test_statue_constant_freshness.py).
GATE_CONSTANT_EVIDENCE = (
    "environments/shared/scripts/zero_action_baseline.py",
    "environments/shared/scripts/stance_quality_baseline.py",
)
#: Files coverage reads its settings from in preference to pyproject.toml
#: (the latter two only when they hold a ``[coverage:*]`` section).
COVERAGERC = ".coveragerc"
OTHER_COVERAGE_CONFIGS = ("setup.cfg", "tox.ini")
_GLOB_CHARACTERS = frozenset("*?[]")


def _coverage_config() -> dict[str, Any]:
    config: dict[str, Any] = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["coverage"]
    return config


def _omit() -> list[str]:
    return list(_coverage_config()["run"]["omit"])


def _named_entries() -> list[str]:
    return [entry for entry in _omit() if entry != TESTS_GLOB]


def _matches(entry: str) -> list[str]:
    return sorted(path.relative_to(REPOSITORY_ROOT).as_posix() for path in REPOSITORY_ROOT.glob(entry))


def _omitted_files() -> set[str]:
    return {path for entry in _named_entries() for path in _matches(entry)}


def _module_name(path: str) -> str:
    """The import name: a package's ``__init__.py`` is imported as the package."""
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _modules_tests_import() -> set[str]:
    """Every absolute module a test file imports (``from pkg import mod`` counts as ``pkg.mod``).

    String constants that are exactly a module name count too, for ``python
    -m`` subprocesses and ``importlib.import_module``.  Importing a module
    imports every package above it, so each dotted prefix counts as well.
    """
    imported: set[str] = set()
    for path in sorted(REPOSITORY_ROOT.glob("environments/**/tests/**/*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("environments.") and node.value.isascii() and " " not in node.value:
                    imported.add(node.value)
    return {".".join(name.split(".")[:end]) for name in imported for end in range(1, name.count(".") + 2)}


def test_the_blanket_omits_are_gone():
    omit = _omit()
    assert TESTS_GLOB in omit
    assert "*/scripts/*" not in omit
    assert "environments/shared/harnesses/*" not in omit
    assert len(omit) == len(set(omit)), "duplicate omit entries"


def test_every_entry_names_files_and_a_star_stands_only_for_the_species():
    for entry in _named_entries():
        parts = entry.split("/")
        assert parts[0] == "environments" and entry.endswith(".py"), entry
        assert not _GLOB_CHARACTERS & set(parts[-1]), f"{entry}: name each file; no wildcard in the file name"
        for index, part in enumerate(parts[:-1]):
            if _GLOB_CHARACTERS & set(part):
                assert index == 1 and part == "*", f"{entry}: a `*` may stand only for the species directory"


def test_every_entry_matches_an_existing_file():
    for entry in _named_entries():
        matches = _matches(entry)
        assert matches, f"{entry} matches no file: delete the stale omit entry"
        assert all((REPOSITORY_ROOT / path).is_file() for path in matches), entry


def test_the_digest_snapshot_stays_omitted_by_name():
    assert DIGEST_SNAPSHOT in _named_entries()
    assert (REPOSITORY_ROOT / DIGEST_SNAPSHOT).is_file()


def test_the_certification_code_counts():
    omitted = _omitted_files()
    for path in CERTIFICATION_CODE + GATE_CONSTANT_EVIDENCE:
        assert (REPOSITORY_ROOT / path).is_file(), path
        assert path not in omitted, f"{path} is certification or gate-constant code: coverage must measure it"


def test_no_omitted_file_is_imported_by_a_test():
    imported = _modules_tests_import()
    hidden = sorted(path for path in _omitted_files() if path != DIGEST_SNAPSHOT and _module_name(path) in imported)
    assert not hidden, f"a test imports {hidden}: delete their omit entries so coverage measures them"


def test_nothing_else_takes_a_file_out_of_the_union():
    config = _coverage_config()
    assert config["run"]["source"] == ["environments"]
    assert "include" not in config["run"], "measure all of environments/; omit names what is left out"
    assert not {"omit", "include"} & set(config.get("report", {})), "a report-time omit would hide measured files"
    assert not (REPOSITORY_ROOT / COVERAGERC).exists(), f"{COVERAGERC} would override pyproject.toml"
    for name in OTHER_COVERAGE_CONFIGS:
        path = REPOSITORY_ROOT / name
        assert not path.is_file() or "[coverage:" not in path.read_text(encoding="utf-8"), f"{name} overrides it too"
