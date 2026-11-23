"""The wheel's package data (pyproject.toml ``[tool.setuptools.package-data]``).

Of the species' ``data/`` directories the wheel ships only the files runtime code reads: the Compsognathus
model parameters (``compsognathus_longipes/model.py``, imported by its env) and the bundled plant manifest
(``plant_contract.constants.BUNDLED_MANIFEST_PATH``).  The rest of each ``data/`` is evidence the docs link to,
read from a checkout; a glob such as ``data/*.json`` would ship about 3 MB of it again.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from environments.shared.paths import ENVIRONMENTS_ROOT, REPOSITORY_ROOT
from environments.shared.plant_contract.constants import BUNDLED_MANIFEST_PATH

COMPSOGNATHUS_LONGIPES_PARAMETERS = ENVIRONMENTS_ROOT / "compsognathus_longipes" / "data" / "model_parameters.json"


def _packages() -> dict[str, Path]:
    """Every package ``packages.find`` takes from ``environments``: a directory with an ``__init__.py``."""
    return {
        ".".join(init.parent.relative_to(REPOSITORY_ROOT).parts): init.parent
        for init in sorted(ENVIRONMENTS_ROOT.rglob("__init__.py"))
    }


def _shipped_data(package_data: dict[str, Any], packages: dict[str, Path]) -> set[Path]:
    """The files under a package's ``data/`` directory that *package_data* matches (``"*"``: every package)."""
    shipped: set[Path] = set()
    for key, patterns in package_data.items():
        if key == "configs":
            continue
        assert key == "*" or key in packages, f"package-data names {key}, which is not a package"
        for directory in packages.values() if key == "*" else [packages[key]]:
            for pattern in patterns:
                shipped.update(
                    path
                    for path in directory.glob(pattern)
                    if path.is_file() and path.relative_to(directory).parts[0] == "data"
                )
    return shipped


def _setuptools() -> dict[str, Any]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return dict(tomllib.load(handle)["tool"]["setuptools"])


def _package_data() -> dict[str, Any]:
    return dict(_setuptools()["package-data"])


def test_package_data_is_the_whole_list() -> None:
    """With include-package-data on, a stale egg-info's SOURCES.txt or setuptools-scm puts every tracked file back."""
    assert _setuptools()["include-package-data"] is False


def test_the_wheel_ships_only_the_data_files_runtime_code_reads() -> None:
    assert _shipped_data(_package_data(), _packages()) == {BUNDLED_MANIFEST_PATH, COMPSOGNATHUS_LONGIPES_PARAMETERS}


def test_the_shipped_data_files_are_the_ones_runtime_code_reads() -> None:
    from environments.compsognathus_longipes import model

    assert model.PARAMETERS == json.loads(COMPSOGNATHUS_LONGIPES_PARAMETERS.read_text())
    assert BUNDLED_MANIFEST_PATH.is_file()


def test_a_data_glob_would_ship_the_evidence_files() -> None:
    """Self-test: the scan sees what a glob over every package's ``data/`` would add."""
    shipped = _shipped_data({"*": ["data/*.json"]}, _packages())
    assert {BUNDLED_MANIFEST_PATH, COMPSOGNATHUS_LONGIPES_PARAMETERS} < shipped
    assert ENVIRONMENTS_ROOT / "compsognathus_longipes" / "data" / "recovery_calibration_v1.json" in shipped
