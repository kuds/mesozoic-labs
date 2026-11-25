"""The one reader of the package version (``config.get_library_version``, decision D-D63 (4)).

A source checkout reports its ``pyproject.toml`` version before ``importlib.metadata``'s, so a stale editable
install records the checkout's version; a non-checkout falls back to the metadata, and a foreign project's
``pyproject.toml`` is ignored.  Every record of a run reads the version there: ``stage_config.json``,
``provenance.json``'s ``dependency_versions["mesozoic_labs"]`` and the ``metrics.json`` payload agree.
"""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import json
import textwrap
import tomllib
from pathlib import Path

import pytest

from environments.shared import config as config_module
from environments.shared.config import DISTRIBUTION_NAME, get_library_version, save_stage_config
from environments.shared.model_id import ModelId
from environments.shared.result_bundle import provenance as result_bundle_provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
#: An installed copy's metadata, older than any checkout here: the stale editable install.
_STALE_METADATA_VERSION = "0.0.1.dev0"


def _write_pyproject(root: Path, *, name: str = DISTRIBUTION_NAME, version: str | None = "9.8.7.dev6") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = ["[project]", f'name = "{name}"']
    if version is not None:
        lines.append(f'version = "{version}"')
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _metadata(monkeypatch: pytest.MonkeyPatch, version: str | None) -> list[str]:
    """Patch ``importlib.metadata.version`` to report *version* for this package (``None``: not installed).

    Returns the names it was asked for, so a test can tell whether the metadata was read at all."""
    asked: list[str] = []
    real = importlib.metadata.version

    def fake(name: str) -> str:
        if name != DISTRIBUTION_NAME:
            return real(name)
        asked.append(name)
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(importlib.metadata, "version", fake)
    return asked


def test_a_checkout_with_a_stale_metadata_version_reports_its_pyproject_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_REPO_ROOT", _write_pyproject(tmp_path / "checkout", version="1.2.3"))
    asked = _metadata(monkeypatch, _STALE_METADATA_VERSION)
    assert get_library_version() == "1.2.3"
    assert asked == []


def test_this_checkout_reports_its_own_pyproject_version_over_the_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject:
        declared = tomllib.load(pyproject)["project"]["version"]
    _metadata(monkeypatch, _STALE_METADATA_VERSION)
    assert get_library_version() == declared


def test_a_non_checkout_falls_back_to_the_installed_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(config_module, "_REPO_ROOT", site_packages)
    asked = _metadata(monkeypatch, "2.0.0")
    assert get_library_version() == "2.0.0"
    assert asked == [DISTRIBUTION_NAME]


def test_a_non_checkout_without_metadata_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_REPO_ROOT", tmp_path)
    _metadata(monkeypatch, None)
    assert get_library_version() == "unknown"


@pytest.mark.parametrize("foreign_name", ["some-other-project", "mesozoic", "mesozoic-labs-extras"])
def test_a_foreign_pyproject_name_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_name: str
) -> None:
    monkeypatch.setattr(config_module, "_REPO_ROOT", _write_pyproject(tmp_path / "parent", name=foreign_name))
    _metadata(monkeypatch, "2.0.0")
    assert get_library_version() == "2.0.0"


@pytest.mark.parametrize("spelling", ["Mesozoic-Labs", "mesozoic_labs", "mesozoic.labs"])
def test_the_project_name_is_compared_in_its_normal_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    monkeypatch.setattr(config_module, "_REPO_ROOT", _write_pyproject(tmp_path / "checkout", name=spelling))
    _metadata(monkeypatch, _STALE_METADATA_VERSION)
    assert get_library_version() == "9.8.7.dev6"


@pytest.mark.parametrize(
    "body",
    [
        b"[project\nname = ",  # malformed TOML
        b'[project]\nname = "mesozoic-labs"\nversion = "1.0"\ndescription = "caf\xe9"\n',  # not UTF-8
        b'[tool.other]\nname = "mesozoic-labs"\n',  # no [project] table
        b'[project]\nname = "mesozoic-labs"\ndynamic = ["version"]\n',  # a dynamic version
        b'[project]\nname = "mesozoic-labs"\nversion = ""\n',  # an empty version
    ],
    ids=["malformed", "non-utf8", "no-project-table", "dynamic-version", "empty-version"],
)
def test_an_unreadable_checkout_version_falls_back_to_the_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    (tmp_path / "pyproject.toml").write_bytes(body)
    monkeypatch.setattr(config_module, "_REPO_ROOT", tmp_path)
    _metadata(monkeypatch, "2.0.0")
    assert get_library_version() == "2.0.0"


def test_every_record_of_a_run_carries_the_one_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``stage_config.json``, ``provenance.json`` and the ``metrics.json`` payload all read ``get_library_version``,
    so a stale editable install's metadata reaches none of them."""
    monkeypatch.setattr(config_module, "_REPO_ROOT", _write_pyproject(tmp_path / "checkout", version="3.4.5"))
    _metadata(monkeypatch, _STALE_METADATA_VERSION)

    config = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {}, "sac_kwargs": {}, "curriculum_kwargs": {}}
    written = save_stage_config(tmp_path / "01_stance", 1, config, "PPO", model_id=VELOCIRAPTOR_MONGOLIENSIS)
    assert json.loads(written.read_text(encoding="utf-8"))["library_version"] == "3.4.5"
    assert result_bundle_provenance._dependency_versions()["mesozoic_labs"] == "3.4.5"


def test_provenance_records_no_version_where_the_reader_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dependency_versions`` keeps its ``None`` for an absent package: the reader's ``"unknown"`` is not a version."""
    monkeypatch.setattr(config_module, "_REPO_ROOT", tmp_path)
    _metadata(monkeypatch, None)
    assert result_bundle_provenance._dependency_versions()["mesozoic_labs"] is None


def _library_version_values(tree: ast.AST) -> list[ast.expr]:
    """Every value a dict literal or a subscript assignment in *tree* stores under ``"library_version"``."""
    values: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            values += [
                value
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and key.value == "library_version"
            ]
    return values


def test_the_metrics_payload_reads_the_one_version() -> None:
    """``train_base._report_hpt_metrics`` records ``get_library_version()``, not a version it reads itself."""
    from environments.shared import train_base

    values = _library_version_values(ast.parse(textwrap.dedent(inspect.getsource(train_base._report_hpt_metrics))))
    assert values, "the metrics payload records no library_version"
    for value in values:
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name), ast.dump(value)
        assert value.func.id == "get_library_version"


def test_no_other_library_module_reads_the_package_version_from_the_metadata() -> None:
    """Only ``config.get_library_version`` asks ``importlib.metadata`` for this package's version."""
    offenders: list[str] = []
    for path in sorted((REPOSITORY_ROOT / "environments").rglob("*.py")):
        if "tests" in path.relative_to(REPOSITORY_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {DISTRIBUTION_NAME, "mesozoic_labs"}
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "version")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "version")
                )
            ):
                offenders.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    assert offenders == []
