"""The lint tools and the SB3 typing environment are pinned once (CU-1).

Wherever ruff and mypy are pinned, they name the same versions: the CI lint
job (both tools), the SB3 job's mypy step (mypy), .pre-commit-config.yaml and
pyproject.toml's ``dev`` extra (both). Otherwise pre-commit formats what CI
rejects (it did, with ruff 0.4.4 against CI's unpinned ruff), or a tool
release turns an unrelated pull request red. The SB3 job's stable-baselines3
pin is the SB3 notebook's, so its mypy step checks the SB3 the notebook trains
with. Every file read here is in both of the workflow's path filters, so a PR
that edits only one of them still runs these checks. Pre-commit's hooks never
touch the digest data files (the MJCF plant sources and meshes the plant
manifest records, the recipe TOMLs, the plant manifests, plant_versions.toml,
the recovery calibrations), and they still see every Python file; the
byte-hashed Python modules stay under the hooks, which CI's pinned ruff keeps
from changing them.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "python-ci.yml"
PRE_COMMIT_CONFIG = REPOSITORY_ROOT / ".pre-commit-config.yaml"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
SB3_NOTEBOOK = REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb"

_HOOK_REPOS = {
    "ruff": "https://github.com/astral-sh/ruff-pre-commit",
    "mypy": "https://github.com/pre-commit/mirrors-mypy",
}
_SB3_PIN = re.compile(r"stable-baselines3\[extra\]==([0-9][0-9.]*)")
PLANT_MANIFEST = REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json"
#: Digest data files besides the plant sources (which come from the committed
#: plant manifest): recipe_sha256's TOMLs, the manifests, plant_versions.toml
#: and the recovery calibrations, as globs under the repository root, each
#: expected to match at least once.
_DIGEST_DATA_GLOBS = (
    "environments/*/data/*.json",
    "configs/plant_manifest.generated.json",
    "configs/plant_versions.toml",
    "configs/*/recovery_calibration.json",
    "configs/*/behaviors/*.toml",
)


def _ci_text() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _ci_pins(tool: str) -> list[str]:
    return re.findall(rf'"{tool}==([0-9][0-9.]*)"', _ci_text())


def _pre_commit_block(tool: str) -> str:
    """The text of *tool*'s ``- repo:`` block in .pre-commit-config.yaml."""
    text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    start = text.index(f"repo: {_HOOK_REPOS[tool]}")
    end = text.find("- repo:", start)
    return text[start : end if end != -1 else len(text)]


def _pre_commit_rev(tool: str) -> str:
    match = re.search(r"^\s*rev:\s*v?([0-9][0-9.]*)\s*$", _pre_commit_block(tool), flags=re.MULTILINE)
    assert match is not None, f"no rev for the {tool} hook repository in {PRE_COMMIT_CONFIG.name}"
    return match.group(1)


def test_every_ci_install_of_ruff_or_mypy_is_pinned_to_one_version() -> None:
    for line in _ci_text().splitlines():
        if "pip install" not in line:
            continue
        unpinned = re.findall(r"(?<![\w.-])(ruff|mypy)(?![\w.-])(?!==)", line)
        assert not unpinned, f"python-ci.yml installs {unpinned} without a version: {line.strip()}"
    ruff_pins, mypy_pins = _ci_pins("ruff"), _ci_pins("mypy")
    assert len(set(ruff_pins)) == 1, f"python-ci.yml pins ruff as {ruff_pins}; expected one version"
    assert len(set(mypy_pins)) == 1, f"python-ci.yml pins mypy as {mypy_pins}; expected one version"
    # The lint job installs mypy, and so does the SB3 job's mypy step.
    assert len(mypy_pins) >= 2, f"python-ci.yml pins mypy {len(mypy_pins)} time(s); the lint and SB3 jobs both need it"


def test_pre_commit_hooks_run_the_ci_versions() -> None:
    for tool in ("ruff", "mypy"):
        (ci_version,) = set(_ci_pins(tool))
        assert _pre_commit_rev(tool) == ci_version, (
            f".pre-commit-config.yaml runs {tool} {_pre_commit_rev(tool)}, CI pins {ci_version}"
        )


def test_pre_commit_ruff_hooks_cover_only_what_ci_checks() -> None:
    # CI runs `ruff check environments/` and `ruff format --check environments/`.
    # The pinned ruff also formats notebooks and Markdown code blocks, so an
    # unscoped hook would rewrite files CI never looks at.
    assert "ruff check environments/" in _ci_text()
    assert "ruff format --check environments/" in _ci_text()
    block = _pre_commit_block("ruff")
    hooks = re.findall(r"^\s*- id:\s*(\S+)", block, flags=re.MULTILINE)
    scoped = re.findall(r"^\s*files:\s*\^environments/\s*$", block, flags=re.MULTILINE)
    assert hooks and len(scoped) == len(hooks), f"each ruff hook ({hooks}) needs `files: ^environments/`"


def test_dev_extra_pins_the_ci_versions() -> None:
    with PYPROJECT.open("rb") as handle:
        dev = tomllib.load(handle)["project"]["optional-dependencies"]["dev"]
    for tool in ("ruff", "mypy"):
        (ci_version,) = set(_ci_pins(tool))
        assert f"{tool}=={ci_version}" in dev, f"pyproject.toml's dev extra must pin {tool}=={ci_version}: {dev}"


def test_sb3_job_pins_the_notebooks_stable_baselines3() -> None:
    notebook = json.loads(SB3_NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    notebook_pins = set(_SB3_PIN.findall(code))
    ci_pins = set(_SB3_PIN.findall(_ci_text()))
    assert len(notebook_pins) == 1, f"the SB3 notebook pins stable-baselines3 as {notebook_pins}"
    assert ci_pins == notebook_pins, f"python-ci.yml pins stable-baselines3 {ci_pins}, the notebook {notebook_pins}"


def _path_filters() -> list[list[str]]:
    """The ``paths:`` lists of python-ci.yml's triggers, one list per trigger."""
    filters: list[list[str]] = []
    current: list[str] | None = None
    for line in _ci_text().splitlines():
        stripped = line.strip()
        if stripped == "paths:":
            current = []
            filters.append(current)
        elif current is not None and stripped.startswith("- "):
            current.append(stripped[2:].strip().strip('"'))
        elif current is not None and stripped and not stripped.startswith("#"):
            current = None
    return filters


def _glob_matches(pattern: str, path: str) -> bool:
    """GitHub's path-filter glob: ``**`` crosses directories, ``*`` does not."""
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(regex, path) is not None


def test_every_file_these_checks_read_triggers_the_workflow() -> None:
    filters = _path_filters()
    assert len(filters) == 2, f"expected the push and pull_request path filters, found {len(filters)}"
    for path in (CI_WORKFLOW, PRE_COMMIT_CONFIG, PYPROJECT, SB3_NOTEBOOK):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in filters:
            assert any(_glob_matches(pattern, relative) for pattern in patterns), (
                f"{relative} is missing from a python-ci.yml paths filter, so a PR editing only it skips these checks"
            )


def _pre_commit_exclude() -> re.Pattern[str]:
    text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    match = re.search(r"^exclude:\s*'([^']+)'\s*$", text, flags=re.MULTILINE)
    assert match is not None, f"{PRE_COMMIT_CONFIG.name} needs a top-level exclude for the digest data files"
    return re.compile(match.group(1))


def _plant_source_paths() -> set[str]:
    """Every MJCF source and asset the committed plant manifest hashes (kept current by plant_contract --check)."""
    paths: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("logical_path"), str):
                paths.add(node["logical_path"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(PLANT_MANIFEST.read_text(encoding="utf-8")))
    return paths


def test_pre_commit_never_rewrites_a_digest_data_file() -> None:
    exclude = _pre_commit_exclude()
    plant_sources = _plant_source_paths()
    assert any(path.endswith(".xml") for path in plant_sources), "the plant manifest records no MJCF source"
    for relative in sorted(plant_sources):
        assert exclude.search(relative), f"pre-commit hooks may rewrite the plant source {relative}"
    for pattern in _DIGEST_DATA_GLOBS:
        files = [path for path in REPOSITORY_ROOT.glob(pattern) if path.is_file()]
        assert files, f"no file matches {pattern}; update _DIGEST_DATA_GLOBS"
        for path in files:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            assert exclude.search(relative), f"pre-commit hooks may rewrite the digest data file {relative}"


def test_pre_commit_exclude_leaves_every_python_file_to_the_hooks() -> None:
    # The ruff and mypy hooks mirror CI's checks of environments/; an exclude
    # that caught Python files would switch them off there silently.
    exclude = _pre_commit_exclude()
    caught = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "environments").rglob("*.py")
        if exclude.search(path.relative_to(REPOSITORY_ROOT).as_posix())
    ]
    assert not caught, f"the pre-commit exclude hides Python files from the ruff and mypy hooks: {caught[:5]}"
