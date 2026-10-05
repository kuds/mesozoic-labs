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
the recovery calibrations), and they still see every Python file: no digest
hashes a Python file's bytes, and the policy-interface digests hash function
tokens, which CI's pinned ruff keeps from changing. Ruff itself skips the
frozen MJX core (pyproject.toml's extend-exclude, D-D17), which
test_plant_contract_frozen_mjx.py checks.

The workflow's structure is pinned here too (CU-14a). The pull_request trigger
reuses the push trigger's path list by a YAML alias, so the two cannot drift.
The test matrix's suites run every directory of pyproject.toml's testpaths
exactly once, installing the ``test`` extra, under the check names
``test (<python>, <suite>)``. At reduced depth the SB3 job's integration step
leaves out exactly the compsognathus_robot parametrisations of
test_compsognathus_training.py (decision 10 (a)), which the nightly schedule, a
manual dispatch and the ``full-ci`` label run.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

from .ci_workflow_helpers import CI_WORKFLOW, ci_text, glob_matches, job_block, job_steps, path_filters
from .notebook_cells import code_cell_sources

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRE_COMMIT_CONFIG = REPOSITORY_ROOT / ".pre-commit-config.yaml"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
SB3_NOTEBOOK = REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb"
COMPSOGNATHUS_TRAINING = REPOSITORY_ROOT / "environments" / "shared" / "tests" / "test_compsognathus_training.py"

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


def _ci_pins(tool: str) -> list[str]:
    return re.findall(rf'"{tool}==([0-9][0-9.]*)"', ci_text())


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
    for line in ci_text().splitlines():
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
    assert "ruff check environments/" in ci_text()
    assert "ruff format --check environments/" in ci_text()
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
    code = "".join(code_cell_sources(SB3_NOTEBOOK))
    notebook_pins = set(_SB3_PIN.findall(code))
    ci_pins = set(_SB3_PIN.findall(ci_text()))
    assert len(notebook_pins) == 1, f"the SB3 notebook pins stable-baselines3 as {notebook_pins}"
    assert ci_pins == notebook_pins, f"python-ci.yml pins stable-baselines3 {ci_pins}, the notebook {notebook_pins}"


def test_every_file_these_checks_read_triggers_the_workflow() -> None:
    filters = path_filters()
    assert len(filters) == 2, f"expected the push and pull_request path filters, found {len(filters)}"
    for path in (CI_WORKFLOW, PRE_COMMIT_CONFIG, PYPROJECT, SB3_NOTEBOOK, COMPSOGNATHUS_TRAINING):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in filters:
            assert any(glob_matches(pattern, relative) for pattern in patterns), (
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


def test_both_triggers_share_one_path_filter() -> None:
    # CU-14a: pull_request names push's anchored list by its alias, so a pattern added to one is in both.
    push, pull_request = path_filters()
    assert push, "python-ci.yml's push trigger filters on no path"
    assert push is pull_request, "the pull_request trigger must reuse the push trigger's paths (paths: *name)"


def test_the_test_matrix_runs_every_test_directory_once() -> None:
    # CU-14a: two suites per Python version, which together run each testpaths directory exactly once.
    with PYPROJECT.open("rb") as handle:
        testpaths = tomllib.load(handle)["tool"]["pytest"]["ini_options"]["testpaths"]
    job = job_block("test")
    listed = re.findall(r"environments/\w+/tests/", job)
    assert len(listed) == len(set(listed)), f"a test directory runs in two suites: {listed}"
    assert sorted(path.rstrip("/") for path in listed) == sorted(testpaths), (
        f"the test matrix's suites run {listed}; pyproject.toml's testpaths are {testpaths}"
    )
    # Each suite gets its `tests` from one include entry: a suite without one would run pytest on no path,
    # that is on every testpaths directory.
    (suites,) = re.findall(r"^\s+suite: \[([^\]]*)\]\s*$", job, re.MULTILINE)
    included = re.findall(r"^\s+- suite: (\S+)\s*\n\s+tests:", job, re.MULTILINE)
    assert sorted(included) == sorted(suite.strip() for suite in suites.split(",")), (suites, included)
    assert re.search(r"^\s+run: pytest \$\{\{ matrix\.tests \}\} ", job, re.MULTILINE), (
        "pytest must run the suite's tests"
    )
    assert 'pip install -e ".[test]"' in job, "the test matrix installs the test extra, not dev"
    # The check names a branch protection rule would require: `test (3.11, shared)` as before CU-14a.
    assert "\n    name: test (${{ matrix.python-version }}, ${{ matrix.suite }})\n" in job, (
        "the test job's explicit name keeps its check names `test (<python>, <suite>)`"
    )


#: The depth idiom of the SB3 job's -k selections: *value* at reduced depth, nothing at full depth.
_REDUCED_DEPTH_ONLY = re.compile(r"\$\{\{ steps\.depth\.outputs\.full != 'true' && '([^']*)' \|\| '' \}\}")


def _selection_at_depth(selection: str, *, full: bool) -> str:
    rendered = _REDUCED_DEPTH_ONLY.sub(lambda match: "" if full else match.group(1), selection)
    assert "${{" not in rendered, f"an expression the depth idiom does not cover: {selection}"
    return rendered


def test_reduced_depth_leaves_out_only_the_robot_training_runs() -> None:
    # Decision 10 (a), CU-14a: pull requests and pushes skip the compsognathus_robot parametrisations of
    # test_compsognathus_training.py's training tests and keep the compsognathus ones; the full depth runs both.
    # A bare `not compsognathus_robot` would also drop the robot's parameters in the step's other files.
    job = job_block("test-sb3")
    for full in ("true", "false"):
        assert f'echo "full={full}" >> "$GITHUB_OUTPUT"' in job, f"the SB3 depth step no longer sets full={full}"
    (step,) = [candidate for candidate in job_steps("test-sb3") if "name: Run SB3 integration tests\n" in candidate]
    assert "environments/shared/tests/test_compsognathus_training.py" in step
    # Either quote style; the selection holds no quote of its own kind.
    ((_, selection),) = re.findall(r"""^\s+-k (["'])(.*)\1 \\$""", step, re.MULTILINE)
    notebook = "not test_actual_notebook_training_stance_and_recovery_reports"
    assert _selection_at_depth(selection, full=True) == notebook
    assert _selection_at_depth(selection, full=False) == (
        f"{notebook} and not (test_compsognathus_training and compsognathus_robot)"
    )
    # The -k terms are the module's name and the robot's parameter id, which the compsognathus id does not contain.
    (species,) = [
        ast.literal_eval(node.value)
        for node in ast.parse(COMPSOGNATHUS_TRAINING.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["SPECIES"]
    ]
    assert set(species) == {"compsognathus", "compsognathus_robot"}, species
