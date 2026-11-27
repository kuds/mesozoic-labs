"""docs/DRIVE_LAYOUT.md stays in step with the code it maps (the maintainer's decision of 2026-11-26).

The page is a living reference: it names, for every file under ``LOG_BASE``,
the code that writes it and the code that reads it, by path and by dotted
name.  These tests fail when a cited module or name no longer exists, when a
model's stage directory is missing from the stage-level section, when the
run-state table drifts from ``reporting.run_index.RUN_STATES`` (or the index's
``unreadable`` group goes unnamed), when the role table drifts from
``run_index.ROLES``, or when a link to or from the page no longer resolves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from environments.shared.reporting import run_index
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PAGE = REPOSITORY_ROOT / "docs" / "DRIVE_LAYOUT.md"
SHARED = REPOSITORY_ROOT / "environments" / "shared"
#: Where a path or a dotted name the page cites is looked up, in this order.
SEARCH_ROOTS = (
    REPOSITORY_ROOT,
    SHARED,
    SHARED / "scripts",
    SHARED / "reporting",
    SHARED / "curriculum",
    SHARED / "harnesses",
)
#: The GitHub link the notebooks and the website use, which renders outside the repository.
GITHUB_PAGE = "https://github.com/kuds/mesozoic-labs/blob/main/docs/DRIVE_LAYOUT.md"


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """The body of the page's ``## <heading>`` section, up to the next ``## `` heading."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", _text(), re.M | re.S)
    assert match, f"DRIVE_LAYOUT.md has no '## {heading}' section"
    return match.group(1)


def _module_file(parts: "list[str]") -> "Path | None":
    """The file of the module ``.``-joined *parts* names, looked up under :data:`SEARCH_ROOTS`."""
    for root in SEARCH_ROOTS:
        candidate = root.joinpath(*parts)
        if candidate.with_suffix(".py").is_file():
            return candidate.with_suffix(".py")
        if (candidate / "__init__.py").is_file():
            return candidate / "__init__.py"
    return None


def _defines(module: Path, name: str) -> bool:
    """Whether *module* defines *name* at any level, or (a package) re-exports it."""
    source = module.read_text(encoding="utf-8")
    if module.name == "__init__.py":
        return re.search(rf"\b{re.escape(name)}\b", source) is not None
    pattern = rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]"
    return re.search(pattern, source, re.M) is not None


def test_every_python_file_the_page_cites_exists() -> None:
    cited = sorted(set(re.findall(r"`([\w./-]+\.py)`", _text())))
    assert cited
    missing = [path for path in cited if not any((root / path).is_file() for root in SEARCH_ROOTS)]
    assert not missing, f"DRIVE_LAYOUT.md cites files that do not exist: {missing}"


def test_every_dotted_name_the_page_cites_is_defined() -> None:
    """``module.name`` (``run_tree.run_kind_of``, ``result_bundle.read_bundle_status``) names a definition."""
    cited = sorted(set(re.findall(r"`((?:[a-z_][a-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*)`", _text())))
    assert cited
    unresolved = []
    for name in cited:
        parts = name.split(".")
        if parts[-1] in {"py", "md", "json", "csv", "txt", "npz", "pkl", "zip", "toml", "png", "mp4", "html"}:
            continue
        if _module_file(parts) is not None:
            continue
        module = _module_file(parts[:-1])
        if module is None or not _defines(module, parts[-1]):
            unresolved.append(name)
    assert not unresolved, f"DRIVE_LAYOUT.md cites names no module defines: {unresolved}"


@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_the_stage_level_names_every_stage_directory_of_the_model(model_id) -> None:
    section = _section("The stage level")
    for entry in load_stage_manifest(model_id).stages:
        name = stage_dirname(model_id, entry.id)
        assert f"`{name}/`" in section, f"{model_id}: the stage level does not name {name}/"


def test_the_run_states_table_follows_the_run_index() -> None:
    section = _section("The reports tree")
    table = section.split("| State | The run holds |", 1)[1].split("\n\n", 1)[0]
    states = re.findall(r"^\| `([a-z_]+)` \|", table, re.M)
    assert states == list(run_index.RUN_STATES), "the state table must list run_index.RUN_STATES, in order"
    # INDEX.md's groups are the states and then the one group that is no state.
    [extra] = [group for group in run_index.INDEX_STATES if group not in run_index.RUN_STATES]
    assert f"adds `{extra}` (`run_index.INDEX_STATES`)" in section
    for schema in (run_index.MODEL_INDEX_SCHEMA, run_index.RUN_INDEX_SCHEMA):
        assert f'"{schema}"' in section
    assert f"{run_index.REPORTS_DIRNAME}/" in section


def test_the_role_table_follows_the_run_index() -> None:
    """Every role ``run_index.ARTIFACT_ROLES`` assigns, and ``unclassified``, has a row, and no row names another."""
    section = _section("The reports tree")
    table = section.split("| Role | Files |", 1)[1].split("\n\n", 1)[0]
    roles = [role for cell in re.findall(r"^\| ([^|]+) \|", table, re.M) for role in re.findall(r"`([a-z_]+)`", cell)]
    assert sorted(roles) == sorted([*run_index.ROLES, run_index.UNCLASSIFIED])
    assert {role for _, _, role in run_index.ARTIFACT_ROLES} == set(run_index.ROLES)


def _markdown_anchors(path: Path) -> "set[str]":
    """The fragments a Markdown file serves: its ``<a id>`` anchors and GitHub's heading slugs."""
    anchors: set[str] = set()
    slugs: dict[str, int] = {}
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        anchors.update(re.findall(r'<a id="([^"]+)"', line))
        heading = re.match(r"#{1,6} +(.+?) *$", line)
        if heading:
            slug = re.sub(r"[^\w\- ]", "", heading.group(1).replace("`", "").lower()).replace(" ", "-")
            anchors.add(f"{slug}-{slugs[slug]}" if slug in slugs else slug)
            slugs[slug] = slugs.get(slug, 0) + 1
    return anchors


def test_links_from_and_to_the_page_resolve() -> None:
    """Every relative link on the page, and every link to it from the READMEs and docs/, reaches its target."""
    link = re.compile(r"\]\(([^)\s]+)\)")
    outbound = [(PAGE, target) for target in link.findall(_text())]
    sources = [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "CONTRIBUTING.md",
        REPOSITORY_ROOT / "results" / "README.md",
        *sorted((REPOSITORY_ROOT / "docs").glob("*.md")),
    ]
    inbound = [
        (source, target)
        for source in sources
        if source != PAGE
        for target in link.findall(source.read_text(encoding="utf-8"))
        if "DRIVE_LAYOUT.md" in target and not target.startswith(("http://", "https://"))
    ]
    assert outbound and inbound
    for source, target in outbound + inbound:
        if target.startswith(("http://", "https://")):
            continue
        path, _, fragment = target.partition("#")
        resolved = (source.parent / path).resolve() if path else source
        assert resolved.exists(), f"{source.name}: {target}"
        if fragment and resolved.suffix == ".md":
            assert fragment in _markdown_anchors(resolved), f"{source.name}: {target}"


@pytest.mark.parametrize(
    "relative",
    [
        "notebooks/sb3_training.ipynb",
        "notebooks/google_drive_summary.ipynb",
        "website/docs/training/recipes.md",
    ],
)
def test_pages_rendered_outside_the_repository_link_the_page_on_github(relative) -> None:
    """The notebooks (Colab) and the website link the page by its GitHub URL, which names a file that exists."""
    assert GITHUB_PAGE in (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
    assert (REPOSITORY_ROOT / GITHUB_PAGE.split("/blob/main/", 1)[1]).is_file()
