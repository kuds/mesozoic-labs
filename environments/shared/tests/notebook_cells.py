"""Notebook cell readers and helpers for the notebook tests, and CI's notebook parse check.

The module imports only the standard library and nothing relative, so CI's
lint job, which does not install the package, runs it by path:

    python environments/shared/tests/notebook_cells.py [NOTEBOOK ...]

With no arguments it checks every ``notebooks/*.ipynb`` of the repository this
file is in (found from the file's location, not the working directory). Every
code cell must parse once its IPython magic and shell lines are replaced by
``pass``, the way nbconvert strips them; nothing is executed.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_notebook(path: Path) -> dict[str, Any]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return notebook


def cell_source(cell: dict[str, Any]) -> str:
    """A cell's source as one string (nbformat stores a list of lines or one string)."""
    return "".join(cell["source"])


def cell_sources(path: Path) -> list[str]:
    """The source of every cell, markdown included."""
    return [cell_source(cell) for cell in load_notebook(path)["cells"]]


def code_cells(path: Path) -> Iterator[tuple[int, str]]:
    """``(index, source)`` of each code cell; the index counts every cell."""
    for index, cell in enumerate(load_notebook(path)["cells"]):
        if cell["cell_type"] == "code":
            yield index, cell_source(cell)


def code_cell_sources(path: Path) -> list[str]:
    return [source for _, source in code_cells(path)]


def cell_index(sources: Sequence[str], marker: str) -> int:
    """The index of the one source containing *marker*."""
    hits = [index for index, source in enumerate(sources) if marker in source]
    assert len(hits) == 1, f"expected exactly one code cell containing {marker!r}, found {len(hits)}"
    return hits[0]


def code_cell(path: Path, marker: str) -> str:
    """The one code cell containing *marker*."""
    sources = code_cell_sources(path)
    return sources[cell_index(sources, marker)]


def first_code_cell(path: Path, marker: str) -> str:
    """The first code cell containing *marker*."""
    for _, source in code_cells(path):
        if marker in source:
            return source
    raise LookupError(f"no code cell of {path.name} contains {marker!r}")


def exec_top_level_def(source: str, name: str, namespace: dict[str, Any]) -> None:
    """Define the one top-level ``def name`` of a code cell's *source* in *namespace*, running nothing else of it.

    For an executed test of a cell that calls a helper another cell defines: the real definition, never a stub.
    """
    defs: list[ast.stmt] = [
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(defs) == 1, f"expected exactly one top-level `def {name}(`, found {len(defs)}"
    exec(compile(ast.Module(body=defs, type_ignores=[]), f"<notebook def {name}>", "exec"), namespace)


def strip_magics(source: str) -> str:
    """*source* with each IPython magic or shell line (``%``, ``!``) replaced by ``pass`` at its indentation."""
    return "\n".join(
        line if not line.lstrip().startswith(("!", "%")) else line[: len(line) - len(line.lstrip())] + "pass"
        for line in source.splitlines()
    )


def parse_failures(paths: Sequence[Path]) -> list[str]:
    """``<path> cell <index>: <error>`` for each code cell of *paths* that does not parse.

    A notebook inside the repository is named relative to its root (``notebooks/<name>.ipynb``).
    """
    failures = []
    for path in paths:
        shown = path.resolve().relative_to(REPO_ROOT) if path.resolve().is_relative_to(REPO_ROOT) else path
        for index, source in code_cells(path):
            try:
                ast.parse(strip_magics(source))
            except SyntaxError as exc:
                failures.append(f"{shown} cell {index}: {exc}")
    return failures


def main(argv: Sequence[str]) -> int:
    paths = [Path(arg) for arg in argv] or sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        print(f"notebook validation failed: no notebooks in {REPO_ROOT / 'notebooks'}", file=sys.stderr)
        return 1
    failures = parse_failures(paths)
    if failures:
        print("notebook validation failed:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("all notebook code cells parse")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
