"""Static checks of the notebooks' code cells: nothing is executed or imported.

Every ``environments`` import in a notebook code cell must resolve: the module
is a file or package directory of this repository, and each imported name is a
submodule or is bound at the module's top level (for a package, in its
``__init__.py``). A library rename a notebook still imports otherwise surfaces
only in a Colab session. The check reads source, so it needs neither the
package's dependencies nor an import of the package.

CI's lint job parses every notebook code cell with ``notebook_cells.py``, run
by path on the standard library alone; the last two tests run it that way.
"""

from __future__ import annotations

import ast
import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from .notebook_cells import REPO_ROOT, code_cells, strip_magics

NOTEBOOKS = [REPO_ROOT / "notebooks" / "sb3_training.ipynb", REPO_ROOT / "notebooks" / "google_drive_summary.ipynb"]
NOTEBOOK_CELLS_PY = Path(__file__).with_name("notebook_cells.py")


def _environments_imports(path: Path) -> list[tuple[int, str, str | None]]:
    """``(cell index, module, name)`` for each ``environments`` import in *path*; name is None for ``import module``."""
    imports: list[tuple[int, str, str | None]] = []
    for index, source in code_cells(path):
        for node in ast.walk(ast.parse(strip_magics(source))):
            pairs: list[tuple[str, str | None]]
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                pairs = [(node.module, alias.name) for alias in node.names]
            elif isinstance(node, ast.Import):
                pairs = [(alias.name, None) for alias in node.names]
            else:
                continue
            imports.extend((index, module, name) for module, name in pairs if module.split(".")[0] == "environments")
    return imports


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _target_names(element)}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _bound_names(body: list[ast.stmt]) -> set[str]:
    """Names *body* binds at run time, inside if/try/with/loop blocks too (not a def, a class or ``if TYPE_CHECKING:``)."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            names.update(_target_names(node.target))
        elif isinstance(node, ast.If) and ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            names.update(_bound_names(node.orelse))
            continue
        for block in ("body", "orelse", "finalbody", "handlers"):
            names.update(_bound_names(getattr(node, block, [])))
    return names


@functools.cache
def _module_names(source: Path) -> frozenset[str]:
    return frozenset(_bound_names(ast.parse(source.read_text(encoding="utf-8")).body))


def _unresolved(module: str, name: str | None) -> str | None:
    """Why ``from module import name`` (``import module`` when name is None) would fail; None when it resolves."""
    target = REPO_ROOT.joinpath(*module.split("."))
    source: Path | None
    if (target / "__init__.py").is_file():
        source = target / "__init__.py"
    elif (target.parent / f"{target.name}.py").is_file():
        source = target.parent / f"{target.name}.py"
    elif target.is_dir():
        source = None  # a namespace package: only its submodules import
    else:
        return f"no module {module}"
    if name is None:
        return None
    if target.is_dir() and ((target / f"{name}.py").is_file() or (target / name).is_dir()):
        return None
    if source is not None and name in _module_names(source):
        return None
    return f"{module} defines no {name}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=["sb3", "drive_summary"])
def test_every_environments_import_in_a_notebook_resolves(path):
    imports = _environments_imports(path)
    assert imports, f"{path.name} imports nothing from environments; the check has nothing to check"
    problems = [f"cell {index}: {problem}" for index, module, name in imports if (problem := _unresolved(module, name))]
    assert not problems, f"{path.name} imports what the library does not define:\n" + "\n".join(problems)


def _run_notebook_cells(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run notebook_cells.py by path with -I -S: no repository, PYTHONPATH or site-packages on ``sys.path``."""
    return subprocess.run(
        [sys.executable, "-I", "-S", str(NOTEBOOK_CELLS_PY), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_ci_notebook_check_parses_the_notebooks_on_the_standard_library_alone(tmp_path):
    result = _run_notebook_cells(cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "all notebook code cells parse\n"


def test_the_ci_notebook_check_fails_on_a_code_cell_that_does_not_parse(tmp_path):
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["not python ("]},
        {"cell_type": "code", "metadata": {}, "source": ["!pip install x\n", "if True:\n", "    %time 1\n"]},
        {"cell_type": "code", "metadata": {}, "source": "def broken(:\n    pass\n"},
    ]
    notebook = tmp_path / "broken.ipynb"
    notebook.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    result = _run_notebook_cells(str(notebook), cwd=tmp_path)
    assert result.returncode == 1
    header, *failures = result.stderr.splitlines()
    assert header == "notebook validation failed:"
    assert [failure.split(": ", 1)[0] for failure in failures] == [f"{notebook} cell 2"], result.stderr
