"""The one repository root (cleanup CU-8c).

``environments/shared/paths.py`` is the one module that computes the
repository root from ``__file__``, apart from the ``sys.path`` bootstraps and
``configs`` anchors listed below.  Every other name for it is that object,
under the name each module already exposed: ``plant_contract.constants``
rebinds it as its own attribute and stays the patch point its consumers read
at call time; ``train_behaviors.REPO_ROOT`` (CI's installed-wheel step imports
it) and ``species_catalog.REPOSITORY_ROOT`` keep their public names.  The
value enters the repository-relative paths ``behavior_identity`` and the
plant source closure hash, so it is pinned here too.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator

import environments.shared.plant_contract as plant_contract
import environments.shared.plant_contract.__main__ as plant_contract_cli
import environments.shared.result_bundle.constants as result_bundle_constants
from environments.compsognathus.scripts import calibrate_recovery
from environments.shared import config, paths, species_catalog, task_fingerprint, train_behaviors
from environments.shared.plant_contract import constants as plant_constants
from environments.shared.plant_contract import versions
from environments.shared.plant_contract.source_layer import _source_payload
from environments.shared.scripts import restamp_recovery_calibration

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = REPO_ROOT / "environments"
PLANT_CONTRACT = ENVIRONMENTS / "shared" / "plant_contract"


def test_every_name_for_the_root_is_the_one_object():
    root = paths.REPOSITORY_ROOT
    aliases = {
        "plant_contract.REPOSITORY_ROOT": plant_contract.REPOSITORY_ROOT,
        "plant_contract.constants.REPOSITORY_ROOT": plant_constants.REPOSITORY_ROOT,
        "result_bundle.constants.REPOSITORY_ROOT": result_bundle_constants.REPOSITORY_ROOT,
        "train_behaviors.REPO_ROOT": train_behaviors.REPO_ROOT,
        "species_catalog.REPOSITORY_ROOT": species_catalog.REPOSITORY_ROOT,
        "config._REPO_ROOT": config._REPO_ROOT,
        "task_fingerprint._REPOSITORY_ROOT": task_fingerprint._REPOSITORY_ROOT,
        "scripts.restamp_recovery_calibration._REPOSITORY_ROOT": restamp_recovery_calibration._REPOSITORY_ROOT,
        # Records repo-relative ``source_sha256`` keys.  (``probe_ppo_updates`` binds it the same way;
        # it is coverage-omitted, so no test imports it, and the static rule below covers it.)
        "compsognathus.scripts.calibrate_recovery.REPO_ROOT": calibrate_recovery.REPO_ROOT,
    }
    assert {name: value is root for name, value in aliases.items()} == dict.fromkeys(aliases, True)
    assert plant_constants._SHARED_ROOT is paths.SHARED_ROOT


def test_the_root_is_the_repository_checkout():
    root = paths.REPOSITORY_ROOT
    assert (root / "pyproject.toml").is_file()
    assert (root / "environments" / "shared" / "paths.py").is_file()
    assert root.name != "environments"
    assert paths.SHARED_ROOT == root / "environments" / "shared"
    assert root == REPO_ROOT


def test_a_patch_of_plant_contract_constants_reaches_its_consumers(monkeypatch, tmp_path):
    """The plant contract's consumers read ``constants.REPOSITORY_ROOT`` at call time, so a test that
    patches it moves them all; a consumer reading ``paths`` (or a name bound at import) would not move."""
    model = tmp_path / "plant" / "model.xml"
    model.parent.mkdir()
    model.write_text('<mujoco model="m"><worldbody/></mujoco>\n', encoding="utf-8")
    monkeypatch.setattr(plant_constants, "REPOSITORY_ROOT", tmp_path)
    assert _source_payload(model)["root"] == "plant/model.xml"
    assert versions._resolve_repo_path("plant/model.xml", field="model") == model.resolve()


def test_the_plant_contract_cli_names_the_written_manifest_from_the_patched_root(monkeypatch, tmp_path, capsys):
    """``python -m environments.shared.plant_contract --write`` reports the manifest it wrote relative
    to ``constants.REPOSITORY_ROOT`` read at call time (the writer is stubbed; nothing is written)."""
    written = tmp_path / "configs" / "plant_manifest.json"
    monkeypatch.setattr(plant_constants, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(plant_contract_cli, "write_plant_manifest", lambda: written)
    monkeypatch.setattr(sys, "argv", ["plant_contract", "--write"])
    assert plant_contract_cli._main() == 0
    assert capsys.readouterr().out == "Wrote configs/plant_manifest.json\n"


def _import_time_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Every node that runs when the module is imported: the module's and each class body's
    statements, decorators and default arguments, but not a function's or a lambda's body."""
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(getattr(node, "decorator_list", []))
            stack.extend(node.args.defaults)
            stack.extend(default for default in node.args.kw_defaults if default is not None)
        else:
            stack.extend(ast.iter_child_nodes(node))


def test_plant_contract_modules_read_the_root_only_through_constants():
    """Static form of the patch-point rule: outside ``constants`` (and ``__init__``, which re-exports
    it for the byte-hashed behavior and sampler modules), no plant-contract module imports the root
    or ``paths`` (``from ... import``, or ``import <...>.paths [as name]``), reads a bare
    ``REPOSITORY_ROOT``, or reads ``<name>.REPOSITORY_ROOT`` when it is imported (at the top level,
    in a class body, a decorator or a default argument), which would bind the value before a
    test's patch; each reads ``constants.REPOSITORY_ROOT`` inside its functions."""
    offenders = []
    for path in sorted(PLANT_CONTRACT.glob("*.py")):
        if path.name in {"constants.py", "__init__.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                (node.module or "").split(".")[-1] == "paths"
                or any(
                    alias.name in {"REPOSITORY_ROOT", "SHARED_ROOT", "_SHARED_ROOT", "paths"} for alias in node.names
                )
            ):
                offenders.append(f"{path.name}:{node.lineno} imports {ast.unparse(node)}")
            elif isinstance(node, ast.Import) and any(alias.name.split(".")[-1] == "paths" for alias in node.names):
                offenders.append(f"{path.name}:{node.lineno} imports {ast.unparse(node)}")
            elif isinstance(node, ast.Name) and node.id == "REPOSITORY_ROOT":
                offenders.append(f"{path.name}:{node.lineno} reads a bare REPOSITORY_ROOT")
        for node in _import_time_nodes(tree):
            if isinstance(node, ast.Attribute) and node.attr == "REPOSITORY_ROOT":
                offenders.append(f"{path.name}:{node.lineno} reads {ast.unparse(node)} at import time")
    assert offenders == []


# --- Tripwire: no second computation of the root from ``__file__``.


def _library_files() -> Iterator[Path]:
    """Every non-test Python file of the package, and the repository's root-level modules."""
    for path in sorted(ENVIRONMENTS.rglob("*.py")):
        if "tests" not in path.relative_to(ENVIRONMENTS).parts:
            yield path
    yield from sorted(REPO_ROOT.glob("*.py"))


def _assignment(node: ast.AST) -> tuple[str, ast.expr] | None:
    """``(name, value)`` for ``name = value`` or ``name: T = value``, else None."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        return node.target.id, node.value
    return None


class _RootResolver:
    """Statically resolves ``Path(__file__)`` chains (and names bound to one) in one file.

    Understands ``Path(__file__)``, ``pathlib.Path(__file__)`` (``pathlib`` under any name) and
    ``Path`` imported under another name (``from pathlib import Path as P``),
    ``.resolve()``/``.absolute()``, ``.parent``, ``.parents[k]``, ``dirname``/``abspath``/``realpath``
    under any spelling (``os.path.dirname``, ``osp.dirname``, a bare ``dirname`` imported from
    ``os.path``), and a name assigned (annotated or not) one of these in the file (a named anchor
    such as ``SHARED_ROOT``, climbed later).  The limit: anything else -- a helper function, a
    loop, a ``".."`` join (``Path(__file__).joinpath("..")``, ``Path(__file__).parent / ".."``,
    ``os.path.join(..., "..")``), a path from another module's ``__file__`` -- is not followed.
    """

    def __init__(self, path: Path, tree: ast.AST) -> None:
        self.file = path.resolve()
        self.path_names = {"Path"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pathlib" and not node.level:
                self.path_names.update(alias.asname for alias in node.names if alias.name == "Path" and alias.asname)
        self.anchors: dict[str, Path] = {}
        for node in ast.walk(tree):
            assignment = _assignment(node)
            if assignment is not None:
                value = self.resolve(assignment[1])
                if value is not None:
                    self.anchors[assignment[0]] = value

    def resolve(self, node: ast.AST) -> Path | None:
        if isinstance(node, ast.Name):
            return self.file if node.id == "__file__" else self.anchors.get(node.id)
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in self.path_names and len(node.args) == 1:
                return self.resolve(node.args[0])
            if isinstance(func, ast.Attribute) and name in {"resolve", "absolute"} and not node.args:
                return self.resolve(func.value)
            if name in {"abspath", "realpath"} and len(node.args) == 1:
                return self.resolve(node.args[0])
            if name == "dirname" and len(node.args) == 1:
                inner = self.resolve(node.args[0])
                return inner.parent if inner is not None else None
            return None
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            inner = self.resolve(node.value)
            return inner.parent if inner is not None else None
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            inner = self.resolve(node.value.value)
            return inner.parents[node.slice.value] if inner is not None else None
        return None

    @staticmethod
    def climbs(node: ast.AST) -> bool:
        """Whether *node* itself moves up a directory (a bare name of a resolved anchor does not):
        ``.parent``, ``.parents[k]`` or a call of anything named ``dirname``, as :meth:`resolve`
        follows it."""
        if isinstance(node, ast.Call):
            func = node.func
            return (func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)) == "dirname"
        return (isinstance(node, ast.Attribute) and node.attr == "parent") or (
            isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents"
        )


def _climbs_to(root: Path, path: Path, tree: ast.AST) -> tuple[set[str | None], bool]:
    """``({assigned name or None}, any bootstrap)`` for the expressions in *tree* (the file at
    *path*) that climb to *root*."""
    resolver = _RootResolver(path, tree)
    owner: dict[int, str] = {}
    bootstrap_nodes: set[int] = set()
    for node in ast.walk(tree):
        assignment = _assignment(node)
        if assignment is not None:
            target, value = assignment
            for inner in ast.walk(value):
                owner[id(inner)] = target
            # The sys.path bootstraps ``_repo_root = str(<root>)`` run before ``environments``
            # is importable, so they cannot import ``paths``.
            if target == "_repo_root" and isinstance(value, ast.Call):
                if getattr(value.func, "id", None) == "str":
                    bootstrap_nodes.update(id(inner) for inner in ast.walk(value))
    names: set[str | None] = set()
    bootstrap = False
    for node in ast.walk(tree):
        if not resolver.climbs(node) or resolver.resolve(node) != root:
            continue
        if id(node) in bootstrap_nodes:
            bootstrap = True
        else:
            names.add(owner.get(id(node)))
    return names, bootstrap


def _root_computations() -> tuple[set[tuple[str, str | None]], set[str]]:
    """``({(file, assigned name)}, {bootstrap file})`` for every expression that climbs to the root."""
    root = REPO_ROOT.resolve()
    computations: set[tuple[str, str | None]] = set()
    bootstraps: set[str] = set()
    for path in _library_files():
        names, bootstrap = _climbs_to(root, path, ast.parse(path.read_text(encoding="utf-8")))
        relative = path.relative_to(REPO_ROOT).as_posix()
        computations.update((relative, name) for name in names)
        if bootstrap:
            bootstraps.add(relative)
    return computations, bootstraps


#: The one computation, and the five "root + configs" anchors CU-8c left under their own names
#: (two of them, stage_manifest._CONFIGS_DIR and recovery_calibration.CONFIGS_ROOT, are patched by
#: tests; folding them is not in the row).
ALLOWED_ROOT_COMPUTATIONS = {
    ("environments/shared/paths.py", "REPOSITORY_ROOT"),
    ("environments/shared/behavior_certification.py", "RULES_PATH"),
    ("environments/shared/recovery_calibration.py", "CONFIGS_ROOT"),
    ("environments/shared/scripts/stance_quality_baseline.py", "CONFIG_ROOT"),
    ("environments/shared/species_names.py", "_MANIFEST_PATH"),
    ("environments/shared/stage_manifest.py", "_CONFIGS_DIR"),
}

#: The files whose ``_repo_root = str(<root>)`` puts the checkout on ``sys.path``: scripts run by
#: path, ``train.py`` and the root ``conftest.py``, which run before ``environments`` is importable
#: and so cannot import ``paths``.  Listed by file, so the form cannot carry a root computation into
#: a library module; a new script run by path adds itself here.
SYS_PATH_BOOTSTRAPS = {
    "conftest.py",
    "environments/brachiosaurus/scripts/test_env.py",
    "environments/brachiosaurus/scripts/train_sb3.py",
    "environments/dibothrosuchus/scripts/test_env.py",
    "environments/dibothrosuchus/scripts/train_sb3.py",
    "environments/shared/scripts/action_bound_report.py",
    "environments/shared/scripts/actuator_saturation_report.py",
    "environments/shared/scripts/compare_run_diagnostics.py",
    "environments/shared/scripts/foot_sensor_report.py",
    "environments/shared/scripts/hw_chassis_study.py",
    "environments/shared/scripts/joint_excursion_report.py",
    "environments/shared/scripts/observation_ablation_report.py",
    "environments/shared/scripts/stance_duty_validation.py",
    "environments/shared/scripts/stance_gate_report.py",
    "environments/shared/scripts/zero_action_baseline.py",
    "environments/shared/train.py",
    "environments/trex/scripts/test_env.py",
    "environments/trex/scripts/train_sb3.py",
    "environments/velociraptor/scripts/actuator_saturation_report.py",
    "environments/velociraptor/scripts/test_env.py",
    "environments/velociraptor/scripts/train_sb3.py",
}


def test_no_module_but_paths_computes_the_repository_root():
    """A second ``Path(__file__)...parents[k]`` that lands on the root, under any name or none (a
    named anchor climbed later included), fails here: import ``paths.REPOSITORY_ROOT`` instead.
    So does a ``_repo_root = str(<root>)`` bootstrap in a file not listed.  Every allowed entry
    must still be found, so neither list can go stale."""
    computations, bootstraps = _root_computations()
    assert computations == ALLOWED_ROOT_COMPUTATIONS
    assert bootstraps == SYS_PATH_BOOTSTRAPS


def test_the_resolver_sees_each_form_it_claims(tmp_path):
    """Self-test: each form the rule claims to follow is flagged in a module at a library depth,
    and a form its docstring names as a limit is not."""
    module = tmp_path / "environments" / "shared" / "probe_module.py"
    module.parent.mkdir(parents=True)
    forms = {
        "A": "Path(__file__).resolve().parents[2]",
        "B": "Path(__file__).parent.parent.parent",
        "C": "os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        "E": "_ANCHOR.parents[1]",
        "F": "_TYPED.parents[1]",
        "H": "Path(dirname(dirname(dirname(abspath(__file__)))))",
        "I": "osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))",
        "J": "_P(__file__).resolve().parents[2]",
        "K": "pathlib_module.Path(__file__).resolve().parents[2]",
    }
    source = (
        "import os.path as osp\n"
        "import pathlib as pathlib_module\n"
        "from os.path import abspath, dirname\n"
        "from pathlib import Path as _P\n"
        "_ANCHOR = Path(__file__).resolve().parent\n"
        "_TYPED: Path = Path(__file__).resolve().parent\n"
        + "".join(f"{k} = {v}\n" for k, v in forms.items())
        + "G: Path = Path(__file__).resolve().parents[2]\n"
        # Documented limits: a ".." join is not followed.
        + 'L = (Path(__file__).parent / ".." / "..").resolve()\n'
        + 'M = Path(__file__).resolve().joinpath("..", "..", "..")\n'
    )
    module.write_text(source, encoding="utf-8")
    names, bootstrap = _climbs_to(tmp_path.resolve(), module, ast.parse(source))
    assert names == set(forms) | {"G"}
    assert not bootstrap
