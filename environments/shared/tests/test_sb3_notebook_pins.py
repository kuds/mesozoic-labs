"""Cell-level pins for ``notebooks/sb3_training.ipynb`` (BEHAVIOR_RECIPES_PLAN §4.7, Phase A WS5).

The SB3 notebook is the Colab driver an operator actually runs, and Phase A
turned its per-stage cells into one behavior-chain loop: ``BEHAVIOR`` names a
deliverable, the manifest resolves the chain, and each node is REUSED (a
certified ancestor in ``RUN_DIR`` or ``TRUNK_FROM``), JUDGED (trained by the
RESUME cell but never gated) or TRAINED from its parent's handoff along the
declared edge — then published BEFORE its verdict is enforced.  Decisions
D-A15 and D-A17..D-A21 (§6.1) amended that loop with retrain-from, chain-by-
digest reuse, the occupied-directory guard, recorded durations and the
ignored-edit warning.

These tests read the notebook JSON and pin its STRUCTURE — the order of calls
inside the ``for NODE in CHAIN:`` body, keyword presence, the absence of
position-keyed names — the way ``test_jax_notebook_pins`` pins the JAX
notebook: a rewording of a print survives; a semantic regression (a node
retrained over a failed verdict, a target reused from another run, the raise
before the bundle write, a load mode inferred from position) fails here
instead of in a Colab session weeks later.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
#: ``train_stage`` trains through ``train_base.train`` (consolidation PR-14c); pins on what it records read its source.
TRAIN_BASE_PATH = REPO_ROOT / "environments" / "shared" / "train_base.py"
JAX_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "jax_training.ipynb"

# Unique markers that identify the cells under test (not their positions —
# cell numbers move when a markdown cell is added).
CONFIG_CELL_MARKER = "# ===== SPECIES SELECTION ====="
STORAGE_CELL_MARKER = "# Storage Configuration"
RESOLVE_CELL_MARKER = "STAGE_CONFIGS = load_all_stages(SPECIES)"
INFRA_CELL_MARKER = "def train_stage("
CHAIN_CELL_MARKER = "# ===== BEHAVIOR CHAIN LOOP ====="
MANUAL_CELL_MARKER = "# ===== MANUAL SINGLE NODE"
RESUME_CELL_MARKER = "# ===== RESUME AN INTERRUPTED STAGE"
COMPLETION_CELL_MARKER = 'print("Training complete!")'
#: The archive-load preflight cell: its FIRST line, deliberately not a `# ===== ` marker. It sits right after the
#: RESOLVE cell and loads a real archive (the trunk run's root handoff, else a throwaway) through `load_sb3_model`
#: before anything is trained (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved them").
PREFLIGHT_CELL_MARKER = '# SB3 archive load preflight (KNOWN_ISSUES "Training / RL": SB3 archives are bound to the interpreter that saved them)'

#: The knobs every ``halt`` / ``disconnect_runtime`` call passes by name, read when it runs (consolidation PR-14b).
DISCONNECT_KNOBS = (("in_colab", "IN_COLAB"), ("auto", "AUTO_DISCONNECT"), ("flush_drive", "USE_GOOGLE_DRIVE"))

#: Every species with a committed stage manifest (the notebook's SPECIES menu).
SPECIES_WITH_MANIFESTS = sorted(path.parent.name for path in (REPO_ROOT / "configs").glob("*/stages.toml"))


def _notebook_text(path: Path = NOTEBOOK_PATH) -> str:
    return path.read_text(encoding="utf-8")


def _cells(path: Path = NOTEBOOK_PATH) -> list[dict]:
    cells: list[dict] = json.loads(_notebook_text(path))["cells"]
    return cells


def _code_cells(path: Path = NOTEBOOK_PATH) -> list[str]:
    return ["".join(cell["source"]) for cell in _cells(path) if cell["cell_type"] == "code"]


def _all_cell_sources(path: Path = NOTEBOOK_PATH) -> list[str]:
    return ["".join(cell["source"]) for cell in _cells(path)]


def _cell_index(cells: list[str], marker: str) -> int:
    hits = [index for index, src in enumerate(cells) if marker in src]
    assert len(hits) == 1, f"expected exactly one code cell containing {marker!r}, found {len(hits)}"
    return hits[0]


def _cell(marker: str) -> str:
    cells = _code_cells()
    return cells[_cell_index(cells, marker)]


def _branch_source(src: str, nodes: list[ast.stmt]) -> str:
    """The source text spanned by a list of statements (an ``if`` body or ``orelse``)."""
    lines = src.splitlines()
    first, last = nodes[0], nodes[-1]
    assert last.end_lineno is not None
    return "\n".join(lines[first.lineno - 1 : last.end_lineno])


def _calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Every ``func_name(...)`` call (a bare name or the attribute of one) under *tree*."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == func_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == func_name)
        )
    ]


def _call(tree: ast.AST, func_name: str) -> ast.Call:
    calls = _calls(tree, func_name)
    assert len(calls) == 1, f"expected exactly one {func_name}(...) call, found {len(calls)}"
    return calls[0]


def _call_segment(src: str, func_name: str) -> str:
    """Source of the (single) call to ``func_name`` in ``src``."""
    segment = ast.get_source_segment(src, _call(ast.parse(src), func_name))
    assert segment is not None
    return segment


def _keyword_source(src: str, call: ast.Call, name: str) -> str:
    """Source of the ``name=`` keyword's value in *call*; fails when the keyword is absent."""
    for keyword in call.keywords:
        if keyword.arg == name:
            segment = ast.get_source_segment(src, keyword.value)
            assert segment is not None
            return segment
    raise AssertionError(f"{_func_name(call)}(...) at line {call.lineno} passes no {name}= keyword")


def _keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def _func_name(call: ast.Call) -> str:
    return call.func.id if isinstance(call.func, ast.Name) else ast.unparse(call.func)


def _top_level_for(src: str, target: str, iterable: str) -> ast.For:
    """The single top-level ``for <target> in <iterable>:`` of *src*."""
    loops = [
        node
        for node in ast.parse(src).body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == target
        and isinstance(node.iter, ast.Name)
        and node.iter.id == iterable
    ]
    assert len(loops) == 1, f"expected exactly one top-level `for {target} in {iterable}:`, found {len(loops)}"
    return loops[0]


def _top_level_def(src: str, name: str) -> ast.FunctionDef:
    defs = [node for node in ast.parse(src).body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(defs) == 1, f"expected exactly one top-level `def {name}(`, found {len(defs)}"
    return defs[0]


def _top_level_assigns(src: str) -> dict[str, ast.expr]:
    """``NAME = value`` statements at the top level of *src*, by name."""
    assigns: dict[str, ast.expr] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node.value
    return assigns


def _ifs(tree: ast.AST, src: str, test_predicate) -> list[ast.If]:
    """Every ``if`` under *tree* whose test SOURCE satisfies *test_predicate*."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and test_predicate(ast.get_source_segment(src, node.test) or "")
    ]


def _the_if(tree: ast.AST, src: str, test_predicate, what: str) -> ast.If:
    hits = _ifs(tree, src, test_predicate)
    assert len(hits) == 1, f"expected exactly one `if` {what}, found {len(hits)}"
    return hits[0]


def _names(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name *tree* binds at any depth: imports, assignment targets, loop/comprehension targets, handler names."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return bound


def _loaded_names(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _raises(tree: ast.AST, exc_name: str) -> list[ast.Raise]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == exc_name
    ]


def _chain_loop() -> tuple[str, ast.For]:
    src = _cell(CHAIN_CELL_MARKER)
    return src, _top_level_for(src, "NODE", "CHAIN")


def _handoff_assigns(tree: ast.AST) -> list[ast.Assign]:
    """``NODE_HANDOFF[<key>] = {...}`` statements under *tree*."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "NODE_HANDOFF"
    ]


def _dict_keys(node: ast.expr) -> list[str]:
    assert isinstance(node, ast.Dict), f"expected a dict literal, got {ast.unparse(node)[:60]!r}"
    return [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]


def _dict_value(node: ast.Dict, key: str) -> ast.expr:
    for candidate, value in zip(node.keys, node.values):
        if isinstance(candidate, ast.Constant) and candidate.value == key:
            return value
    raise AssertionError(f"dict literal has no {key!r} key")


#: The three branch decisions of the chain loop, located structurally.
def _reuse_if(src: str, loop: ast.For) -> ast.If:
    return _the_if(loop, src, lambda test: test == "ancestor is not None", "on `ancestor is not None` (REUSE)")


def _verdict_if(src: str, loop: ast.For) -> ast.If:
    return _the_if(
        loop,
        src,
        lambda test: test.startswith("verdict is not None"),
        "on an existing verdict (never retrained over)",
    )


def _judge_if(src: str, loop: ast.For) -> ast.If:
    """The ``if`` whose body holds the JUDGE branch (its orelse chain ends in TRAIN)."""
    hits = [
        node
        for node in loop.body
        if isinstance(node, ast.If)
        and _calls(ast.Module(body=node.body, type_ignores=[]), "evaluate_stage_checkpoints")
    ]
    assert len(hits) == 1, "expected exactly one top-level `if` of the loop body holding the JUDGE branch"
    return hits[0]


def _train_branch(judge_if: ast.If) -> list[ast.stmt]:
    """The final ``else`` of the JUDGE if/elif chain: the TRAIN branch."""
    node: ast.If = judge_if
    while len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
    assert node.orelse, "the JUDGE chain must end in an `else:` (TRAIN)"
    return node.orelse


class TestBehaviorKnob:
    """§4.7: the config cell declares the behavior recipe knobs; the recovery pilot knob is gone."""

    def test_behavior_and_trunk_from_are_declared_in_the_config_cell(self):
        assigns = _top_level_assigns(_cell(CONFIG_CELL_MARKER))
        for name in ("BEHAVIOR", "TRUNK_FROM", "RETRAIN_FROM", "RUN_LABEL", "RUN_ID"):
            assert name in assigns, f"the config cell must declare {name}"
            assert isinstance(assigns[name], ast.Constant) and isinstance(assigns[name].value, str), (
                f"{name} is a plain string constant an operator edits"
            )
        # Manual overrides remain off. Automatic selection can reuse certified
        # ancestors; the target still trains here, and RUN_ID = "" mints a fresh run
        # (consolidation PR-14a moved the knob here from the storage cell).
        for name in ("RETRAIN_FROM", "RUN_LABEL", "RUN_ID"):
            assert assigns[name].value == "", f"{name} must default to the empty string (off)"
        # D-A25: the trunk is selected automatically unless a run is pinned or "" turns reuse off.
        assert assigns["TRUNK_FROM"].value == "auto"
        # Consolidation PR-4 took the certified-library knobs with the canonical publish wrapper; PR-5 took
        # SOURCE_SELECTION with the library itself.
        for name in (
            "CERTIFIED_LIBRARY_ROOT",
            "PUBLISH_CERTIFIED",
            "CERTIFIED_COMPARISON_EPISODES",
            "SOURCE_SELECTION",
        ):
            assert name not in assigns, f"{name} left with the certified library (consolidation PR-4/PR-5)"
        # D-D14: widening is the command-line tool's job; the notebook's widen knobs left with the widen cell
        # (consolidation PR-14a). D-C17's fail-closed bound of 1 stays the tool's default (--max-revision-gap),
        # pinned here for the shared matrix too: test_widen_checkpoint.py is SB3-gated at module level.
        from environments.shared.scripts.widen_checkpoint import DEFAULT_MAX_REVISION_GAP

        for name in ("WIDEN_FROM", "WIDEN_MAX_REVISION_GAP"):
            assert name not in assigns, f"{name} left the notebook with the widen cell (D-D14)"
        assert DEFAULT_MAX_REVISION_GAP == 1
        # The default behavior resolves on every committed manifest to the
        # terminal deliverable, so a Run-all still walks stance -> ... -> behavior.
        behavior = assigns["BEHAVIOR"].value
        assert behavior
        for species in SPECIES_WITH_MANIFESTS:
            manifest = load_stage_manifest(species)
            assert manifest.resolve_behavior(behavior).id == "behavior", (
                f"BEHAVIOR={behavior!r} must resolve to the terminal deliverable of {species}"
            )

    def test_run_recovery_stage_knob_is_gone(self):
        for index, src in enumerate(_all_cell_sources()):
            assert "RUN_RECOVERY_STAGE" not in src, (
                f"cell {index} mentions RUN_RECOVERY_STAGE: recovery is a chain node under BEHAVIOR, not an opt-in pilot"
            )

    def test_the_certified_library_knobs_and_prose_are_gone_from_every_cell(self):
        """Consolidation PR-5: no cell, markdown included, names the deleted library, its knobs or its CLI flags.

        No structural pin reads the markdown cells, so a stale sentence telling the operator that blank source
        paths select a library recommendation would otherwise survive every test."""
        for index, src in enumerate(_all_cell_sources()):
            for token in (
                "SOURCE_SELECTION",
                "CERTIFIED_LIBRARY_ROOT",
                "PUBLISH_CERTIFIED",
                "CERTIFIED_COMPARISON_EPISODES",
                "certified_library",
                "certified library recommendation",
                "library recommendation",
                "--auto-source",
                "--publish-certified",
                "--certified-library",
                "--comparison-episodes",
                "auto_source",
                "publish_certified",
                "certification_skip_reason",
            ):
                assert token not in src, f"cell {index} still names {token!r}: the certified library left with PR-5"

    def test_the_widen_cell_and_its_knobs_are_gone_from_every_cell(self):
        """Decision D-D14 (consolidation PR-14a): widening is the command-line tool's job. No cell, markdown
        included, names the deleted knobs or calls the tool, and no copy of the old widen-seed remedy (restart the
        runtime or delete the memo, then delete the stray directory) survives."""
        for index, src in enumerate(_all_cell_sources()):
            for token in (
                "WIDEN_FROM",
                "WIDEN_MAX_REVISION_GAP",
                "widen_checkpoint(",
                "del _ACTIVE_RUN_ID",
                "stray directory",
                "restart the runtime (or",
            ):
                assert token not in src, f"cell {index} still names {token!r}"
        for index, src in enumerate(_code_cells()):
            assert "environments.shared.scripts.widen_checkpoint" not in src, (
                f"code cell {index} imports the widen tool"
            )

    def test_the_label_knob_reaches_every_training_call(self):
        """D-A21: every ``train_stage`` call outside the definition threads ``label=RUN_LABEL or None``."""
        cells = _code_cells()
        infra_at = _cell_index(cells, INFRA_CELL_MARKER)
        seen = 0
        for index, src in enumerate(cells):
            if index == infra_at:
                continue
            for call in _calls(ast.parse(src), "train_stage"):
                seen += 1
                assert _keyword_source(src, call, "label") == "RUN_LABEL or None", f"cell {index}"
        assert seen == 3, "train_stage is called from exactly the chain loop, the manual cell and the RESUME cell"


class TestChainResolution:
    """The chain is resolved ONCE, in the resolve cell, through the manifest."""

    def test_the_manifest_mechanism_the_notebook_relies_on(self):
        trex = load_stage_manifest("trex")
        assert trex.resolve_behavior("hunt").id == "behavior"
        assert tuple(node.id for node in trex.chain_for("behavior")) == ("stance", "locomotion", "behavior")
        assert trex.resolve_behavior("stand").id == "recovery", "a label resolves to its DEEPEST deliverable"
        assert tuple(node.id for node in trex.chain_for("recovery")) == ("stance", "recovery")
        assert trex.resolve_behavior("walk").id == "locomotion"
        assert tuple(node.id for node in trex.chain_for("locomotion")) == ("stance", "locomotion")
        # An id resolves to itself; the chain never visits a sibling branch.
        assert trex.resolve_behavior("stance").id == "stance"
        assert tuple(node.id for node in trex.chain_for("stance")) == ("stance",)
        for species in SPECIES_WITH_MANIFESTS:
            manifest = load_stage_manifest(species)
            assert manifest.resolve_behavior("hunt").id == "behavior", species
            has_recovery = any(entry.id == "recovery" for entry in manifest.stages)
            assert manifest.resolve_behavior("stand").id == ("recovery" if has_recovery else "stance"), species
            assert manifest.chain_for("behavior")[0].id == "stance", "every chain is root-first"
            with pytest.raises(StageManifestError):
                manifest.resolve_behavior("sprint")

    def test_behavior_is_resolved_once_through_the_manifest(self):
        cells = _code_cells()
        resolve_src = cells[_cell_index(cells, RESOLVE_CELL_MARKER)]
        assigns = _top_level_assigns(resolve_src)
        assert ast.unparse(assigns["MANIFEST"]) == "load_stage_manifest(SPECIES)"
        assert ast.unparse(assigns["TARGET_NODE"]) == "MANIFEST.resolve_behavior(BEHAVIOR)"
        assert ast.unparse(assigns["CHAIN"]) == "MANIFEST.chain_for(TARGET_NODE.id)"
        # D-A19: RETRAIN_FROM is resolved beside the chain and must lie on it.
        assert ast.unparse(assigns["RETRAIN_NODE"]) == "MANIFEST.resolve(RETRAIN_FROM) if RETRAIN_FROM else None"
        guard = _the_if(
            ast.parse(resolve_src),
            resolve_src,
            lambda test: "RETRAIN_NODE" in test and "not in CHAIN" in test,
            "refusing a RETRAIN_NODE off the chain",
        )
        assert _raises(guard, "RuntimeError"), "a RETRAIN_FROM off the chain is a RuntimeError, not a silent no-op"
        # Nowhere else resolves a behavior or builds a chain — the loop walks CHAIN.
        resolve_at = _cell_index(cells, RESOLVE_CELL_MARKER)
        for index, src in enumerate(cells):
            if index == resolve_at:
                continue
            tree = ast.parse(src)
            assert not _calls(tree, "resolve_behavior"), f"cell {index} re-resolves BEHAVIOR"
            assert not _calls(tree, "chain_for"), f"cell {index} rebuilds the chain"
            assert not _calls(tree, "load_stage_manifest"), f"cell {index} reloads the manifest"
        assert resolve_at < _cell_index(cells, INFRA_CELL_MARKER) < _cell_index(cells, CHAIN_CELL_MARKER)


class TestReuseRule:
    """§4.2 / D-A17 / D-A18 / D-A19: which directories a node may be satisfied from, and how."""

    def test_trunk_from_resolves_to_a_bundle_before_training(self):
        cells = _code_cells()
        storage_at = _cell_index(cells, STORAGE_CELL_MARKER)
        assert storage_at < _cell_index(cells, CHAIN_CELL_MARKER)
        src = cells[storage_at]
        assigns = _top_level_assigns(src)
        assert isinstance(assigns["TRUNK_DIR"], ast.Constant) and assigns["TRUNK_DIR"].value is None
        tree = ast.parse(src)
        trunk_if = _the_if(tree, src, lambda test: test == "TRUNK_FROM", "resolving TRUNK_FROM")
        body = _branch_source(src, trunk_if.body)
        assert "load_provenance(TRUNK_DIR)" in body, "a trunk is a run WITH a provenance.json"
        assert "except ResultBundleError" in body
        assert "RUN_DIR.resolve()" in body, "a trunk is an EARLIER run, never this one"
        assert "canonical_algorithm(ALGORITHM)" in body
        assert '"stable-baselines3"' in body
        assert len(_raises(trunk_if, "RuntimeError")) >= 3, (
            "missing provenance, this run's own directory, and a species/algorithm/backend mismatch each refuse"
        )
        # The resolution happens BEFORE any node runs, at the top level of the storage cell.
        assert trunk_if in tree.body

    def test_reuse_checks_verdict_plant_and_task_hash(self):
        src, loop = _chain_loop()
        find = _call(loop, "find_certified_ancestor")
        assert {
            "species",
            "entry",
            "current_task_sha256",
            "plant_identity",
            "current_gate_config",
            "parent_model_sha256",
        } <= _keyword_names(find)
        assert _keyword_source(src, find, "entry") == "NODE"
        assert _keyword_source(src, find, "current_task_sha256") == "task_sha256"
        assert _keyword_source(src, find, "plant_identity") == "PLANT_IDENTITY"
        # D-A22 (rule 7): the gate this session would judge the node under is the
        # block save_stage_config records as 'curriculum' — the same source.
        assert _keyword_source(src, find, "current_gate_config") == 'config.get("curriculum_kwargs", {})'
        # The task digest comes from the CURRENT config through the shared derivation, for this backend,
        # from exactly the sources train_base.train records it from (train_stage trains through it with
        # SPECIES_CFG and STAGE_CONFIGS): a drift (env_kwargs={} here, say) would silently refuse every
        # reuse with "judged under task".
        train_src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        recorded = _call(_top_level_def(train_src, "train"), "derive_stage_task_fingerprint")
        assert {kw.arg: ast.get_source_segment(train_src, kw.value) for kw in recorded.keywords} == {
            "species": "species",
            "stage": "stage",
            "backend": '"stable-baselines3"',
            "env_kwargs": 'config.get("env_kwargs", {})',
            "plant_identity": "plant_identity.to_dict()",
        }
        fingerprint = _call(loop, "derive_stage_task_fingerprint")
        loop_sources = {kw.arg: ast.get_source_segment(src, kw.value) for kw in fingerprint.keywords}
        assert loop_sources == {
            "species": "SPECIES",
            "stage": "stage",
            "backend": '"stable-baselines3"',
            "env_kwargs": 'config.get("env_kwargs", {})',
            "plant_identity": "PLANT_IDENTITY.to_dict()",
        }
        task_assign = next(
            node for node in loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "task_sha256"
        )
        assert isinstance(task_assign.value, ast.Subscript) and task_assign.value.value is fingerprint
        assert ast.unparse(task_assign.value.slice) == "'task_sha256'"
        # Candidates: RUN_DIR first, then TRUNK_DIR (when set) — for a non-target, non-covered node.
        covered_if = _the_if(loop, src, lambda test: test == "covered", "on `covered`")
        assert len(covered_if.orelse) == 1 and isinstance(covered_if.orelse[0], ast.If)
        target_if = covered_if.orelse[0]
        assert ast.get_source_segment(src, target_if.test) == "NODE.id == TARGET_NODE.id"
        else_src = _branch_source(src, target_if.orelse)
        else_assign = next(
            node
            for node in target_if.orelse
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "candidates"
        )
        assert ast.unparse(else_assign.value).startswith("[RUN_DIR]"), "RUN_DIR is tried first"
        assert "TRUNK_DIR" in else_src and "TRUNK_DIR is not None" in else_src
        # Every refusal is printed (never silent), and the walk stops at the first success.
        handlers = [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.ExceptHandler)
            and node.type is not None
            and ast.unparse(node.type) == "AncestorReuseError"
        ]
        assert len(handlers) == 1
        assert _calls(handlers[0], "print"), "an AncestorReuseError refusal is printed with its reason"
        candidate_loop = next(
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "candidate"
        )
        assert ast.unparse(candidate_loop.iter) == "candidates"
        assert any(isinstance(node, ast.Break) for node in ast.walk(candidate_loop))

    def test_the_target_node_is_only_ever_reused_from_this_run(self):
        """D-A18: an earlier run's certified target is THAT run's deliverable."""
        src, loop = _chain_loop()
        covered_if = _the_if(loop, src, lambda test: test == "covered", "on `covered`")
        target_if = covered_if.orelse[0]
        assert isinstance(target_if, ast.If)
        assert ast.get_source_segment(src, target_if.test) == "NODE.id == TARGET_NODE.id"
        target_assign = next(
            node
            for node in target_if.body
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "candidates"
        )
        assert isinstance(target_assign.value, ast.List)
        assert [ast.unparse(elt) for elt in target_assign.value.elts] == ["RUN_DIR"], (
            "TARGET_NODE's reuse candidates are [RUN_DIR] only — never TRUNK_DIR"
        )

    def test_retrain_from_covers_a_node_and_its_descendants(self):
        """D-A19: a covered node has NO reuse candidates and goes to the train branch."""
        src, loop = _chain_loop()
        covered = next(
            node for node in loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "covered"
        )
        # The exact boolean, not its substrings: `or` -> `and` would make RETRAIN_FROM a silent no-op
        # (the named node is never its own ancestor) while every substring still appeared.
        assert ast.unparse(covered.value) == (
            "RETRAIN_NODE is not None and (NODE.id == RETRAIN_NODE.id or RETRAIN_NODE in MANIFEST.ancestors(NODE.id))"
        ), "the named node itself and every descendant of it are covered; nothing else is"
        covered_if = _the_if(loop, src, lambda test: test == "covered", "on `covered`")
        candidates = next(
            node
            for node in covered_if.body
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "candidates"
        )
        assert isinstance(candidates.value, ast.List) and candidates.value.elts == [], (
            "a covered node has no reuse candidates at all (neither RUN_DIR nor TRUNK_DIR)"
        )
        assert _calls(covered_if, "print"), "the loop says RETRAIN_FROM covers the node"
        assert "RETRAIN_FROM" in _branch_source(src, covered_if.body)
        # A covered node also never JUDGES or trips over an existing verdict: it trains. The exact tests,
        # so a dropped `not` (an UNcovered failed node skipping the never-retrained raise, a covered node
        # raising instead of training) fails here.
        assert ast.get_source_segment(src, _verdict_if(src, loop).test) == "verdict is not None and not covered"
        assert (ast.get_source_segment(src, _judge_if(src, loop).test) or "").startswith(
            "not covered and verdict is None and "
        )

    def test_reuse_chains_by_digest_onto_the_parent_resolved_here(self):
        """D-A17: a non-root node passes its parent's handoff digest; every NODE_HANDOFF entry carries one."""
        src, loop = _chain_loop()
        find = _call(loop, "find_certified_ancestor")
        parent_digest = next(kw.value for kw in find.keywords if kw.arg == "parent_model_sha256")
        assert isinstance(parent_digest, ast.IfExp), "parent_model_sha256 is the parent's digest, or None for a root"
        assert ast.unparse(parent_digest.body) == "parent_handoff['model_sha256']"
        assert ast.unparse(parent_digest.test) == "parent_handoff is not None"
        assert isinstance(parent_digest.orelse, ast.Constant) and parent_digest.orelse.value is None
        handoffs = _handoff_assigns(loop)
        assert len(handoffs) == 2, "one NODE_HANDOFF entry for a reused node, one for a trained/judged node"
        expected_keys = [
            "model",
            "vecnorm",
            "stage_dir",
            "run_dir",
            "run_id",
            "model_sha256",
            "normalization_sha256",
            "reused",
        ]
        by_reused: dict[bool, ast.Dict] = {}
        for assign in handoffs:
            assert ast.unparse(assign.targets[0]) == "NODE_HANDOFF[NODE.id]"
            assert isinstance(assign.value, ast.Dict)
            assert _dict_keys(assign.value) == expected_keys, ast.unparse(assign.value)
            reused = _dict_value(assign.value, "reused")
            assert isinstance(reused, ast.Constant) and isinstance(reused.value, bool)
            by_reused[reused.value] = assign.value
        assert set(by_reused) == {True, False}
        assert ast.unparse(_dict_value(by_reused[True], "model_sha256")) == "ancestor.model_sha256"
        assert ast.unparse(_dict_value(by_reused[True], "run_id")) == "None if same_run else ancestor.run_id", (
            "run_id names the ancestor's run only for a CROSS-run reuse"
        )
        same_run = next(
            node
            for node in ast.walk(_reuse_if(src, loop))
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "same_run"
        )
        assert ast.unparse(same_run.value) == "candidate == RUN_DIR", (
            "same-run means the candidate that succeeded was RUN_DIR itself; anything else (`True`) would "
            "leave a TRUNK_DIR ancestor unrecorded and its children with parent_run_id=None"
        )
        trained_digest = _dict_value(by_reused[False], "model_sha256")
        assert isinstance(trained_digest, ast.Call) and _func_name(trained_digest) == "sha256_file"
        assert "handoff_stem" in ast.unparse(trained_digest), "the digest is of the handoff zip the next node loads"
        trained_run_id = _dict_value(by_reused[False], "run_id")
        assert isinstance(trained_run_id, ast.Constant) and trained_run_id.value is None
        # The child passes that run id on as its recorded parent.
        train = _call(loop, "train_stage")
        assert _keyword_source(src, train, "parent_run_id") == (
            'parent_handoff["run_id"] if parent_handoff is not None else None'
        )

    def test_an_ignored_hyperparameter_edit_is_named_after_a_successful_reuse(self):
        """D-A21: reuse carries the ancestor's recipe; an edit since is IGNORED and said so, never refused."""
        src, loop = _chain_loop()
        reuse_if = _reuse_if(src, loop)
        diff = _call(reuse_if, "hyperparameter_diff")
        assert [ast.unparse(arg) for arg in diff.args] == ["config", "ALGORITHM", "_recorded"]
        recorded = next(
            node
            for node in ast.walk(reuse_if)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_recorded"
        )
        assert "ancestor.stage_dir / 'stage_config.json'" in ast.unparse(recorded.value), (
            "the diff is taken against the ANCESTOR's recorded stage_config.json (a diff of config against "
            "itself is always empty, so the warning could never fire)"
        )
        assert recorded.lineno < diff.lineno
        assert '"<unreadable stage_config.json>"' in _branch_source(src, reuse_if.body)
        warning_if = _the_if(reuse_if, src, lambda test: test == "ignored_edits", "on `ignored_edits`")
        warning = _branch_source(src, warning_if.body)
        assert _calls(warning_if, "print") and "RETRAIN_FROM" in warning, (
            "the warning says the edit is ignored and that RETRAIN_FROM trains the node here"
        )
        assert not _raises(warning_if, "RuntimeError"), "an ignored edit is a warning, not a refusal"
        # Ordered: reuse succeeded -> warn -> record -> hand off; all before the `continue`.
        find_at = _call(loop, "find_certified_ancestor").lineno
        assert find_at < diff.lineno < _handoff_assigns(reuse_if)[0].lineno
        assert warning_if.lineno < _call(reuse_if, "record_ancestor").lineno

    def test_only_the_trunk_candidate_follows_ancestor_records(self):
        """Decision D-A23: ``find_certified_ancestor`` can follow a run's own
        ``ancestors/<stage_id>/ancestor.json`` to the run that certified the node — opt-in through
        ``follow_records`` (library default False).  The loop tries RUN_DIR before TRUNK_DIR with one
        call and reads ``same_run = candidate == RUN_DIR`` off the CANDIDATE, so RUN_DIR must never
        follow: a re-run after a pass that reused stance from the trunk would otherwise get the
        trunk's stance back with ``same_run`` True — its results re-entered as trained here, the
        bundle refusing the node as both trained and reused, later nodes recording no
        ``parent_run_id``, and for the target (``candidates = [RUN_DIR]``) a cross-run reuse (D-A18).
        The call therefore passes exactly ``follow_records=candidate is not RUN_DIR``: the trunk
        candidate composes (a trunk that itself reused stance resolves it to the run that certified
        it), this run's own directory never does."""
        import inspect

        from environments.shared.ancestors import find_certified_ancestor

        src, loop = _chain_loop()
        find = _call(loop, "find_certified_ancestor")
        assert "follow_records" in _keyword_names(find), "the trunk candidate must opt in to following records"
        assert _keyword_source(src, find, "follow_records") == "candidate is not RUN_DIR", (
            "RUN_DIR must never follow its own record; only the trunk candidate may"
        )
        assert inspect.signature(find_certified_ancestor).parameters["follow_records"].default is False, (
            "following ancestor records stays opt-in in the library"
        )
        # And ``same_run`` is still read off the candidate, which is why RUN_DIR must not follow.
        reuse_if = _reuse_if(src, loop)
        same_run = next(
            node
            for node in reuse_if.body
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "same_run"
        )
        assert ast.unparse(same_run.value) == "candidate == RUN_DIR"

    def test_cross_run_reuse_records_ancestors_and_never_copies_checkpoints(self):
        src, loop = _chain_loop()
        record = _call(loop, "record_ancestor")
        assert [ast.unparse(arg) for arg in record.args] == ["RUN_DIR", "ancestor"]
        reuse_if = _reuse_if(src, loop)
        assert record in ast.walk(reuse_if)
        same_run_if = _the_if(reuse_if, src, lambda test: test == "same_run", "on `same_run`")
        assert record in [node for stmt in same_run_if.orelse for node in ast.walk(stmt)]
        assert record not in [node for stmt in same_run_if.body for node in ast.walk(stmt)], (
            "a same-run node re-enters the bundle from its verdict; only a CROSS-run ancestor is recorded"
        )
        assert "NODE_RESULTS[NODE.id]" in _branch_source(src, same_run_if.body)
        assert 'ancestor.verdict.get("stage_result")' in _branch_source(src, same_run_if.body)
        # Never a copy: the checkpoint is loaded from where it lives (A10). The canonical
        # library wrapper that copied it under certified_inputs/ left with consolidation PR-4.
        # D-A25: the loop consults no library; its only cross-run candidate is TRUNK_DIR.
        assert "resolve_canonical_parent" not in src and "SOURCE_SELECTION" not in src
        assert "copy_canonical_ancestor" not in src and "certified_canonical" not in src
        assert "certified_inputs" not in src
        assert "shutil" not in src
        copy_calls = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"copy", "copy2", "copyfile", "copytree", "replace", "rename"}
        ]
        assert not copy_calls, "a reused ancestor's checkpoint is never copied into this run (A10)"
        assert ".zip" not in ast.unparse(_dict_value(_handoff_assigns(reuse_if)[0].value, "model")), (
            "the handoff is the ancestor's own stem, in its own run (A10), without a duplicate extension"
        )

    def test_the_library_rule_the_notebook_relies_on(self, tmp_path):
        """Invariant 6, thin: the seven-rule reuse check the loop delegates to is fail-closed."""
        from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor

        from .reporting_helpers import make_plant_identity
        from .test_ancestors import OTHER_TASK, STANCE_CURRICULUM, STANCE_TASK, build_trunk_run, trunk_plant

        stance = load_stage_manifest("trex").by_id("stance")

        def find(run_dir, **overrides):
            kwargs = dict(
                species="trex",
                entry=stance,
                current_task_sha256=STANCE_TASK,
                plant_identity=trunk_plant(),
                current_gate_config=STANCE_CURRICULUM,
            )
            kwargs.update(overrides)
            return find_certified_ancestor(run_dir, **kwargs)

        passed = tmp_path / "passed"
        passed.mkdir()
        stage_dir = build_trunk_run(passed)
        ancestor = find(passed)
        assert ancestor.stage_dir == stage_dir and ancestor.stage_id == "stance"
        assert ancestor.model_sha256.startswith("sha256:") and ancestor.verdict["passed"] is True

        failed = tmp_path / "failed"
        failed.mkdir()
        build_trunk_run(failed, passed=False)
        with pytest.raises(AncestorReuseError, match="FAILED gate"):
            find(failed)
        with pytest.raises(AncestorReuseError, match="judged under task"):
            find(passed, current_task_sha256=OTHER_TASK)
        other_plant = make_plant_identity(
            species="trex", model_path="environments/trex/assets/trex.xml", physics_sha256="sha256:" + "9" * 64
        )
        with pytest.raises(AncestorReuseError, match="incompatible with the current trex plant"):
            find(passed, plant_identity=other_plant)
        # D-A22 (rule 7): a verdict judged under another gate is refused naming the threshold.
        with pytest.raises(AncestorReuseError, match="differing thresholds"):
            find(passed, current_gate_config={**STANCE_CURRICULUM, "max_unsupported_duty_ucb": 0.05})
        # Absence of a verdict never reads as a pass.
        unjudged = tmp_path / "unjudged"
        unjudged.mkdir()
        build_trunk_run(unjudged, verdict=False)
        with pytest.raises(AncestorReuseError, match="no gate_verdict.json"):
            find(unjudged)


class _Clock:
    """Stands in for ``datetime`` in the storage cell: every ``now()`` is one second later, so two executions
    never mint the same timestamp and only the ``_ACTIVE_RUN_ID`` memo can keep a run."""

    def __init__(self):
        self.ticks = 0

    def now(self):
        from datetime import datetime

        self.ticks += 1
        return datetime(2026, 1, 1, 0, 0, self.ticks)


def _storage_namespace(tmp_path, monkeypatch, **knobs):
    """Execute-ready namespace for the storage cell, with ``initialize_result_bundle`` recorded instead of run."""
    import types

    from environments.shared import plant_contract, result_bundle

    identity = {"species": "trex", "test_identity": True}
    monkeypatch.setattr(
        plant_contract, "current_plant_identity", lambda species: types.SimpleNamespace(to_dict=lambda: identity)
    )
    calls: list = []

    def initialize(directory, **kwargs):
        calls.append((directory, kwargs))
        return directory / "provenance.json"

    monkeypatch.setattr(result_bundle, "initialize_result_bundle", initialize)
    namespace = {
        "Path": Path,
        "datetime": _Clock(),
        "repo_root": tmp_path,
        "IN_COLAB": False,
        "USE_GOOGLE_DRIVE": False,
        "SPECIES": "trex",
        "ALGORITHM": "ppo",
        "SEED": 42,
        "N_ENVS": 1,
        "QUICK_TEST": False,
        "TRUNK_FROM": "",
        "RUN_ID": "",
        **knobs,
    }
    return namespace, calls, identity


class TestStorageCellRerun:
    """``RUN_ID`` is a configuration-cell knob (consolidation PR-14a). The storage cell resolves it into
    ``_ACTIVE_RUN_ID`` -- the memo every later cell reads as the run's id -- and never rebinds the knob, so a rerun in
    the same kernel keeps ``RUN_DIR`` even after the configuration cell reset ``RUN_ID`` to ``""``: certified nodes of
    an EARLIER run come in through ``TRUNK_FROM``, never by pointing ``RUN_ID`` at that run.
    Ported from the deleted certified-library notebook test (consolidation PR-4)."""

    def test_a_rerun_keeps_the_run_directory(self, tmp_path, monkeypatch, capsys):
        """The clock ticks a second per call, so only the memo can keep the run: a cell that dropped the memo would
        mint a second timestamp here (at 2ebed89 two executions in the same second minted the same id, so this
        test could not tell)."""
        src = _cell(STORAGE_CELL_MARKER)
        namespace, provenance_calls, identity = _storage_namespace(tmp_path, monkeypatch)
        exec(compile(src, "sb3_storage", "exec"), namespace)
        first = namespace["RUN_DIR"]
        assert first.name == "20260101_000001" and first.is_dir() and first.is_relative_to(tmp_path / "logs")
        assert namespace["TRUNK_DIR"] is None
        assert namespace["RUN_ID"] == "", "the storage cell never rebinds the knob"
        assert f"Run directory: {first} (new run)" in capsys.readouterr().out
        # The current plant identity is handed to the bundle initializer (as the deleted test pinned).
        assert provenance_calls[0][0] == first and provenance_calls[0][1]["plant_identity"] == identity
        assert provenance_calls[0][1]["run_id"] == first.name
        # Re-running the configuration cell resets the knob to ""; the memo keeps the run.
        namespace["RUN_ID"] = ""
        exec(compile(src, "sb3_storage_rerun", "exec"), namespace)
        assert namespace["RUN_DIR"] == first
        assert namespace["_ACTIVE_RUN_ID"] == first.name
        assert provenance_calls[1][1]["run_id"] == first.name, "never an empty run_id into the provenance"
        assert "(re-entering run '20260101_000001': holds no directory; bundle status none)" in capsys.readouterr().out
        # The certified-library path the cell used to derive left with consolidation PR-4.
        assert "CERTIFIED_LIBRARY" not in namespace and "CERTIFIED_LIBRARY" not in src

    def test_an_explicit_run_id_beats_the_memo_and_a_path_is_refused(self, tmp_path, monkeypatch, capsys):
        src = _cell(STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch)
        exec(compile(src, "sb3_storage", "exec"), namespace)
        assert namespace["_ACTIVE_RUN_ID"] == "20260101_000001"
        # An explicit id beats the memo: the remedy every refusal names (a fresh RUN_ID) takes effect ...
        namespace["RUN_ID"] = "20260924_120000"
        exec(compile(src, "sb3_storage_explicit", "exec"), namespace)
        assert namespace["RUN_DIR"] == tmp_path / "logs" / "trex" / "ppo" / "20260924_120000"
        assert namespace["_ACTIVE_RUN_ID"] == "20260924_120000"
        assert "(new run: RUN_ID names no existing run)" in capsys.readouterr().out
        # ... and is memoised in turn (D-D15): RUN_ID = "" now re-enters it.
        namespace["RUN_ID"] = ""
        exec(compile(src, "sb3_storage_after_explicit", "exec"), namespace)
        assert namespace["RUN_DIR"].name == namespace["_ACTIVE_RUN_ID"] == "20260924_120000"
        assert [call[1]["run_id"] for call in provenance_calls] == [
            "20260101_000001",
            "20260924_120000",
            "20260924_120000",
        ]
        # A path, "." / ".." or surrounding whitespace is not a run id: refused before anything moves.
        for bad in ("..", ".", "../20260101_000000", str(tmp_path / "elsewhere"), " 20260101_000000", "x "):
            namespace["RUN_ID"] = bad
            with pytest.raises(ValueError, match="is not a run id"):
                exec(compile(src, "sb3_storage_bad", "exec"), namespace)
        assert len(provenance_calls) == 3 and not (tmp_path / "elsewhere").exists()
        assert namespace["_ACTIVE_RUN_ID"] == "20260924_120000"

    def test_a_widened_root_under_another_seed_refuses_before_anything_is_written(self, tmp_path, monkeypatch):
        """D-C14 without the widen cell: a root widened on the command line keeps its parent's seed, and the storage
        cell refuses another SEED before ``initialize_result_bundle`` and before it moves RUN_DIR or the memo."""
        from environments.shared.result_bundle import ResultBundleError

        run_dir = tmp_path / "logs" / "trex" / "ppo" / "20260924_000000"
        stage_dir = run_dir / "01_stance"
        stage_dir.mkdir(parents=True)
        (stage_dir / "stage_config.json").write_text(
            json.dumps({"run": {"seed": 44, "n_envs": 4, "widened_from_run_id": "20260815_205206"}})
        )
        src = _cell(STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch)
        exec(compile(src, "sb3_storage_earlier_run", "exec"), namespace)  # this runtime's earlier run
        earlier = namespace["RUN_DIR"]
        namespace["RUN_ID"] = run_dir.name
        with pytest.raises(ResultBundleError, match="SEED = 42.*20260815_205206.*seed 44"):
            exec(compile(src, "sb3_storage", "exec"), namespace)
        assert len(provenance_calls) == 1 and sorted(p.name for p in run_dir.iterdir()) == ["01_stance"]
        assert namespace["RUN_DIR"] == earlier and namespace["_ACTIVE_RUN_ID"] == earlier.name
        namespace["SEED"] = 44
        exec(compile(src, "sb3_storage", "exec"), namespace)
        assert provenance_calls[1][1]["seed"] == 44 and provenance_calls[1][1]["run_id"] == run_dir.name
        assert namespace["RUN_DIR"] == run_dir

    def test_a_refused_run_id_never_reaches_the_memo(self, tmp_path, monkeypatch):
        """An explicit RUN_ID whose provenance refuses this session (another run's seed or plant) leaves the memo and
        RUN_DIR where they were, so ``RUN_ID = ""`` goes back to this runtime's run instead of re-entering the refused
        one; and a TRUNK_FROM naming the run RUN_ID resolved to names RUN_ID as the knob to change."""
        from environments.shared import result_bundle
        from environments.shared.result_bundle import ResultBundleError

        src = _cell(STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch)
        exec(compile(src, "sb3_storage", "exec"), namespace)
        earlier = namespace["RUN_DIR"]
        other = tmp_path / "logs" / "trex" / "ppo" / "20260815_205206"
        other.mkdir(parents=True)

        def initialize(directory, **kwargs):
            provenance_calls.append((directory, kwargs))
            if directory == other:
                raise ResultBundleError("run directory already belongs to a different run: {'training_seed': (44, 42)}")
            return directory / "provenance.json"

        monkeypatch.setattr(result_bundle, "initialize_result_bundle", initialize)
        namespace["RUN_ID"] = other.name
        with pytest.raises(ResultBundleError, match="belongs to a different run"):
            exec(compile(src, "sb3_storage_refused", "exec"), namespace)
        assert namespace["_ACTIVE_RUN_ID"] == earlier.name and namespace["RUN_DIR"] == earlier
        namespace["RUN_ID"] = ""
        exec(compile(src, "sb3_storage_back", "exec"), namespace)
        assert namespace["RUN_DIR"] == earlier and provenance_calls[-1][1]["run_id"] == earlier.name
        # RUN_ID and TRUNK_FROM both naming one run: the refusal names RUN_ID, the knob that was wrong.
        namespace.update(RUN_ID=earlier.name, TRUNK_FROM=earlier.name)
        with pytest.raises(RuntimeError, match=r"is the run RUN_ID resolved to .*set RUN_ID to a new id"):
            exec(compile(src, "sb3_storage_own_trunk", "exec"), namespace)

    def test_a_quick_test_run_lives_outside_the_tree_trunks_and_replicates_come_from(self, tmp_path, monkeypatch):
        """A 50k-step QUICK_TEST stance can pass a statue-level gate, so it must never be what ``TRUNK_FROM = "auto"``
        selects or what seed replication counts: its run lives under ``<algo>_quick_test/``, beside the ``<algo>/``
        tree ``select_trunk`` and ``discover_replicates_for_run`` scan, while a pinned trunk still resolves there."""
        real = tmp_path / "logs" / "trex" / "ppo" / "20260101_000000"
        real.mkdir(parents=True)
        (real / "provenance.json").write_text(
            json.dumps({"species": "trex", "algorithm": "PPO", "backend": "stable-baselines3", "run_id": real.name})
        )
        src = _cell(STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(
            tmp_path, monkeypatch, QUICK_TEST=True, TRUNK_FROM=real.name
        )
        exec(compile(src, "sb3_storage_quick", "exec"), namespace)
        quick = namespace["RUN_DIR"]
        assert quick == tmp_path / "logs" / "trex" / "ppo_quick_test" / "20260101_000001" and quick.is_dir()
        assert provenance_calls[0][0] == quick and provenance_calls[0][1]["run_id"] == quick.name
        assert namespace["TRUNK_DIR"] == real, "a pinned trunk resolves under the real <algo>/ tree"
        assert sorted(path.name for path in (tmp_path / "logs" / "trex" / "ppo").iterdir()) == [real.name]
        # A real session writes into <algo>/ exactly as before, and the refusal names the tree RUN_ID lives in.
        namespace.update(QUICK_TEST=False, RUN_ID="20260924_120000", TRUNK_FROM="")
        exec(compile(src, "sb3_storage_real", "exec"), namespace)
        assert namespace["RUN_DIR"] == tmp_path / "logs" / "trex" / "ppo" / "20260924_120000"
        namespace.update(QUICK_TEST=True, RUN_ID="../x")
        with pytest.raises(ValueError, match=r"names a run directory under .*ppo_quick_test"):
            exec(compile(src, "sb3_storage_bad_quick", "exec"), namespace)

    def test_the_memo_counts_only_in_the_tree_it_was_opened_in(self, tmp_path, monkeypatch, capsys):
        """With ``RUN_ID = ""`` the memo re-enters this runtime's run only in the tree SPECIES, ALGORITHM and
        QUICK_TEST select: a quick test first and then the real session (the usual order in one runtime) gives the
        real run its own timestamp, instead of opening ``ppo/<the quick test's id>``; a rerun in one tree keeps its
        run as before."""
        src = _cell(STORAGE_CELL_MARKER)
        namespace, provenance_calls, _ = _storage_namespace(tmp_path, monkeypatch, QUICK_TEST=True)
        exec(compile(src, "sb3_storage_quick", "exec"), namespace)
        quick = namespace["RUN_DIR"]
        assert quick == tmp_path / "logs" / "trex" / "ppo_quick_test" / "20260101_000001"
        namespace.update(QUICK_TEST=False, RUN_ID="")
        exec(compile(src, "sb3_storage_real", "exec"), namespace)
        real = namespace["RUN_DIR"]
        assert real == tmp_path / "logs" / "trex" / "ppo" / "20260101_000002", "a fresh timestamp, not the memo's id"
        assert namespace["_ACTIVE_RUN_ID"] == real.name and "(new run)" in capsys.readouterr().out
        exec(compile(src, "sb3_storage_real_rerun", "exec"), namespace)
        assert namespace["RUN_DIR"] == real, "a rerun in the same tree keeps the run"
        assert [call[1]["run_id"] for call in provenance_calls] == [quick.name, real.name, real.name]

    def test_the_trunk_and_replicate_scans_never_follow_run_dir_into_the_quick_test_tree(self):
        """The two scans the quick-test tree hides from: the auto-trunk selection names the real ``<algo>/`` tree (not
        ``RUN_DIR``'s parent), and replicate discovery scans ``RUN_DIR``'s own siblings."""
        resolve = _cell(RESOLVE_CELL_MARKER)
        select = _call(ast.parse(resolve), "select_trunk")
        assert ast.unparse(select.args[0]) == "LOG_BASE / SPECIES / ALGORITHM.lower()"
        storage = _cell(STORAGE_CELL_MARKER)
        assigns = _top_level_assigns(storage)
        assert ast.unparse(assigns["_runs_root"]) == (
            "LOG_BASE / SPECIES / (ALGORITHM.lower() + ('_quick_test' if QUICK_TEST else ''))"
        )
        assert ast.unparse(assigns["_run_dir"]) == "_runs_root / _run_id"

    def test_later_cells_read_the_resolved_run_id_never_the_knob(self):
        cells = _code_cells()
        config_at = _cell_index(cells, CONFIG_CELL_MARKER)
        storage_at = _cell_index(cells, STORAGE_CELL_MARKER)
        for index, src in enumerate(cells):
            tree = ast.parse(src)
            if index != config_at:
                assert "RUN_ID" not in _bound_names(tree), f"code cell {index} rebinds the RUN_ID knob"
            if index not in (config_at, storage_at):
                assert "RUN_ID" not in _loaded_names(tree), f"code cell {index} reads RUN_ID; read _ACTIVE_RUN_ID"
        storage = cells[storage_at]
        tree = ast.parse(storage)
        initialize = _call(tree, "initialize_result_bundle")
        assert [ast.unparse(arg) for arg in initialize.args] == ["_run_dir"]
        assert _keyword_source(storage, initialize, "run_id") == "_run_id"
        infra = cells[_cell_index(cells, INFRA_CELL_MARKER)]
        forwarded = _call(_top_level_def(infra, "save_run_bundle"), "_lib_save_result_bundle")
        assert _keyword_source(infra, forwarded, "run_id") == "_ACTIVE_RUN_ID"
        # The widened-root seed check runs on the resolved directory before the cell creates the directory or mints
        # the provenance, and RUN_DIR and the memo are bound only once the provenance accepted the run.
        seed_check = _call(tree, "refuse_widened_seed_mismatch")
        assert [ast.unparse(arg) for arg in seed_check.args] == ["_run_dir"]
        assert _keyword_source(storage, seed_check, "seed") == "SEED"
        (bind,) = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "(_ACTIVE_RUN_ID, RUN_DIR)"
        ]
        assert ast.unparse(bind.value) == "(_run_id, _run_dir)"
        mkdir = next(call for call in _calls(tree, "mkdir") if ast.unparse(call.func) == "_run_dir.mkdir")
        assert seed_check.lineno < mkdir.lineno < initialize.lineno < bind.lineno
        own_trunk = _call(tree, "load_provenance")  # the TRUNK_FROM block reads RUN_DIR, bound above it
        assert bind.lineno < own_trunk.lineno


class TestAutoTrunk:
    """Decision D-A25: ``TRUNK_FROM = "auto"`` selects the trunk run in the resolve cell."""

    def test_the_storage_cell_defers_auto_to_the_resolve_cell(self):
        src = _cell(STORAGE_CELL_MARKER)
        assigns = _top_level_assigns(src)
        assert isinstance(assigns["AUTO_TRUNK"], ast.Constant) and assigns["AUTO_TRUNK"].value is False
        tree = ast.parse(src)
        trunk_if = _the_if(tree, src, lambda test: test == "TRUNK_FROM", "resolving TRUNK_FROM")
        auto_if = _the_if(trunk_if, src, lambda test: test == 'TRUNK_FROM == "auto"', "on the auto sentinel")
        auto_body = _branch_source(src, auto_if.body)
        assert "AUTO_TRUNK = True" in auto_body
        assert "TRUNK_DIR" not in auto_body, "auto resolves no directory here: the chain is not known yet"
        # The explicit path is the else branch, unchanged: provenance, own-directory and identity refusals.
        explicit = _branch_source(src, auto_if.orelse)
        assert "load_provenance(TRUNK_DIR)" in explicit and "RUN_DIR.resolve()" in explicit

    def test_the_resolve_cell_selects_the_trunk_after_the_chain_and_before_the_loop(self):
        cells = _code_cells()
        storage_at = _cell_index(cells, STORAGE_CELL_MARKER)
        resolve_at = _cell_index(cells, RESOLVE_CELL_MARKER)
        assert storage_at < resolve_at < _cell_index(cells, CHAIN_CELL_MARKER)
        src = cells[resolve_at]
        tree = ast.parse(src)
        auto_if = _the_if(tree, src, lambda test: test == 'globals().get("AUTO_TRUNK", False)', "selecting the trunk")
        assert auto_if in tree.body, "the selection runs at the top level of the resolve cell"
        select = _call(auto_if, "select_trunk")
        assert ast.unparse(select.args[0]) == "LOG_BASE / SPECIES / ALGORITHM.lower()", (
            "the siblings under this species/algorithm log directory are scanned"
        )
        assert _keyword_source(src, select, "chain") == "CHAIN"
        assert _keyword_source(src, select, "stage_configs") == "STAGE_CONFIGS"
        assert _keyword_source(src, select, "plant_identity") == "PLANT_IDENTITY"
        assert _keyword_source(src, select, "exclude") == "(RUN_DIR,)", "this run is never its own trunk"
        assert _keyword_source(src, select, "retrain_from") == "RETRAIN_NODE", "D-A19 bounds what is consulted"
        assert "widen_from" not in _keyword_names(select), "D-D14: no widen session is special-cased"
        assert _keyword_source(src, select, "algorithm") == "ALGORITHM", "a SAC run never trunks a PPO chain"
        body = _branch_source(src, auto_if.body)
        assert "TRUNK_DIR = TRUNK_SELECTION.run_dir" in body
        assert "TRUNK_SELECTION.describe()" in body, "the choice and every refusal are printed"
        # The chain and RETRAIN_NODE the selection reads are bound earlier in the same cell.
        assigns = _top_level_assigns(src)
        assert "CHAIN" in assigns and "RETRAIN_NODE" in assigns
        assert assigns["CHAIN"].lineno < auto_if.lineno and assigns["RETRAIN_NODE"].lineno < auto_if.lineno

    def test_the_chain_loop_rederives_the_auto_trunk_from_the_selection(self):
        """A storage-cell rerun resets TRUNK_DIR to None (only a pinned TRUNK_FROM fills it there), so
        under "auto" the loop takes TRUNK_DIR from the resolve cell's selection for THIS log directory
        and refuses a missing or foreign selection."""
        src, loop = _chain_loop()
        tree = ast.parse(src)
        auto_if = _the_if(tree, src, lambda test: test == 'globals().get("AUTO_TRUNK", False)', "re-deriving the trunk")
        assert auto_if in tree.body and auto_if.lineno < loop.lineno, "before the loop, at the top level"
        body = _branch_source(src, auto_if.body)
        assert 'globals().get("TRUNK_SELECTION")' in body
        assert "LOG_BASE / SPECIES / ALGORITHM.lower()" in body, "a selection for another log directory is refused"
        assert "TRUNK_DIR = _selection.run_dir" in body
        assert len(_raises(auto_if, "RuntimeError")) == 1
        assert "WIDEN_FROM" not in src, "D-D14: widening is the command-line tool's; no widen session is special-cased"

    def test_the_resolve_cell_ends_with_the_complete_run_and_widened_root_refusals(self):
        cells = _code_cells()
        resolve_at = _cell_index(cells, RESOLVE_CELL_MARKER)
        src = cells[resolve_at]
        tree = ast.parse(src)
        auto_if = _the_if(tree, src, lambda test: test == 'globals().get("AUTO_TRUNK", False)', "selecting the trunk")
        guard = _the_if(tree, src, lambda test: test == 'globals().get("RUN_DIR") is not None', "on a bound RUN_DIR")
        assert guard is tree.body[-1] and guard.lineno > auto_if.end_lineno, "last, once TRUNK_DIR is final"
        complete = _call(guard, "refuse_complete_run_session")
        widened = _call(guard, "refuse_trunk_over_unjudged_widened_root")
        assert complete.lineno < widened.lineno
        for call in (complete, widened):
            assert [ast.unparse(arg) for arg in call.args] == ["RUN_DIR"]
            for keyword, value in (
                ("species", "SPECIES"),
                ("chain", "CHAIN"),
                ("target", "TARGET_NODE"),
                ("retrain_from", "RETRAIN_NODE"),
                ("trunk_dir", "TRUNK_DIR"),
            ):
                assert _keyword_source(src, call, keyword) == value, keyword
        # A plain refusal: no disconnect (imported later, by the infrastructure cell), and the loop has not run yet.
        assert not _calls(tree, "disconnect_runtime") and not _calls(tree, "halt")
        assert resolve_at < _cell_index(cells, CHAIN_CELL_MARKER)

    def test_the_resolve_cell_refuses_a_complete_run_before_anything_is_written(self, tmp_path):
        """Executed: a complete run re-entered for a node it lacks refuses; a reuse-only session passes."""
        from environments.shared.config import load_all_stages
        from environments.shared.result_bundle import DEFAULT_MANIFEST_NAME, ResultBundleError
        from environments.shared.stage_manifest import stage_dirname

        run_dir = tmp_path / "20260920_010912"
        stance_dir = run_dir / stage_dirname("velociraptor", 1)
        stance_dir.mkdir(parents=True)
        (stance_dir / "gate_verdict.json").write_text("{}")
        (run_dir / DEFAULT_MANIFEST_NAME).write_text(json.dumps({"status": "complete"}))
        before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
        namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        namespace.update(RUN_DIR=run_dir, TRUNK_DIR=None, BEHAVIOR="stance")
        exec(_cell(RESOLVE_CELL_MARKER), namespace)  # stance is reused in place: nothing to refuse
        namespace["BEHAVIOR"] = "walk"
        with pytest.raises(ResultBundleError, match="would judge or train 'locomotion' here") as excinfo:
            exec(_cell(RESOLVE_CELL_MARKER), namespace)
        assert 'TRUNK_FROM = "20260920_010912"' in str(excinfo.value)
        assert {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == before

    def test_the_resolve_cell_refuses_a_trunk_over_an_unjudged_widened_root(self, tmp_path):
        """Executed, D-C13 without the widen cell's ``TRUNK_DIR = None``: a root widened into this run on the command
        line is judged by the chain loop before any trunk may stand in for it."""
        from environments.shared.config import load_all_stages
        from environments.shared.result_bundle import ResultBundleError
        from environments.shared.stage_manifest import stage_dirname

        run_dir = tmp_path / "20260924_000000"
        stance_dir = run_dir / stage_dirname("velociraptor", 1)
        stance_dir.mkdir(parents=True)
        (stance_dir / "stage_config.json").write_text(
            json.dumps({"run": {"seed": 42, "n_envs": 4, "widened_from_run_id": "20260815_205206"}})
        )
        namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        namespace.update(RUN_DIR=run_dir, TRUNK_DIR=None, BEHAVIOR="walk")
        exec(_cell(RESOLVE_CELL_MARKER), namespace)  # no trunk: the loop judges the widened root
        namespace["TRUNK_DIR"] = tmp_path / "20260914_123816"
        with pytest.raises(ResultBundleError, match="never judge the widened root"):
            exec(_cell(RESOLVE_CELL_MARKER), namespace)

    def test_the_selection_it_delegates_to(self, tmp_path, monkeypatch):
        """Invariant, thin: the selector prefers coverage, then recency, and reuses the seven-rule check."""
        from environments.shared import task_fingerprint as task_fingerprint_module
        from environments.shared.ancestors import select_trunk

        from .test_ancestors import (
            LOCOMOTION_CURRICULUM,
            LOCOMOTION_TASK,
            STANCE_CURRICULUM,
            STANCE_TASK,
            build_chained_trunk,
            build_trunk_run,
            trunk_plant,
        )

        build_trunk_run(tmp_path / "20260910_000000")
        build_chained_trunk(tmp_path / "20260905_000000")
        digests = {1: STANCE_TASK, 2: LOCOMOTION_TASK}
        monkeypatch.setattr(
            task_fingerprint_module,
            "derive_stage_task_fingerprint",
            lambda **kwargs: {"task_sha256": digests[kwargs["stage"]]},
        )
        manifest = load_stage_manifest("trex")
        selection = select_trunk(
            tmp_path,
            species="trex",
            chain=tuple(manifest.by_id(stage_id) for stage_id in ("stance", "locomotion", "behavior")),
            stage_configs={
                1: {"env_kwargs": {}, "curriculum_kwargs": STANCE_CURRICULUM},
                2: {"env_kwargs": {}, "curriculum_kwargs": LOCOMOTION_CURRICULUM},
                3: {"env_kwargs": {}, "curriculum_kwargs": {}},
            },
            plant_identity=trunk_plant(),
        )
        assert selection.run_dir == tmp_path / "20260905_000000"
        assert [candidate.coverage for candidate in selection.candidates] == [1, 2]


class TestLoadModeByEdge:
    """The load mode is DECLARED by the edge, never inferred from a position."""

    def test_train_stage_requires_a_declared_load_mode(self):
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        kwonly = [arg.arg for arg in train_stage.args.kwonlyargs]
        assert "task_load_mode" in kwonly, "task_load_mode must be keyword-only"
        assert train_stage.args.kw_defaults[kwonly.index("task_load_mode")] is None, "task_load_mode has no default"
        for name in ("load_path", "run_dir", "vecnorm_path", "parent_run_id", "label"):
            assert name in kwonly, f"{name} is a keyword-only argument of train_stage"
        assert "if task_load_mode is None:" not in src, "no inference block: the mode is declared or refused"
        unknown = _the_if(
            train_stage,
            src,
            lambda test: test.startswith("task_load_mode not in"),
            "refusing an unknown task_load_mode",
        )
        assert _raises(unknown, "ValueError")

    def test_no_position_name_survives_in_the_training_or_chain_cell(self):
        for marker in (INFRA_CELL_MARKER, CHAIN_CELL_MARKER, MANUAL_CELL_MARKER, RESUME_CELL_MARKER):
            src = _cell(marker)
            tree = ast.parse(src)
            assert "stage_position" not in _names(tree), f"{marker!r}: stage_position is back"
            position_compares = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Compare)
                and any(
                    isinstance(operand, ast.Attribute) and operand.attr == "position"
                    for operand in (node.left, *node.comparators)
                )
            ]
            assert not position_compares, f"{marker!r} decides something by comparing a `.position`"

    def test_shaping_is_keyed_on_the_edge_and_applied_to_sac(self):
        """train_stage trains through train_base.train (consolidation PR-14c), whose shaping this pins."""
        src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        train = _top_level_def(src, "train")
        shaping = _call(train, "_stage_entry_shaping_callbacks")
        assert _keyword_names(shaping) == {"task_load_mode", "parent_id", "load_path"}
        assert _keyword_source(src, shaping, "parent_id") == "entry.warm_start_from", (
            "shaping fires on the declared EDGE (warm_start_from), verbatim like every CLI caller"
        )
        assert _keyword_source(src, shaping, "task_load_mode") == "task_load_mode"
        extend = next(call for call in _calls(train, "extend") if shaping in call.args)
        assert ast.unparse(extend.func) == "callbacks.extend" and len(extend.args) == 1, (
            "the shaping callbacks are applied unfiltered — SAC gets the same warm-up the CLI gives it (DU1)"
        )
        for index, cell in enumerate(_code_cells()):
            assert "StageWarmupCallback" not in cell, f"cell {index} filters or names StageWarmupCallback"

    def test_a_root_node_or_missing_parent_checkpoint_refuses_initialize_next_stage(self):
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        no_load = _the_if(
            train_stage,
            src,
            lambda test: test == 'task_load_mode == "initialize_next_stage" and not load_path',
            "refusing initialize_next_stage without a load_path",
        )
        root = _the_if(
            train_stage,
            src,
            lambda test: test == 'task_load_mode == "initialize_next_stage" and NODE.warm_start_from is None',
            "refusing initialize_next_stage on a root",
        )
        pair = _the_if(
            train_stage,
            src,
            lambda test: test == "bool(load_path) != bool(vecnorm_path)",
            "refusing a checkpoint without its sidecar",
        )
        for guard in (no_load, root, pair):
            assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise), (
                "each refusal is an unconditional raise"
            )
            assert _raises(guard, "ValueError")
            assert guard.lineno < _call(train_stage, "train").lineno, (
                "argument refusals happen before train_base.train touches the stage directory"
            )
        # The chain loop declares the edge from the parent's presence, never from a number.
        chain_src, loop = _chain_loop()
        edge_if = _the_if(
            loop,
            chain_src,
            lambda test: test == "parent_handoff is not None",
            "choosing the load mode from the parent's handoff",
        )
        assert '"initialize_next_stage"' in _branch_source(chain_src, edge_if.body)
        assert '"resume_same_stage"' in _branch_source(chain_src, edge_if.orelse)
        assert 'parent_handoff["model"]' in _branch_source(chain_src, edge_if.body)
        assert 'parent_handoff["vecnorm"]' in _branch_source(chain_src, edge_if.body)
        train = _call(loop, "train_stage")
        assert _keyword_source(chain_src, train, "task_load_mode") == "task_load_mode"
        assert _keyword_source(chain_src, train, "load_path") == "load_path"
        assert _keyword_source(chain_src, train, "vecnorm_path") == "vecnorm_path"
        assert _keyword_source(chain_src, train, "run_dir") == "RUN_DIR"


class TestTrainStageRecordKeeping:
    """``train_stage`` trains through ``train_base.train`` (consolidation PR-14c): D-A20, the declared-parent check,
    the stage record, D-D11's seed and D-A15's duration are train()'s (pinned in test_train_base.py)."""

    def test_train_stage_trains_through_train_base(self):
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        call = _call(train_stage, "train")
        assert ast.unparse(call.func) == "train_base.train"
        assert [ast.unparse(arg) for arg in call.args] == ["SPECIES_CFG", "STAGE_CONFIGS", "stage", "timesteps"]
        assert {keyword.arg: ast.get_source_segment(src, keyword.value) for keyword in call.keywords} == {
            "n_envs": "N_ENVS",
            "seed": "SEED",
            "load_path": "load_path",
            "vecnorm_path": "vecnorm_path",
            "eval_freq": "eval_freq",
            "save_freq": "save_freq",
            "verbose": "VERBOSE",
            "algorithm": "ALGORITHM.lower()",
            "output_dir": "str(stage_dir)",
            "task_load_mode": "task_load_mode",
            "parent_run_id": "parent_run_id",
            "label": "label",
            # evaluate_stage_checkpoints replaces the HPT report and metrics.json, and a stop propagates before
            # the final save: the node keeps its periodic checkpoints for the RESUME cell and is never judged on
            # a truncated final.
            "report_metrics": "False",
            "save_on_interrupt": "False",
        }
        tree = ast.parse(src)
        for name in ("save_stage_config", "refuse_occupied_stage_dir", "validate_declared_parent", "learn"):
            assert not _calls(tree, name), f"the infrastructure cell calls {name} itself instead of through train()"
        assert "stamp_canonical_training" not in src and "canonical_training_origin" not in src

    def test_the_evaluation_reports_the_recorded_duration(self):
        """D-A15: train() records the duration at its final save (test_train_base.py); the evaluation reads the
        record back, as the chain loop's JUDGE branch does."""
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        evaluate = _call(train_stage, "evaluate_stage_checkpoints")
        assert _call(train_stage, "train").lineno < evaluate.lineno
        assert _keyword_source(src, evaluate, "duration_seconds") == "read_stage_duration(stage_dir)"
        assert _keyword_source(src, evaluate, "timesteps") == "int(model.num_timesteps)"


class TestJudgeBranch:
    """The evaluation tail is a top-level function the JUDGE branch can call without training."""

    def test_evaluate_stage_checkpoints_is_the_librarys_and_train_stage_returns_its_tuple(self):
        """The evaluation tail is ``reporting.evaluate_stage_checkpoints`` (consolidation PR-14c; its internals are
        pinned in test_reporting_stage_artifacts.py), called with this session's species, plant and seeds."""
        src = _cell(INFRA_CELL_MARKER)
        assert "from environments.shared.reporting import evaluate_stage_checkpoints, generate_stage_artifacts" in src
        assert "def evaluate_stage_checkpoints(" not in src
        for name in ("_eval_forward_vel", "save_evaluation_episodes", "load_sb3_model", "eval_policy"):
            assert name not in src, f"the infrastructure cell evaluates by itself ({name})"
        train_stage = _top_level_def(src, "train_stage")
        delegate = _call(train_stage, "evaluate_stage_checkpoints")
        assert [ast.unparse(arg) for arg in delegate.args] == [
            "SPECIES_CFG",
            "config",
            "stage",
            "ALGORITHM",
            "stage_dir",
        ]
        assert _keyword_source(src, delegate, "plant_identity") == "PLANT_IDENTITY"
        # SEED + 3000: it equals the library's PUBLICATION_SEED_START only for SEED 42.
        assert _keyword_source(src, delegate, "evaluation_seed") == "EVALUATION_SEED"
        assert _keyword_source(src, delegate, "model") == "model", "the in-memory model is passed after training"
        # train_stage returns the unchanged 6-tuple.
        returns = [node for node in ast.walk(train_stage) if isinstance(node, ast.Return)]
        assert len(returns) == 1 and isinstance(returns[0].value, ast.Tuple) and len(returns[0].value.elts) == 6

    def test_the_loop_judges_from_disk_with_the_recorded_duration(self):
        src, loop = _chain_loop()
        judge_if = _judge_if(src, loop)
        call = _call(judge_if, "evaluate_stage_checkpoints")
        assert [ast.unparse(arg) for arg in call.args] == [
            "SPECIES_CFG",
            "STAGE_CONFIGS[stage]",
            "stage",
            "ALGORITHM",
            "stage_dir",
        ]
        assert _keyword_source(src, call, "plant_identity") == "PLANT_IDENTITY"
        assert _keyword_source(src, call, "evaluation_seed") == "EVALUATION_SEED"
        assert "model" not in _keyword_names(call), "the JUDGE branch has no in-memory model: it loads the final zip"
        assert _keyword_source(src, call, "final_path") == "final_stem"
        assert _keyword_source(src, call, "final_vecnorm_path") == "final_vecnorm"
        assert "read_stage_duration(stage_dir)" in _keyword_source(src, call, "duration_seconds"), (
            "D-A15: a judged node reports the duration train_stage recorded"
        )
        test = ast.get_source_segment(src, judge_if.test) or ""
        assert "verdict is None" in test
        assert "final_stem" in test and ".exists()" in test and "final_vecnorm" in test, (
            "JUDGE needs the final checkpoint AND its sidecar"
        )

    def test_the_loop_judges_a_root_widened_on_the_command_line(self):
        """What the command-line widen tool relies on downstream (decision D-D14): the loop REUSE-refuses a
        verdict-less directory (printed, never silent), reads no verdict, and takes the JUDGE branch on the
        ``<stage_label>_final`` pair the tool wrote."""
        from environments.shared.scripts.widen_checkpoint import FORBIDDEN_OUTPUT_FILES

        assert "gate_verdict.json" in FORBIDDEN_OUTPUT_FILES, "the tool never mints a verdict: the loop must judge"
        src, loop = _chain_loop()
        judge_if = _judge_if(src, loop)
        test = ast.get_source_segment(src, judge_if.test) or ""
        assert "verdict is None" in test and "final_stem" in test and "final_vecnorm" in test
        final_assign = next(
            node for node in loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "final_stem"
        )
        assert "stage_label(stage)" in ast.unparse(final_assign.value) and "_final" in ast.unparse(
            final_assign.value
        ), "JUDGE keys on <stage_label>_final — the byte-identical copies widen_checkpoint writes (D-C9)"
        # Reuse of the widened directory is refused by the library (no verdict) and the loop prints the reason.
        handler = next(
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.ExceptHandler)
            and node.type is not None
            and ast.unparse(node.type) == "AncestorReuseError"
        )
        assert _calls(handler, "print")

    def test_the_widen_tool_documents_how_the_notebook_judges_its_output(self, capsys):
        """D-D14 made the command line the widen path: ``--to-stage-dir``'s help names the run directory the storage
        cell resolves for ``RUN_ID`` and the knobs the judging session sets."""
        from environments.shared.scripts import widen_checkpoint as widen_module

        with pytest.raises(SystemExit) as excinfo:
            widen_module.main(["--help"])
        assert excinfo.value.code == 0
        help_text = " ".join(capsys.readouterr().out.split())
        for phrase in (
            "<LOG_BASE>/<species>/<algo>/<new run",
            "<stage_dirname(species, root)>",
            "has not opened yet",
            'RUN_ID = "<new run id>"',
            "SEED = the parent's run seed",
            'TRUNK_FROM = ""',
            "YYYYMMDD_HHMMSS",
        ):
            assert phrase in help_text, phrase
        doc = " ".join((widen_module.__doc__ or "").split())
        for phrase in (
            "D-D14",
            "--label",
            "/content/mesozoic-labs",
            "D-C13",
            "D-C14",
            "YYYYMMDD_HHMMSS",
            'drive.mount("/content/drive")',
            "never the storage cell",
            "/content/drive/MyDrive/mesozoic-labs/logs",
        ):
            assert phrase in doc, phrase


class TestChainLoop:
    """The loop walks CHAIN root-first and does exactly one of REUSE / JUDGE / TRAIN per node."""

    def test_the_loop_walks_the_chain_in_manifest_order(self):
        src, loop = _chain_loop()  # _top_level_for: the one top-level `for NODE in CHAIN:`
        for index, cell in enumerate(_code_cells()):
            for node in ast.walk(ast.parse(cell)):
                if not isinstance(node, ast.For):
                    continue
                iterable = node.iter
                assert not (
                    isinstance(iterable, (ast.Tuple, ast.List))
                    and iterable.elts
                    and all(isinstance(elt, ast.Constant) and isinstance(elt.value, int) for elt in iterable.elts)
                ), f"cell {index} walks a literal tuple of stage numbers"
                assert not (
                    isinstance(iterable, ast.Call)
                    and isinstance(iterable.func, ast.Name)
                    and iterable.func.id == "range"
                    and iterable.args
                    and all(isinstance(arg, ast.Constant) for arg in iterable.args)
                ), f"cell {index} walks range(...) of stage numbers"
        # The parent is resolved through the manifest and must be certified in this session.
        body = _branch_source(src, loop.body)
        assert "MANIFEST.parent_of(NODE.id)" in body
        parent_if = _the_if(
            loop,
            src,
            lambda test: test == "parent is not None and parent.id not in NODE_HANDOFF",
            "on an uncertified parent",
        )
        assert _raises(parent_if, "RuntimeError")

    def test_branches_are_reuse_then_judge_then_train(self):
        src, loop = _chain_loop()
        body = _branch_source(src, loop.body)
        order = [
            "find_certified_ancestor(",
            "read_gate_verdict(",
            "evaluate_stage_checkpoints(",
            "train_stage(",
            "generate_stage_artifacts(",
            "save_run_bundle(",
            "publication_gate_passed",
            "NODE_HANDOFF[NODE.id] =",
        ]
        # First occurrence of each step; the handoff assignment is the trained node's (last) one —
        # the reused node's copy sits inside the REUSE branch, before `continue`.
        positions = [body.rindex(item) if item.startswith("NODE_HANDOFF") else body.index(item) for item in order]
        assert positions == sorted(positions), f"chain-loop order regressed: {list(zip(order, positions))}"
        last = loop.body[-1]
        assert isinstance(last, ast.Assign) and ast.unparse(last.targets[0]) == "NODE_HANDOFF[NODE.id]", (
            "the certified handoff is the LAST statement of the body: nothing after the gate check can be skipped"
        )
        assert len(_calls(loop, "train_stage")) == 1
        assert len(_calls(loop, "generate_stage_artifacts")) == 1
        assert len(_calls(loop, "find_certified_ancestor")) == 1

    def test_a_reused_node_continues_without_training(self):
        src, loop = _chain_loop()
        reuse_if = _reuse_if(src, loop)
        assert isinstance(reuse_if.body[-1], ast.Continue), "a reused node ends its iteration with `continue`"
        for name in ("train_stage", "generate_stage_artifacts", "evaluate_stage_checkpoints", "freeze_recovery_gate"):
            assert not _calls(reuse_if, name), f"the REUSE branch calls {name}"
        assert not _calls(reuse_if, "save_run_bundle"), "a reused node writes no bundle; the next trained node does"
        handoffs = _handoff_assigns(reuse_if)
        assert len(handoffs) == 1
        reused = _dict_value(handoffs[0].value, "reused")
        assert isinstance(reused, ast.Constant) and reused.value is True
        assert "completed_stages.append(" in _branch_source(src, reuse_if.body), (
            "a same-run reused node re-enters this run's results"
        )

    def test_the_judge_branch_never_trains_and_the_train_branch_never_judges_first(self):
        src, loop = _chain_loop()
        judge_if = _judge_if(src, loop)
        judge_body = ast.Module(body=judge_if.body, type_ignores=[])
        assert not _calls(judge_body, "train_stage")
        assert not _calls(judge_body, "freeze_recovery_gate")
        train = ast.Module(body=_train_branch(judge_if), type_ignores=[])
        assert _calls(train, "train_stage")
        assert not _calls(train, "evaluate_stage_checkpoints"), "TRAIN evaluates through train_stage, never twice"
        assert not _calls(train, "read_gate_verdict")
        # The verdict is read ONCE, before the JUDGE/TRAIN decision, and after REUSE.
        verdict_read = _call(loop, "read_gate_verdict")
        assert _reuse_if(src, loop).lineno < verdict_read.lineno < judge_if.lineno
        # Both branches feed the same post-training tail (panel, artifacts, bundle, gate).
        artifacts = _call(loop, "generate_stage_artifacts")
        assert artifacts in [node for stmt in loop.body for node in ast.walk(stmt)]
        assert artifacts.lineno > judge_if.end_lineno, "artifacts are generated after the branch, for both"
        assert _keyword_source(src, artifacts, "stage_results") == "results"
        assert _keyword_source(src, artifacts, "stage_dir") == "stage_dir"

    def test_an_interrupted_or_failed_node_is_never_retrained_silently(self):
        src, loop = _chain_loop()
        verdict_if = _verdict_if(src, loop)
        assert verdict_if.lineno < _judge_if(src, loop).lineno
        # Every path through an existing verdict raises: FAILED, or passed-but-refused by the reuse rule.
        failed_if = _the_if(verdict_if, src, lambda test: test == 'not verdict["passed"]', "on a FAILED verdict")
        assert len(failed_if.body) == 1 and isinstance(failed_if.body[0], ast.Raise)
        assert _raises(failed_if, "RuntimeError")
        assert isinstance(verdict_if.body[-1], ast.Raise), (
            "a passed verdict the reuse rule refused is not retrained over"
        )
        assert len(_raises(verdict_if, "RuntimeError")) == 2
        assert not _calls(verdict_if, "train_stage")
        # Periodic checkpoints without a final one: an interrupted node, pointed at the RESUME cell.
        judge_if = _judge_if(src, loop)
        interrupted = judge_if.orelse[0]
        assert isinstance(interrupted, ast.If)
        test = ast.get_source_segment(src, interrupted.test) or ""
        assert "verdict is None" in test and '.glob("*.zip")' in test
        assert len(interrupted.body) == 1 and isinstance(interrupted.body[0], ast.Raise)
        assert "RESUME_STAGE" in ast.unparse(interrupted.body[0]), "the message names the RESUME cell's knob"

    def test_a_write_into_a_complete_run_is_refused_before_it_happens(self):
        """A complete bundle is immutable (consolidation PR-14a). The resolve cell predicts from the directory alone;
        what it cannot see without the reuse rule -- a node held only as an ``ancestors/`` record that the trunk no
        longer certifies, or a trunk's copy recorded into a run that lacks the record -- is refused here, before
        the write."""
        src, loop = _chain_loop()
        record_guard, backstop = sorted(_calls(loop, "refuse_write_into_complete_run"), key=lambda call: call.lineno)
        for call in (record_guard, backstop):
            assert [ast.unparse(arg) for arg in call.args] == ["RUN_DIR"]
        # The record guard: in REUSE, only for a record the run does not hold yet (record_ancestor is a no-op for
        # the same record and refuses a different one), before record_ancestor writes it.
        reuse_if = _reuse_if(src, loop)
        record_if = _the_if(
            reuse_if,
            src,
            lambda test: test == "not (Path(RUN_DIR) / ANCESTORS_DIRNAME / NODE.id / ANCESTOR_RECORD_NAME).is_file()",
            "on a record the run does not hold",
        )
        assert record_if.body[0].value is record_guard
        assert record_if.end_lineno < _call(reuse_if, "record_ancestor").lineno
        # The backstop: a top-level statement after the verdict refusals, before JUDGE, RESUME's refusal and TRAIN.
        assert any(isinstance(stmt, ast.Expr) and stmt.value is backstop for stmt in loop.body)
        assert _verdict_if(src, loop).end_lineno < backstop.lineno < _judge_if(src, loop).lineno

    @staticmethod
    def _complete_run_loop(tmp_path, monkeypatch, *, find, prepare):
        """Execute the configuration, resolve and chain-loop cells for a complete velociraptor run under
        ``BEHAVIOR = "walk"`` with a pinned trunk run; ``find`` stands in for ``find_certified_ancestor`` (the reuse
        rule) and ``prepare(run_dir)`` adds what the run holds beside its judged locomotion. Returns the run
        directory, its files before the loop, and the ``train_stage`` / ``record_ancestor`` calls made."""
        from environments.shared import ancestors
        from environments.shared.config import load_all_stages
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.stage_manifest import stage_dirname, stage_label

        run_dir = tmp_path / "20260920_010912"
        locomotion = run_dir / stage_dirname("velociraptor", "locomotion")
        locomotion.mkdir(parents=True)
        (locomotion / "gate_verdict.json").write_text("{}")
        (run_dir / "artifact_manifest.json").write_text(json.dumps({"status": "complete"}))
        prepare(run_dir)
        before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
        namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        namespace.update(RUN_DIR=run_dir, TRUNK_DIR=tmp_path / "20260921_000000", BEHAVIOR="walk")
        exec(_cell(RESOLVE_CELL_MARKER), namespace)  # the directory alone predicts a reuse-only session
        trained: list = []
        recorded: list = []
        monkeypatch.setattr(ancestors, "find_certified_ancestor", find)
        monkeypatch.setattr(ancestors, "record_ancestor", lambda directory, ancestor: recorded.append(ancestor))
        namespace.update(
            Path=Path,
            stage_dirname=stage_dirname,
            stage_label=stage_label,
            PLANT_IDENTITY=current_plant_identity("velociraptor"),
            NODE_RESULTS={},
            completed_stages=[],
            NODE_HANDOFF={},
            train_stage=lambda **kwargs: trained.append(kwargs),
            QUICK_TEST=True,
        )
        try:
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        finally:
            after = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
            assert after == before, "nothing is written into the complete run"
            assert not trained and not recorded, "nothing is trained or recorded into the complete run"

    def test_executed_a_record_the_trunk_no_longer_certifies_is_never_trained_into_a_complete_run(
        self, tmp_path, monkeypatch
    ):
        from environments.shared.ancestors import AncestorReuseError
        from environments.shared.result_bundle import ResultBundleError

        def refuse(candidate, **kwargs):
            raise AncestorReuseError("not certified here")

        def record_stance(run_dir):
            record = run_dir / "ancestors" / "stance" / "ancestor.json"
            record.parent.mkdir(parents=True)
            record.write_text("{}")

        with pytest.raises(ResultBundleError, match="Judging or training 'stance' would write into run") as excinfo:
            self._complete_run_loop(tmp_path, monkeypatch, find=refuse, prepare=record_stance)
        assert 'TRUNK_FROM = "20260920_010912"' in str(excinfo.value)

    def test_executed_a_trunk_copy_is_never_recorded_into_a_complete_run(self, tmp_path, monkeypatch):
        """The run's own stance verdict is refused by the reuse rule while the trunk's copy is accepted."""
        import types

        from environments.shared.ancestors import AncestorReuseError
        from environments.shared.result_bundle import ResultBundleError
        from environments.shared.stage_manifest import stage_dirname

        trunk = tmp_path / "20260921_000000"
        copy = types.SimpleNamespace(stage_id="stance", run_id=trunk.name, stage_dir=trunk / "01_stance")

        def find(candidate, **kwargs):
            if candidate == trunk and kwargs["entry"].id == "stance":
                return copy
            raise AncestorReuseError("refused by the reuse rule")

        def judge_stance(run_dir):
            stance = run_dir / stage_dirname("velociraptor", "stance")
            stance.mkdir(parents=True)
            (stance / "gate_verdict.json").write_text("{}")

        with pytest.raises(ResultBundleError, match="Recording 'stance' from run 20260921_000000 would write into"):
            self._complete_run_loop(tmp_path, monkeypatch, find=find, prepare=judge_stance)


class TestFrozenNullFlow:
    """A frozen-null gate kind freezes BEFORE training and rolls its panel after, keyed on the kind set."""

    def test_freeze_before_train_roll_after(self):
        src, loop = _chain_loop()
        train_branch = _train_branch(_judge_if(src, loop))
        train_module = ast.Module(body=train_branch, type_ignores=[])
        freeze = _call(train_module, "freeze_recovery_gate")
        validate = _call(train_module, "validate_recovery_resolution")
        train = _call(train_module, "train_stage")
        assert freeze.lineno < validate.lineno < train.lineno, (
            "pre-registration: freeze, validate the resolution, THEN spend the budget"
        )
        panel = _call(loop, "roll_policy_panel")
        evidence = _call(loop, "write_recovery_evidence")
        artifacts = _call(loop, "generate_stage_artifacts")
        assert train.lineno < panel.lineno < evidence.lineno < artifacts.lineno
        assert _keyword_source(src, artifacts, "recovery_successes_by_seed") == "panel_successes"
        assert "panel_successes = None" in _branch_source(src, loop.body), "a non-frozen kind passes no panel"
        # The freeze uses the PARENT's handoff as the brace null; a root under the kind is refused first.
        assert "parent_handoff" in _keyword_source(src, freeze, "policy_zip")
        assert "parent_handoff" in _keyword_source(src, freeze, "vecnorm")
        assert _keyword_source(src, freeze, "stage") == "stage"
        root_if = _the_if(
            train_module,
            src,
            lambda test: "FROZEN_NULL_GATE_KINDS" in test and "parent_handoff is None" in test,
            "refusing a root under a frozen-null kind",
        )
        assert _raises(root_if, "RuntimeError") and root_if.lineno < freeze.lineno
        # Every frozen-null step is guarded by the kind SET, never by the id.
        for call in (freeze, validate, panel, evidence):
            guard = next(
                node
                for node in ast.walk(loop)
                if isinstance(node, ast.If) and call in [inner for stmt in node.body for inner in ast.walk(stmt)]
            )
            assert "FROZEN_NULL_GATE_KINDS" in _names(guard.test), f"{_func_name(call)} is not keyed on the kind set"
        freeze_guard = next(
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.If) and freeze in [inner for stmt in node.body for inner in ast.walk(stmt)]
        )
        assert "gate_resolution.json" in (ast.get_source_segment(src, freeze_guard.test) or ""), (
            "an existing resolution is validated, never re-frozen"
        )
        for marker in (CHAIN_CELL_MARKER, MANUAL_CELL_MARKER):
            assert '== "recovery"' not in _cell(marker), "the frozen-null flow is keyed on the kind, not the id"

    def test_the_gate_kind_set_the_notebook_relies_on(self):
        from environments.shared.curriculum import FROZEN_NULL_GATE_KINDS, GATE_KINDS
        from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND

        assert FROZEN_NULL_GATE_KINDS == frozenset({"recovery_quality/v1"})
        assert FROZEN_NULL_GATE_KINDS == frozenset({RECOVERY_GATE_KIND})
        assert FROZEN_NULL_GATE_KINDS <= set(GATE_KINDS)
        assert isinstance(FROZEN_NULL_GATE_KINDS, frozenset)
        # The trex recovery stage declares that kind, so BEHAVIOR="stand" takes the frozen-null flow.
        from environments.shared.config import load_all_stages

        recovery = load_all_stages("trex")["recovery"]
        assert recovery["curriculum_kwargs"]["gate_kind"] in FROZEN_NULL_GATE_KINDS


class TestPublication:
    """Invariant 5 (notebook side): publish every certified deliverable, then enforce, then stop."""

    def test_every_bundle_write_passes_chain_results_and_the_target_deliverable(self):
        cells = _code_cells()
        bundle_calls = summary_calls = 0
        for index, src in enumerate(cells):
            tree = ast.parse(src)
            assert "curriculum_results" not in src, f"cell {index} still names curriculum_results"
            for call in _calls(tree, "save_run_bundle"):
                bundle_calls += 1
                assert call.args and ast.unparse(call.args[0]) == "chain_results()", f"cell {index} line {call.lineno}"
                assert _keyword_source(src, call, "species") == "SPECIES"
            for call in _calls(tree, "write_training_summary"):
                summary_calls += 1
                assert [ast.unparse(arg) for arg in call.args] == ["RUN_DIR", "chain_results()"], (
                    f"cell {index} line {call.lineno}"
                )
        assert bundle_calls == 2 and summary_calls == 2, "the chain loop and the manual cell each write once"
        infra = cells[_cell_index(cells, INFRA_CELL_MARKER)]
        bundle_def = _top_level_def(infra, "save_run_bundle")
        forwarded = _call(bundle_def, "_lib_save_result_bundle")
        assert _keyword_source(infra, forwarded, "target_deliverable") == "TARGET_NODE.key", (
            "the bundle status is judged against BEHAVIOR's node (result schema v4)"
        )
        assert ast.unparse(forwarded.args[0]) == "stage_results_list"
        chain_def = _top_level_def(infra, "chain_results")
        assert "MANIFEST.stages" in ast.unparse(chain_def) and "NODE_RESULTS" in ast.unparse(chain_def), (
            "chain_results() is NODE_RESULTS in manifest order"
        )

    def test_gate_failure_writes_the_bundle_then_halts(self):
        src, loop = _chain_loop()
        gate_if = _the_if(
            loop, src, lambda test: test == 'not results["publication_gate_passed"]', "enforcing the verdict"
        )
        assert gate_if in loop.body, "the gate check is a top-level statement of the loop body"
        kinds = [type(stmt) for stmt in gate_if.body]
        assert kinds == [ast.Assign, ast.Expr], "the message, then halt (release the runtime, then raise)"
        assert ast.unparse(gate_if.body[0].targets[0]) == "_gate_msg"
        assert '"; ".join(results["gate_failures"])' in ast.get_source_segment(src, gate_if.body[0])
        halt = gate_if.body[1].value
        assert isinstance(halt, ast.Call) and _func_name(halt) == "halt"
        assert [ast.unparse(arg) for arg in halt.args] == ["_gate_msg"]
        # The knobs are read when the gate refuses (consolidation PR-14b), never bound earlier.
        for keyword, value in DISCONNECT_KNOBS:
            assert _keyword_source(src, halt, keyword) == value, keyword
        # Publication before enforcement: summary and bundle are written before the check.
        assert _call(loop, "write_training_summary").lineno < _call(loop, "save_run_bundle").lineno < gate_if.lineno
        assert _call(loop, "generate_stage_artifacts").lineno < _call(loop, "write_training_summary").lineno
        # And the handoff — what the next node loads — is set only after the gate passed.
        assert gate_if.end_lineno is not None
        reuse_nodes = list(ast.walk(_reuse_if(src, loop)))
        trained_handoffs = [assign for assign in _handoff_assigns(loop) if assign not in reuse_nodes]
        assert len(trained_handoffs) == 1 and trained_handoffs[0].lineno > gate_if.end_lineno
        # The gate refusal is the loop's only runtime release: the certified-library
        # publish block and its own disconnect left with consolidation PR-4.
        assert len(_calls(loop, "halt")) == 1 and not _calls(loop, "disconnect_runtime")
        assert "PUBLISH_CERTIFIED" not in src and "publish_canonical_stage" not in src
        assert "CERTIFIED_LIBRARY" not in src and "certified_publications" not in src
        # The results dict recorded is the gated one (generate_stage_artifacts returns the verdict).
        results_assign = next(
            node
            for node in loop.body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "results"
            and isinstance(node.value, ast.Call)
            and _func_name(node.value) == "generate_stage_artifacts"
        )
        assert results_assign.lineno < gate_if.lineno
        assert "NODE_RESULTS[NODE.id] = results" in _branch_source(src, loop.body)

    def test_one_disconnect_path(self):
        """The helpers live in ``environments.shared.notebook_runtime`` (consolidation PR-14b): the infrastructure
        cell's one import binds them, no cell rebinds one, and every call passes what the signatures take."""
        helpers = {"disconnect_runtime", "halt", "display_stage_videos"}
        cells = _code_cells()
        infra = _cell_index(cells, INFRA_CELL_MARKER)
        for index, src in enumerate(cells):
            tree = ast.parse(src)
            if index == infra:
                imports = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module == "environments.shared.notebook_runtime"
                ]
                assert len(imports) == 1 and {alias.name for alias in imports[0].names} == helpers
                tree.body.remove(imports[0])
            assert not _bound_names(tree) & helpers, f"cell {index} rebinds a helper"
            assert "unassign(" not in src and "import mediapy" not in src, f"cell {index}"
            for call in _calls(tree, "display_stage_videos"):
                assert len(call.args) == 1 and not call.keywords, f"cell {index}: display_stage_videos(stage_dir)"
            for call in _calls(tree, "halt") + _calls(tree, "disconnect_runtime"):
                assert len(call.args) == 1, f"cell {index}: the reason is the one positional argument"
                for keyword, value in DISCONNECT_KNOBS:
                    assert _keyword_source(src, call, keyword) == value, f"cell {index}: {keyword}"

    def test_no_random_baseline_cell(self):
        """The random-action baseline cell and the markdown sentence pointing at it left with consolidation PR-14b:
        the zero-action cell measures the floor that decides whether stage 1 learned anything."""
        for index, src in enumerate(_code_cells()):
            assert not _calls(ast.parse(src), "sample"), f"cell {index} rolls random actions"
        assert not [src for src in _all_cell_sources() if "**random** policy" in src]


class TestSeedReplication:
    """Plan §4.5 (WS-B4): the certification_panel role and the discovered replicates reach the bundle."""

    PROVENANCE_CELL_MARKER = "PROVENANCE_PATH = initialize_result_bundle("

    def test_seed_roles_declare_the_certification_panel(self):
        """Both seed_roles dicts carry certification_panel sourced from PUBLICATION_SEED_START (D-B17)."""
        for marker, func_name in (
            (self.PROVENANCE_CELL_MARKER, "initialize_result_bundle"),
            (INFRA_CELL_MARKER, "_lib_save_result_bundle"),
        ):
            src = _cell(marker)
            call = _call(ast.parse(src), func_name)
            roles = next(keyword.value for keyword in call.keywords if keyword.arg == "seed_roles")
            assert isinstance(roles, ast.Dict), f"{func_name}(seed_roles=...) must be a literal dict"
            by_key = {
                key.value: ast.get_source_segment(src, value)
                for key, value in zip(roles.keys, roles.values)
                if isinstance(key, ast.Constant)
            }
            assert by_key.get("certification_panel") == "PUBLICATION_SEED_START", by_key
            assert {"training", "checkpoint_selection_evaluation", "publication_evaluation"} <= set(by_key)
        assert "from environments.shared.constants import PUBLICATION_SEED_START" in _cell(self.PROVENANCE_CELL_MARKER)

    def test_the_checkpoint_selection_seed_role_is_the_seed_train_evaluates_on(self):
        """``train_stage`` trains through ``train_base.train`` (consolidation PR-14c), whose eval env is seeded
        ``seed + 1000``; the storage cell records the same value as the checkpoint-selection seed role."""
        assert ast.unparse(_top_level_assigns(_cell(STORAGE_CELL_MARKER))["CHECKPOINT_SELECTION_SEED"]) == "SEED + 1000"
        train_src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        eval_env = next(
            node.value
            for node in ast.walk(_top_level_def(train_src, "train"))
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "eval_env"
        )
        assert _func_name(eval_env) == "create_vec_env" and ast.unparse(eval_env.args[4]) == "seed + 1000"

    def test_the_bundle_write_passes_discovered_replicates(self):
        """save_run_bundle hands the sibling replicates to the writer (D-B10/D-B16)."""
        src = _cell(INFRA_CELL_MARKER)
        call = _call(ast.parse(src), "_lib_save_result_bundle")
        assert "replicates" in _keyword_names(call)
        assert _keyword_source(src, call, "replicates").startswith("discover_replicates_for_run(run_dir, ")
        assert "from environments.shared.replication import discover_replicates_for_run" in src


class TestEscapeHatch:
    """The manual single-node cell: off by default, records but never enforces, never feeds the chain."""

    def test_the_manual_cell_is_off_by_default_and_declares_its_load_mode(self):
        src = _cell(MANUAL_CELL_MARKER)
        assigns = _top_level_assigns(src)
        for name in ("MANUAL_NODE", "MANUAL_LOAD_PATH", "MANUAL_VECNORM_PATH", "MANUAL_TIMESTEPS"):
            assert isinstance(assigns[name], ast.Constant) and assigns[name].value is None, f"{name} defaults to None"
        assert isinstance(assigns["MANUAL_LOAD_MODE"], ast.Constant)
        assert assigns["MANUAL_LOAD_MODE"].value == "initialize_next_stage"
        top = ast.parse(src).body
        gates = [node for node in top if isinstance(node, ast.If)]
        assert len(gates) == 1 and ast.get_source_segment(src, gates[0].test) == "MANUAL_NODE is not None"
        assert all(isinstance(node, (ast.Assign, ast.If)) for node in top), (
            "everything but the knobs sits under `if MANUAL_NODE is not None:`"
        )
        train = _call(gates[0], "train_stage")
        mode = next(kw.value for kw in train.keywords if kw.arg == "task_load_mode")
        assert isinstance(mode, ast.IfExp)
        assert ast.unparse(mode.test) == "MANUAL_LOAD_PATH"
        assert ast.unparse(mode.body) == "MANUAL_LOAD_MODE"
        assert isinstance(mode.orelse, ast.Constant) and mode.orelse.value == "resume_same_stage", (
            "a manual node with no load trains from scratch under the declared same-stage mode"
        )
        assert _keyword_source(src, train, "load_path") == "MANUAL_LOAD_PATH"
        assert _keyword_source(src, train, "vecnorm_path") == "MANUAL_VECNORM_PATH"
        assert _keyword_source(src, train, "run_dir") == "RUN_DIR"
        assert "MANIFEST.resolve(MANUAL_NODE)" in src, "the manual node is resolved through the manifest"

    def test_the_manual_cell_records_but_never_enforces_the_verdict(self):
        src = _cell(MANUAL_CELL_MARKER)
        tree = ast.parse(src)
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Raise)], (
            "the manual cell never raises on a verdict (its one refusal, a complete run, is a helper call before any write)"
        )
        assert not _calls(tree, "disconnect_runtime") and not _calls(tree, "halt"), (
            "the manual cell never releases the runtime"
        )
        assert _calls(tree, "generate_stage_artifacts"), "the verdict is still judged and written"
        assert "NODE_RESULTS[_manual_entry.id]" in src or re.search(r"NODE_RESULTS\[[^\]]+\] =", src)
        assert "completed_stages.append(" in src
        assert '"publication_gate_passed"' in src and '"gate_failures"' in src, "the verdict is printed"
        tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        assert len(tries) == 1
        try_module = ast.Module(body=tries[0].body, type_ignores=[])
        assert _calls(try_module, "write_training_summary") and _calls(try_module, "save_run_bundle")
        handler = tries[0].handlers[0]
        assert handler.type is not None and ast.unparse(handler.type) == "ResultBundleError"
        assert _calls(handler, "print"), "a bundle the manual node cannot enter is reported, never swallowed"
        # The frozen-null flow is the loop's: freeze before, roll after, keyed on the kind set.
        assert _call(tree, "freeze_recovery_gate").lineno < _call(tree, "train_stage").lineno
        assert _call(tree, "train_stage").lineno < _call(tree, "roll_policy_panel").lineno
        artifacts = _call(tree, "generate_stage_artifacts")
        assert "recovery_successes_by_seed" in _keyword_names(artifacts)

    def test_the_manual_cell_refuses_a_complete_run_before_it_writes(self):
        """A complete bundle is immutable (consolidation PR-14a): refused before the freeze or the training."""
        src = _cell(MANUAL_CELL_MARKER)
        tree = ast.parse(src)
        gate = next(node for node in tree.body if isinstance(node, ast.If))
        guard = _call(gate, "refuse_write_into_complete_run")
        assert [ast.unparse(arg) for arg in guard.args] == ["RUN_DIR"]
        assert any(isinstance(node, ast.Expr) and node.value is guard for node in gate.body), "a top-level statement"
        entry = next(
            node
            for node in gate.body
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_manual_entry"
        )
        assert (
            entry.lineno < guard.lineno < _call(tree, "freeze_recovery_gate").lineno < _call(tree, "train_stage").lineno
        )

    def test_the_manual_cell_never_feeds_the_chain(self):
        src = _cell(MANUAL_CELL_MARKER)
        tree = ast.parse(src)
        assert not _handoff_assigns(tree)
        assert "NODE_HANDOFF" not in _names(tree), "the manual cell must not read or write NODE_HANDOFF"
        for node in ast.walk(tree):
            if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                assert "NODE_HANDOFF" not in ast.unparse(node.target)

    def test_exactly_one_escape_hatch_after_the_chain_loop_and_resume_before_it(self):
        """RESUME sits between the helpers that define ``train_stage`` and the chain loop, so "set ``RUN_ID`` and
        ``RESUME_STAGE``, then Run all" finishes the interrupted node before the loop judges it; the manual node
        follows the loop (decision D-D16)."""
        cells = _code_cells()
        infra_at = _cell_index(cells, INFRA_CELL_MARKER)
        chain_at = _cell_index(cells, CHAIN_CELL_MARKER)
        manual_at = _cell_index(cells, MANUAL_CELL_MARKER)
        resume_at = _cell_index(cells, RESUME_CELL_MARKER)
        assert infra_at < resume_at < chain_at < manual_at
        assert sum(cell.startswith("# ===== ") for cell in cells) == 4, (
            "the four `# ===== ` cells: species selection, chain loop, manual node, resume"
        )
        assert len([cell for cell in cells if "for NODE in CHAIN:" in cell]) == 1
        callers = [index for index, cell in enumerate(cells) if _calls(ast.parse(cell), "train_stage")]
        assert callers == [resume_at, chain_at, manual_at], "no other cell trains a node"


class TestResumeCell:
    """The RESUME cell resumes THIS node's own periodic checkpoint under a declared same-stage mode."""

    def test_the_resume_cell_still_declares_resume_same_stage_and_never_infers(self):
        src = _cell(RESUME_CELL_MARKER)
        assigns = _top_level_assigns(src)
        assert isinstance(assigns["RESUME_STAGE"], ast.Constant) and assigns["RESUME_STAGE"].value is None
        tree = ast.parse(src)
        train = _call(tree, "train_stage")
        mode = next(kw.value for kw in train.keywords if kw.arg == "task_load_mode")
        assert isinstance(mode, ast.Constant) and mode.value == "resume_same_stage"
        assert _keyword_source(src, train, "load_path") == "str(ckpt_res)", "the node's OWN periodic checkpoint"
        assert "vecnorm_path" in _keyword_names(train)
        assert _keyword_source(src, train, "run_dir") == "RUN_DIR"
        assert "parent_run_id" not in _keyword_names(train), "a same-stage resume has no parent run"
        assert "stage_position" not in _names(tree)
        # It never judges: no verdict, no bundle — the chain loop's JUDGE branch does that.
        for name in (
            "generate_stage_artifacts",
            "save_run_bundle",
            "write_training_summary",
            "evaluate_stage_checkpoints",
        ):
            assert not _calls(tree, name), f"the RESUME cell calls {name}; judging belongs to the chain loop"
        # A finished node -- a verdict or the final pair -- is skipped before the checkpoint scan: printed, never
        # raised, never trained, and pointed at the chain loop's JUDGE branch (decision D-D16).
        finished = _the_if(
            tree,
            src,
            lambda test: (
                test == "verdict_res is not None or (final_pair_res[0].exists() and final_problem_res is None)"
            ),
            "on a finished node",
        )
        finished_src = _branch_source(src, finished.body)
        assert "chain loop" in finished_src and "JUDGE" in finished_src and "fresh RUN_ID" in finished_src
        finished_body = ast.Module(finished.body, [])
        assert _calls(finished_body, "print") and not [
            node for node in ast.walk(finished_body) if isinstance(node, ast.Raise)
        ]
        assert train in [node for stmt in finished.orelse for node in ast.walk(stmt)]
        # ... and the checkpoint scan (with its no-checkpoint refusal) runs only for an unfinished node.
        scans = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and "model_dir_res.glob" in ast.unparse(node.iter)
        ]
        assert len(scans) == 1 and scans[0] in [node for stmt in finished.orelse for node in ast.walk(stmt)]
        assert "read_gate_verdict(stage_dir_res)" in src
        # One integrity check for every pair: the final pair and each periodic candidate.
        checks = _calls(tree, "pair_problem_res")
        assert [ast.unparse(call) for call in checks] == [
            "pair_problem_res(*final_pair_res)",
            "pair_problem_res(cand_ckpt, cand_vecnorm)",
        ]
        # A spent budget WITHOUT the final pair (the runtime stopped between the last periodic save and the final
        # one) is refused: the JUDGE branch needs the final pair, and there is nothing left to train.
        spent = _the_if(tree, src, lambda test: test == "remaining_res == 0", "on a spent budget")
        spent_src = _branch_source(src, spent.body)
        assert "JUDGE" in spent_src and "fresh RUN_ID" in spent_src
        spent_body = ast.Module(spent.body, [])
        assert _raises(spent_body, "RuntimeError") and not _calls(spent_body, "train_stage")
        assert train in [node for stmt in spent.orelse for node in ast.walk(stmt)]
        assert "entry_res = MANIFEST.resolve(RESUME_STAGE)" in src and "stage_res = entry_res.reference" in src
        assert "stage_dirname(SPECIES, stage_res)" in src and "stage_label(stage_res)" in src

    @staticmethod
    def _write_pair(zip_path: Path, vecnorm_path: Path) -> None:
        """An intact checkpoint pair as the RESUME cell checks one: SB3's outer members and an unpickling sidecar."""
        import pickle
        import zipfile

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("data", "{}")
            archive.writestr("policy.pth", b"")
        vecnorm_path.write_bytes(pickle.dumps({}))

    @classmethod
    def _run_resume_cell(
        cls,
        tmp_path,
        species: str,
        behavior: str,
        resume_stage,
        reference,
        *,
        steps: int | None = 100_000,
        retrain_from: str | None = None,
    ) -> list[dict]:
        """Execute the RESUME cell against one intact periodic pair of *reference* at *steps* (none when *steps* is
        None: a widened root holds only its handoff and final pairs); return the train_stage calls. Whatever else the
        stage directory holds (a verdict, a final pair) is the caller's."""
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname, stage_label

        models = tmp_path / stage_dirname(species, reference) / "models"
        models.mkdir(parents=True, exist_ok=True)
        label = stage_label(reference)
        if steps is not None:
            cls._write_pair(models / f"{label}_{steps}_steps.zip", models / f"{label}_vecnormalize_{steps}_steps.pkl")
        manifest = load_stage_manifest(species)
        calls: list[dict] = []

        def train_stage(**kwargs):
            calls.append(kwargs)
            return (None,) * 6

        namespace = {
            "SPECIES": species,
            "BEHAVIOR": behavior,
            "MANIFEST": manifest,
            "CHAIN": manifest.chain_for(manifest.resolve_behavior(behavior).id),
            "STAGE_CONFIGS": load_all_stages(species),
            "QUICK_TEST": False,
            "RUN_DIR": tmp_path,
            "RUN_LABEL": "",
            "RETRAIN_NODE": manifest.resolve(retrain_from) if retrain_from else None,
            "train_stage": train_stage,
        }
        src = _cell(RESUME_CELL_MARKER).replace("RESUME_STAGE = None", f"RESUME_STAGE = {resume_stage!r}", 1)
        exec(compile(src, "sb3_resume", "exec"), namespace)
        return calls

    @pytest.mark.parametrize(
        ("behavior", "resume_stage", "reference"),
        [("walk", "locomotion", 2), ("walk", 2, 2), ("stand", "recovery", "recovery")],
    )
    def test_the_resume_cell_takes_a_stage_number_or_id(self, tmp_path, behavior, resume_stage, reference):
        """Executed: ``RESUME_STAGE = "locomotion"`` resumes the same node as ``2`` (it used to raise ``KeyError``),
        finding the ``stage2_*`` checkpoints the chain loop wrote; a stage without a number keeps its id."""
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname, stage_label

        species = "compsognathus_robot"
        [call] = self._run_resume_cell(tmp_path, species, behavior, resume_stage, reference)
        models = tmp_path / stage_dirname(species, reference) / "models"
        assert call["stage"] == reference
        assert call["load_path"] == str(models / f"{stage_label(reference)}_100000_steps.zip")
        budget = load_all_stages(species)[reference]["curriculum_kwargs"]["timesteps"]
        assert call["timesteps"] == budget - 100_000

    def test_the_resume_cell_refuses_a_node_off_the_chain_before_it_trains(self, tmp_path):
        """Executed: the chain loop visits only CHAIN, so a node off it would train and never be judged — a
        ``stand`` run's recovery reopened under ``walk`` is refused with the knobs to restore."""
        with pytest.raises(RuntimeError, match=r"'recovery', which is not on the chain of behavior 'walk'") as refused:
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "recovery", "recovery")
        assert "BEHAVIOR" in str(refused.value) and "re-run sections 2-3" in str(refused.value)

    @pytest.mark.parametrize("steps", [100_000, None])
    @pytest.mark.parametrize("passed", [True, False])
    def test_a_judged_node_is_never_resumed(self, tmp_path, capsys, passed, steps):
        """Executed: a node holding ``gate_verdict.json`` -- passed or failed -- trains nothing, whatever its periodic
        checkpoints say, and before the checkpoint scan (a judged widened root holds no periodic pair); the chain loop
        reuses or refuses it. Before D-D16 an early-stopped node's short periodic step (1.45M of 6M in session 4)
        trained the rest of the budget into the judged directory, rewriting its final pair and, whenever a
        post-resume evaluation beat the seeded best, the handoff pair its verdict hashes."""
        from environments.shared.result_bundle import GATE_VERDICT_SCHEMA
        from environments.shared.stage_manifest import stage_dirname

        stage_dir = tmp_path / stage_dirname("compsognathus_robot", 2)
        stage_dir.mkdir(parents=True)
        (stage_dir / "gate_verdict.json").write_text(
            json.dumps({"schema": GATE_VERDICT_SCHEMA, "passed": passed, "failures": [] if passed else ["too slow"]})
        )
        assert self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=steps) == []
        out = capsys.readouterr().out
        assert "Nothing to resume: 'locomotion' already holds a gate verdict" in out and "JUDGE" in out

    @pytest.mark.parametrize("steps", [1_450_000, 2_999_970, None])
    def test_a_finished_node_is_never_resumed_whatever_its_periodic_steps(self, tmp_path, capsys, steps):
        """Executed: an intact final pair decides that a node's training finished, not periodic arithmetic. An early
        stop (1.45M) or ``N_ENVS = 3`` (a 33,333-call cadence whose newest save is 2,999,970 of 3M) leaves the newest
        periodic step short of the budget, and a widened root holds no periodic pair at all (``None``); the loop's
        JUDGE branch judges the final pair instead."""
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_final.zip", models / "stage2_final_vecnorm.pkl")
        assert self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", 2, 2, steps=steps) == []
        assert "already holds its final pair stage2_final.zip" in capsys.readouterr().out

    def test_a_half_written_final_pair_is_resumed_from_the_periodic_checkpoint(self, tmp_path):
        """Executed: only the whole pair marks a finished node, as in the chain loop's JUDGE test; a final zip without
        its sidecar is an interrupted save, resumed from the newest intact periodic pair."""
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        models.mkdir(parents=True)
        (models / "stage2_final.zip").write_bytes(b"truncated")
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_800_000)
        assert call["timesteps"] == 200_000 and call["load_path"].endswith("stage2_2800000_steps.zip")

    @pytest.mark.parametrize("broken", ["zip", "sidecar"])
    def test_a_final_pair_cut_short_is_resumed_over(self, tmp_path, capsys, broken):
        """Executed: the final pair is the one checkpoint ``train()`` writes straight to the mount, so a reclaim during
        the final save can truncate it. It is checked like a periodic pair; a broken one is not a finished node
        (the JUDGE branch could not load it), so the cell resumes from the newest intact periodic pair and warns."""
        import pickle

        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_final.zip", models / "stage2_final_vecnorm.pkl")
        if broken == "zip":
            data = (models / "stage2_final.zip").read_bytes()
            (models / "stage2_final.zip").write_bytes(data[: len(data) // 2])
        else:
            (models / "stage2_final_vecnorm.pkl").write_bytes(pickle.dumps({"obs_rms": list(range(50))})[:20])
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_800_000)
        assert call["timesteps"] == 200_000 and call["load_path"].endswith("stage2_2800000_steps.zip")
        assert "WARNING: the final pair of 'locomotion' is incomplete" in capsys.readouterr().out

    def test_a_spent_budget_with_a_final_pair_cut_short_names_it(self, tmp_path):
        """Executed: with nothing left to train, the refusal names the broken final pair and why, not "never saved"."""
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_final.zip", models / "stage2_final_vecnorm.pkl")
        (models / "stage2_final_vecnorm.pkl").write_bytes(b"\x80\x04truncated")
        budget = load_all_stages("compsognathus_robot")[2]["curriculum_kwargs"]["timesteps"]
        with pytest.raises(
            RuntimeError, match=r"stage2_final\.zip is incomplete \(VecNormalize sidecar .* fresh RUN_ID"
        ):
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", 2, 2, steps=budget)

    @pytest.mark.parametrize("retrain_from", ["stance", "locomotion"])
    def test_a_node_retrain_from_covers_is_refused_before_it_trains(self, tmp_path, retrain_from):
        """Executed: the chain loop trains a node ``RETRAIN_FROM`` covers from its parent instead of judging it, and
        D-A20 refuses its occupied directory, so a resume under it could never be judged."""
        with pytest.raises(RuntimeError, match=r"which RETRAIN_FROM .* covers.*Set RETRAIN_FROM = \"\""):
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, retrain_from=retrain_from)

    def test_a_spent_budget_without_the_final_pair_is_refused(self, tmp_path):
        """Executed: a runtime stopped between the last periodic save (at the budget) and the final save leaves
        nothing to train and nothing the JUDGE branch can judge; the cell says so instead of pointing at the loop,
        which would send the operator straight back here."""
        from environments.shared.config import load_all_stages

        budget = load_all_stages("compsognathus_robot")[2]["curriculum_kwargs"]["timesteps"]
        with pytest.raises(RuntimeError, match=r"never saved .* fresh RUN_ID"):
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", 2, 2, steps=budget)

    def test_the_resume_cell_refuses_a_complete_run_before_it_trains(self):
        """A complete bundle is immutable (consolidation PR-14a): nothing is resumed into it; the spent-budget branch
        and the checkpoint scan stay read-only."""
        src = _cell(RESUME_CELL_MARKER)
        tree = ast.parse(src)
        spent = _the_if(tree, src, lambda test: test == "remaining_res == 0", "on a spent budget")
        guard = _call(tree, "refuse_write_into_complete_run")
        assert [ast.unparse(arg) for arg in guard.args] == ["RUN_DIR"]
        first = spent.orelse[0]
        assert isinstance(first, ast.Expr) and first.value is guard, "the first statement of the resuming branch"
        assert guard.lineno < _call(tree, "train_stage").lineno

    def test_the_resume_prose_routes_old_checkpoints_to_the_command_line_widen(self):
        """The markdown right before the RESUME cell: ``RUN_ID`` is set in the configuration cell, never into a
        complete run; an earlier run's certified nodes come in through ``TRUNK_FROM`` in a new run; a pre-bump
        checkpoint is widened on the command line (D-C17's bound as the tool's flag) into a new run judged here
        under the parent's seed (D-C14, D-D14). No copy of the old restart / memo-reset remedy survives."""
        every = _all_cell_sources()
        prose = every[every.index(_cell(RESUME_CELL_MARKER)) - 1]
        assert prose.startswith("## "), "a markdown section header sits right before the RESUME cell"
        for phrase in (
            "`RUN_ID` in the configuration cell",
            "`complete`",
            "`TRUNK_FROM`",
            "environments.shared.scripts.widen_checkpoint",
            "--max-revision-gap",
            "--label",
            "D-C17",
            "D-C14",
            "D-D14",
            "`SEED`",
            'TRUNK_FROM = ""',
            "re-entering run",
        ):
            assert phrase in prose, f"the RESUME prose no longer names {phrase}"
        assert "**new** `RUN_ID`" in prose and "never by pointing `RUN_ID` at the old run" in prose
        for gone in ("WIDEN_FROM", "WIDEN_MAX_REVISION_GAP", "restart the runtime", "_ACTIVE_RUN_ID", "stray"):
            assert gone not in prose, f"the RESUME prose still names {gone!r}"


def _preflight_cell() -> tuple[str, ast.Module]:
    src = _cell(PREFLIGHT_CELL_MARKER)
    return src, ast.parse(src)


class TestArchiveLoadPreflightCell:
    """The cell that proves SB3 archives load on this runtime before anything is trained.

    Two Colab sessions on 2026-09-19 died inside the widen tool's self-verification because the runtime image had
    moved to another Python minor version and a bare ``PPO.load`` executed the parent archive's cloudpickled
    schedule bytecode. The preflight used to live in the infrastructure cell and loaded a throwaway model saved by
    the same interpreter, so it could never see the fault. It runs right after the RESOLVE cell, on the trunk run's
    real root handoff (else a throwaway), through ``policy_loading.load_sb3_model`` -- the one loader every
    repository load goes through -- with the print flushed first so a kernel death is attributable. (Until
    consolidation PR-14a it loaded the WIDEN_FROM parent's handoff first; D-D14 moved widening to the command
    line.)"""

    def test_the_preflight_follows_the_resolve_cell(self):
        src, tree = _preflight_cell()
        assert src.splitlines()[0] == PREFLIGHT_CELL_MARKER
        assert not src.startswith("# ===== "), "not one of the four `# ===== ` cells"
        cells = _code_cells()
        preflight_at = _cell_index(cells, PREFLIGHT_CELL_MARKER)
        resolve_at = _cell_index(cells, RESOLVE_CELL_MARKER)
        assert preflight_at == resolve_at + 1, "right after the RESOLVE cell (CHAIN and TRUNK_DIR exist)"
        assert preflight_at < _cell_index(cells, INFRA_CELL_MARKER) < _cell_index(cells, CHAIN_CELL_MARKER)
        every = _all_cell_sources()
        # Only the section-4 header sits between them (decision D-D16 regrouped the sections).
        assert every.index(cells[preflight_at]) == every.index(cells[resolve_at]) + 2
        assert every[every.index(cells[resolve_at]) + 1].startswith("## 4. Preflights")
        # The notebook-only PR-12 slice removed the direction/terrain guard: the marker is the raw first line.
        assert "COMMAND_TERRAIN_BEHAVIOR" not in src

    def test_the_preflight_loads_a_real_archive_through_the_loader_with_the_print_flushed_first(self):
        src, tree = _preflight_cell()
        assert "from environments.shared.policy_loading import inspect_sb3_archive, load_sb3_model" in src
        loads = _calls(tree, "load_sb3_model")
        assert len(loads) == 1, "exactly one load, the one being proven"
        (load,) = loads
        # No bare algorithm load anywhere in the cell: the loader is the path under test.
        assert not re.search(r"\b(PPO|SAC|AlgoClass|alg_cls)\.load\(", src)
        # The flushed print immediately precedes the load, and names what is loaded and by which Python it was saved.
        prints = [node for node in _calls(tree, "print") if node.lineno < load.lineno]
        assert prints, "a print precedes the load"
        last = prints[-1]
        assert any(keyword.arg == "flush" and ast.literal_eval(keyword.value) is True for keyword in last.keywords), (
            "the print before the load is flushed so a kernel death is attributable to the load"
        )
        printed = ast.get_source_segment(src, last)
        assert "saved_python_text" in printed and "bytecode_members" in printed and "kernel death HERE" in printed
        # The archive is the trunk run's root handoff when there is one, else a throwaway (D-D14 took the
        # WIDEN_FROM parent's branch with the widen cell); a trunk without a complete pair falls back, never raises.
        assert "WIDEN_FROM" not in src
        trunk_if = _the_if(tree, src, lambda test: test == "TRUNK_DIR is not None", "on a trunk run")
        assert trunk_if in tree.body and "_preflight_root_handoff(TRUNK_DIR)" in _branch_source(src, trunk_if.body)
        assert not _raises(tree, "RuntimeError")
        assert "select_handoff_checkpoint(" in src and "stage_dir_candidates(SPECIES, CHAIN[0].reference)" in src
        assert "tempfile.TemporaryDirectory(" in src, (
            "the throwaway model lives in a temporary directory the cell removes"
        )
        assert 'load_sb3_model(_preflight_archive, device="cpu")' in src

    def test_the_preflight_trains_nothing_writes_nothing_and_reads_only_earlier_names(self):
        import builtins

        src, tree = _preflight_cell()
        for name in (
            "train_stage",
            "widen_checkpoint",
            "evaluate_stage_checkpoints",
            "generate_stage_artifacts",
            "learn",
            "initialize_result_bundle",
            "save_stage_config",
            "write_gate_verdict",
            "disconnect_runtime",
            "halt",
            "open",
        ):
            assert not _calls(tree, name), f"the preflight cell calls {name}"
        for name in ("NODE_HANDOFF", "NODE_RESULTS", "completed_stages", "RUN_DIR"):
            assert name not in _names(tree), f"the preflight cell touches {name}"
        bound = _bound_names(tree)
        earlier: set[str] = set()
        for marker in ("# Add repo root to path", CONFIG_CELL_MARKER, STORAGE_CELL_MARKER, RESOLVE_CELL_MARKER):
            earlier |= _bound_names(ast.parse(_cell(marker)))
        # The cell defines a helper and a throwaway env class: their parameters are bound locally, not read.
        parameters = {
            argument.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for argument in (*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs)
        }
        free = _loaded_names(tree) - bound - parameters - set(dir(builtins))
        assert free <= earlier, f"the preflight cell reads names no earlier cell binds: {sorted(free - earlier)}"
        assert free <= {"SPECIES", "CHAIN", "TRUNK_DIR", "gym", "np", "sys"}, sorted(free)


class TestCommandSliceReseed:
    """Amendment A12 / invariant 8 (BEHAVIOR_RECIPES_PLAN §4.6): the loaded statistics' command slice is reseeded
    whenever the node's command_mode is not "none" — EXCEPT on a same-stage resume, whose sidecar already holds the
    statistics the policy trained under. ``train_base._load_vecnorm_into_envs`` applies the rule
    (test_command_frame.py) and ``train_base.train``, which ``train_stage`` trains through (consolidation PR-14c),
    feeds it the stage config's mode and the notebook's sidecar — always False in Phase C."""

    def test_train_feeds_the_rule_the_stage_config_and_the_sidecar(self):
        import inspect

        from environments.shared.curriculum import load_vecnorm_stats

        src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        load = _call(_top_level_def(src, "train"), "_load_vecnorm_into_envs")
        assert _keyword_source(src, load, "command_mode") == (
            'str(config.get("env_kwargs", {}).get("command_mode", "none"))'
        )
        assert _keyword_source(src, load, "task_load_mode") == "task_load_mode"
        assert _keyword_source(src, load, "vecnorm_path") == "vecnorm_path"
        # Without it a sidecar loads with plant validation skipped (the notebook's manual cell names any sidecar).
        assert _keyword_source(src, load, "plant_identity") == "plant_identity"
        assert [ast.unparse(arg) for arg in load.args] == ["load_path", "train_env", "eval_env"], (
            "both destinations are passed, so the reseed reaches the train AND the eval wrapper"
        )
        rule = ast.get_source_segment(src, _top_level_def(src, "_load_vecnorm_into_envs")) or ""
        assert 'if command_mode != "none" and task_load_mode != "resume_same_stage":' in rule
        parameters = inspect.signature(load_vecnorm_stats).parameters
        assert parameters["reseed_command_slice"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["reseed_command_slice"].default is False, "reseeding stays opt-in in the library"

    def test_every_committed_stage_is_command_mode_none_in_phase_c(self):
        """The flag is False on every committed config today; Phase D flips it per stage, not the notebook."""
        from environments.shared.config import load_all_stages

        for species in SPECIES_WITH_MANIFESTS:
            for reference, config in load_all_stages(species).items():
                assert config.get("env_kwargs", {}).get("command_mode", "none") == "none", (species, reference)


class TestDeliverableAwareCells:
    """The tail cells evaluate BEHAVIOR's deliverable, replay its chain, and plot this run's nodes."""

    def _evaluation_cell(self) -> tuple[str, ast.Call]:
        hits = [
            (src, call)
            for src in _code_cells()
            for call in _calls(ast.parse(src), "evaluate")
            if "model_path" in _keyword_names(call)
        ]
        assert len(hits) == 1, "expected exactly one evaluate(model_path=...) cell"
        return hits[0]

    def test_evaluation_targets_the_behaviors_deliverable(self):
        src, call = self._evaluation_cell()
        assert _keyword_source(src, call, "stage") == "TARGET_NODE.reference"
        assert _keyword_source(src, call, "model_path") == 'NODE_HANDOFF[TARGET_NODE.id]["model"] + ".zip"'
        guard = _the_if(
            ast.parse(src), src, lambda test: test == "TARGET_NODE.id not in NODE_HANDOFF", "on an uncertified target"
        )
        assert _raises(guard, "RuntimeError") and guard.lineno < call.lineno
        for index, cell in enumerate(_code_cells()):
            for other in _calls(ast.parse(cell), "evaluate"):
                for keyword in other.keywords:
                    if keyword.arg == "stage":
                        assert not isinstance(keyword.value, ast.Constant), f"cell {index} evaluates a literal stage"
            assert not re.search(r"\bpath_3\b", cell), f"cell {index} names path_3"

    def test_replay_walks_the_chain_and_curves_stay_in_this_run(self):
        cells = _code_cells()
        chain_at = _cell_index(cells, CHAIN_CELL_MARKER)
        replay = [
            src
            for index, src in enumerate(cells)
            if index != chain_at
            and any(
                isinstance(node, ast.For) and ast.unparse(node.iter) == "CHAIN" and _calls(node, "display_stage_videos")
                for node in ast.parse(src).body
            )
        ]
        assert len(replay) == 1, "exactly one cell besides the loop replays the chain's videos"
        loop = _top_level_for(replay[0], "entry", "CHAIN")
        videos = _call(loop, "display_stage_videos")
        assert [ast.unparse(arg) for arg in videos.args] == ["handoff['stage_dir']"], (
            "a reused ancestor plays from its OWN run's stage directory"
        )
        assert "NODE_HANDOFF.get(entry.id)" in replay[0]
        curves = [
            src
            for src in cells
            if any(
                isinstance(node, ast.For)
                and ast.unparse(node.iter) == "completed_stages"
                and _calls(node, "plot_training_curves")
                for node in ast.parse(src).body
            )
        ]
        assert len(curves) == 1, "exactly one cell re-plots this run's curves"
        assert "MANIFEST.resolve(stage_ref).id" in curves[0], "curves are labelled by node id"
        assert "CHAIN" not in _names(ast.parse(curves[0])), (
            "reused ancestors keep their curves in the run that trained them"
        )
        # The completion cell reports every chain node from NODE_HANDOFF and keeps the complete-bundle check.
        completion = _cell(COMPLETION_CELL_MARKER)
        assert "validate_result_bundle(RUN_DIR, require_complete=True)" in completion
        assert "require_publishable=True" in completion
        _top_level_for(completion, "entry", "CHAIN")
        assert "NODE_HANDOFF.get(entry.id)" in completion
        # The disconnect names the behavior and reads the knobs when it runs (consolidation PR-14b).
        cells_after = cells[_cell_index(cells, COMPLETION_CELL_MARKER) + 1 :]
        assert cells_after and "BEHAVIOR" in _names(ast.parse(cells_after[-1]))
        disconnect = _call(ast.parse(cells_after[-1]), "disconnect_runtime")
        for keyword, value in DISCONNECT_KNOBS:
            assert _keyword_source(cells_after[-1], disconnect, keyword) == value, keyword

    def test_the_curves_cell_writes_nothing_into_the_sealed_bundle(self, tmp_path):
        """The chain loop seals the bundle with each node's declared ``figures/`` set; the curves
        cell runs after that, so any file it wrote would be undeclared and the cleanup cell's
        ``validate_result_bundle`` would raise before the auto-disconnect (the flat PNGs every
        completed Run all left in its stage directories until 2026-09-23)."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname

        cells = _code_cells()
        visualization = [src for src in cells if "def plot_training_curves(" in src]
        curves = [
            src
            for src in cells
            if any(
                isinstance(node, ast.For) and ast.unparse(node.iter) == "completed_stages"
                for node in ast.parse(src).body
            )
        ]
        assert len(visualization) == 1 and len(curves) == 1
        species = "velociraptor"
        run_dir = tmp_path / "run"
        stage_dir = run_dir / stage_dirname(species, 2)
        stage_dir.mkdir(parents=True)
        np.savez(
            stage_dir / "evaluations.npz",
            timesteps=np.array([50_000, 100_000]),
            results=np.array([[1.0, 2.0], [3.0, 4.0]]),
            ep_lengths=np.array([[500, 600], [700, 800]]),
        )
        before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
        namespace = {
            "SPECIES": species,
            "ALGORITHM": "PPO",
            "STAGE_CONFIGS": load_all_stages(species),
            "MANIFEST": load_stage_manifest(species),
            "completed_stages": [(2, str(stage_dir))],
            "plt": plt,
            "Path": Path,
        }
        figures = len(plt.get_fignums())
        try:
            exec(visualization[0], namespace)
            exec(curves[0], namespace)
            assert len(plt.get_fignums()) > figures, "the cell still draws the curves inline"
        finally:
            plt.close("all")
        assert sorted(path.relative_to(run_dir) for path in run_dir.rglob("*")) == before

    def test_no_hand_threaded_stage_variables_remain(self):
        forbidden = [
            r"\bresults_[123]\b",
            r"\bpath_[123]\b",
            r"\bdir_[123]\b",
            r"\bresults_r\b",
            r"\bvecnorm_[123]\b",
            r"\bmodel_[123]\b",
            r"\bstage_dir_[123]\b",
            r"\bRUN_RECOVERY_STAGE\b",
            r"\bcurriculum_results\b",
        ]
        for index, src in enumerate(_code_cells()):
            for pattern in forbidden:
                assert not re.search(pattern, src), (
                    f"code cell {index} matches {pattern}: a hand-threaded stage variable"
                )


def test_no_f_string_stage_n_in_any_code_cell():
    for index, src in enumerate(_code_cells()):
        assert 'f"stage{' not in src and "f'stage{" not in src, (
            f"code cell {index} rebuilds a stage{{N}} directory name; use stage_dirname / stage_label"
        )


class TestNotebookWithoutTheDirectionTerrainMode:
    """The notebook-only PR-12 slice (decision D-D13) removed the direction/terrain mode.

    The canonical halves of the deleted test_behavior_notebook.py live on here: the configuration
    defaults, free-form stage ids, the dropdown's JSON annotations and the setup cell's ``REPO_REF``
    safety. The pilots run from the command line only (``environments.shared.train_behaviors``)
    until PR-11 gives them manifest nodes.
    """

    PILOT_BEHAVIORS = (
        "difficult_terrain",
        "follow_direction_difficult_terrain",
        "follow_direction",
        "follow_direction_speed",
        "terrain_contact",
        "sloped_terrain",
        "bumps_terrain",
        "depressions_terrain",
        "mixed_terrain",
        "combined_terrain",
        "combined_mixed_terrain",
    )

    def test_the_configuration_defaults_and_a_free_form_stage_resolve(self):
        from environments.shared.config import load_all_stages

        namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        assert namespace["BEHAVIOR"] == "hunt"
        assert namespace["SPECIES"] == "velociraptor"
        assert namespace["N_ENVS"] == 4 and namespace["SEED"] == 42
        assert namespace["RUN_ID"] == "", "a fresh run by default (consolidation PR-14a)"
        assert "WIDEN_FROM" not in namespace and "WIDEN_MAX_REVISION_GAP" not in namespace
        assert "COMMAND_TERRAIN_BEHAVIOR" not in namespace
        assert not [name for name in namespace if name.startswith("BEHAVIOR_")]
        namespace["BEHAVIOR"] = "stance"
        exec(_cell("EnvClass = SPECIES_CFG.env_class"), namespace)
        assert namespace["TARGET_NODE"].id == "stance"
        assert [node.id for node in namespace["CHAIN"]] == ["stance"]

    def test_the_behavior_dropdown_allows_free_input_and_every_param_annotation_is_json(self):
        config = _cell(CONFIG_CELL_MARKER)
        behavior_line = next(line for line in config.splitlines() if line.startswith("BEHAVIOR ="))
        options, trailing = json.JSONDecoder().raw_decode(behavior_line.split("# @param ", 1)[1])
        assert {"stand", "walk", "hunt"} <= set(options)
        assert not set(options) & set(self.PILOT_BEHAVIORS), "the pilots have no notebook path until PR-11"
        assert json.loads(behavior_line.split("# @param ", 1)[1][trailing:].strip()) == {"allow-input": True}
        for source in (_cell("REPO_REF ="), config):
            for line in source.splitlines():
                if "# @param {" in line:
                    assert "type" in json.loads(line.split("# @param ", 1)[1])

    def test_the_direction_terrain_mode_is_gone_from_every_cell(self):
        knobs = r"\bBEHAVIOR_(LOAD_MODE|CHECKPOINT|VECNORMALIZE|SEED|STEPS|EVAL_ONLY|EVAL_EPISODES|RECORD_VIDEO|VIDEO_FPS|RUN_ID)\b"
        for index, src in enumerate(_all_cell_sources()):
            for token in (
                "COMMAND_TERRAIN_BEHAVIOR",
                "behavior_notebook",
                "BEHAVIOR_PLAN",
                "BEHAVIOR_RESULT",
                "PILOT_",
            ):
                assert token not in src, f"cell {index} names {token}"
            assert not re.search(knobs, src), f"cell {index} names a direction/terrain knob"

    @pytest.mark.parametrize(
        "dirty,loaded,expected", [(True, False, "local edits"), (False, True, "Restart"), (False, False, None)]
    )
    def test_colab_ref_change_is_explicit_and_preserves_edits(self, tmp_path, monkeypatch, dirty, loaded, expected):
        """Execute the notebook's actual Git setup block with controlled Git replies."""
        import subprocess
        import types

        checkout = tmp_path / "checkout"
        (checkout / ".git").mkdir(parents=True)
        source = _cell("REPO_REF =")
        tree = ast.parse(source)
        colab_block = next(node for node in tree.body if isinstance(node, ast.If))
        start = next(
            i
            for i, node in enumerate(colab_block.body)
            if isinstance(node, ast.Import) and node.names[0].name == "pathlib"
        )
        body = colab_block.body[start:]
        for node in body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "repo_dir"
            ):
                node.value = ast.Call(
                    func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(checkout))], keywords=[]
                )
        commands = []

        def fake_output(command, **kwargs):
            if command[1:3] == ["rev-parse", "FETCH_HEAD^{commit}"]:
                return "new-commit\n"
            if command[1:3] == ["rev-parse", "HEAD"]:
                return "old-commit\n"
            if command[1:3] == ["status", "--porcelain"]:
                return " M notebook.ipynb\n" if dirty else ""
            raise AssertionError(command)

        monkeypatch.setattr(subprocess, "check_output", fake_output)
        monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
        namespace = {
            "Path": Path,
            "REPO_REF": "feature-branch",
            "sys": types.SimpleNamespace(modules={"environments": object()} if loaded else {}),
            "importlib": types.SimpleNamespace(util=types.SimpleNamespace(find_spec=lambda name: object())),
        }
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "setup", "exec")
        if expected:
            with pytest.raises(RuntimeError, match=expected):
                exec(code, namespace)
            assert not any(command[1] == "checkout" for command in commands)
        else:
            exec(code, namespace)
            assert ["git", "checkout", "--detach", "new-commit"] in commands
        assert commands[0] == ["git", "fetch", "origin", "feature-branch"]
        assert not any("--force" in command or "reset" in command or "clean" in command for command in commands)


@pytest.mark.parametrize("path", [NOTEBOOK_PATH, JAX_NOTEBOOK_PATH], ids=["sb3", "jax"])
def test_the_notebook_round_trips_through_json_dump_indent_1(path):
    """Every notebook edit goes through json.load -> json.dump(indent=1, ensure_ascii=False) + newline."""
    text = _notebook_text(path)
    notebook = json.loads(text)
    assert json.dumps(notebook, indent=1, ensure_ascii=False) + "\n" == text, (
        f"{path.name} is not in the canonical json.dump(indent=1, ensure_ascii=False) form"
    )
    if path == NOTEBOOK_PATH:
        # Every SB3 code cell is plain Python (the JAX install cell keeps a `!pip` Colab magic).
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path.name}[code cell {index}]")
