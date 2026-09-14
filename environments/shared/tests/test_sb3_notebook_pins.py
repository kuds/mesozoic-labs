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
#: The Phase C widen cell (D-C13): its FIRST line, deliberately not a `# ===== ` marker.
WIDEN_CELL_MARKER = "# Widen an earlier run's certified root checkpoint (BEHAVIOR_RECIPES_PLAN §4.6)"
#: The names the widen cell may take from the cells that run before it (amendment A14a).
WIDEN_CELL_INHERITED_NAMES = frozenset(
    {
        "SPECIES",
        "ALGORITHM",
        "RUN_DIR",
        "RUN_LABEL",
        "LOG_BASE",
        "CHAIN",
        "SEED",
        "WIDEN_FROM",
        "WIDEN_MAX_REVISION_GAP",
    }
)

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
        for name in ("BEHAVIOR", "TRUNK_FROM", "WIDEN_FROM", "RETRAIN_FROM", "RUN_LABEL"):
            assert name in assigns, f"the config cell must declare {name}"
            assert isinstance(assigns[name], ast.Constant) and isinstance(assigns[name].value, str), (
                f"{name} is a plain string constant an operator edits"
            )
        # The optional knobs are OFF by default: an unattended Run-all trains
        # the chain here, reuses nothing from another run, widens nothing
        # (D-C13), and labels nothing.
        for name in ("TRUNK_FROM", "WIDEN_FROM", "RETRAIN_FROM", "RUN_LABEL"):
            assert assigns[name].value == "", f"{name} must default to the empty string (off)"
        # D-C17: the revision-gap bound is an integer constant defaulting to the tool's fail-closed 1 (the Phase C
        # bump alone); a widen session for an r11 trex stance parent raises it to 2 by hand.
        from environments.shared.scripts.widen_checkpoint import DEFAULT_MAX_REVISION_GAP

        assert "WIDEN_MAX_REVISION_GAP" in assigns, "the config cell must declare WIDEN_MAX_REVISION_GAP"
        gap = assigns["WIDEN_MAX_REVISION_GAP"]
        assert isinstance(gap, ast.Constant) and type(gap.value) is int, "an integer constant (not a bool)"
        assert gap.value == 1 == DEFAULT_MAX_REVISION_GAP, "WIDEN_MAX_REVISION_GAP defaults to the tool's bound of 1"
        assert re.search(r"WIDEN_MAX_REVISION_GAP = 1\s+#.*D-C17", _cell(CONFIG_CELL_MARKER)), (
            "the knob's comment names decision D-C17"
        )
        # A14b: the config cell tells the operator a widen session sets SEED to the parent's seed.
        config_src = _cell(CONFIG_CELL_MARKER)
        assert re.search(r"#.*widen.*SEED.*parent", config_src, re.IGNORECASE), (
            "the config cell must say a widen session sets SEED to the parent run's seed (D-C14)"
        )
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
        # from exactly the sources train_stage records it from: a drift (env_kwargs={} here, say) would
        # silently refuse every reuse with "judged under task".
        fingerprint = _call(loop, "derive_stage_task_fingerprint")
        loop_sources = {kw.arg: ast.get_source_segment(src, kw.value) for kw in fingerprint.keywords}
        infra_src = _cell(INFRA_CELL_MARKER)
        infra_fingerprint = _call(_top_level_def(infra_src, "train_stage"), "derive_stage_task_fingerprint")
        infra_sources = {kw.arg: ast.get_source_segment(infra_src, kw.value) for kw in infra_fingerprint.keywords}
        assert loop_sources == infra_sources, "the loop derives the task digest exactly as train_stage records it"
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
        expected_keys = ["model", "vecnorm", "stage_dir", "run_dir", "run_id", "model_sha256", "reused"]
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
        # Never a copy: the checkpoint is loaded from where it lives (A10).
        assert "shutil" not in src
        copy_calls = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"copy", "copy2", "copyfile", "copytree", "replace", "rename"}
        ]
        assert not copy_calls, "a reused ancestor's checkpoint is never copied into this run"
        assert ".zip" not in ast.unparse(_dict_value(_handoff_assigns(reuse_if)[0].value, "model")), (
            "the handoff is the ancestor's own stem, in its own run"
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
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        shaping = _call(train_stage, "_stage_entry_shaping_callbacks")
        assert _keyword_names(shaping) == {"task_load_mode", "parent_id", "load_path"}
        assert _keyword_source(src, shaping, "parent_id") == "NODE.warm_start_from", (
            "shaping fires on the declared EDGE (warm_start_from), verbatim like every CLI caller"
        )
        assert _keyword_source(src, shaping, "task_load_mode") == "task_load_mode"
        extend = _call(train_stage, "extend")
        assert ast.unparse(extend.func) == "callbacks.extend"
        assert len(extend.args) == 1 and isinstance(extend.args[0], ast.Name) and extend.args[0].id == "shaping", (
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
            assert guard.lineno < _call(train_stage, "refuse_occupied_stage_dir").lineno, (
                "argument refusals happen before the stage directory is touched"
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

    def test_declared_parent_mismatch_refuses(self):
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        validate = _call(train_stage, "validate_declared_parent")
        assert ast.unparse(validate.args[0]) == "read_checkpoint_task_fingerprint(load_path)"
        assert {"declared_parent", "species", "child_stage", "artifact"} <= _keyword_names(validate)
        assert "parent_of(stage)" in _branch_source(src, [train_stage]).split("validate_declared_parent(")[0], (
            "the declared parent is the manifest's parent_of(stage)"
        )
        assert _keyword_source(src, validate, "child_stage") == "NODE.reference"
        guard = next(node for node in ast.walk(train_stage) if isinstance(node, ast.If) and validate in ast.walk(node))
        assert ast.get_source_segment(src, guard.test) == "entering_from_parent"
        assert validate.lineno < _call(train_stage, "save_stage_config").lineno, (
            "the parent is validated BEFORE the stage record is written"
        )


class TestTrainStageRecordKeeping:
    """D-A20 and D-A15 inside ``train_stage``: no silent overwrite; the real duration is recorded."""

    def test_an_occupied_stage_dir_is_refused_before_the_config_is_written(self):
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        refuse = _call(train_stage, "refuse_occupied_stage_dir")
        assert ast.unparse(refuse.args[0]) == "stage_dir"
        assert _keyword_source(src, refuse, "task_load_mode") == "task_load_mode if load_path else None", (
            "the guard sees the EFFECTIVE mode — the one save_stage_config records"
        )
        mkdir_calls = [node for node in _calls(train_stage, "mkdir") if ast.unparse(node.func) == "stage_dir.mkdir"]
        assert mkdir_calls and refuse.lineno < min(node.lineno for node in mkdir_calls)
        assert refuse.lineno < _call(train_stage, "save_stage_config").lineno
        assert "refuse_occupied_stage_dir" in _cell(INFRA_CELL_MARKER).split("def ")[0], (
            "refuse_occupied_stage_dir is imported from environments.shared.config, not redefined"
        )

    def test_the_resume_re_save_keeps_the_edge_the_node_entered_on(self, tmp_path):
        """A RESUME-cell load (``task_load_mode="resume_same_stage"`` of the node's own periodic
        checkpoint) re-saves stage_config.json through ``save_stage_config`` with the EFFECTIVE mode,
        and the library keeps the ``initialize_next_stage`` edge the node entered on (recording the
        resume under ``RESUME_LINEAGE_KEYS``), so a resumed-then-judged node still chains by digest
        (ancestors rule 4) for a later same-run pass or as a TRUNK_FROM ancestor."""
        import zipfile

        from environments.shared.config import LOAD_LINEAGE_KEYS, RESUME_LINEAGE_KEYS, save_stage_config

        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        save = _call(train_stage, "save_stage_config")
        assert ast.unparse(save.args[0]) == "stage_dir"
        assert _keyword_source(src, save, "load_path") == "load_path"
        assert _keyword_source(src, save, "load_mode") == "task_load_mode if load_path else None"
        resume_src = _cell(RESUME_CELL_MARKER)
        resume_train = _call(ast.parse(resume_src), "train_stage")
        assert _keyword_source(resume_src, resume_train, "task_load_mode") == '"resume_same_stage"'
        assert _keyword_source(resume_src, resume_train, "load_path") == "str(ckpt_res)"

        def zipped(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data", "{}")
                archive.writestr("policy.pth", b"w")
            return path

        stage_config = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {}, "curriculum_kwargs": {}}
        stage_dir = tmp_path / "02_locomotion"
        parent = zipped(tmp_path / "01_stance" / "models" / "stance_final.zip")
        save_stage_config(stage_dir, 2, stage_config, "PPO", load_path=str(parent), load_mode="initialize_next_stage")
        entered = json.loads((stage_dir / "stage_config.json").read_text())["run"]
        periodic = zipped(stage_dir / "models" / "locomotion_100_steps.zip")
        save_stage_config(stage_dir, 2, stage_config, "PPO", load_path=str(periodic), load_mode="resume_same_stage")
        resumed = json.loads((stage_dir / "stage_config.json").read_text())["run"]
        assert {key: resumed.get(key) for key in LOAD_LINEAGE_KEYS} == {
            key: entered.get(key) for key in LOAD_LINEAGE_KEYS
        }
        assert resumed["load_mode"] == "initialize_next_stage"
        assert resumed["resume_load_path"] == str(periodic) and set(RESUME_LINEAGE_KEYS) <= set(resumed)

    def test_the_stage_duration_is_recorded_on_exit(self):
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        record = _call(train_stage, "record_stage_duration")
        assert ast.unparse(record.args[0]) == "stage_dir"
        learn = _call(train_stage, "learn")
        evaluate = _call(train_stage, "evaluate_stage_checkpoints")
        assert learn.lineno < record.lineno < evaluate.lineno, (
            "the duration is recorded after learn() returns and BEFORE evaluation, so a judged node reports it"
        )
        # A same-stage resume accumulates on the record read BEFORE the config re-save.
        prior = _call(train_stage, "read_stage_duration")
        assert prior.lineno < _call(train_stage, "save_stage_config").lineno
        recorded = ast.unparse(record.args[1])
        duration_assign = next(
            node
            for node in ast.walk(train_stage)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == recorded
        )
        assert "prior_duration" in ast.unparse(duration_assign.value), (
            "a resume's duration accumulates on the prior one"
        )
        assert _keyword_source(src, evaluate, "duration_seconds") == recorded, (
            "the judged results carry the recorded value"
        )


class TestJudgeBranch:
    """The evaluation tail is a top-level function the JUDGE branch can call without training."""

    def test_evaluate_stage_checkpoints_is_split_out_and_train_stage_returns_its_tuple(self):
        src = _cell(INFRA_CELL_MARKER)
        judge = _top_level_def(src, "evaluate_stage_checkpoints")
        assert [arg.arg for arg in judge.args.args] == ["stage", "stage_dir"]
        kwonly = [arg.arg for arg in judge.args.kwonlyargs]
        assert kwonly == ["final_path", "final_vecnorm_path", "timesteps", "duration_seconds", "model"]
        defaults = dict(zip(kwonly, judge.args.kw_defaults))
        for name in ("final_path", "final_vecnorm_path", "timesteps", "duration_seconds"):
            assert defaults[name] is None, f"{name} has no default"
        assert isinstance(defaults["model"], ast.Constant) and defaults["model"].value is None
        # model=None loads the final zip back and validates its plant before any rollout.
        load_if = _the_if(judge, src, lambda test: test == "model is None", "loading the final model")
        load_src = _branch_source(src, load_if.body)
        assert ".load(" in load_src and "validate_model_plant(" in load_src
        assert load_if.lineno < _calls(judge, "_eval_forward_vel")[0].lineno
        # The evidence writers live in the split-out tail only.
        train_stage = _top_level_def(src, "train_stage")
        assert len(_calls(judge, "_lib_save_evaluation_episodes")) == 3
        assert not _calls(train_stage, "_lib_save_evaluation_episodes")
        assert not _calls(train_stage, "_eval_forward_vel")
        # train_stage delegates and returns the unchanged 6-tuple.
        delegate = _call(train_stage, "evaluate_stage_checkpoints")
        assert _keyword_source(src, delegate, "model") == "model", "the in-memory model is passed after training"
        returns = [node for node in ast.walk(train_stage) if isinstance(node, ast.Return)]
        assert len(returns) == 1 and isinstance(returns[0].value, ast.Tuple) and len(returns[0].value.elts) == 6
        judge_returns = [node for node in ast.walk(judge) if isinstance(node, ast.Return)]
        assert len(judge_returns) == 1 and isinstance(judge_returns[0].value, ast.Tuple)
        assert len(judge_returns[0].value.elts) == 5

    def test_the_loop_judges_from_disk_with_the_recorded_duration(self):
        src, loop = _chain_loop()
        judge_if = _judge_if(src, loop)
        call = _call(judge_if, "evaluate_stage_checkpoints")
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

    def test_gate_failure_writes_the_bundle_then_disconnects_then_raises(self):
        src, loop = _chain_loop()
        gate_if = _the_if(
            loop, src, lambda test: test == 'not results["publication_gate_passed"]', "enforcing the verdict"
        )
        assert gate_if in loop.body, "the gate check is a top-level statement of the loop body"
        kinds = [type(stmt) for stmt in gate_if.body]
        assert kinds == [ast.Assign, ast.Expr, ast.Raise], "message, disconnect, raise — in that order"
        assert ast.unparse(gate_if.body[0].targets[0]) == "_gate_msg"
        assert '"; ".join(results["gate_failures"])' in ast.get_source_segment(src, gate_if.body[0])
        assert isinstance(gate_if.body[1].value, ast.Call) and _func_name(gate_if.body[1].value) == "disconnect_runtime"
        assert ast.unparse(gate_if.body[1].value.args[0]) == "_gate_msg"
        assert ast.unparse(gate_if.body[2]) == "raise RuntimeError(_gate_msg)"
        # Publication before enforcement: summary and bundle are written before the check.
        assert _call(loop, "write_training_summary").lineno < _call(loop, "save_run_bundle").lineno < gate_if.lineno
        assert _call(loop, "generate_stage_artifacts").lineno < _call(loop, "write_training_summary").lineno
        # And the handoff — what the next node loads — is set only after the gate passed.
        assert gate_if.end_lineno is not None
        reuse_nodes = list(ast.walk(_reuse_if(src, loop)))
        trained_handoffs = [assign for assign in _handoff_assigns(loop) if assign not in reuse_nodes]
        assert len(trained_handoffs) == 1 and trained_handoffs[0].lineno > gate_if.end_lineno
        assert len(_calls(loop, "disconnect_runtime")) == 1
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
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Raise)], "the manual cell never raises"
        assert not _calls(tree, "disconnect_runtime"), "the manual cell never releases the runtime"
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

    def test_the_manual_cell_never_feeds_the_chain(self):
        src = _cell(MANUAL_CELL_MARKER)
        tree = ast.parse(src)
        assert not _handoff_assigns(tree)
        assert "NODE_HANDOFF" not in _names(tree), "the manual cell must not read or write NODE_HANDOFF"
        for node in ast.walk(tree):
            if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                assert "NODE_HANDOFF" not in ast.unparse(node.target)

    def test_exactly_one_escape_hatch_and_it_follows_the_chain_loop(self):
        cells = _code_cells()
        chain_at = _cell_index(cells, CHAIN_CELL_MARKER)
        manual_at = _cell_index(cells, MANUAL_CELL_MARKER)
        resume_at = _cell_index(cells, RESUME_CELL_MARKER)
        assert chain_at < manual_at < resume_at
        assert sum(cell.startswith("# ===== ") for cell in cells) == 4, (
            "the four `# ===== ` cells: species selection, chain loop, manual node, resume"
        )
        assert len([cell for cell in cells if "for NODE in CHAIN:" in cell]) == 1
        callers = [index for index, cell in enumerate(cells) if _calls(ast.parse(cell), "train_stage")]
        assert callers == [chain_at, manual_at, resume_at], "no other cell trains a node"


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
        assert "load_path" in _keyword_names(train) and "vecnorm_path" in _keyword_names(train)
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
        spent = _the_if(tree, src, lambda test: test == "remaining_res == 0", "on a spent budget")
        spent_src = _branch_source(src, spent.body)
        assert "chain loop" in spent_src and "JUDGE" in spent_src, (
            "a spent budget points the operator at the loop's JUDGE branch"
        )
        assert _calls(spent, "print") and not [node for node in ast.walk(spent) if isinstance(node, ast.Raise)]
        assert train in [node for stmt in spent.orelse for node in ast.walk(stmt)]
        assert "stage_dirname(SPECIES, RESUME_STAGE)" in src and "stage_label(RESUME_STAGE)" in src

    def test_the_resume_prose_routes_old_checkpoints_to_the_widen_knobs(self):
        """The markdown right before the RESUME cell sends a pre-bump checkpoint to ``WIDEN_FROM`` in a new run,
        under the ``WIDEN_MAX_REVISION_GAP`` bound (D-C17) and the parent's seed (D-C14), never to ``RUN_ID``."""
        every = _all_cell_sources()
        prose = every[every.index(_cell(RESUME_CELL_MARKER)) - 1]
        assert prose.startswith("## "), "a markdown section header sits right before the RESUME cell"
        for phrase in ("`WIDEN_FROM`", "`WIDEN_MAX_REVISION_GAP`", "default 1", "D-C17", "D-C14", "`SEED`"):
            assert phrase in prose, f"the RESUME prose no longer names {phrase}"
        assert "**new** `RUN_ID`" in prose and "never by pointing `RUN_ID` at the old run" in prose


def _widen_cell() -> tuple[str, ast.Module]:
    src = _cell(WIDEN_CELL_MARKER)
    return src, ast.parse(src)


def _root_name(tree: ast.AST) -> str:
    """The name the widen cell binds to ``CHAIN[0]`` — the chain's root, the node it widens."""
    roots = [
        target
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and ast.unparse(node.value) == "CHAIN[0]"
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert len(roots) == 1, "the widen cell binds the chain's root exactly once, from CHAIN[0]"
    return roots[0].id


class TestWidenCell:
    """Phase C (BEHAVIOR_RECIPES_PLAN §4.6; decisions D-C13, D-C14; amendment A14): ``WIDEN_FROM`` widens an
    earlier run's certified ROOT handoff into THIS run's root stage directory — before the chain loop, in a fresh
    RUN_ID — as a judge-ready node the loop then REUSE-refuses (no verdict) and JUDGES from its ``<stage_label>_final``
    pair.  The cell never trains, never certifies, never writes into the parent, and refuses a re-run in the same
    RUN_DIR."""

    def test_exactly_one_widen_cell_between_resolve_and_chain(self):
        cells = _code_cells()
        hits = [index for index, src in enumerate(cells) if "widen_checkpoint(" in src]
        assert len(hits) == 1, "exactly one code cell calls widen_checkpoint(...)"
        widen_at = hits[0]
        src = cells[widen_at]
        assert src.splitlines()[0] == WIDEN_CELL_MARKER, "the widen cell is identified by its first line"
        resolve_at = _cell_index(cells, RESOLVE_CELL_MARKER)
        assert widen_at == resolve_at + 1, "the widen cell is the code cell right after the RESOLVE cell (CHAIN exists)"
        assert widen_at < _cell_index(cells, INFRA_CELL_MARKER) < _cell_index(cells, CHAIN_CELL_MARKER)
        # Immediately after in the full cell list too: no markdown or other cell sits between them.
        every = _all_cell_sources()
        assert every.index(src) == every.index(cells[resolve_at]) + 1
        # The tool is imported from the scripts module, the way the zero-action baseline cell imports its script.
        assert "from environments.shared.scripts.widen_checkpoint import" in src
        # With WIDEN_FROM empty the cell prints one line and does nothing else; everything else sits under the else.
        tree = ast.parse(src)
        guard = _the_if(tree, src, lambda test: test == "not WIDEN_FROM", "on an empty WIDEN_FROM")
        assert guard in tree.body, "the WIDEN_FROM guard is a top-level statement of the cell"
        assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Expr)
        assert isinstance(guard.body[0].value, ast.Call) and _func_name(guard.body[0].value) == "print"
        widen = _call(tree, "widen_checkpoint")
        assert widen in [node for stmt in guard.orelse for node in ast.walk(stmt)], (
            "the widening happens under the else"
        )

    def test_widen_cell_is_not_an_escape_hatch_and_trains_nothing(self):
        src, tree = _widen_cell()
        assert not src.startswith("# ===== "), "the widen cell is not one of the four `# ===== ` cells"
        for name in (
            "train_stage",
            "evaluate_stage_checkpoints",
            "generate_stage_artifacts",
            "save_run_bundle",
            "write_training_summary",
            "freeze_recovery_gate",
            "roll_policy_panel",
            "learn",
            "find_certified_ancestor",
            "record_ancestor",
        ):
            assert not _calls(tree, name), f"the widen cell calls {name}"
        for name in ("NODE_HANDOFF", "NODE_RESULTS", "completed_stages", "TRUNK_DIR"):
            assert name not in _names(tree), f"the widen cell touches {name}: it feeds the chain only through the disk"
        # The four `# ===== ` cells and the three train_stage callers are exactly as before (A23).
        cells = _code_cells()
        assert sum(cell.startswith("# ===== ") for cell in cells) == 4
        callers = [index for index, cell in enumerate(cells) if _calls(ast.parse(cell), "train_stage")]
        assert len(callers) == 3 and cells.index(src) not in callers
        # It never certifies or records the node itself: no verdict, no provenance, no resolution, no write of any
        # kind — the tool writes the judge-ready root; the chain loop's JUDGE branch mints the verdict.
        assert "gate_verdict" not in src
        for name in (
            "write_gate_verdict",
            "initialize_result_bundle",
            "update_provenance",
            "save_stage_config",
            "open",
        ):
            assert not _calls(tree, name), f"the widen cell calls {name}"
        writers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {
                "write_text",
                "write_bytes",
                "mkdir",
                "copy",
                "copy2",
                "copyfile",
                "copytree",
                "rename",
                "replace",
                "unlink",
                "rmtree",
                "touch",
            }
        ]
        assert not writers, f"the widen cell writes or moves files itself: {[ast.unparse(node) for node in writers]}"
        assert "shutil" not in _names(tree)

    def test_widen_cell_refuses_occupied_target_and_requires_provenance(self):
        src, tree = _widen_cell()
        root = _root_name(tree)
        widen = _call(tree, "widen_checkpoint")
        # D-A20 / D-C13: the target is refused when it already records a stage, BEFORE the tool runs, with the
        # effective mode None (nothing is loaded into it) — so a re-run in the same RUN_DIR refuses, never overwrites.
        refuse = _call(tree, "refuse_occupied_stage_dir")
        mode = next(keyword.value for keyword in refuse.keywords if keyword.arg == "task_load_mode")
        assert isinstance(mode, ast.Constant) and mode.value is None
        assert refuse.lineno < widen.lineno, "the occupied-directory guard runs before the widening"
        target = ast.unparse(refuse.args[0])
        assert target == _keyword_source(src, widen, "target_stage_dir"), (
            "the guard checks the directory the tool writes"
        )
        target_assign = next(
            node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == target
        )
        assert ast.unparse(target_assign.value) == f"RUN_DIR / stage_dirname(SPECIES, {root}.reference)", (
            "the widened root lands in THIS run under the stage's directory name — where the chain loop looks"
        )
        # The tool's own refusals (WidenError) are never caught: a refusal halts Run-all with the tool's message.
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try) and widen in [n for s in node.body for n in ast.walk(s)]
        ], "widen_checkpoint(...) is not wrapped in a try"
        assert "WidenError" not in _names(tree)
        # A parent is a run WITH a provenance.json: load_provenance inside try/except ResultBundleError, refusing.
        load = _call(tree, "load_provenance")
        tries = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try) and load in [n for s in node.body for n in ast.walk(s)]
        ]
        assert len(tries) == 1
        handlers = tries[0].handlers
        assert [ast.unparse(handler.type) for handler in handlers if handler.type is not None] == ["ResultBundleError"]
        assert _raises(handlers[0], "RuntimeError") and "provenance.json" in _branch_source(src, handlers[0].body)
        # Never this run's own directory.
        own = _the_if(tree, src, lambda test: "RUN_DIR.resolve()" in test, "refusing this run's own directory")
        assert _raises(own, "RuntimeError") and own.lineno < load.lineno
        # Same species, algorithm and backend as this run — mirrored from the TRUNK_FROM block.
        assert "canonical_algorithm(ALGORITHM)" in src and '"stable-baselines3"' in src
        identity_ifs = [
            node
            for node in _ifs(tree, src, lambda test: "!=" in test)
            if _raises(node, "RuntimeError") and "species, algorithm and backend" in _branch_source(src, node.body)
        ]
        assert len(identity_ifs) == 1, "a species/algorithm/backend mismatch refuses"
        assert identity_ifs[0].lineno < widen.lineno
        # The source stage directory is located under the PARENT run through stage_dir_candidates (any layout
        # generation), and a parent without that stage refuses instead of widening something else.
        source_name = _keyword_source(src, widen, "parent_stage_dir")
        source_assign = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == source_name
        )
        candidates = _calls(source_assign, "stage_dir_candidates")
        assert candidates and [ast.unparse(arg) for arg in candidates[0].args] == ["SPECIES", f"{root}.reference"]
        assert ".is_dir()" in ast.unparse(source_assign.value)
        missing = _the_if(
            tree, src, lambda test: test == f"{source_name} is None", "on a parent without the root stage"
        )
        assert _raises(missing, "RuntimeError") and missing.lineno < widen.lineno
        # Every refusal is a raise, never a print-and-continue.
        assert len(_raises(tree, "RuntimeError")) >= 4 and len(_raises(tree, "ValueError")) == 1

    def test_widen_cell_threads_the_label_knob(self):
        """D-A21 mirrored: the widen call records ``RUN_LABEL or None``; every other keyword is the run's own."""
        import inspect

        from environments.shared.scripts.widen_checkpoint import widen_checkpoint

        src, tree = _widen_cell()
        root = _root_name(tree)
        widen = _call(tree, "widen_checkpoint")
        assert not widen.args, "widen_checkpoint is keyword-only; the cell passes keywords"
        keywords = {keyword.arg: ast.get_source_segment(src, keyword.value) for keyword in widen.keywords}
        assert keywords["label"] == "RUN_LABEL or None"
        assert keywords["species"] == "SPECIES"
        assert keywords["stage"] == f"{root}.reference"
        assert keywords["algorithm"] == "ALGORITHM"
        assert "run_id" in (keywords.get("parent_run_id") or ""), "the parent is named by its provenance run_id"
        assert keywords["max_revision_gap"] == "WIDEN_MAX_REVISION_GAP", "the cell-6 bound is threaded as-is (D-C17)"
        assert set(keywords) == {
            "species",
            "stage",
            "parent_stage_dir",
            "target_stage_dir",
            "algorithm",
            "label",
            "parent_run_id",
            "max_revision_gap",
        }, "the notebook never passes the explicit-pair form, invents run facts, or allows a legacy plant"
        parameters = inspect.signature(widen_checkpoint).parameters
        assert set(keywords) <= set(parameters)
        assert parameters["allow_legacy_plant"].default is False
        assert parameters["max_revision_gap"].default == 1, "the tool fails closed at one revision without the knob"
        assert (
            'print(f"  policy-interface revisions crossed: {_widen_result.revision_gap} '
            '(WIDEN_MAX_REVISION_GAP={WIDEN_MAX_REVISION_GAP})")' in src
        ), "the revisions crossed are printed from the returned result, next to the bound they were checked against"
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())
        # The report numbers are printed from the returned result, not re-read from disk.
        assert _calls(tree, "print") and "max_action_delta" in src and "observation_dim" in src

    def test_widen_cell_imports_every_name_it_uses(self):
        """A14a: the cell runs BEFORE the infrastructure cell, so it imports its own names; the only free names are the
        knobs and constants the config, storage and resolve cells bind before it."""
        import builtins

        src, tree = _widen_cell()
        bound = _bound_names(tree)
        assert {
            "Path",
            "refuse_occupied_stage_dir",
            "load_provenance",
            "ResultBundleError",
            "canonical_algorithm",
            "widen_checkpoint",
            "stage_dir_candidates",
            "stage_dirname",
        } <= bound, "the widen cell imports the tool and every helper it calls itself"
        earlier: set[str] = set()
        for marker in (CONFIG_CELL_MARKER, STORAGE_CELL_MARKER, RESOLVE_CELL_MARKER):
            earlier |= _bound_names(ast.parse(_cell(marker)))
        free = _loaded_names(tree) - bound - set(dir(builtins))
        assert free <= earlier, f"the widen cell reads names no earlier cell binds: {sorted(free - earlier)}"
        assert free <= WIDEN_CELL_INHERITED_NAMES, (
            f"the widen cell leans on more than the documented knobs and constants: {sorted(free - WIDEN_CELL_INHERITED_NAMES)}"
        )
        # It relies on nothing the infrastructure cell defines (it has not run yet).
        infra_defs = {
            node.name for node in ast.parse(_cell(INFRA_CELL_MARKER)).body if isinstance(node, ast.FunctionDef)
        }
        assert not (_loaded_names(tree) & infra_defs)

    def test_widen_cell_refuses_a_seed_other_than_the_parents(self):
        """A14b / D-C14: the parent stage's recorded run.seed must equal SEED — this run's provenance publishes
        training_seed = SEED, and seed replication counts distinct seeds; a mismatch raises naming both values."""
        src, tree = _widen_cell()
        widen = _call(tree, "widen_checkpoint")
        seed_ifs = _ifs(tree, src, lambda test: re.fullmatch(r"(\w+) != SEED|SEED != (\w+)", test) is not None)
        assert len(seed_ifs) == 1, "exactly one `if <parent seed> != SEED:` guard"
        seed_if = seed_ifs[0]
        match = re.fullmatch(r"(\w+) != SEED|SEED != (\w+)", ast.get_source_segment(src, seed_if.test) or "")
        assert match is not None
        parent_seed = match.group(1) or match.group(2)
        assert len(seed_if.body) == 1 and isinstance(seed_if.body[0], ast.Raise), "an unconditional raise"
        assert _raises(seed_if, "ValueError")
        message = _names(seed_if.body[0])
        assert {"SEED", parent_seed} <= message, "the message names BOTH the notebook SEED and the parent's seed"
        # The parent's seed is read from the parent stage's stage_config.json run block, never from this run.
        seed_assign = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == parent_seed
        )
        seed_src = ast.get_source_segment(src, seed_assign.value) or ""
        assert '"seed"' in seed_src
        run_block = seed_src.split(".get")[0].split("[")[0]
        run_assign = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == run_block
        )
        run_src = ast.get_source_segment(src, run_assign.value) or ""
        assert "stage_config.json" in run_src and '"run"' in run_src
        assert _keyword_source(src, widen, "parent_stage_dir") in run_src, "read from the PARENT's stage directory"
        # Refused before anything is touched: the seed check precedes the occupied-target guard and the widening.
        assert seed_if.lineno < _call(tree, "refuse_occupied_stage_dir").lineno < widen.lineno

    def test_the_chain_loop_judges_a_widened_root(self):
        """What the widen cell relies on downstream: the loop REUSE-refuses a verdict-less directory (printed, never
        silent), reads no verdict, and takes the JUDGE branch on the ``<stage_label>_final`` pair the tool wrote."""
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


class TestCommandSliceReseed:
    """Amendment A12 / invariant 8 (BEHAVIOR_RECIPES_PLAN §4.6): the notebook's direct ``load_vecnorm_stats`` call
    reseeds the command slice whenever the node's command_mode is not "none" — EXCEPT on a same-stage resume, whose
    sidecar already holds the statistics the policy trained under (the rule ``train_base._load_vecnorm_into_envs``
    applies, and the RESUME cell calls ``train_stage`` with ``task_load_mode="resume_same_stage"``) — always False in
    Phase C."""

    def test_train_stage_reseeds_the_command_slice_from_the_stage_config(self):
        import inspect

        from environments.shared.curriculum import load_vecnorm_stats

        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        load = _call(train_stage, "load_vecnorm_stats")
        assert " ".join(_keyword_source(src, load, "reseed_command_slice").split()) == (
            'config.get("env_kwargs", {}).get("command_mode", "none") != "none" '
            'and task_load_mode != "resume_same_stage"'
        ), (
            "the flag derives from the stage config's command_mode and is never set on a same-stage resume, exactly "
            "as train_base._load_vecnorm_into_envs"
        )
        assert _keyword_source(src, load, "carry_ret_rms") == 'task_load_mode == "resume_same_stage"'
        assert _keyword_source(src, load, "current_plant") == "PLANT_IDENTITY"
        assert [ast.unparse(arg) for arg in load.args] == ["vecnorm_path", "train_env", "eval_env"], (
            "both destinations are passed, so the reseed reaches the train AND the eval wrapper"
        )
        parameters = inspect.signature(load_vecnorm_stats).parameters
        assert parameters["reseed_command_slice"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["reseed_command_slice"].default is False, "reseeding stays opt-in in the library"

    def test_every_committed_stage_is_command_mode_none_in_phase_c(self):
        """The flag is False on every committed config today; Phase D flips it per stage, not the notebook."""
        from environments.shared.config import load_all_stages

        for species in SPECIES_WITH_MANIFESTS:
            for reference, config in load_all_stages(species).items():
                assert config.get("env_kwargs", {}).get("command_mode", "none") == "none", (species, reference)


def _widen_cell_namespace(
    log_base: Path,
    run_dir: Path,
    chain,
    *,
    seed: int,
    widen_from: str,
    max_revision_gap: int = 1,
) -> dict:
    """What the config, storage and resolve cells bind before the widen cell runs (``WIDEN_CELL_INHERITED_NAMES``)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    namespace = {
        "SPECIES": "trex",
        "ALGORITHM": "ppo",
        "LOG_BASE": log_base,
        "RUN_DIR": run_dir,
        "RUN_LABEL": None,
        "SEED": seed,
        "CHAIN": chain,
        "WIDEN_FROM": widen_from,
        "WIDEN_MAX_REVISION_GAP": max_revision_gap,
    }
    assert set(namespace) == WIDEN_CELL_INHERITED_NAMES
    return namespace


class TestWidenCellExecution:
    """The widen cell's SOURCE executed over a real narrow (r-1) parent — the fixture ``test_widen_checkpoint`` builds —
    with the namespace the config, storage and resolve cells would have bound.  SB3-bound (~40 s, plus one r-2 parent
    for the D-C17 case)."""

    def test_the_widen_cell_widens_a_narrow_parent_and_refuses_a_rerun(self, tmp_path, capsys):
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("torch")
        from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor
        from environments.shared.config import (
            LOAD_LINEAGE_KEYS,
            WIDEN_LINEAGE_KEYS,
            StageDirectoryOccupiedError,
            load_stage_config,
        )
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.result_bundle import initialize_result_bundle, read_gate_verdict
        from environments.shared.scripts.widen_checkpoint import FORBIDDEN_OUTPUT_FILES
        from environments.shared.task_fingerprint import derive_stage_task_fingerprint

        from .test_widen_checkpoint import PARENT_RUN_NAME, PARENT_SEED, build_narrow_parent

        src = _cell(WIDEN_CELL_MARKER)
        code = compile(src, "sb3_training.ipynb[widen cell]", "exec")
        log_base = tmp_path / "logs"
        parent = build_narrow_parent(log_base / "trex" / "ppo", "ppo")
        # The parent is a run WITH a provenance.json, minted by the repository's own writer (as cell 7 does).
        initialize_result_bundle(
            parent["run_dir"],
            species="trex",
            algorithm="ppo",
            backend="stable-baselines3",
            seed=PARENT_SEED,
            plant_identity=parent["narrow"].to_dict(),
            run_id=PARENT_RUN_NAME,
            repository_root=REPO_ROOT,
        )
        manifest = load_stage_manifest("trex")
        chain = manifest.chain_for("behavior")
        assert chain[0].id == "stance"

        def namespace(run_dir: Path, *, seed: int = PARENT_SEED, widen_from: str = PARENT_RUN_NAME) -> dict:
            return _widen_cell_namespace(log_base, run_dir, chain, seed=seed, widen_from=widen_from)

        # (1) The widening: a judge-ready root under this run's stage directory name.
        run_dir = log_base / "trex" / "ppo" / "20260914_000000"
        exec(code, namespace(run_dir))  # noqa: S102 - the notebook cell under test
        out = capsys.readouterr().out
        widths = f"{parent['narrow'].observation_dim} -> {parent['current'].observation_dim}"
        assert widths in out and "robust_best_model" in out and "max action delta" in out
        target = run_dir / "01_stance"
        for name in (
            "models/robust_best_model.zip",
            "models/robust_best_model_vecnorm.pkl",
            "models/stage1_final.zip",
            "models/stage1_final_vecnorm.pkl",
            "stage_config.json",
            "plant_identity.json",
            "task_fingerprint.json",
            "widen_report.json",
        ):
            assert (target / name).is_file(), name
        for name in FORBIDDEN_OUTPUT_FILES:
            assert not (target / name).exists(), f"the widen cell left {name} behind"
        run_block = json.loads((target / "stage_config.json").read_text(encoding="utf-8"))["run"]
        assert run_block["seed"] == PARENT_SEED
        assert run_block["widened_from_run_id"] == PARENT_RUN_NAME
        assert set(WIDEN_LINEAGE_KEYS) <= set(run_block) and not (set(LOAD_LINEAGE_KEYS) & set(run_block)), (
            "a widened node is a root (D-C8): widen lineage, never load lineage"
        )
        assert run_block.get("label") is None, "RUN_LABEL=None reaches the run block as no label"
        assert (parent["stage_dir"] / "models" / "robust_best_model.zip").read_bytes() != (
            target / "models" / "robust_best_model.zip"
        ).read_bytes()
        assert not any(path.name in FORBIDDEN_OUTPUT_FILES for path in parent["stage_dir"].iterdir())

        # (2) What the chain loop then sees: REUSE is refused (no verdict), and JUDGE's precondition holds.
        entry = manifest.resolve("stance")
        stance_config = load_stage_config("trex", "stance")
        task_sha256 = derive_stage_task_fingerprint(
            species="trex",
            stage="stance",
            backend="stable-baselines3",
            env_kwargs=stance_config.get("env_kwargs", {}),
            plant_identity=current_plant_identity("trex").to_dict(),
        )["task_sha256"]
        with pytest.raises(AncestorReuseError, match="no gate_verdict.json"):
            find_certified_ancestor(
                run_dir,
                species="trex",
                entry=entry,
                current_task_sha256=task_sha256,
                plant_identity=current_plant_identity("trex"),
                current_gate_config=stance_config.get("curriculum_kwargs", {}),
                parent_model_sha256=None,
            )
        assert read_gate_verdict(target) is None
        assert (target / "models" / "stage1_final.zip").exists() and (
            target / "models" / "stage1_final_vecnorm.pkl"
        ).exists()

        # (3) A second run of the cell in the same RUN_DIR refuses (D-A20 / D-C13) and changes nothing.
        before = sorted(path.relative_to(target) for path in target.rglob("*"))
        with pytest.raises(StageDirectoryOccupiedError, match="already records a stage"):
            exec(code, namespace(run_dir))  # noqa: S102
        assert sorted(path.relative_to(target) for path in target.rglob("*")) == before

        # (4) SEED != parent run.seed refuses BEFORE anything is written, naming both values (D-C14 / A14b).
        other = log_base / "trex" / "ppo" / "20260914_000001"
        with pytest.raises(ValueError, match=rf"SEED={PARENT_SEED + 1}.*seed {PARENT_SEED}"):
            exec(code, namespace(other, seed=PARENT_SEED + 1))  # noqa: S102
        assert not (other / "01_stance").exists()

        # (5) WIDEN_FROM empty: one line, nothing written.
        idle = log_base / "trex" / "ppo" / "20260914_000002"
        capsys.readouterr()
        exec(code, namespace(idle, widen_from=""))  # noqa: S102
        assert capsys.readouterr().out.count("\n") == 1 and not (idle / "01_stance").exists()

        # (6) A parent without provenance.json, or this run's own directory, refuses with the reason.
        bare = log_base / "trex" / "ppo" / "bare_parent"
        (bare / "01_stance").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="not a run with a provenance.json"):
            exec(code, namespace(log_base / "trex" / "ppo" / "20260914_000003", widen_from="bare_parent"))  # noqa: S102
        with pytest.raises(RuntimeError, match="this run's own directory"):
            exec(code, namespace(run_dir, widen_from=str(run_dir)))  # noqa: S102

        # (7) A parent stage without a stage_config.json, or without a recorded run seed, refuses BEFORE the tool
        # runs and before the SEED comparison (otherwise a raw FileNotFoundError, or a SEED mismatch against None,
        # would mask the tool's own refusal); nothing is written.
        import shutil

        def _drop_config(config_path: Path) -> None:
            config_path.unlink()

        def _drop_seed(config_path: Path) -> None:
            record = json.loads(config_path.read_text(encoding="utf-8"))
            record["run"] = {}
            config_path.write_text(json.dumps(record), encoding="utf-8")

        for name, damage, reason in (
            ("no_config", _drop_config, "records no stage_config.json"),
            ("no_seed", _drop_seed, "records no run seed"),
        ):
            damaged = log_base / "trex" / "ppo" / f"parent_{name}"
            shutil.copytree(parent["run_dir"], damaged)
            damage(damaged / parent["stage_dir"].name / "stage_config.json")
            fresh = log_base / "trex" / "ppo" / f"20260914_{name}"
            with pytest.raises(RuntimeError, match=reason):
                exec(code, namespace(fresh, widen_from=damaged.name))  # noqa: S102
            assert not (fresh / "01_stance").exists()

    def test_the_knob_bounds_the_revision_gap(self, tmp_path, capsys):
        """D-C17: an r-2 parent — the shape of the certified trex r11 stance parents at r13 — is refused under the
        default ``WIDEN_MAX_REVISION_GAP = 1`` with the tool's own message naming the gap and the flag, and widens under
        ``2`` with the report and the result recording ``revision_gap`` 2.  One r-2 parent is built for both halves."""
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("torch")
        from environments.shared.config import WIDEN_LINEAGE_KEYS
        from environments.shared.result_bundle import initialize_result_bundle
        from environments.shared.scripts.widen_checkpoint import WidenError

        from .test_widen_checkpoint import PARENT_SEED, build_narrow_parent

        src = _cell(WIDEN_CELL_MARKER)
        code = compile(src, "sb3_training.ipynb[widen cell]", "exec")
        log_base = tmp_path / "logs"
        parent_run = "20260101_000002"
        parent = build_narrow_parent(log_base / "trex" / "ppo", "ppo", run_name=parent_run, revision_gap=2)
        current = parent["current"]
        narrow = parent["narrow"]
        assert narrow.policy_interface_revision == current.policy_interface_revision - 2
        assert narrow.policy_interface_sha256 != current.policy_interface_sha256
        initialize_result_bundle(
            parent["run_dir"],
            species="trex",
            algorithm="ppo",
            backend="stable-baselines3",
            seed=PARENT_SEED,
            plant_identity=narrow.to_dict(),
            run_id=parent_run,
            repository_root=REPO_ROOT,
        )
        chain = load_stage_manifest("trex").chain_for("behavior")

        def namespace(run_dir: Path, *, max_revision_gap: int) -> dict:
            return _widen_cell_namespace(
                log_base, run_dir, chain, seed=PARENT_SEED, widen_from=parent_run, max_revision_gap=max_revision_gap
            )

        # Under the default bound the tool refuses (the cell never catches WidenError): the message names both
        # revisions, the measured gap, the bound and the opt-in flag; nothing is written into the run.
        refused = log_base / "trex" / "ppo" / "20260914_000010"
        with pytest.raises(WidenError) as info:
            exec(code, namespace(refused, max_revision_gap=1))  # noqa: S102 - the notebook cell under test
        message = str(info.value)
        for fragment in (
            "is not the current trex plant one interface-only revision behind",
            f"policy_interface_revision: parent={narrow.policy_interface_revision}, "
            f"current={current.policy_interface_revision}",
            "gap 2 exceeds max_revision_gap=1",
            "--max-revision-gap 2 / max_revision_gap=2",
            "plant_versions.toml",
        ):
            assert fragment in message, fragment
        assert not (refused / "01_stance").exists()

        # Under WIDEN_MAX_REVISION_GAP = 2 the same parent widens; the gap is recorded everywhere the tool writes it.
        widened = log_base / "trex" / "ppo" / "20260914_000011"
        capsys.readouterr()
        bound = namespace(widened, max_revision_gap=2)
        exec(code, bound)  # noqa: S102
        out = capsys.readouterr().out
        assert "policy-interface revisions crossed: 2 (WIDEN_MAX_REVISION_GAP=2)" in out
        target = widened / "01_stance"
        result = bound["_widen_result"]
        assert result.revision_gap == 2 and result.target_stage_dir == target
        report = json.loads((target / "widen_report.json").read_text(encoding="utf-8"))
        assert report["revision_gap"] == 2 and report["max_revision_gap"] == 2
        assert result.report["revision_gap"] == 2 and result.report["max_revision_gap"] == 2
        run_block = json.loads((target / "stage_config.json").read_text(encoding="utf-8"))["run"]
        assert run_block["widened_from_policy_interface_revision"] == narrow.policy_interface_revision
        assert run_block["widened_from_run_id"] == parent_run and run_block["seed"] == PARENT_SEED
        assert set(WIDEN_LINEAGE_KEYS) <= set(run_block)
        assert (target / "models" / "stage1_final.zip").is_file() and not (target / "gate_verdict.json").exists()


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
        assert [ast.unparse(arg) for arg in videos.args] == ["entry.reference", "handoff['stage_dir']"], (
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
        # The disconnect names the behavior.
        cells_after = cells[_cell_index(cells, COMPLETION_CELL_MARKER) + 1 :]
        assert cells_after and "BEHAVIOR" in _names(ast.parse(cells_after[-1]))
        assert _calls(ast.parse(cells_after[-1]), "disconnect_runtime")

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
