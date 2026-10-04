"""Cell-level pins for ``notebooks/sb3_training.ipynb`` (BEHAVIOR_RECIPES_PLAN §4.7, Phase A WS5).

The SB3 notebook is the Colab driver an operator actually runs, and Phase A
turned its per-stage cells into one behavior-chain loop: ``BEHAVIOR`` names a
deliverable, the manifest resolves the chain, and each node is REUSED (a
certified ancestor in ``RUN_DIR`` or ``TRUNK_FROM``), JUDGED (trained by the
RESUME cell but never gated) or TRAINED from its parent's handoff along the
declared edge — then published BEFORE its verdict is enforced.  Decisions
D-A15 and D-A17..D-A21 (§6.1) amended that loop with retrain-from, chain-by-
digest reuse, the occupied-directory guard, recorded durations and the
ignored-edit warning; cleanup ROW-4/6's decision 6 (b) (amending D-A17 and
D-C13) has it judge a node ``RUN_DIR`` holds trained but unjudged before it
consults the trunk, and its decision 4 (a) has the resolve cell record the
trunk in ``RUN_DIR/trunk_run.json``, which the RESUME cell checks a resume
against.

These tests read the notebook JSON and pin its STRUCTURE — the order of calls
inside the ``for NODE in CHAIN:`` body, keyword presence, the absence of
position-keyed names: a rewording of a print survives; a semantic regression (a node
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

from environments.shared.stage_manifest import load_stage_manifest

from .notebook_cells import cell_index as _cell_index
from .notebook_cells import cell_sources, code_cell, code_cell_sources, code_cells, exec_top_level_def

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
#: The Drive summary checks out ``REPO_REF`` with this notebook's Git block (cleanup CU-15), pinned here.
DRIVE_SUMMARY_PATH = REPO_ROOT / "notebooks" / "google_drive_summary.ipynb"
#: ``train_stage`` trains through ``train_base.train`` (consolidation PR-14c); pins on what it records read its source.
TRAIN_BASE_PATH = REPO_ROOT / "environments" / "shared" / "train_base.py"

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
#: RESOLVE cell and calls `policy_loading.sb3_archive_load_preflight`, which loads a real archive (the trunk run's root
#: handoff, else a throwaway) through `load_sb3_model` before anything is trained (KNOWN_ISSUES, "SB3 archives are
#: bound to the interpreter that saved them"; the cell's body moved into the library with cleanup CU-6).
PREFLIGHT_CELL_MARKER = '# SB3 archive load preflight (KNOWN_ISSUES "Training / RL": SB3 archives are bound to the interpreter that saved them)'

#: The knobs every ``halt`` / ``disconnect_runtime`` call passes by name, read when it runs (consolidation PR-14b).
DISCONNECT_KNOBS = (("in_colab", "IN_COLAB"), ("auto", "AUTO_DISCONNECT"), ("flush_drive", "USE_GOOGLE_DRIVE"))

#: Every species with a committed stage manifest (the notebook's SPECIES menu).
SPECIES_WITH_MANIFESTS = sorted(path.parent.name for path in (REPO_ROOT / "configs").glob("*/stages.toml"))


def _code_cells() -> list[str]:
    return code_cell_sources(NOTEBOOK_PATH)


def _all_cell_sources() -> list[str]:
    return cell_sources(NOTEBOOK_PATH)


def _cell(marker: str) -> str:
    return code_cell(NOTEBOOK_PATH, marker)


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


def _define_node_budget(namespace: dict) -> None:
    """The infrastructure cell's real ``node_budget`` (never a stub), for an executed cell that reads the budget."""
    exec_top_level_def(_cell(INFRA_CELL_MARKER), "node_budget", namespace)


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
        # The task digest comes from the CURRENT config and this session's plant identity through the one
        # stage-level derivation (task_fingerprint.stage_task_fingerprint, cleanup CU-8a), the helper
        # train_base.train records it through (train_stage trains through it with SPECIES_CFG and
        # STAGE_CONFIGS). A drift (env_kwargs={} in place of stage_config=config, say) would silently
        # refuse every reuse with "judged under task". Executed:
        # test_executed_the_loop_derives_the_task_through_the_stage_helper below.
        fingerprint = _call(loop, "stage_task_fingerprint")
        assert [ast.unparse(arg) for arg in fingerprint.args] == ["SPECIES", "stage"]
        assert {kw.arg: ast.get_source_segment(src, kw.value) for kw in fingerprint.keywords} == {
            "stage_config": "config",
            "plant_identity": "PLANT_IDENTITY",
        }
        assert not _calls(ast.parse(src), "derive_stage_task_fingerprint"), "the cell derives only through the helper"
        task_assign = next(
            node for node in loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "task_sha256"
        )
        assert isinstance(task_assign.value, ast.Subscript) and task_assign.value.value is fingerprint
        assert ast.unparse(task_assign.value.slice) == "'task_sha256'"
        # Candidates: RUN_DIR first, then TRUNK_DIR (when set) — for a non-target, non-covered node, unless RUN_DIR
        # holds it trained but unjudged (cleanup ROW-4/6, decision 6 (b), amending D-A17 and D-C13): such a node is
        # judged here, only on the parent resolved here, and the trunk is not consulted for it (executed:
        # TestJudgeFirst).
        covered_if = _the_if(loop, src, lambda test: test == "covered", "on `covered`")
        assert len(covered_if.orelse) == 1 and isinstance(covered_if.orelse[0], ast.If)
        target_if = covered_if.orelse[0]
        assert ast.get_source_segment(src, target_if.test) == "NODE.id == TARGET_NODE.id"
        else_assign = next(
            node
            for node in target_if.orelse
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "candidates"
        )
        assert ast.unparse(else_assign.value) == (
            "[RUN_DIR] + ([TRUNK_DIR] if TRUNK_DIR is not None and unjudged is None else [])"
        ), "RUN_DIR is tried first; the trunk only when one is set and RUN_DIR holds no unjudged node"
        unjudged_assign, lineage_if, last = target_if.orelse
        assert last is else_assign
        assert isinstance(unjudged_assign, ast.Assign) and ast.unparse(unjudged_assign.targets[0]) == "unjudged"
        assert ast.unparse(unjudged_assign.value) == "unjudged_stage_dir(RUN_DIR, species=SPECIES, entry=NODE)"
        assert isinstance(lineage_if, ast.If)
        assert ast.get_source_segment(src, lineage_if.test) == "unjudged is not None and TRUNK_DIR is not None"
        assert not lineage_if.orelse
        refuse, printed = lineage_if.body
        assert isinstance(refuse, ast.Expr) and refuse.value is _call(loop, "refuse_judging_off_the_resolved_parent")
        assert [ast.unparse(arg) for arg in refuse.value.args] == ["unjudged"]
        assert _keyword_names(refuse.value) == {"entry", "parent_model_sha256"}
        assert _keyword_source(src, refuse.value, "entry") == "NODE"
        assert _keyword_source(src, refuse.value, "parent_model_sha256") == _keyword_source(
            src, _call(loop, "find_certified_ancestor"), "parent_model_sha256"
        ), "rule 4 checks the unjudged node against the parent the reuse rule would chain it onto"
        assert isinstance(printed, ast.Expr) and _func_name(printed.value) == "print", "why the trunk is skipped"
        # The header states the exemption: a node the run holds as an ancestors/ record still consults the trunk
        # (result_bundle.unjudged_stage_dir). The candidates comment states it, and that a root widened beside it
        # is refused.
        assert "holds trained but unjudged (and not as an ancestors/ record) never comes from" in src
        assert "or refused as an interrupted node or (rule 4) as trained on another parent." in src
        assert "One the run holds as an ancestors/ record still consults the trunk (a root widened beside it" in src
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
        judge_test = " ".join((ast.get_source_segment(src, _judge_if(src, loop).test) or "").split())
        assert judge_test.startswith("not covered and verdict is None and ")

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
        # The library's guarded check (config.ignored_hyperparameter_edits, cleanup CU-8a): the diff is
        # taken against the ANCESTOR's recorded stage_config.json (a diff of config against itself is
        # always empty, so the warning could never fire), and a record that is missing, unreadable or
        # not a JSON object is named, never raised on. Executed:
        # test_executed_a_reused_ancestor_whose_record_is_not_an_object_is_named_not_raised below.
        diff = _call(reuse_if, "ignored_hyperparameter_edits")
        assert [ast.unparse(arg) for arg in diff.args] == ["config", "ALGORITHM", "ancestor.stage_dir"]
        assert not diff.keywords
        assign = next(node for node in ast.walk(reuse_if) if isinstance(node, ast.Assign) and node.value is diff)
        assert ast.unparse(assign.targets[0]) == "ignored_edits"
        assert not _calls(ast.parse(src), "hyperparameter_diff"), "the cell uses the guarded check, not its own copy"
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

    @staticmethod
    def _run_chain_loop(namespace: dict, run_dir: Path, *, behavior: str, trunk_dir: "Path | None") -> None:
        """Execute the configuration, resolve and chain-loop cells for velociraptor under *behavior* into
        *namespace*; the caller patches the reuse rule (and whatever it spies on) first."""
        from environments.shared.config import load_all_stages
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.stage_manifest import stage_dirname, stage_label

        run_dir.mkdir(parents=True, exist_ok=True)
        namespace.update(load_all_stages=load_all_stages, load_stage_manifest=load_stage_manifest)
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        # RUN_DIR implies the storage cell ran, which binds LOG_BASE (the resolve cell records the trunk under it).
        namespace.update(LOG_BASE=run_dir.parent / "logs", RUN_DIR=run_dir, TRUNK_DIR=trunk_dir, BEHAVIOR=behavior)
        exec(_cell(RESOLVE_CELL_MARKER), namespace)
        namespace.update(
            Path=Path,
            stage_dirname=stage_dirname,
            stage_label=stage_label,
            PLANT_IDENTITY=current_plant_identity("velociraptor"),
            NODE_RESULTS={},
            completed_stages=[],
            NODE_HANDOFF={},
            QUICK_TEST=True,
        )
        _define_node_budget(namespace)
        exec(_cell(CHAIN_CELL_MARKER), namespace)

    def test_executed_the_loop_derives_the_task_through_the_stage_helper(self, tmp_path, monkeypatch):
        """Cleanup CU-8a: the digest the reuse rule compares is the one stage-level derivation's
        (``task_fingerprint.stage_task_fingerprint``, the helper ``train_base.train`` records through),
        called once per node with this session's in-memory config and plant identity, and it reaches
        ``derive_stage_task_fingerprint`` with exactly the arguments training records."""
        from environments.shared import ancestors, task_fingerprint

        class Asked(Exception):
            """The reuse rule was consulted: what follows is pinned elsewhere."""

        real_helper = task_fingerprint.stage_task_fingerprint
        real_derive = task_fingerprint.derive_stage_task_fingerprint
        helper_calls: list = []
        derived: list = []
        asked: list = []

        def helper(*args, **kwargs):
            helper_calls.append((args, kwargs))
            return real_helper(*args, **kwargs)

        def derive(**kwargs):
            derived.append(kwargs)
            return real_derive(**kwargs)

        def find(candidate, **kwargs):
            asked.append(kwargs["current_task_sha256"])
            raise Asked

        monkeypatch.setattr(task_fingerprint, "stage_task_fingerprint", helper)
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", derive)
        monkeypatch.setattr(ancestors, "find_certified_ancestor", find)
        namespace: dict = {}
        with pytest.raises(Asked):
            self._run_chain_loop(namespace, tmp_path / "20260930_000000", behavior="stance", trunk_dir=None)

        reference = load_stage_manifest("velociraptor").resolve("stance").reference
        config = namespace["STAGE_CONFIGS"][reference]
        identity = namespace["PLANT_IDENTITY"]
        ((args, kwargs),) = helper_calls
        assert args == ("velociraptor", reference)
        assert set(kwargs) == {"stage_config", "plant_identity"}
        # The session's own objects, not copies re-read from disk.
        assert kwargs["stage_config"] is config and kwargs["plant_identity"] is identity
        assert derived == [
            {
                "species": "velociraptor",
                "stage": reference,
                "backend": "stable-baselines3",
                "env_kwargs": config.get("env_kwargs", {}),
                "plant_identity": identity.to_dict(),
            }
        ]
        assert asked == [real_derive(**derived[0])["task_sha256"]]

    def test_executed_a_reused_ancestor_whose_record_is_not_an_object_is_named_not_raised(
        self, tmp_path, monkeypatch, capsys
    ):
        """A certified ancestor whose ``stage_config.json`` is readable JSON but not an object (``[]``)
        is named as ``<unreadable stage_config.json>`` in the ignored-edit warning, and the reuse goes
        on (cleanup CU-8a). The cell's own unguarded copy of the check raised AttributeError here,
        after the reuse had succeeded."""
        import types

        from environments.shared import ancestors
        from environments.shared.ancestors import AncestorReuseError

        class Recorded(Exception):
            """The reuse reached record_ancestor: what follows is pinned elsewhere."""

        trunk = tmp_path / "20260921_000000"
        stance_dir = trunk / "01_stance"
        stance_dir.mkdir(parents=True)
        (stance_dir / "stage_config.json").write_text("[]", encoding="utf-8")
        copy = types.SimpleNamespace(stage_id="stance", run_id=trunk.name, stage_dir=stance_dir)

        def find(candidate, **kwargs):
            if candidate == trunk and kwargs["entry"].id == "stance":
                return copy
            raise AncestorReuseError("refused by the reuse rule")

        def record(run_dir, ancestor):
            assert ancestor is copy
            raise Recorded

        monkeypatch.setattr(ancestors, "find_certified_ancestor", find)
        monkeypatch.setattr(ancestors, "record_ancestor", record)
        with pytest.raises(Recorded):
            self._run_chain_loop({}, tmp_path / "20260930_000000", behavior="walk", trunk_dir=trunk)
        assert (
            "WARNING: reusing certified 'stance' from run 20260921_000000 ignores this run's hyperparameter edit: "
            "<unreadable stage_config.json> differ"
        ) in capsys.readouterr().out


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
        # The same for the other two knobs that pick the tree.
        namespace.update(ALGORITHM="sac")
        exec(compile(src, "sb3_storage_sac", "exec"), namespace)
        assert namespace["RUN_DIR"] == tmp_path / "logs" / "trex" / "sac" / "20260101_000003"
        namespace.update(SPECIES="velociraptor")
        exec(compile(src, "sb3_storage_species", "exec"), namespace)
        assert namespace["RUN_DIR"] == tmp_path / "logs" / "velociraptor" / "sac" / "20260101_000004"

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

    def test_the_resolve_cell_ends_with_the_complete_run_refusal_then_the_trunk_record(self):
        """The guard on a bound ``RUN_DIR`` is last, once ``TRUNK_DIR`` is final. It refuses a session that would
        write into a complete run, then records the trunk this session resolved in ``RUN_DIR/trunk_run.json``
        (cleanup ROW-4/6, decision 4 (a); executed: TestTrunkRecord), so a refused session writes nothing, and section
        3's markdown says so. D-C13's widened-root refusal left it with ROW-4/6's decision 6 (b): the chain loop
        judges a widened root before it consults any trunk (executed: TestJudgeFirst). The function stays exported
        for a notebook copy older than that change."""
        import environments.shared.result_bundle as result_bundle

        cells = _code_cells()
        resolve_at = _cell_index(cells, RESOLVE_CELL_MARKER)
        src = cells[resolve_at]
        tree = ast.parse(src)
        auto_if = _the_if(tree, src, lambda test: test == 'globals().get("AUTO_TRUNK", False)', "selecting the trunk")
        guard = _the_if(tree, src, lambda test: test == 'globals().get("RUN_DIR") is not None', "on a bound RUN_DIR")
        assert guard is tree.body[-1] and guard.lineno > auto_if.end_lineno, "last, once TRUNK_DIR is final"
        complete = _call(guard, "refuse_complete_run_session")
        assert [ast.unparse(arg) for arg in complete.args] == ["RUN_DIR"]
        for keyword, value in (
            ("species", "SPECIES"),
            ("chain", "CHAIN"),
            ("target", "TARGET_NODE"),
            ("retrain_from", "RETRAIN_NODE"),
            ("trunk_dir", "TRUNK_DIR"),
        ):
            assert _keyword_source(src, complete, keyword) == value, keyword
        imported, refused, recorded = guard.body
        assert isinstance(imported, ast.ImportFrom) and [alias.name for alias in imported.names] == [
            "record_trunk_run",
            "refuse_complete_run_session",
        ]
        assert isinstance(refused, ast.Expr) and refused.value is complete, "the refusal comes first"
        assert isinstance(recorded, ast.Expr) and ast.unparse(recorded) == (
            "print(record_trunk_run(RUN_DIR, trunk_dir=TRUNK_DIR, log_dir=LOG_BASE / SPECIES / ALGORITHM.lower()))"
        ), "then the trunk record, printed, under the directory a TRUNK_FROM id resolves under"
        every = _all_cell_sources()
        section = every[every.index(_cell(STORAGE_CELL_MARKER)) - 1]
        assert section.startswith("## 3. ") and "records the trunk in the run's `trunk_run.json`" in section
        for index, cell in enumerate(every):
            assert "refuse_trunk_over_unjudged_widened_root" not in cell, f"cell {index} names D-C13's refusal"
        assert "refuse_trunk_over_unjudged_widened_root" in result_bundle.__all__, "it stays exported"
        # A plain refusal: no disconnect (imported later, by the infrastructure cell), and the loop has not run yet.
        assert not _calls(tree, "disconnect_runtime") and not _calls(tree, "halt")
        assert resolve_at < _cell_index(cells, CHAIN_CELL_MARKER)

    def test_the_resolve_cell_refuses_a_complete_run_before_anything_is_written(self, tmp_path):
        """Executed: a complete run re-entered for a node it lacks refuses; a reuse-only session passes. Neither
        writes a trunk record into it (decision 4 (a): a complete bundle is immutable)."""
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
        namespace.update(LOG_BASE=run_dir.parent / "logs", RUN_DIR=run_dir, TRUNK_DIR=None, BEHAVIOR="stance")
        exec(_cell(RESOLVE_CELL_MARKER), namespace)  # stance is reused in place: nothing to refuse
        namespace["BEHAVIOR"] = "walk"
        with pytest.raises(ResultBundleError, match="would judge or train 'locomotion' here") as excinfo:
            exec(_cell(RESOLVE_CELL_MARKER), namespace)
        assert 'TRUNK_FROM = "20260920_010912"' in str(excinfo.value)
        assert {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == before
        assert not (run_dir / "trunk_run.json").exists(), "nothing is recorded into a complete run"


class TestTrunkRecord:
    """Cleanup ROW-4/6, decision 4 (a): the resolve cell records the trunk this session resolved in
    ``RUN_DIR/trunk_run.json``, as the ``TRUNK_FROM`` value that reproduces it, and prints the line
    ``result_bundle.record_trunk_run`` returns. Executed through the configuration and resolve cells for velociraptor,
    with ``RUN_DIR`` and the trunk under ``LOG_BASE/velociraptor/ppo/`` as the storage cell lays them out (the table of
    writes is pinned in ``test_result_bundle_trunk_record.py``)."""

    RUN_ID = "20261003_000000"
    TRUNK_ID = "20260921_000000"

    @classmethod
    def _resolve(cls, tmp_path, *, trunk: "Path | str | None", namespace: "dict | None" = None, auto=False) -> dict:
        """Execute the configuration and resolve cells with ``TRUNK_DIR`` = *trunk* (a run id names a run under
        ``LOG_BASE/velociraptor/ppo/``); with *auto*, ``TRUNK_FROM = "auto"`` and the trunk is what the stubbed
        ``select_trunk`` returns (the caller patches it). Returns the namespace, which a second call may reuse."""
        from environments.shared.config import load_all_stages

        log_dir = tmp_path / "logs" / "velociraptor" / "ppo"
        run_dir = log_dir / cls.RUN_ID
        run_dir.mkdir(parents=True, exist_ok=True)
        trunk_dir = log_dir / trunk if isinstance(trunk, str) else trunk
        if namespace is None:
            namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
            exec(_cell(CONFIG_CELL_MARKER), namespace)
        namespace.update(LOG_BASE=tmp_path / "logs", RUN_DIR=run_dir, TRUNK_DIR=trunk_dir, BEHAVIOR="walk")
        if auto:
            from environments.shared.plant_contract import current_plant_identity

            namespace.update(AUTO_TRUNK=True, PLANT_IDENTITY=current_plant_identity("velociraptor"))
        exec(_cell(RESOLVE_CELL_MARKER), namespace)
        return namespace

    @classmethod
    def _run_dir(cls, tmp_path: Path) -> Path:
        return tmp_path / "logs" / "velociraptor" / "ppo" / cls.RUN_ID

    @staticmethod
    def _ancestor_record(run_dir: Path, node: str) -> None:
        record = run_dir / "ancestors" / node / "ancestor.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}")

    @staticmethod
    def _recorded(run_dir: Path) -> str:
        from environments.shared.result_bundle import TRUNK_RECORD_SCHEMA

        record = json.loads((run_dir / "trunk_run.json").read_text(encoding="utf-8"))
        assert set(record) == {"schema", "trunk_from"} and record["schema"] == TRUNK_RECORD_SCHEMA
        return str(record["trunk_from"])

    def test_executed_a_fresh_run_records_the_trunk_id_and_a_session_on_the_same_trunk_writes_nothing(
        self, tmp_path, capsys
    ):
        namespace = self._resolve(tmp_path, trunk=self.TRUNK_ID)  # the trunk need not exist: nothing is read
        path = self._run_dir(tmp_path) / "trunk_run.json"
        assert path.read_bytes() == (
            b'{\n  "schema": "mesozoic.trunk-run/v1",\n  "trunk_from": "' + self.TRUNK_ID.encode() + b'"\n}\n'
        )
        assert f'Trunk record:  wrote TRUNK_FROM = "{self.TRUNK_ID}" to {path}.' in capsys.readouterr().out
        written, stamp = path.read_bytes(), path.stat().st_mtime_ns
        self._resolve(tmp_path, trunk=self.TRUNK_ID, namespace=namespace)
        assert path.read_bytes() == written and path.stat().st_mtime_ns == stamp, "the same trunk: nothing written"
        assert f'names TRUNK_FROM = "{self.TRUNK_ID}", this session\'s trunk.' in capsys.readouterr().out
        assert sorted(entry.name for entry in self._run_dir(tmp_path).iterdir()) == ["trunk_run.json"]

    def test_executed_no_trunk_is_recorded_as_the_empty_value(self, tmp_path):
        self._resolve(tmp_path, trunk=None)
        assert self._recorded(self._run_dir(tmp_path)) == ""

    def test_executed_a_trunk_outside_the_log_directory_is_recorded_as_its_absolute_directory(self, tmp_path):
        """A pinned ``TRUNK_FROM`` may be an absolute path to a run anywhere; its name alone would resolve under
        ``LOG_BASE/<species>/<algo>/``, another directory."""
        elsewhere = tmp_path / "elsewhere" / self.TRUNK_ID
        self._resolve(tmp_path, trunk=elsewhere)
        assert self._recorded(self._run_dir(tmp_path)) == str(elsewhere.resolve())

    def test_executed_the_auto_selected_trunk_is_recorded_never_auto(self, tmp_path, monkeypatch, capsys):
        import types

        from environments.shared import ancestors

        selected = tmp_path / "logs" / "velociraptor" / "ppo" / self.TRUNK_ID
        monkeypatch.setattr(
            ancestors,
            "select_trunk",
            lambda *args, **kwargs: types.SimpleNamespace(run_dir=selected, describe=lambda: "stubbed selection"),
        )
        self._resolve(tmp_path, trunk=None, auto=True)
        assert self._recorded(self._run_dir(tmp_path)) == self.TRUNK_ID
        assert f'wrote TRUNK_FROM = "{self.TRUNK_ID}"' in capsys.readouterr().out

    def test_executed_a_changed_trunk_is_recorded_again_while_the_run_holds_no_ancestor_record(self, tmp_path, capsys):
        namespace = self._resolve(tmp_path, trunk=None)
        self._resolve(tmp_path, trunk=self.TRUNK_ID, namespace=namespace)
        assert self._recorded(self._run_dir(tmp_path)) == self.TRUNK_ID
        assert f'TRUNK_FROM = "" -> "{self.TRUNK_ID}"' in capsys.readouterr().out

    def test_executed_the_recorded_trunk_is_kept_once_the_run_holds_an_ancestor_record(self, tmp_path, capsys):
        """A session under another trunk is warned, not refused (only the RESUME cell refuses), and the record keeps
        the recorded trunk."""
        run_dir = self._run_dir(tmp_path)
        namespace = self._resolve(tmp_path, trunk=self.TRUNK_ID)
        self._ancestor_record(run_dir, "stance")
        written = (run_dir / "trunk_run.json").read_bytes()
        capsys.readouterr()
        self._resolve(tmp_path, trunk="20260922_000000", namespace=namespace)
        assert (run_dir / "trunk_run.json").read_bytes() == written
        out = capsys.readouterr().out
        assert (
            f"WARNING: this run's ancestor records ('stance') were reused through TRUNK_FROM = \"{self.TRUNK_ID}\""
            in out
        )
        assert f'Set TRUNK_FROM = "{self.TRUNK_ID}" in the configuration cell' in out

    def test_executed_a_run_with_ancestor_records_but_no_trunk_record_is_never_recorded_with_a_guess(
        self, tmp_path, capsys
    ):
        """A run opened before ROW-4/6 that holds records: the resume recipe's manual route applies."""
        run_dir = self._run_dir(tmp_path)
        self._ancestor_record(run_dir, "stance")
        self._resolve(tmp_path, trunk=self.TRUNK_ID)
        assert not (run_dir / "trunk_run.json").exists()
        assert "Trunk record:  none; this run holds ancestor records ('stance')" in capsys.readouterr().out

    def test_executed_an_unreadable_trunk_record_in_a_run_with_records_is_kept_and_warned_about(self, tmp_path, capsys):
        run_dir = self._run_dir(tmp_path)
        self._ancestor_record(run_dir, "stance")
        (run_dir / "trunk_run.json").write_text("{")
        self._resolve(tmp_path, trunk=self.TRUNK_ID)
        assert (run_dir / "trunk_run.json").read_text() == "{"
        out = capsys.readouterr().out
        assert "WARNING: cannot read the trunk record" in out and "the RESUME cell refuses a resume" in out


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
        assert train_stage.args.kw_defaults[kwonly.index("run_dir")] is None, (
            "run_dir has no default: a node trains only into a run, never into a fallback directory outside it"
        )
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
        """train_stage trains through train_base.train (consolidation PR-14c), whose shaping this pins: train()
        hands its edge, load mode and load to the stage body it trains through (cleanup CU-10b), and the body
        builds the shaping from exactly those."""
        src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        train = _top_level_def(src, "train")
        body = _call(train, "_train_stage_body")
        assert _keyword_source(src, body, "parent_id") == "entry.warm_start_from", (
            "shaping fires on the declared EDGE (warm_start_from), verbatim like every CLI caller"
        )
        assert _keyword_source(src, body, "task_load_mode") == "task_load_mode"
        assert _keyword_source(src, body, "load_path") == "load_path"
        assert not _calls(train, "_stage_entry_shaping_callbacks"), "train() builds no shaping of its own"
        stage_body = _top_level_def(src, "_train_stage_body")
        shaping = _call(stage_body, "_stage_entry_shaping_callbacks")
        assert _keyword_names(shaping) == {"task_load_mode", "parent_id", "load_path"}
        for name in ("parent_id", "task_load_mode", "load_path"):
            assert _keyword_source(src, shaping, name) == name, name
        extend = next(call for call in _calls(stage_body, "extend") if shaping in call.args)
        assert ast.unparse(extend.func) == "callbacks.extend" and len(extend.args) == 1, (
            "the shaping callbacks are applied unfiltered — SAC gets the same warm-up the CLI gives it (DU1)"
        )
        assert any(isinstance(statement, ast.Expr) and statement.value is extend for statement in stage_body.body), (
            "applied on every path through the body, under no condition"
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
            # evaluate_stage_checkpoints replaces the metrics.json report, and a stop propagates before
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

    def test_train_stage_evaluates_by_default_and_only_the_resume_cell_opts_out(self):
        """Cleanup CU-6: ``evaluate`` is keyword-only and ``True`` by default; ``False`` skips only the
        ``evaluate_stage_checkpoints`` call, which sits in an ``if evaluate:`` without ``else``, right after the
        values the tuple holds without it, so the one 6-tuple ``return`` stays. The chain loop's TRAIN branch and the
        manual cell keep the default (their ``stage_results`` feed the artifacts); only the RESUME cell passes
        ``evaluate=False``, since the JUDGE branch evaluates every node the RESUME cell trains, from disk, before
        anything certifies it (cleanup ROW-4/6)."""
        src = _cell(INFRA_CELL_MARKER)
        train_stage = _top_level_def(src, "train_stage")
        kwonly = [arg.arg for arg in train_stage.args.kwonlyargs]
        assert "evaluate" in kwonly, "evaluate is a keyword-only argument of train_stage"
        default = train_stage.args.kw_defaults[kwonly.index("evaluate")]
        assert isinstance(default, ast.Constant) and default.value is True
        guard = _the_if(train_stage, src, lambda test: test == "evaluate", "on evaluate")
        assert guard in train_stage.body and not guard.orelse
        assert _call(train_stage, "evaluate_stage_checkpoints") in [n for stmt in guard.body for n in ast.walk(stmt)]
        assert ast.unparse(train_stage.body[train_stage.body.index(guard) - 1]) == (
            "_model, _handoff_stem, _final_model_path, _handoff_vecnorm, _stage_results = "
            "(model, None, str(final_path), None, None)"
        ), "without the evaluation: the trained model, no handoff, the final model path, no results"
        passed = {}
        for marker in (CHAIN_CELL_MARKER, MANUAL_CELL_MARKER, RESUME_CELL_MARKER):
            cell_src = _cell(marker)
            call = _call(ast.parse(cell_src), "train_stage")
            passed[marker] = _keyword_source(cell_src, call, "evaluate") if "evaluate" in _keyword_names(call) else None
        assert passed == {CHAIN_CELL_MARKER: None, MANUAL_CELL_MARKER: None, RESUME_CELL_MARKER: "False"}

    def test_train_stage_without_evaluate_trains_the_same_and_skips_only_the_evaluation(self, tmp_path, capsys):
        """Executed (cleanup CU-6): the infrastructure cell's real ``train_stage``, with ``train_base.train`` and the
        evaluation stubbed. ``evaluate=False`` makes the same ``train`` call and the same prints as the default and
        skips exactly ``evaluate_stage_checkpoints`` (up to 60 episodes and both ``evaluation_*.csv`` files, which the
        JUDGE branch redoes); the 6-tuple keeps its shape, with the in-memory model, the final model path and the
        stage directory in their slots. The default evaluates once, on the in-memory model."""
        from types import SimpleNamespace

        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname, stage_label

        species = "compsognathus_robot"
        model = SimpleNamespace(num_timesteps=1_234)
        trained: list[tuple[tuple, dict]] = []
        evaluated: list[tuple[tuple, dict]] = []

        def train(*args, **kwargs):
            trained.append((args, kwargs))
            return model

        def evaluate_stage_checkpoints(*args, **kwargs):
            evaluated.append((args, kwargs))
            return "evaluated model", "handoff stem", "final model path", "handoff sidecar", {"stage": 2}

        namespace = {
            "train_base": SimpleNamespace(train=train),
            "evaluate_stage_checkpoints": evaluate_stage_checkpoints,
            "read_stage_duration": lambda stage_dir: 12.5,
            "Path": Path,
            "stage_dirname": stage_dirname,
            "stage_label": stage_label,
            "MANIFEST": load_stage_manifest(species),
            "SPECIES": species,
            "STAGE_CONFIGS": load_all_stages(species),
            "SPECIES_CFG": object(),
            "ALGORITHM": "PPO",
            "N_ENVS": 4,
            "SEED": 42,
            "VERBOSE": 0,
            "PLANT_IDENTITY": object(),
            "EVALUATION_SEED": 3042,
        }
        exec_top_level_def(_cell(INFRA_CELL_MARKER), "train_stage", namespace)
        stage_dir = tmp_path / stage_dirname(species, 2)

        evaluated_tuple = namespace["train_stage"](2, 1_000, run_dir=tmp_path, task_load_mode="resume_same_stage")
        printed = capsys.readouterr().out
        assert evaluated_tuple == (
            "evaluated model",
            "handoff stem",
            "final model path",
            stage_dir,
            "handoff sidecar",
            {"stage": 2},
        )
        [(args, kwargs)] = evaluated
        assert args[4] == stage_dir and kwargs["model"] is model
        assert kwargs["timesteps"] == 1_234 and kwargs["duration_seconds"] == 12.5

        skipped = namespace["train_stage"](
            2, 1_000, run_dir=tmp_path, task_load_mode="resume_same_stage", evaluate=False
        )
        assert skipped == (model, None, str(stage_dir / "models" / f"{stage_label(2)}_final"), stage_dir, None, None)
        assert len(evaluated) == 1, "no evaluation without evaluate"
        assert len(trained) == 2 and trained[0] == trained[1], "the training itself is the same"
        assert capsys.readouterr().out == printed and "Final model saved to: " in printed

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
        ``<stage_label>_final`` pair the tool wrote, under a trunk too: it judges the widened root before it consults
        any trunk (cleanup ROW-4/6, decision 6 (b), which replaced D-C13's resolve-cell refusal; executed:
        TestJudgeFirst)."""
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
            # Decision 6 (b): a node RUN_DIR holds trained but unjudged is found, and checked against the parent
            # resolved here, before the reuse rule may consult the trunk.
            "unjudged_stage_dir(",
            "refuse_judging_off_the_resolved_parent(",
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
        assert len(_calls(loop, "unjudged_stage_dir")) == 1
        assert len(_calls(loop, "refuse_judging_off_the_resolved_parent")) == 1

    def test_the_branch_numbers_follow_the_loop_order_in_the_code_and_the_prose(self):
        """Section 6 numbers the branches in the order the loop tries them, and the chain cell's labels agree (they
        carried BEHAVIOR_RECIPES_PLAN §4.7's numbers, 1 REUSE, 2 TRAIN, 3 JUDGE, until cleanup CU-5). Its Judge item
        says the trunk is not consulted for a node this run trained but never judged (cleanup ROW-4/6, decision
        6 (b))."""
        src = _cell(CHAIN_CELL_MARKER)
        labels = re.findall(r"#\s+\((\d)\) (REUSE|JUDGE|TRAIN)\b", src)
        assert labels == [("1", "REUSE"), ("2", "JUDGE"), ("3", "TRAIN")] * 2, "the header, then the branches"
        every = _all_cell_sources()
        prose = every[every.index(src) - 1]
        assert prose.startswith("## 6. ")
        numbered = re.findall(r"^(\d)\. \*\*(\w+)\*\*", prose, re.MULTILINE)
        assert numbered == [("1", "Reuse"), ("2", "Judge"), ("3", "Train")]
        judge = prose[prose.index("2. **Judge**") : prose.index("3. **Train**")]
        assert "never replaced by a trunk's copy: the trunk is not consulted for it" in judge
        assert "decision 6 (b)" in judge

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
        namespace.update(
            LOG_BASE=run_dir.parent / "logs", RUN_DIR=run_dir, TRUNK_DIR=tmp_path / "20260921_000000", BEHAVIOR="walk"
        )
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
        _define_node_budget(namespace)
        try:
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        finally:
            after = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
            assert after == before, "nothing is written into the complete run"
            assert not trained and not recorded, "nothing is trained or recorded into the complete run"

    @pytest.mark.parametrize("broken", ["zip", "sidecar", "missing sidecar", None])
    def test_executed_a_final_pair_cut_short_is_an_interrupted_node_never_judged(self, tmp_path, monkeypatch, broken):
        """The JUDGE branch loads the final pair, so a pair a reclaim cut short during the final save would fail
        there on every Run all with a bare load error. The loop runs the RESUME cell's check first and sends such a
        node to the interrupted-node refusal, which names what is wrong and RESUME_STAGE (decision D-D16, amended)."""
        import pickle

        from environments.shared import ancestors
        from environments.shared.ancestors import AncestorReuseError
        from environments.shared.config import load_all_stages
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.stage_manifest import stage_dirname, stage_label

        run_dir = tmp_path / "20260925_000000"
        reference = load_stage_manifest("velociraptor").resolve("stance").reference
        models = run_dir / stage_dirname("velociraptor", reference) / "models"
        final = models / f"{stage_label(reference)}_final"
        TestResumeCell._write_pair(Path(f"{final}.zip"), Path(f"{final}_vecnorm.pkl"))
        if broken == "zip":
            Path(f"{final}.zip").write_bytes(Path(f"{final}.zip").read_bytes()[:30])
        elif broken == "sidecar":
            Path(f"{final}_vecnorm.pkl").write_bytes(pickle.dumps({"obs_rms": list(range(20))})[:12])
        elif broken == "missing sidecar":
            Path(f"{final}_vecnorm.pkl").unlink()
        namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        namespace.update(LOG_BASE=run_dir.parent / "logs", RUN_DIR=run_dir, TRUNK_DIR=None, BEHAVIOR="stance")
        exec(_cell(RESOLVE_CELL_MARKER), namespace)

        def refuse(candidate, **kwargs):
            raise AncestorReuseError("no gate_verdict.json")

        def never(*args, **kwargs):
            raise AssertionError("a final pair cut short must never reach the JUDGE branch or be trained over")

        class Judged(Exception):
            pass

        judged: list[dict] = []

        def judge(*args, **kwargs):
            judged.append(kwargs)
            raise Judged  # stop the loop here: what follows JUDGE is pinned elsewhere

        monkeypatch.setattr(ancestors, "find_certified_ancestor", refuse)
        namespace.update(
            Path=Path,
            stage_dirname=stage_dirname,
            stage_label=stage_label,
            PLANT_IDENTITY=current_plant_identity("velociraptor"),
            NODE_RESULTS={},
            completed_stages=[],
            NODE_HANDOFF={},
            evaluate_stage_checkpoints=never if broken else judge,
            train_stage=never,
            QUICK_TEST=False,
            SPECIES_CFG=None,
            ALGORITHM="ppo",
            EVALUATION_SEED=3042,
            read_stage_duration=lambda stage_dir: None,
        )
        _define_node_budget(namespace)
        if broken is None:  # an intact final pair is judged exactly as before
            with pytest.raises(Judged):
                exec(_cell(CHAIN_CELL_MARKER), namespace)
            assert [call["final_path"] for call in judged] == [final]
            return
        with pytest.raises(
            RuntimeError, match=r"no completed stage \((bad/truncated|VecNormalize sidecar|missing matched)"
        ):
            exec(_cell(CHAIN_CELL_MARKER), namespace)

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


class TestJudgeFirst:
    """Cleanup ROW-4/6, decision 6 (b), amending D-A17 and D-C13: the chain loop judges a node ``RUN_DIR`` holds
    trained but unjudged (any ``models/*.zip`` and no ``gate_verdict.json``) before it consults the trunk, and only on
    the parent resolved here. Executed against a trunk whose copy of every node the reuse rule accepts, so a trunk
    the loop asks would stand in for the node: the trunk must not be asked for such a node at all."""

    RUN_ID = "20261003_000000"
    TRUNK_ID = "20260921_000000"
    #: The handoff digest of the trunk's (stubbed) copy of every node.
    TRUNK_SHA = "sha256:" + "a" * 64

    class Judged(Exception):
        """The loop reached the JUDGE branch: what follows it is pinned elsewhere."""

    class Trained(Exception):
        """The loop reached the TRAIN branch: what follows it is pinned elsewhere."""

    @classmethod
    def _loop(cls, tmp_path, monkeypatch, *, behavior, prepare, trunk=True, retrain_from=""):
        """Execute the configuration, resolve and chain-loop cells for velociraptor under *behavior*, with
        ``prepare(run_dir)`` writing what the run holds first and the trunk pinned (or no trunk). The reuse rule is
        stubbed: this run's directory never holds a reusable copy (the cases below hold none, or a verdict the rule
        is taken to refuse), and the trunk's copy of every node is accepted. Returns the reuse rule's questions as
        ``(run id, node id)``, the recorded ancestors, the final paths JUDGE was given and the stages TRAIN was
        given; the loop is left to raise (``Judged``, ``Trained`` or a refusal) to the caller."""
        import types

        from environments.shared import ancestors
        from environments.shared.ancestors import AncestorReuseError
        from environments.shared.config import load_all_stages
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.stage_manifest import stage_dirname, stage_label

        run_dir = tmp_path / "logs" / "velociraptor" / "ppo" / cls.RUN_ID
        trunk_dir = run_dir.parent / cls.TRUNK_ID if trunk else None
        run_dir.mkdir(parents=True)
        prepare(run_dir)
        asked: list = []
        recorded: list = []
        judged: list = []
        trained: list = []

        def find(candidate, **kwargs):
            entry = kwargs["entry"]
            asked.append((Path(candidate).name, entry.id))
            if trunk_dir is not None and candidate == trunk_dir:
                stage_dir = trunk_dir / stage_dirname("velociraptor", entry.reference)
                return types.SimpleNamespace(
                    stage_id=entry.id,
                    run_id=cls.TRUNK_ID,
                    stage_dir=stage_dir,
                    model_stem=str(stage_dir / "models" / "handoff"),
                    normalization_path=stage_dir / "models" / "handoff_vecnorm.pkl",
                    source_run_dir=trunk_dir,
                    model_sha256=cls.TRUNK_SHA,
                    normalization_sha256="sha256:" + "b" * 64,
                    handoff_name="handoff",
                    verdict={},
                )
            raise AncestorReuseError("refused by the stubbed reuse rule")

        def judge(*args, **kwargs):
            judged.append(kwargs["final_path"])
            raise cls.Judged

        def train(*args, **kwargs):
            trained.append(kwargs["stage"])
            raise cls.Trained

        monkeypatch.setattr(ancestors, "find_certified_ancestor", find)
        monkeypatch.setattr(
            ancestors, "record_ancestor", lambda directory, ancestor: recorded.append(ancestor.stage_id)
        )
        namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
        exec(_cell(CONFIG_CELL_MARKER), namespace)
        # RUN_DIR implies the storage cell ran, which binds LOG_BASE (the trunk lies under it, as TRUNK_FROM's id).
        namespace.update(
            LOG_BASE=tmp_path / "logs",
            RUN_DIR=run_dir,
            TRUNK_DIR=trunk_dir,
            BEHAVIOR=behavior,
            RETRAIN_FROM=retrain_from,
        )
        exec(_cell(RESOLVE_CELL_MARKER), namespace)
        namespace.update(
            Path=Path,
            stage_dirname=stage_dirname,
            stage_label=stage_label,
            PLANT_IDENTITY=current_plant_identity("velociraptor"),
            NODE_RESULTS={},
            completed_stages=[],
            NODE_HANDOFF={},
            evaluate_stage_checkpoints=judge,
            train_stage=train,
            QUICK_TEST=False,
            SPECIES_CFG=None,
            ALGORITHM="ppo",
            EVALUATION_SEED=3042,
            read_stage_duration=lambda stage_dir: None,
        )
        _define_node_budget(namespace)
        return namespace, asked, recorded, judged, trained

    @staticmethod
    def _stage_dir(run_dir: Path, node: str) -> Path:
        from environments.shared.stage_manifest import stage_dirname

        return run_dir / stage_dirname("velociraptor", load_stage_manifest("velociraptor").resolve(node).reference)

    @classmethod
    def _final_pair(cls, run_dir: Path, node: str, *, run_block: "dict | None" = None) -> Path:
        """An intact ``<stage_label>_final`` pair of *node* in *run_dir* (the stem the JUDGE branch is given), with
        *run_block* as its ``stage_config.json`` run block when given."""
        from environments.shared.stage_manifest import stage_label

        stage_dir = cls._stage_dir(run_dir, node)
        final = (
            stage_dir / "models" / f"{stage_label(load_stage_manifest('velociraptor').resolve(node).reference)}_final"
        )
        TestResumeCell._write_pair(Path(f"{final}.zip"), Path(f"{final}_vecnorm.pkl"))
        if run_block is not None:
            (stage_dir / "stage_config.json").write_text(json.dumps({"run": run_block}), encoding="utf-8")
        return final

    @classmethod
    def _edge(cls, parent_sha256: str) -> dict:
        """The run block of a non-root trained along its edge from the parent checkpoint *parent_sha256*."""
        return {"load_mode": "initialize_next_stage", "parent_checkpoint_sha256": parent_sha256, "load_path": "x"}

    @staticmethod
    def _record(run_dir: Path, node: str) -> None:
        record = run_dir / "ancestors" / node / "ancestor.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}", encoding="utf-8")

    def test_executed_an_unjudged_root_is_judged_before_a_certifying_trunk(self, tmp_path, monkeypatch, capsys):
        """A node a trunk's copy stood in for before ROW-4/6: the trunk certifies stance, yet the loop never asks it
        and judges this run's own copy; nothing is recorded, and the reason is printed."""
        finals: dict = {}
        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="walk",
            prepare=lambda run: finals.update(stance=self._final_pair(run, "stance")),
        )
        with pytest.raises(self.Judged):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert judged == [finals["stance"]]
        assert asked == [(self.RUN_ID, "stance")], "the trunk is never asked for a node this run trained"
        assert recorded == [] and trained == []
        assert f"Not consulting the trunk for 'stance': {self._stage_dir(namespace['RUN_DIR'], 'stance')}" in (
            capsys.readouterr().out
        )

    def test_executed_an_unjudged_ancestor_on_the_resolved_parent_is_judged(self, tmp_path, monkeypatch):
        """A non-root trained here on the parent this session resolved (the trunk's stance) is judged; the trunk
        still stands in for stance, which this run never trained."""
        finals: dict = {}
        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="hunt",
            prepare=lambda run: finals.update(
                locomotion=self._final_pair(run, "locomotion", run_block=self._edge(self.TRUNK_SHA))
            ),
        )
        with pytest.raises(self.Judged):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert judged == [finals["locomotion"]]
        assert recorded == ["stance"]
        assert asked == [(self.RUN_ID, "stance"), (self.TRUNK_ID, "stance"), (self.RUN_ID, "locomotion")]

    def test_executed_an_unjudged_ancestor_on_another_parent_is_refused_before_the_trunk_is_asked(
        self, tmp_path, monkeypatch
    ):
        """Rule 4 before the trunk is skipped: JUDGE applies no chain check, so a node trained on another parent
        than the one resolved here is refused, with the remedy that works (a fresh RUN_ID), not judged; and the
        trunk's copy may not stand in for it either."""
        from environments.shared.result_bundle import ResultBundleError

        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="hunt",
            prepare=lambda run: self._final_pair(run, "locomotion", run_block=self._edge("sha256:" + "c" * 64)),
        )
        with pytest.raises(ResultBundleError, match="does not chain onto the parent this session resolved") as excinfo:
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert "fresh RUN_ID" in str(excinfo.value) and "gate_verdict.json" not in str(excinfo.value)
        assert judged == [] and trained == []
        assert asked == [(self.RUN_ID, "stance"), (self.TRUNK_ID, "stance")], "locomotion is never looked up"

    @pytest.mark.parametrize("held", ["periodic checkpoints only", "a final pair without its sidecar"])
    def test_executed_an_interrupted_node_under_a_certifying_trunk_is_refused_toward_resume(
        self, tmp_path, monkeypatch, held
    ):
        """Any checkpoint without a verdict keeps the trunk out, not only an intact final pair: an interrupted node
        reaches the interrupted-node refusal, which names RESUME_STAGE, instead of being abandoned for the trunk's
        copy."""
        from environments.shared.stage_manifest import stage_label

        def prepare(run_dir):
            if held == "periodic checkpoints only":
                models = self._stage_dir(run_dir, "stance") / "models"
                label = stage_label(load_stage_manifest("velociraptor").resolve("stance").reference)
                TestResumeCell._write_pair(
                    models / f"{label}_100000_steps.zip", models / f"{label}_vecnormalize_100000_steps.pkl"
                )
            else:
                Path(f"{self._final_pair(run_dir, 'stance')}_vecnorm.pkl").unlink()

        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path, monkeypatch, behavior="walk", prepare=prepare
        )
        with pytest.raises(RuntimeError, match="an interrupted node. Set RESUME_STAGE"):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert asked == [(self.RUN_ID, "stance")]
        assert recorded == [] and judged == [] and trained == []

    @pytest.mark.parametrize(
        "verdict",
        ["{}", json.dumps({"passed": False, "failures": ["x"]}), json.dumps({"passed": True})],
        ids=["malformed", "failed", "passed"],
    )
    def test_executed_a_node_with_any_verdict_still_consults_the_trunk(self, tmp_path, monkeypatch, capsys, verdict):
        """A verdict, whatever it says, is tested by presence: the node is judged already, so the reuse rule decides
        as before (a malformed one is never read before REUSE), and the trunk's copy stands in when this run's
        verdict is refused."""

        def prepare(run_dir):
            self._final_pair(run_dir, "stance")
            (self._stage_dir(run_dir, "stance") / "gate_verdict.json").write_text(verdict, encoding="utf-8")

        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path, monkeypatch, behavior="walk", prepare=prepare
        )
        with pytest.raises(self.Trained):  # the target, locomotion, is trained next
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert asked == [(self.RUN_ID, "stance"), (self.TRUNK_ID, "stance"), (self.RUN_ID, "locomotion")]
        assert recorded == ["stance"] and judged == []
        assert "Not consulting the trunk" not in capsys.readouterr().out

    def test_executed_a_node_the_run_holds_as_an_ancestor_record_keeps_the_trunk(self, tmp_path, monkeypatch, capsys):
        """A run that took the trunk's copy over its own before ROW-4/6 holds the node's unjudged pair and its
        ``ancestors/`` record: the record is the node in this run, and judging the directory too would make every
        later bundle write refuse it as both trained and reused."""

        def prepare(run_dir):
            self._final_pair(run_dir, "stance")
            self._record(run_dir, "stance")

        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path, monkeypatch, behavior="walk", prepare=prepare
        )
        with pytest.raises(self.Trained):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert judged == [], "a record-held node is never judged"
        assert asked[:2] == [(self.RUN_ID, "stance"), (self.TRUNK_ID, "stance")] and recorded == ["stance"]
        assert "Not consulting the trunk" not in capsys.readouterr().out

    def test_executed_a_covered_node_trains_and_no_trunk_is_asked(self, tmp_path, monkeypatch, capsys):
        """D-A19 unchanged: a node RETRAIN_FROM covers has no reuse candidates and trains, whatever RUN_DIR holds
        (D-A20 then refuses an occupied directory in train_stage)."""
        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="walk",
            retrain_from="stance",
            prepare=lambda run: self._final_pair(run, "stance"),
        )
        with pytest.raises(self.Trained):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert asked == [] and recorded == [] and judged == []
        assert trained == [load_stage_manifest("velociraptor").resolve("stance").reference]
        assert "Not consulting the trunk" not in capsys.readouterr().out

    def test_executed_the_target_is_judged_from_this_run_as_before(self, tmp_path, monkeypatch, capsys):
        """D-A18 unchanged: the target is only ever looked for in this run, so there is no trunk to skip."""
        finals: dict = {}
        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="stance",
            prepare=lambda run: finals.update(stance=self._final_pair(run, "stance")),
        )
        with pytest.raises(self.Judged):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert judged == [finals["stance"]] and asked == [(self.RUN_ID, "stance")] and recorded == []
        assert "Not consulting the trunk" not in capsys.readouterr().out

    def test_executed_without_a_trunk_the_loop_judges_as_before(self, tmp_path, monkeypatch, capsys):
        """No trunk, nothing to skip: no rule-4 check and no print, and JUDGE takes the node as before."""
        finals: dict = {}
        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="walk",
            trunk=False,
            prepare=lambda run: finals.update(stance=self._final_pair(run, "stance")),
        )
        with pytest.raises(self.Judged):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert judged == [finals["stance"]] and asked == [(self.RUN_ID, "stance")] and recorded == []
        assert "Not consulting the trunk" not in capsys.readouterr().out

    def test_executed_a_widened_root_under_a_trunk_passes_the_resolve_cell_and_is_judged(
        self, tmp_path, monkeypatch, capsys
    ):
        """D-C13 as amended by decision 6 (b): the resolve cell no longer refuses a trunk over a root widened into
        this run on the command line; the chain loop judges the widened root (rule 4 passes it as a root: it records
        no load lineage) before it consults the trunk, on the ``<stage_label>_final`` pair the tool wrote, and records
        nothing."""
        finals: dict = {}
        widened = {"seed": 42, "n_envs": 4, "widened_from_run_id": "20260815_205206"}
        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path,
            monkeypatch,
            behavior="walk",
            prepare=lambda run: finals.update(stance=self._final_pair(run, "stance", run_block=widened)),
        )  # the resolve cell ran with the trunk set, and passed
        with pytest.raises(self.Judged):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert judged == [finals["stance"]]
        assert asked == [(self.RUN_ID, "stance")] and recorded == [] and trained == []
        assert "Not consulting the trunk for 'stance'" in capsys.readouterr().out

    def test_executed_a_widened_root_cut_short_under_a_trunk_is_an_interrupted_node(self, tmp_path, monkeypatch):
        """A widened pair without its sidecar is not judged, and not replaced by the trunk's copy either: it is
        refused as an interrupted node."""

        def prepare(run_dir):
            widened = {"seed": 42, "n_envs": 4, "widened_from_run_id": "20260815_205206"}
            Path(f"{self._final_pair(run_dir, 'stance', run_block=widened)}_vecnorm.pkl").unlink()

        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path, monkeypatch, behavior="walk", prepare=prepare
        )
        with pytest.raises(RuntimeError, match="an interrupted node. Set RESUME_STAGE"):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert asked == [(self.RUN_ID, "stance")] and recorded == [] and judged == []

    @pytest.mark.parametrize("trunk", [True, False], ids=["trunk", "no-trunk"])
    def test_executed_a_widened_root_the_run_holds_as_an_ancestor_record_is_refused(self, tmp_path, monkeypatch, trunk):
        """The widen tool does not refuse a run that already took the root from a trunk: a root widened there would
        be neither judged (the record stays the node in the run, which D-C13 refused before ROW-4/6) nor judgeable
        (the bundle refuses a node both trained and reused). Refused before the trunk is asked or anything is judged,
        recorded or trained, with or without a trunk, naming a new run id."""
        from environments.shared.result_bundle import ResultBundleError

        def prepare(run_dir):
            widened = {"seed": 42, "n_envs": 4, "widened_from_run_id": "20260815_205206"}
            self._final_pair(run_dir, "stance", run_block=widened)
            self._record(run_dir, "stance")

        namespace, asked, recorded, judged, trained = self._loop(
            tmp_path, monkeypatch, behavior="walk", prepare=prepare, trunk=trunk
        )
        with pytest.raises(ResultBundleError, match=r"already holds 'stance' as a reused ancestor.*new run id"):
            exec(_cell(CHAIN_CELL_MARKER), namespace)
        assert asked == recorded == judged == trained == []


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
        assert _keyword_source(infra, forwarded, "run_dir") == "RUN_DIR", "the bundle is this run's, never another"
        summary_def = _top_level_def(infra, "write_training_summary")
        assert _keyword_source(infra, _call(summary_def, "_lib_write_training_summary"), "species") == "SPECIES"
        # No dead parameters (cleanup CU-5): the run directory and species are forwarded, never taken and ignored.
        assert [arg.arg for arg in bundle_def.args.args] == ["stage_results_list", "species"]
        assert [arg.arg for arg in summary_def.args.args] == ["run_dir", "stage_results_list"]
        assert not bundle_def.args.kwonlyargs and not summary_def.args.kwonlyargs
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
        ``eval_env_seed(seed)`` in the stage body train() trains through (cleanup CU-10b); the storage cell
        records the same value, ``SEED + 1000``, as the checkpoint-selection seed role."""
        from environments.shared.train_base import eval_env_seed

        selection = _top_level_assigns(_cell(STORAGE_CELL_MARKER))["CHECKPOINT_SELECTION_SEED"]
        assert ast.unparse(selection) == "SEED + 1000"
        notebook_value = compile(ast.Expression(selection), "CHECKPOINT_SELECTION_SEED", "eval")
        for seed in (0, 7, 42, 1_000_003):
            assert eval(notebook_value, {"SEED": seed}) == eval_env_seed(seed), seed
        train_src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        body = _call(_top_level_def(train_src, "train"), "_train_stage_body")
        assert _keyword_source(train_src, body, "seed") == "seed"
        eval_env = next(
            node.value
            for node in ast.walk(_top_level_def(train_src, "_train_stage_body"))
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "eval_env"
        )
        assert _func_name(eval_env) == "create_vec_env" and ast.unparse(eval_env.args[4]) == "eval_env_seed(seed)"

    def test_the_bundle_write_passes_discovered_replicates(self):
        """save_run_bundle hands the sibling replicates to the writer (D-B10/D-B16)."""
        src = _cell(INFRA_CELL_MARKER)
        call = _call(ast.parse(src), "_lib_save_result_bundle")
        assert "replicates" in _keyword_names(call)
        # save_run_bundle writes RUN_DIR's bundle (its dead run_dir=None parameter went in cleanup CU-5).
        assert _keyword_source(src, call, "replicates").startswith("discover_replicates_for_run(RUN_DIR, ")
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


class TestNodeBudget:
    """One budget derivation: the chain loop, the RESUME cell and the manual cell read ``node_budget``, so a resume
    measures what is left against the budget the chain loop trains and judges a node to (decision D-D16)."""

    def test_every_budget_site_reads_node_budget(self):
        cells = _code_cells()
        infra = cells[_cell_index(cells, INFRA_CELL_MARKER)]
        budget_def = _top_level_def(infra, "node_budget")
        for index, src in enumerate(cells):
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Subscript) and ast.unparse(node).endswith("['curriculum_kwargs']['timesteps']"):
                    assert src == infra and budget_def.lineno <= node.lineno <= (budget_def.end_lineno or 0), (
                        f"code cell {index} line {node.lineno} derives a node budget outside node_budget()"
                    )
        for marker, target, value in (
            (CHAIN_CELL_MARKER, "budget", "node_budget(stage)"),
            (RESUME_CELL_MARKER, "budget_res", "node_budget(stage_res)"),
            (MANUAL_CELL_MARKER, "_manual_budget", "MANUAL_TIMESTEPS or node_budget(_manual_stage)"),
        ):
            assigns = [
                node
                for node in ast.walk(ast.parse(_cell(marker)))
                if isinstance(node, ast.Assign) and [ast.unparse(each) for each in node.targets] == [target]
            ]
            assert [ast.unparse(node.value) for node in assigns] == [value], marker

    def test_executed_the_budget_is_the_toml_budget_or_the_quick_test_budget(self):
        from environments.shared.config import load_all_stages

        for species in SPECIES_WITH_MANIFESTS:
            stage_configs = load_all_stages(species)
            for quick_test in (False, True):
                namespace = {"QUICK_TEST": quick_test, "STAGE_CONFIGS": stage_configs}
                _define_node_budget(namespace)
                for entry in load_stage_manifest(species).stages:
                    toml_budget = stage_configs[entry.reference]["curriculum_kwargs"]["timesteps"]
                    expected = 50_000 if quick_test else toml_budget
                    assert namespace["node_budget"](entry.reference) == expected, (species, entry.id, quick_test)


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
        # Cleanup CU-6: the chain loop's JUDGE branch evaluates the resumed node from disk before it judges it (since
        # ROW-4/6 for every node this cell trains), so this cell's train_stage skips its own evaluation.
        assert _keyword_source(src, train, "evaluate") == "False", "the RESUME cell evaluates nothing"
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
        # ... and the checkpoint walk (with its two refusals) runs only for an unfinished node, over this node's own
        # models directory and label. Since cleanup CU-6 the walk is the library's, which reads the trainer's
        # checkpoint-name pattern (test_curriculum_checkpoints.py): the cell globs and parses no name itself.
        walks = _calls(tree, "newest_intact_periodic_pair")
        assert [ast.unparse(call) for call in walks] == ["newest_intact_periodic_pair(model_dir_res, label_res)"]
        assert walks[0] in [node for stmt in finished.orelse for node in ast.walk(stmt)]
        assert not _calls(tree, "glob") and not _calls(tree, "rglob")
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert "re" not in imported and "re" not in _names(tree), "the cell compiles no pattern of its own"
        assert "read_gate_verdict(stage_dir_res)" in src
        # One integrity check for every pair: the final pair here, each periodic candidate inside the walk.
        checks = _calls(tree, "checkpoint_pair_problem")
        assert [ast.unparse(call) for call in checks] == ["checkpoint_pair_problem(*final_pair_res)"]
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
        calls: list[dict] | None = None,
        trunk_dir: "Path | None" = None,
    ) -> list[dict]:
        """Execute the RESUME cell against one intact periodic pair of *reference* at *steps* (none when *steps* is
        None: a widened root holds only its handoff and final pairs); return the train_stage calls. Whatever else the
        stage directory holds (a verdict, a final pair, an ``ancestors/`` record, ``trunk_run.json``) is the
        caller's. ``RUN_DIR`` is *tmp_path*, and ``LOG_BASE`` and ``ALGORITHM`` are bound as the storage and
        configuration cells bind them, so a trunk under ``LOG_BASE/<species>/ppo/`` is recorded by its id;
        ``TRUNK_DIR`` is *trunk_dir* (cleanup ROW-4/6, decision 4 (a))."""
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname, stage_label

        models = tmp_path / stage_dirname(species, reference) / "models"
        models.mkdir(parents=True, exist_ok=True)
        label = stage_label(reference)
        if steps is not None:
            cls._write_pair(models / f"{label}_{steps}_steps.zip", models / f"{label}_vecnormalize_{steps}_steps.pkl")
        manifest = load_stage_manifest(species)
        made: list[dict] = [] if calls is None else calls

        def train_stage(save_freq=100_000, **kwargs):  # the notebook's default checkpoint cadence
            made.append(kwargs)
            return (None,) * 6

        namespace = {
            "SPECIES": species,
            "BEHAVIOR": behavior,
            "MANIFEST": manifest,
            "CHAIN": manifest.chain_for(manifest.resolve_behavior(behavior).id),
            "STAGE_CONFIGS": load_all_stages(species),
            "QUICK_TEST": False,
            "RUN_DIR": tmp_path,
            "LOG_BASE": tmp_path / "logs",
            "ALGORITHM": "ppo",
            "TRUNK_DIR": trunk_dir,
            "RUN_LABEL": "",
            "RETRAIN_NODE": manifest.resolve(retrain_from) if retrain_from else None,
            "train_stage": train_stage,
        }
        _define_node_budget(namespace)
        src = _cell(RESUME_CELL_MARKER).replace("RESUME_STAGE = None", f"RESUME_STAGE = {resume_stage!r}", 1)
        exec(compile(src, "sb3_resume", "exec"), namespace)
        return made

    @pytest.mark.parametrize(
        ("behavior", "resume_stage", "reference"),
        [("walk", "locomotion", 2), ("walk", 2, 2), ("stand", "recovery", "recovery")],
    )
    def test_the_resume_cell_takes_a_stage_number_or_id(self, tmp_path, capsys, behavior, resume_stage, reference):
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
        assert call["evaluate"] is False, "the chain loop's JUDGE branch evaluates the resumed node (cleanup CU-6)"
        assert "WARNING" not in capsys.readouterr().out, "an ordinary resume warns about nothing"

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
        its sidecar is an interrupted save, resumed from the newest intact periodic pair (here within one checkpoint
        cadence of the budget, the state a reclaim during the final save leaves)."""
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        models.mkdir(parents=True)
        (models / "stage2_final.zip").write_bytes(b"truncated")
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_900_000)
        assert call["timesteps"] == 100_000 and call["load_path"].endswith("stage2_2900000_steps.zip")

    @pytest.mark.parametrize("broken", ["zip", "sidecar"])
    def test_a_final_pair_cut_short_is_resumed_over(self, tmp_path, capsys, broken):
        """Executed: a reclaim during the final save can leave the final pair broken (off a mount ``train()`` writes it in
        place and can truncate it; on one it stages the pair and leaves a zip without its sidecar). It is checked like a periodic pair; a broken one is not a finished node
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
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_900_000)
        assert call["timesteps"] == 100_000 and call["load_path"].endswith("stage2_2900000_steps.zip")
        assert "WARNING: the final pair of 'locomotion' is incomplete" in capsys.readouterr().out

    def test_a_final_pair_cut_short_after_an_early_stop_is_refused(self, tmp_path):
        """Executed: a broken final pair means learn() returned; its newest intact periodic pair 200k short of the
        budget (beyond the 100k cadence) says the node stopped early, and training it further is a new attempt
        (D-D16), never the rest of the budget in place."""
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_final.zip", models / "stage2_final_vecnorm.pkl")
        (models / "stage2_final_vecnorm.pkl").unlink()
        calls: list[dict] = []
        with pytest.raises(RuntimeError, match=r"200,000 steps short .* early stop .* fresh RUN_ID"):
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", 2, 2, steps=2_800_000, calls=calls)
        assert calls == []

    @pytest.mark.parametrize("reclaimed_in", ["sidecar removal", "local save", "zip copy", "sidecar copy"])
    @pytest.mark.parametrize(("steps", "refused"), [(2_800_000, True), (2_900_000, False)])
    def test_a_staged_final_save_cut_short_on_a_mount_reads_as_one_written_in_place(
        self, tmp_path, monkeypatch, reclaimed_in, steps, refused
    ):
        """Executed against what ``train_base._save_final_and_sync_tb`` leaves: on a Drive mount it stages the final
        pair and publishes it (CU-3, D-D20). A reclaim at any point of that save must leave what a save straight to the
        mount left, a final zip cut short, so the cell resumes over it within one checkpoint cadence of the budget and
        refuses it beyond as an early stop (D-D16). Without the placeholder zip ``train()`` writes first, a reclaim
        before the zip landed left no final zip, and an early-stopped node was trained further in place."""
        import pickle
        import tempfile
        import zipfile

        from environments.shared import file_io, train_base
        from environments.shared.stage_manifest import stage_dirname

        class Reclaimed(BaseException):
            pass

        class Model:
            def save(self, path):
                if reclaimed_in == "local save":
                    Path(f"{path}.zip").write_bytes(b"PK\x03\x04 half")
                    raise Reclaimed()
                with zipfile.ZipFile(f"{path}.zip", "w") as archive:
                    archive.writestr("data", "{}")
                    archive.writestr("policy.pth", b"")

        class Env:
            def save(self, path):
                Path(path).write_bytes(pickle.dumps({}))

        failing_copy = {"zip copy": 1, "sidecar copy": 2}.get(reclaimed_in)
        real_copy, copies = file_io.atomic_copy, []

        def copy(src, dst):
            copies.append(dst)
            if len(copies) == failing_copy:
                raise Reclaimed()
            real_copy(src, dst)

        monkeypatch.setattr(file_io, "atomic_copy", copy)
        root = tmp_path / "content" / "drive"
        monkeypatch.setattr(train_base, "_REMOTE_MOUNT_ROOTS", (str(root),))
        (tmp_path / "scratch").mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
        run_dir = root / "MyDrive" / "compsognathus_robot" / "ppo" / "20260926_000000"
        models = run_dir / stage_dirname("compsognathus_robot", 2) / "models"
        models.mkdir(parents=True)
        if reclaimed_in == "sidecar removal":
            # A previous final pair (here as a save straight to the mount left it: zip cut short, sidecar intact);
            # the reclaim lands after the placeholder, before the old sidecar goes.
            (models / "stage2_final.zip").write_bytes(b"PK\x03\x04 half")
            Env().save(str(models / "stage2_final_vecnorm.pkl"))
            real_unlink = Path.unlink

            def unlink(self, *args, **kwargs):
                if self.name == "stage2_final_vecnorm.pkl":
                    raise Reclaimed()
                return real_unlink(self, *args, **kwargs)

            monkeypatch.setattr(Path, "unlink", unlink)

        with pytest.raises(Reclaimed):
            train_base._save_final_and_sync_tb(Model(), Env(), models, 2, None, root / "tb")
        monkeypatch.undo()
        assert (models / "stage2_final.zip").exists()
        assert (models / "stage2_final_vecnorm.pkl").exists() == (reclaimed_in == "sidecar removal")

        calls: list[dict] = []
        if refused:
            with pytest.raises(RuntimeError, match=r"200,000 steps short .* early stop .* fresh RUN_ID"):
                self._run_resume_cell(run_dir, "compsognathus_robot", "walk", 2, 2, steps=steps, calls=calls)
            assert calls == []
        else:
            [call] = self._run_resume_cell(run_dir, "compsognathus_robot", "walk", 2, 2, steps=steps, calls=calls)
            assert call["timesteps"] == 100_000 and call["load_path"].endswith("stage2_2900000_steps.zip")

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
        calls: list[dict] = []
        with pytest.raises(RuntimeError, match=r"which RETRAIN_FROM .* covers.*Set RETRAIN_FROM = \"\"") as excinfo:
            self._run_resume_cell(
                tmp_path, "compsognathus_robot", "walk", "locomotion", 2, retrain_from=retrain_from, calls=calls
            )
        assert calls == [], "refused before anything trains"
        # Decision 6 (b): the loop judges the resumed node before it consults any trunk, so the BEHAVIOR route left.
        assert "BEHAVIOR" not in str(excinfo.value)
        assert "judges the resumed node before it consults any trunk" in str(excinfo.value)

    def test_a_retrain_from_below_the_resumed_node_does_not_cover_it(self, tmp_path):
        """Executed: RETRAIN_FROM covers the named node and its descendants only; resuming an ancestor of it
        trains as usual."""
        [call] = self._run_resume_cell(
            tmp_path, "compsognathus_robot", "hunt", "locomotion", 2, retrain_from="behavior"
        )
        assert call["stage"] == 2

    def test_a_broken_newest_periodic_pair_falls_back_to_the_next_intact_one(self, tmp_path, capsys):
        """Executed: every periodic candidate goes through the same check; a newest pair a reclaim cut short is
        skipped with a warning and the next older intact step-point is resumed."""
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_2800000_steps.zip", models / "stage2_vecnormalize_2800000_steps.pkl")
        (models / "stage2_2800000_steps.zip").write_bytes(b"PK\x03\x04 truncated")
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_700_000)
        assert call["load_path"].endswith("stage2_2700000_steps.zip") and call["timesteps"] == 300_000
        assert "WARNING: skipping stage2_2800000_steps.zip: bad/truncated checkpoint zip" in capsys.readouterr().out

    def test_the_resume_report_follows_the_warnings_in_order(self, tmp_path, capsys):
        """Executed (cleanup CU-6): the library walks, and the operator sees what the cell printed when it walked
        itself: each skipped candidate's WARNING, then the pair resumed, its sidecar, the skipped count and the
        budget arithmetic, in that order."""
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_2800000_steps.zip", models / "stage2_vecnormalize_2800000_steps.pkl")
        (models / "stage2_vecnormalize_2800000_steps.pkl").unlink()
        self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_700_000)
        budget = load_all_stages("compsognathus_robot")[2]["curriculum_kwargs"]["timesteps"]
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("Resuming 'locomotion' (stage 2) from ")
        assert lines[1:6] == [
            "WARNING: skipping stage2_2800000_steps.zip: missing matched VecNormalize sidecar "
            "stage2_vecnormalize_2800000_steps.pkl",
            f"Newest intact periodic checkpoint: {models / 'stage2_2700000_steps.zip'}",
            f"Matched VecNormalize:              {models / 'stage2_vecnormalize_2700000_steps.pkl'}",
            "(1 newer candidate(s) skipped as incomplete/corrupt — see warnings above)",
            f"Checkpoint steps: 2,700,000 of {budget:,} — remaining budget: {budget - 2_700_000:,}",
        ]

    def test_a_node_without_a_periodic_checkpoint_is_refused_with_the_knobs_to_check(self, tmp_path, capsys):
        """Executed (cleanup CU-6; no test ran this refusal before): no ``stage2_*_steps.zip`` at all usually means
        ``RUN_ID`` or ``RESUME_STAGE`` names the wrong run or node, so the refusal names both knobs and what the number
        means; nothing trains and nothing is warned about. The library walk returns no pair and no skip for it."""
        calls: list[dict] = []
        with pytest.raises(RuntimeError) as refused:
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=None, calls=calls)
        message = str(refused.value)
        assert message.startswith("No periodic checkpoint stage2_*_steps.zip in ")
        assert "RUN_ID in the configuration cell names the interrupted run" in message
        assert "run the storage cell (section 3) after changing it" in message
        assert "RESUME_STAGE names its node (it resolved to 'locomotion'" in message
        assert "not the resolve table's # or the NN_ prefix" in message
        assert calls == [] and "WARNING" not in capsys.readouterr().out

    def test_a_node_without_an_intact_periodic_pair_is_refused_naming_every_skipped_file(self, tmp_path, capsys):
        """Executed (cleanup CU-6; no test ran this refusal before): when every candidate fails the pair check the
        cell warns about each, newest first, and refuses naming them all; it never trains a broken pair."""
        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_2800000_steps.zip", models / "stage2_vecnormalize_2800000_steps.pkl")
        (models / "stage2_2800000_steps.zip").write_bytes(b"PK\x03\x04 truncated")
        self._write_pair(models / "stage2_2700000_steps.zip", models / "stage2_vecnormalize_2700000_steps.pkl")
        (models / "stage2_vecnormalize_2700000_steps.pkl").unlink()
        calls: list[dict] = []
        with pytest.raises(FileNotFoundError) as refused:
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=None, calls=calls)
        message = str(refused.value)
        assert message.startswith(
            "No intact periodic checkpoint pair stage2_<steps>_steps.zip + stage2_vecnormalize_<steps>_steps.pkl in "
        )
        skipped = message.split(". Skipped: ", 1)[1]
        assert skipped.startswith("stage2_2800000_steps.zip: bad/truncated checkpoint zip stage2_2800000_steps.zip (")
        assert skipped.endswith(
            "; stage2_2700000_steps.zip: missing matched VecNormalize sidecar stage2_vecnormalize_2700000_steps.pkl. "
            "Deleting a corrupt newest pair by hand is no longer needed — this cell already fell back through every "
            "older step-point."
        )
        warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("WARNING: skipping ")]
        assert [line.removeprefix("WARNING: skipping ").split(":")[0] for line in warnings] == [
            "stage2_2800000_steps.zip",
            "stage2_2700000_steps.zip",
        ]
        assert calls == []

    @pytest.mark.parametrize("members", [("data",), ("policy.pth",)])
    def test_a_final_zip_without_sb3_members_is_resumed_over(self, tmp_path, members):
        """Executed: a readable zip is not enough; SB3's outer ``data`` and ``policy.pth`` members are required."""
        import zipfile

        from environments.shared.stage_manifest import stage_dirname

        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_final.zip", models / "stage2_final_vecnorm.pkl")
        with zipfile.ZipFile(models / "stage2_final.zip", "w") as archive:
            for member in members:
                archive.writestr(member, b"x")
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, steps=2_900_000)
        assert call["timesteps"] == 100_000

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
        and the checkpoint scan stay read-only. Then, before anything trains, the trunk record (cleanup ROW-4/6,
        decision 4 (a)): a node the run holds as an ``ancestors/`` record, or another trunk than the recorded one, is
        refused (executed below)."""
        src = _cell(RESUME_CELL_MARKER)
        tree = ast.parse(src)
        spent = _the_if(tree, src, lambda test: test == "remaining_res == 0", "on a spent budget")
        guard = _call(tree, "refuse_write_into_complete_run")
        assert [ast.unparse(arg) for arg in guard.args] == ["RUN_DIR"]
        first = spent.orelse[0]
        assert isinstance(first, ast.Expr) and first.value is guard, "the first statement of the resuming branch"
        assert guard.lineno < _call(tree, "train_stage").lineno
        recorded = _call(tree, "refuse_trunk_other_than_recorded")
        second = spent.orelse[1]
        assert isinstance(second, ast.Expr) and second.value is recorded, "right after the complete-run guard"
        assert ast.unparse(recorded) == (
            "refuse_trunk_other_than_recorded(RUN_DIR, trunk_dir=globals().get('TRUNK_DIR'), "
            "log_dir=LOG_BASE / SPECIES / ALGORITHM.lower(), what=f'Resuming {stage_res!r}', resumed=entry_res)"
        ), "the session's trunk, under the directory a TRUNK_FROM id resolves under, and the node to resume"
        assert recorded.lineno < _call(tree, "train_stage").lineno

    #: The trunk the ``trunk_run.json`` of the resume tests below records, under ``LOG_BASE/compsognathus_robot/ppo/``.
    RECORDED_TRUNK = "20260921_000000"

    @staticmethod
    def _ancestor_record(run_dir: Path, node: str) -> None:
        record = run_dir / "ancestors" / node / "ancestor.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}")

    @staticmethod
    def _trunk_record(run_dir: Path, trunk_from: str) -> None:
        from environments.shared.result_bundle import TRUNK_RECORD_SCHEMA

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trunk_run.json").write_text(
            json.dumps({"schema": TRUNK_RECORD_SCHEMA, "trunk_from": trunk_from}, indent=2, sort_keys=True) + "\n"
        )

    @pytest.mark.parametrize("session", ["no trunk", "another run", "the same name elsewhere"])
    def test_executed_a_resume_under_another_trunk_than_the_run_records_is_refused_before_it_trains(
        self, tmp_path, session
    ):
        """Cleanup ROW-4/6, decision 4 (a): once the run holds an ``ancestors/`` record, its ``trunk_run.json`` keeps
        the recorded trunk, the one those records came through when the cells ran in order. Under another, the chain
        loop would refuse a second parent for a recorded node that trunk certifies only after this resume trained
        (``record_ancestor``), and train a recorded node it does not certify (every one, under ``TRUNK_FROM = ""``)
        again here; a run directory with the recorded name elsewhere is another trunk."""
        from environments.shared.result_bundle import ResultBundleError

        self._ancestor_record(tmp_path, "stance")
        self._trunk_record(tmp_path, self.RECORDED_TRUNK)
        written = (tmp_path / "trunk_run.json").read_bytes()
        trunk_dir = {
            "no trunk": None,
            "another run": tmp_path / "logs" / "compsognathus_robot" / "ppo" / "20260922_000000",
            "the same name elsewhere": tmp_path / "elsewhere" / self.RECORDED_TRUNK,
        }[session]
        calls: list[dict] = []
        with pytest.raises(
            ResultBundleError, match=r"Resuming 2 under TRUNK_FROM = .* is refused before anything"
        ) as excinfo:
            self._run_resume_cell(
                tmp_path, "compsognathus_robot", "walk", "locomotion", 2, calls=calls, trunk_dir=trunk_dir
            )
        assert f'Set TRUNK_FROM = "{self.RECORDED_TRUNK}" in the configuration cell' in str(excinfo.value)
        assert calls == [], "refused before anything trains"
        assert (tmp_path / "trunk_run.json").read_bytes() == written, "the refusal writes nothing"

    @pytest.mark.parametrize("recorded", ["", RECORDED_TRUNK])
    def test_executed_a_resume_under_the_trunk_the_run_records_trains(self, tmp_path, capsys, recorded):
        self._ancestor_record(tmp_path, "stance")
        self._trunk_record(tmp_path, recorded)
        trunk_dir = tmp_path / "logs" / "compsognathus_robot" / "ppo" / recorded if recorded else None
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, trunk_dir=trunk_dir)
        assert call["stage"] == 2
        assert "WARNING" not in capsys.readouterr().out

    @pytest.mark.parametrize("held", ["another trunk", "an unreadable file"])
    def test_executed_a_run_without_ancestor_records_resumes_under_any_trunk(self, tmp_path, held):
        """Only a reuse from a trunk ties a run to that trunk, and every such reuse writes a record."""
        if held == "another trunk":
            self._trunk_record(tmp_path, self.RECORDED_TRUNK)
        else:
            (tmp_path / "trunk_run.json").write_text("{")
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2)
        assert call["stage"] == 2

    def test_executed_a_run_with_records_but_no_trunk_record_resumes_by_the_manual_route(self, tmp_path):
        """A run opened before ROW-4/6: nothing records its trunk, and the recipe's manual route pins it."""
        self._ancestor_record(tmp_path, "stance")
        [call] = self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2)
        assert call["stage"] == 2

    def test_executed_an_unreadable_trunk_record_in_a_run_with_records_is_refused(self, tmp_path):
        from environments.shared.result_bundle import ResultBundleError

        self._ancestor_record(tmp_path, "stance")
        (tmp_path / "trunk_run.json").write_text("{")
        calls: list[dict] = []
        with pytest.raises(ResultBundleError, match=r"cannot read the trunk record .* Remove that file") as excinfo:
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, calls=calls)
        assert calls == [] and "manual route (section 5, step 1)" in str(excinfo.value)

    def test_executed_a_node_the_run_holds_as_an_ancestor_record_is_never_resumed(self, tmp_path):
        """A reused ancestor's record stays the node in the run (``result_bundle.unjudged_stage_dir``'s exemption: the
        chain loop takes it while a trunk certifies the node, and a bundle refuses a node both reused and trained), so a
        resume of it would train and never be judged into the run's bundle. Refused under the trunk the run records,
        too."""
        from environments.shared.result_bundle import ResultBundleError

        self._ancestor_record(tmp_path, "locomotion")
        self._trunk_record(tmp_path, "")
        calls: list[dict] = []
        with pytest.raises(ResultBundleError, match=r"holds 'locomotion' as a reused ancestor .* fresh RUN_ID"):
            self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2, calls=calls)
        assert calls == []

    def test_executed_a_finished_record_held_node_left_in_resume_stage_still_only_prints(self, tmp_path, capsys):
        """D-D16: a ``RESUME_STAGE`` left set on a finished node never stops a later Run all, record or not; the
        refusal sits in the resuming branch only."""
        from environments.shared.stage_manifest import stage_dirname

        self._ancestor_record(tmp_path, "locomotion")
        models = tmp_path / stage_dirname("compsognathus_robot", 2) / "models"
        self._write_pair(models / "stage2_final.zip", models / "stage2_final_vecnorm.pkl")
        assert self._run_resume_cell(tmp_path, "compsognathus_robot", "walk", "locomotion", 2) == []
        assert "Nothing to resume" in capsys.readouterr().out

    def test_the_resume_prose_routes_old_checkpoints_to_the_command_line_widen(self):
        """The markdown right before the RESUME cell: ``RUN_ID`` is set in the configuration cell, never into a
        complete run; an earlier run's certified nodes come in through ``TRUNK_FROM`` in a new run; a pre-bump
        checkpoint is widened on the command line (D-C17's bound as the tool's flag) into a new run judged here
        under the parent's seed (D-C14, D-D14), and since cleanup ROW-4/6 the chain loop judges it before it consults
        any trunk (decision 6 (b)), where the resolve cell refused a trunk before. No copy of the old restart /
        memo-reset remedy survives."""
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
            "the chain loop judges the widened root before it consults any trunk",
        ):
            assert phrase in prose, f"the RESUME prose no longer names {phrase}"
        assert "**new** `RUN_ID`" in prose and "never by pointing `RUN_ID` at the old run" in prose
        for gone in (
            "WIDEN_FROM",
            "WIDEN_MAX_REVISION_GAP",
            "restart the runtime",
            "_ACTIVE_RUN_ID",
            "stray",
            "the resolve cell refuses a trunk",
        ):
            assert gone not in prose, f"the RESUME prose still names {gone!r}"

    def test_the_resume_prose_keeps_the_trunk_recovery_and_d_d16_rules(self):
        """The recipe tells the operator which trunk a resume must re-supply and states D-D16's rules; cleanup CU-5's
        cut of the markdown kept them. Since cleanup ROW-4/6 the run's ``trunk_run.json`` records the trunk (decision
        4 (a)) and is the route; the manual route (the trunk the resolve cell printed, or the nearest ancestor record)
        is the fallback for a run opened before that file existed. The chain loop judges a node this run trained
        before it consults any trunk (decision 6 (b)), so the recipe no longer routes such a node through
        ``BEHAVIOR``, and a node the run holds as an ancestor record, beside its own checkpoints, is never resumed."""
        every = _all_cell_sources()
        prose = every[every.index(_cell(RESUME_CELL_MARKER)) - 1]
        for phrase in (
            # The run records its trunk (decision 4 (a)), and the resume takes TRUNK_FROM from that record ...
            "`TRUNK_FROM` set to the value the run's `trunk_run.json` records",
            "decision 4 (a)",
            # ... and the RESUME cell refuses any other trunk once the run holds a record (decision 4 (a)'s reader);
            # if those records came through a trunk the file does not name (cells run out of order), remove the file.
            "this cell refuses a resume under any other trunk",
            "remove the file and pin `TRUNK_FROM`",
            "A run without `trunk_run.json`",
            # ... and without that file, how to find the trunk, and the two ways a guess goes wrong (#559).
            "nearest the interrupted node",
            "not necessarily the trunk",
            "may select a newer run",
            "never follows this run's own ancestor records",
            # D-D16 (a) and its amendment.
            "is never retrained",
            'Set `RETRAIN_FROM = ""`',
            "within one checkpoint cadence",
            "spent budget without an intact final pair",
            "a new attempt, in a fresh `RUN_ID`",
            "never resumed without its sidecar",
            "`gate_verdict.json` or an intact final pair",
            # A node trained here is judged before any trunk is consulted (decision 6 (b)).
            "decision 6 (b)",
            "is judged here too",
            "A reused ancestor's record stays the node",
            "holds as an `ancestors/` record is never resumed (this cell refuses it)",
            # The widened root is judged here (D-C13 as amended by D-D14).
            "D-C13",
        ):
            assert phrase in prose, f"the RESUME prose no longer says {phrase!r}"
        assert "`BEHAVIOR` set to it" not in prose, "the BEHAVIOR route left with decision 6 (b)"


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
    line.) Since cleanup CU-6 the cell is one call of ``policy_loading.sb3_archive_load_preflight``, whose load,
    flushed print, archive choice and temporary directory test_policy_loading.py pins and executes."""

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
        """Since cleanup CU-6 the cell is one call of the library's preflight on this session's species, chain root
        and trunk; the one load, the flushed print right before it, the trunk's root handoff under either directory
        naming else a throwaway in a temporary directory, and the fall-back that never raises are the function's,
        pinned and executed in test_policy_loading.py. The cell itself neither loads nor prints."""
        src, tree = _preflight_cell()
        assert "from environments.shared.policy_loading import sb3_archive_load_preflight" in src
        (call,) = _calls(tree, "sb3_archive_load_preflight")
        assert ast.unparse(call) == "sb3_archive_load_preflight(SPECIES, CHAIN[0].reference, trunk_dir=TRUNK_DIR)"
        statements = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
        assert len(statements) == 1 and isinstance(statements[0], ast.Assign) and statements[0].value is call, (
            "one top-level call and nothing around it, its result bound so the notebook shows no repr"
        )
        assert not _calls(tree, "load_sb3_model") and not _calls(tree, "print")
        # No bare algorithm load anywhere in the cell: the loader is the path under test.
        assert not re.search(r"\b(PPO|SAC|AlgoClass|alg_cls)\.load\(", src)
        # D-D14 took the WIDEN_FROM parent's branch with the widen cell; a trunk without a pair falls back.
        assert "WIDEN_FROM" not in src and not _raises(tree, "RuntimeError")

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
        # Parameters of any helper the cell defines are bound locally, not read (until cleanup CU-6 the cell defined
        # a handoff helper and a throwaway env class).
        parameters = {
            argument.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for argument in (*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs)
        }
        free = _loaded_names(tree) - bound - parameters - set(dir(builtins))
        assert free <= earlier, f"the preflight cell reads names no earlier cell binds: {sorted(free - earlier)}"
        assert free <= {"SPECIES", "CHAIN", "TRUNK_DIR"}, sorted(free)


class TestCommandSliceReseed:
    """Amendment A12 / invariant 8 (BEHAVIOR_RECIPES_PLAN §4.6): the loaded statistics' command slice is reseeded
    whenever the node's command_mode is not "none" — EXCEPT on a same-stage resume, whose sidecar already holds the
    statistics the policy trained under. ``train_base._load_vecnorm_into_envs`` applies the rule
    (test_command_frame.py) and ``train_base.train``, which ``train_stage`` trains through (consolidation PR-14c),
    feeds it the stage config's mode and the notebook's sidecar — always False in Phase C.  Since cleanup CU-10b
    train() does so through the stage body it hands its load, sidecar, load mode and plant identity."""

    def test_train_feeds_the_rule_the_stage_config_and_the_sidecar(self):
        import inspect

        from environments.shared.curriculum import load_vecnorm_stats

        src = TRAIN_BASE_PATH.read_text(encoding="utf-8")
        train = _top_level_def(src, "train")
        assert not _calls(train, "_load_vecnorm_into_envs"), "train() loads the statistics only through the body"
        body_call = _call(train, "_train_stage_body")
        for name, value in (
            ("vecnorm_load_path", "load_path"),
            ("vecnorm_path", "vecnorm_path"),
            ("task_load_mode", "task_load_mode"),
            ("plant_identity", "plant_identity"),
        ):
            assert _keyword_source(src, body_call, name) == value, name
        assert [ast.unparse(arg) for arg in body_call.args[1:4]] == ["species_cfg", "stage_configs", "stage"]
        body = _top_level_def(src, "_train_stage_body")
        # The body's `config` is the stage config of the stage it trains.
        assert any(
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "config"
            and ast.unparse(node.value) == "stage_configs[stage]"
            for node in body.body
        )
        load = _call(body, "_load_vecnorm_into_envs")
        assert _keyword_source(src, load, "command_mode") == (
            'str(config.get("env_kwargs", {}).get("command_mode", "none"))'
        )
        assert _keyword_source(src, load, "task_load_mode") == "task_load_mode"
        assert _keyword_source(src, load, "vecnorm_path") == "vecnorm_path"
        # Without it a sidecar loads with plant validation skipped (the notebook's manual cell names any sidecar).
        assert _keyword_source(src, load, "plant_identity") == "plant_identity"
        assert [ast.unparse(arg) for arg in load.args] == ["vecnorm_load_path", "train_env", "eval_env"], (
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
        cell runs after that, so any file it wrote would be undeclared and the bundle-verification
        cell's ``validate_result_bundle`` would raise before the auto-disconnect (the flat PNGs every
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
        # Display only by construction: neither wrapper takes a save path or forwards one (cleanup CU-5).
        for name in ("plot_training_curves", "plot_diagnostics_graphs"):
            wrapper = _top_level_def(visualization[0], name)
            assert [arg.arg for arg in wrapper.args.args] == ["stage_dirs", "stage_configs", "algo_name"], name
            assert not wrapper.args.kwonlyargs, name
            assert not {"save_path", "save_dir", "show"} & _keyword_names(_call(wrapper, f"_lib_{name}")), name
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
        "dirty,loaded,indexed,expected",
        [
            (True, False, True, "local edits"),
            (False, True, True, "Restart"),
            (False, False, True, None),
            # A clone whose first fetch failed: no index, HEAD already at the fetched commit, and
            # every file staged as deleted. Setup checks it out instead of reporting an empty tree.
            (True, False, False, None),
        ],
    )
    @pytest.mark.parametrize("path", [NOTEBOOK_PATH, DRIVE_SUMMARY_PATH], ids=["sb3", "drive_summary"])
    def test_colab_ref_change_is_explicit_and_preserves_edits(
        self, tmp_path, monkeypatch, path, dirty, loaded, indexed, expected
    ):
        """Execute the notebook's actual Git setup block with controlled Git replies (both notebooks')."""
        import subprocess
        import types

        checkout = tmp_path / "checkout"
        (checkout / ".git").mkdir(parents=True)
        if indexed:
            (checkout / ".git" / "index").touch()
        source = code_cell(path, "REPO_REF =")
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
                return "old-commit\n" if indexed else "new-commit\n"
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


def _colab_setup(path: Path) -> tuple[ast.Assign, list[ast.stmt]]:
    """The setup cell's ``IN_COLAB`` assignment, and its Git block: in its ``if IN_COLAB:`` block, from
    ``import pathlib`` through the print of the checked-out ref and commit."""
    tree = ast.parse(code_cell(path, "REPO_REF ="))
    (in_colab,) = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["IN_COLAB"]
    ]
    (colab_block,) = [node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "IN_COLAB"]
    body = colab_block.body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Import) and node.names[0].name == "pathlib")
    end = next(i for i, node in enumerate(body) if "Repository ref: " in ast.unparse(node))
    return in_colab, body[start : end + 1]


def test_the_drive_summary_checks_out_repo_ref_with_the_sb3_notebooks_git_block():
    """The Drive summary's setup cell runs the SB3 setup cell's Git block statement for statement (CU-15).

    The block installs the package, so it cannot live in it; the two copies are pinned equal as ASTs
    instead: the ``--no-checkout`` clone, the fetch of ``REPO_REF``, ``FETCH_HEAD^{commit}``, the
    refusals after an import of the package or over local edits, and the detached checkout. Both cells
    declare ``REPO_REF = "main"`` and the same ``IN_COLAB``. The ref-change test above runs the block.
    """
    sb3_in_colab, sb3_block = _colab_setup(NOTEBOOK_PATH)
    drive_in_colab, drive_block = _colab_setup(DRIVE_SUMMARY_PATH)
    assert len(sb3_block) >= 10, "the SB3 Git block was not found whole"
    assert [ast.dump(node) for node in drive_block] == [ast.dump(node) for node in sb3_block], (
        "google_drive_summary.ipynb's Git block differs from sb3_training.ipynb's:\n"
        + ast.unparse(ast.Module(body=drive_block, type_ignores=[]))
    )
    assert ast.dump(drive_in_colab) == ast.dump(sb3_in_colab)
    for path in (NOTEBOOK_PATH, DRIVE_SUMMARY_PATH):
        assert 'REPO_REF = "main"  # @param {"type":"string"}' in code_cell(path, "REPO_REF =").splitlines()


@pytest.mark.parametrize("path", [NOTEBOOK_PATH, DRIVE_SUMMARY_PATH], ids=["sb3", "drive_summary"])
def test_the_notebook_round_trips_through_json_dump_indent_1(path):
    """Every notebook edit goes through json.load -> json.dump(indent=1, ensure_ascii=False) + newline."""
    text = path.read_text(encoding="utf-8")
    notebook = json.loads(text)
    assert json.dumps(notebook, indent=1, ensure_ascii=False) + "\n" == text, (
        f"{path.name} is not in the canonical json.dump(indent=1, ensure_ascii=False) form"
    )
    if path == NOTEBOOK_PATH:
        # Every SB3 code cell is plain Python.
        for index, source in code_cells(path):
            ast.parse(source, filename=f"{path.name}[code cell {index}]")
