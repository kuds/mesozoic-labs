"""The one lenient ``stage_config.json`` reader (cleanup CU-8c).

``file_io.read_json_object`` is the lenient JSON-object reader and
``config.read_recorded_stage_config`` applies it to a stage directory's
``stage_config.json``.  Every library reader that took a bad record as
"proves nothing" goes through them; the Drive summary notebook's own readers
(CU-15's, kept by the cleanup plan's section 4.9) are left alone, two of them
lenient (``_infer_algorithm``, ``_stage_name``).  These tests hold the shared
reader and each former reader to the answers they gave before on the eleven
cases the CU-8c probe measured (missing, a directory, bad JSON, not UTF-8, a
byte-order mark, the four non-object JSON values, an object holding NaN, an
object), and pin the strict readers' new handling of a file that is not
UTF-8.  Two static tripwires keep further copies out (their docstrings say
what they cannot see, and a self-test runs both on each form), and a third
pins the library self-parsers' explicit UTF-8 decoding.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pytest

from environments.shared import ancestors, config, file_io
from environments.shared.reporting import gates, save_result_bundle, stage_artifacts
from environments.shared.result_bundle import (
    ResultBundleError,
    audit_result_bundle,
    reentry,
    validate_evaluation_evidence,
)

from .notebook_cells import code_cells, strip_magics
from .result_bundle_helpers import _complete_bundle, _complete_bundle_inputs, _plant_identity

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = REPO_ROOT / "environments"
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))

_OBJECT = (
    b'{"run": {"duration_seconds": 1.5, "load_mode": "initialize_next_stage", "load_path": "p"}, '
    b'"algorithm": "PPO", "curriculum": {"x": 1}, "task_fingerprint": {"task_sha256": "t"}}'
)
_NAN_OBJECT = (
    b'{"run": {"duration_seconds": NaN, "load_mode": "initialize_next_stage", "load_path": "p"}, '
    b'"curriculum": {"x": NaN}}'
)
_NOT_UTF8 = b'{"a": "\xff"}'
#: The CU-8c probe's eleven files.  ``None`` is no file; ``"DIR"`` a directory in its place.
CASES: dict[str, bytes | str | None] = {
    "missing": None,
    "directory": "DIR",
    "bad-json": b"{",
    "not-utf-8": _NOT_UTF8,
    "byte-order-mark": b'\xef\xbb\xbf{"run": {}}',
    "list": b"[]",
    "null": b"null",
    "string": b'"x"',
    "number": b"3",
    "object-with-nan": _NAN_OBJECT,
    "object": _OBJECT,
}
#: The cases a lenient reader answers with "nothing".
UNREADABLE = [case for case in CASES if not case.startswith("object")]


def _write_case(stage_dir: Path, case: str) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "stage_config.json"
    payload = CASES[case]
    if payload == "DIR":
        path.mkdir()
    elif isinstance(payload, bytes):
        path.write_bytes(payload)
    return path


def _canonical(value: Any) -> str:
    """A comparable rendering: NaN compares unequal to itself, its JSON spelling does not."""
    return json.dumps(value, sort_keys=True, allow_nan=True)


@pytest.mark.parametrize("case", list(CASES))
def test_read_json_object_returns_only_a_json_object(tmp_path: Path, case: str) -> None:
    """``None`` for every case a record proves nothing; the object itself, NaN accepted, otherwise."""
    path = _write_case(tmp_path, case)
    for argument in (path, str(path)):
        value = file_io.read_json_object(argument)
        if case in UNREADABLE:
            assert value is None
        else:
            payload = CASES[case]
            assert isinstance(value, dict) and isinstance(payload, bytes)
            assert _canonical(value) == _canonical(json.loads(payload))
    if case == "object-with-nan":
        value = file_io.read_json_object(path)
        assert value is not None and math.isnan(value["run"]["duration_seconds"])


@pytest.mark.parametrize("case", list(CASES))
def test_read_recorded_stage_config_reads_the_stage_directorys_record(tmp_path: Path, case: str) -> None:
    """The stage-directory form of the same reader, on ``<stage_dir>/stage_config.json``."""
    assert config.STAGE_CONFIG_FILENAME == "stage_config.json"
    path = _write_case(tmp_path, case)
    for stage_dir in (tmp_path, str(tmp_path)):
        assert _canonical(config.read_recorded_stage_config(stage_dir)) == _canonical(file_io.read_json_object(path))


_LINEAGE = {"load_path": "p", "load_mode": "initialize_next_stage"}
_RUN = {"duration_seconds": 1.5, "load_mode": "initialize_next_stage", "load_path": "p"}
_NAN_RUN = {"duration_seconds": math.nan, "load_mode": "initialize_next_stage", "load_path": "p"}
_EDIT_CONFIG: dict[str, Any] = {"ppo_kwargs": {}, "curriculum_kwargs": {}}

#: Each former reader, the answer it gives an unreadable record, and its answers to the two
#: objects -- measured by the CU-8c probe on the base commit, where each read the file itself.
FORMER_READERS: dict[str, tuple[Callable[[Path], Any], Any, Any, Any]] = {
    "config.read_stage_duration": (config.read_stage_duration, None, None, 1.5),
    "config.ignored_hyperparameter_edits": (
        lambda stage_dir: config.ignored_hyperparameter_edits(_EDIT_CONFIG, "PPO", stage_dir),
        ["<unreadable stage_config.json>"],
        [],
        [],
    ),
    "config._recorded_edge_lineage": (config._recorded_edge_lineage, {}, _LINEAGE, _LINEAGE),
    "ancestors._recorded_load_lineage": (ancestors._recorded_load_lineage, {}, _LINEAGE, _LINEAGE),
    "result_bundle.reentry._run_block": (reentry._run_block, None, _NAN_RUN, _RUN),
    "reporting.gates._current_task_sha256": (gates._current_task_sha256, None, None, "t"),
    "reporting.stage_artifacts._recorded_curriculum_block": (
        stage_artifacts._recorded_curriculum_block,
        None,
        {"x": math.nan},
        {"x": 1},
    ),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("reader", list(FORMER_READERS))
def test_each_former_reader_keeps_its_answers(tmp_path: Path, reader: str, case: str) -> None:
    """The public ``read_stage_duration`` and ``ignored_hyperparameter_edits`` and the private
    readers rewired onto the shared one answer every case as they did when each parsed the file."""
    function, unreadable, nan_object, valid_object = FORMER_READERS[reader]
    _write_case(tmp_path, case)
    expected = {"object-with-nan": nan_object, "object": valid_object}.get(case, unreadable)
    assert _canonical(function(tmp_path)) == _canonical(expected)


# --- The strict readers: a file that is not UTF-8 is their own refusal, not a UnicodeDecodeError.


def _save(run_dir: Path, stage_results: Any, stage_configs: Any) -> Any:
    return save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
    )


def test_save_result_bundle_refuses_a_stage_config_that_is_not_utf8(tmp_path: Path, stable_provenance: None) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    (run_dir / "stage1" / "stage_config.json").write_bytes(_NOT_UTF8)
    with pytest.raises(ResultBundleError, match=r"cannot read resolved stage config .*stage1/stage_config\.json: "):
        _save(run_dir, stage_results, stage_configs)


def _complete_bundle_with_undecodable_stage1_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths, _, _ = _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    (run_dir / "stage1" / "stage_config.json").write_bytes(_NOT_UTF8)
    return summary, provenance


def test_evaluation_evidence_refuses_a_stage_config_that_is_not_utf8(tmp_path: Path, stable_provenance: None) -> None:
    run_dir = tmp_path / "run"
    summary, provenance = _complete_bundle_with_undecodable_stage1_config(run_dir)
    with pytest.raises(ResultBundleError, match=r"cannot read publication gate config for stage 1: 'utf-8' codec"):
        validate_evaluation_evidence(run_dir, summary, provenance)


def test_the_audit_records_a_stage_config_that_is_not_utf8(tmp_path: Path, stable_provenance: None) -> None:
    """The audit classifies rather than raises: both its evidence check and its own read of the
    config record the decode failure.  ``verify_hashes=False`` because the rewritten file no
    longer hashes as the manifest says, which would stop the audit before it reads the file."""
    run_dir = tmp_path / "run"
    _complete_bundle_with_undecodable_stage1_config(run_dir)
    report = audit_result_bundle(run_dir, verify_hashes=False)
    assert report["status"] == "canonical-conflict"
    errors = report["errors"]
    assert any(error.startswith("cannot read publication gate config for stage 1: 'utf-8' codec") for error in errors)
    assert any(error.startswith("'utf-8' codec can't decode") for error in errors), errors


# --- Tripwires.


def _non_test_sources() -> Iterator[tuple[str, list[ast.AST]]]:
    """``(name, trees)``: every non-test module under ``environments/`` (one tree) and every
    notebook (one tree per code cell; the cells run in one namespace, so they share imports)."""
    for path in sorted(ENVIRONMENTS.rglob("*.py")):
        if "tests" in path.relative_to(ENVIRONMENTS).parts:
            continue
        yield path.relative_to(REPO_ROOT).as_posix(), [ast.parse(path.read_text(encoding="utf-8"))]
    for notebook in NOTEBOOKS:
        trees: list[ast.AST] = [ast.parse(strip_magics(source)) for _, source in code_cells(notebook)]
        yield notebook.relative_to(REPO_ROOT).as_posix(), trees


def _scopes(tree: ast.AST) -> Iterator[tuple[str, list[ast.AST]]]:
    """``(name, nodes)`` for each function, then ``("<module>", nodes)`` for the code outside every
    function (a script's or a notebook cell's top level)."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes = list(ast.walk(node))
            inside.update(id(inner) for inner in nodes)
            yield node.name, nodes
    yield "<module>", [node for node in ast.walk(tree) if id(node) not in inside]


_LOADS = ("load", "loads")


class _JsonLoads:
    """Recognises a module's ``json.load``/``json.loads`` calls under the names it imports them by:
    ``json`` as each name it is imported as (``import json``, ``import json as _json``) and a bare
    ``load``/``loads`` (``from json import loads [as parse]``).  Imports anywhere in the module
    count, and a notebook's cells share theirs."""

    def __init__(self, trees: list[ast.AST]) -> None:
        self.modules: set[str] = set()
        self.functions: set[str] = set()
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "json":
                            self.modules.add(alias.asname or "json")
                        elif alias.name.startswith("json.") and alias.asname is None:
                            self.modules.add("json")
                elif isinstance(node, ast.ImportFrom) and node.module == "json" and not node.level:
                    self.functions.update(alias.asname or alias.name for alias in node.names if alias.name in _LOADS)

    def __call__(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in self.functions
        return (
            isinstance(func, ast.Attribute)
            and func.attr in _LOADS
            and isinstance(func.value, ast.Name)
            and func.value.id in self.modules
        )


def _names_the_stage_config(node: ast.AST) -> bool:
    """A string constant (an f-string's literal parts and glob patterns included) ending in
    ``stage_config.json``, or the ``STAGE_CONFIG_FILENAME`` constant."""
    return (
        (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("stage_config.json"))
        or (isinstance(node, ast.Name) and node.id == "STAGE_CONFIG_FILENAME")
        or (isinstance(node, ast.Attribute) and node.attr == "STAGE_CONFIG_FILENAME")
    )


def _stage_config_self_parsers(sources: Iterable[tuple[str, list[ast.AST]]]) -> set[str]:
    """``"<file>::<function>"`` for each function (or top level) that both names the record and
    calls ``json.load``/``json.loads``."""
    found = set()
    for name, trees in sources:
        is_load = _JsonLoads(trees)
        for tree in trees:
            for scope, nodes in _scopes(tree):
                if any(_names_the_stage_config(node) for node in nodes) and any(is_load(node) for node in nodes):
                    found.add(f"{name}::{scope}")
    return found


#: The functions that name the record and parse it themselves, each on purpose.
STAGE_CONFIG_SELF_PARSERS = {
    # A read-modify-write: a missing or unreadable record is an error.
    "environments/shared/config.py::record_stage_duration",
    # The recovery-gate species check refuses an unreadable record and ignores a non-object.
    "environments/shared/curriculum/gate_resolver.py::_resolution_species",
    # The bundle writer, audit and evidence reader refuse a bad record with their own errors.
    "environments/shared/reporting/bundles.py::save_result_bundle",
    "environments/shared/result_bundle/audit.py::audit_result_bundle",
    "environments/shared/result_bundle/evidence.py::validate_evaluation_evidence",
    # The stage-results builder raises on an unreadable or non-object record rather than reading
    # it as absent (an absent file or key leaves plant_identity out).
    "environments/shared/reporting/stage_artifacts.py::build_stage_results_from_eval_data",
    # The Drive summary notebook's own readers (CU-15's; kept by the cleanup plan's section 4.9):
    # scan_run and show_full_config raise on a bad record; _infer_algorithm and _stage_name fall
    # back (to the directory name and to "") without the shared reader.
    "notebooks/google_drive_summary.ipynb::_infer_algorithm",
    "notebooks/google_drive_summary.ipynb::scan_run",
    "notebooks/google_drive_summary.ipynb::_stage_name",
    "notebooks/google_drive_summary.ipynb::show_full_config",
}


def test_only_the_listed_functions_parse_the_stage_config_themselves():
    """No other function (nor a module's or a notebook cell's top level) both names
    ``stage_config.json`` (or ``STAGE_CONFIG_FILENAME``) and calls ``json.load``/``json.loads``: a
    lenient library read goes through ``config.read_recorded_stage_config``.

    It sees ``json`` under any name a module imports it by and a bare ``load``/``loads`` imported
    from it, and any string constant that ends in ``stage_config.json`` (an f-string's literal
    part, a glob pattern).  The limits: the rule sees one function at a time, so a helper that
    takes a path and parses it, called from a function that names the file, escapes it (the
    deleted ``ancestors._read_json_mapping`` was one); the next test catches such a helper only
    when it is lenient itself (a ``try`` or ``suppress`` around the parse), so a function that
    calls a strict helper on the record and swallows the helper's error escapes both rules.  A
    file name the function reaches only through another name (a module constant ``_NAME =
    "stage_config.json"``, a concatenation) and a parser other than ``json.load``/``json.loads``
    escape too.
    """
    assert _stage_config_self_parsers(_non_test_sources()) == STAGE_CONFIG_SELF_PARSERS


#: Functions that read some OTHER record leniently with their own ``try``, outside the rule
#: below's purpose (none of them reads ``stage_config.json``); ``audit_result_bundle``, which
#: records every unreadable bundle file, ``stage_config.json`` included (S5 in the list above), as
#: an audit error; and the shared reader itself.
OTHER_LENIENT_JSON_READERS = {
    "environments/shared/file_io.py::read_json_object",
    "environments/shared/result_bundle/audit.py::audit_result_bundle",
    "environments/shared/curriculum/baseline_watch.py::read_zero_action_baseline",
    "environments/shared/harnesses/freeze_recovery_gate.py::_validated_recovery_resolution",
    "environments/shared/result_bundle/provenance.py::initialize_result_bundle",
    "environments/shared/result_bundle/reentry.py::_provenance_training_seed",
    "environments/shared/task_fingerprint.py::read_checkpoint_attribute",
}

#: The exceptions a handler (or ``suppress``) catching a failed read or parse names: ``OSError``
#: under each of its names, the decode errors and the catch-alls.  A narrower ``OSError`` such as
#: ``FileNotFoundError`` is not one: it excuses only a missing file, as an ``is_file`` guard does.
_READ_OR_DECODE_ERRORS = {
    "OSError",
    "IOError",
    "EnvironmentError",
    "ValueError",
    "JSONDecodeError",
    "UnicodeError",
    "UnicodeDecodeError",
    "Exception",
    "BaseException",
}


def _exception_names(nodes: Iterable[ast.expr]) -> set[str]:
    return {node.id if isinstance(node, ast.Name) else getattr(node, "attr", "") for node in nodes}


def _catches(handler: ast.ExceptHandler) -> set[str]:
    if handler.type is None:
        return {"BaseException"}
    return _exception_names(handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])


def _suppresses_a_read_error(expr: ast.expr) -> bool:
    """``contextlib.suppress(...)`` or a bare ``suppress(...)`` naming a read or decode error."""
    if not isinstance(expr, ast.Call):
        return False
    callee = expr.func.attr if isinstance(expr.func, ast.Attribute) else getattr(expr.func, "id", None)
    return callee == "suppress" and bool(_exception_names(expr.args) & _READ_OR_DECODE_ERRORS)


def _carries_on_after_a_failed_load(node: ast.AST, is_load: _JsonLoads) -> bool:
    """A ``try`` around a load whose read-or-decode-error handler does not raise, or a
    ``with suppress(<read or decode error>)`` block around a load."""

    def loads(statements: list[ast.stmt]) -> bool:
        return any(is_load(inner) for statement in statements for inner in ast.walk(statement))

    if isinstance(node, ast.Try):
        return loads(node.body) and any(
            _catches(handler) & _READ_OR_DECODE_ERRORS
            and not any(isinstance(inner, ast.Raise) for statement in handler.body for inner in ast.walk(statement))
            for handler in node.handlers
        )
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return loads(node.body) and any(_suppresses_a_read_error(item.context_expr) for item in node.items)
    return False


def _lenient_json_readers(sources: Iterable[tuple[str, list[ast.AST]]]) -> set[str]:
    """``"<file>::<function>"`` for each library function (or top level) that carries on after a
    failed ``json.load``/``json.loads``; notebooks are skipped."""
    found = set()
    for name, trees in sources:
        if name.endswith(".ipynb"):
            continue
        is_load = _JsonLoads(trees)
        for tree in trees:
            for scope, nodes in _scopes(tree):
                if any(_carries_on_after_a_failed_load(node, is_load) for node in nodes):
                    found.add(f"{name}::{scope}")
    return found


def test_no_new_lenient_json_reader_beside_the_shared_one():
    """No other library function (nor a module's top level) carries on after a failed
    ``json.load``/``json.loads`` -- a ``try`` whose handler for an ``OSError`` (``IOError``,
    ``EnvironmentError``) or a decode error does not raise, or a ``with suppress(...)`` of one --
    with ``json`` under any imported name: that is a lenient-reader copy, and a bad record should
    read the same everywhere.  Catches a re-added ``replication._read_stage_config`` or
    ``ancestors._read_json_mapping`` under any name.

    What it does not see: a function that calls a strict helper on the record (or any helper
    that is not ``json.load``/``json.loads``) inside a ``try`` and swallows the helper's error; an
    ``is_file`` guard, or a handler naming only a narrower ``OSError`` such as
    ``FileNotFoundError``, plus a parse that may raise (lenient only about a missing file); a
    handler that raises; a parser other than ``json.load``/``json.loads``.  Notebooks are left
    out: the Drive summary's readers are CU-15's.
    """
    assert _lenient_json_readers(_non_test_sources()) == OTHER_LENIENT_JSON_READERS


def _record_reads(function: ast.AST, is_load: _JsonLoads) -> list[ast.Call]:
    """Each ``read_text``/``open`` call in *function* that opens the record and whose result
    reaches ``json.load``/``json.loads`` (an argument of the load, the ``as`` name of a ``with`` the
    load reads, or a name assigned the text the load parses).  The record's path is followed
    through the function's assignments and loops from a name of the record."""
    nodes = list(ast.walk(function))
    record_names: set[str] = set()

    def names_the_record(expr: ast.AST) -> bool:
        return any(
            _names_the_stage_config(inner) or (isinstance(inner, ast.Name) and inner.id in record_names)
            for inner in ast.walk(expr)
        )

    grown = True
    while grown:
        grown = False
        for node in nodes:
            pairs: list[tuple[ast.AST, ast.AST]] = []
            if isinstance(node, ast.Assign):
                pairs = [(target, node.value) for target in node.targets]
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and node.value is not None:
                pairs = [(node.target, node.value)]
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                pairs = [(node.target, node.iter)]
            for target, value in pairs:
                if names_the_record(value):
                    for inner in ast.walk(target):
                        if isinstance(inner, ast.Name) and inner.id not in record_names:
                            record_names.add(inner.id)
                            grown = True

    loads = [node for node in nodes if isinstance(node, ast.Call) and is_load(node)]
    loaded = [inner for load in loads for argument in load.args for inner in ast.walk(argument)]
    loaded_ids = {id(inner) for inner in loaded}
    loaded_names = {inner.id for inner in loaded if isinstance(inner, ast.Name)}
    reaching: list[ast.expr] = []
    for node in nodes:
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name) and item.optional_vars.id in loaded_names:
                    reaching.append(item.context_expr)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in loaded_names:
                reaching.append(node.value)
    reaching.extend(node for node in nodes if isinstance(node, ast.Call) and id(node) in loaded_ids)
    reads = []
    for call in reaching:
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "open"):
            path: ast.AST | None = func.value
        elif isinstance(func, ast.Name) and func.id == "open" and call.args:
            path = call.args[0]
        else:
            continue
        if path is not None and names_the_record(path):
            reads.append(call)
    return reads


def _decodes_as_utf8(call: ast.Call) -> bool:
    return any(
        keyword.arg == "encoding" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "utf-8"
        for keyword in call.keywords
    )


def _record_reads_by_function(sources: Iterable[tuple[str, list[ast.AST]]], wanted: set[str]) -> dict[str, list[bool]]:
    """``{"<file>::<function>": [whether each record read passes encoding="utf-8"]}`` for *wanted*."""
    reads: dict[str, list[bool]] = {}
    for name, trees in sources:
        is_load = _JsonLoads(trees)
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and f"{name}::{node.name}" in wanted:
                    reads[f"{name}::{node.name}"] = [_decodes_as_utf8(call) for call in _record_reads(node, is_load)]
    return reads


def test_the_library_self_parsers_decode_the_record_as_utf8():
    """Each library function above that parses the record itself opens it with
    ``encoding="utf-8"`` (the stage-results builder's was added by CU-8c), so it decodes the same
    under any locale.  The suite runs in UTF-8 mode, where a read without it behaves the same, so
    nothing else notices the keyword's loss.  Only reads of the record count (the builder's
    ``metrics.json`` read is not this pin's), and each listed function must have one; the Drive
    summary's readers are CU-15's."""
    library = {entry for entry in STAGE_CONFIG_SELF_PARSERS if entry.startswith("environments/")}
    reads = _record_reads_by_function(_non_test_sources(), library)
    assert {entry: bool(flags) for entry, flags in reads.items()} == dict.fromkeys(library, True)
    assert {entry: flags for entry, flags in reads.items() if not all(flags)} == {}


# --- The tripwires' self-test: what each catches and what it is documented not to see.

_LENIENT_COPY = """
{imports}
def _read(stage_dir):
    try:
        return {load}((stage_dir / "stage_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
"""

#: ``(name, source, caught by the self-parser rule, caught by the lenient rule)``.
TRIPWIRE_FORMS = [
    ("plain json", _LENIENT_COPY.format(imports="import json", load="json.loads"), True, True),
    ("json as an alias", _LENIENT_COPY.format(imports="import json as jsonlib", load="jsonlib.loads"), True, True),
    ("bare loads", _LENIENT_COPY.format(imports="from json import loads", load="loads"), True, True),
    ("bare loads aliased", _LENIENT_COPY.format(imports="from json import loads as parse", load="parse"), True, True),
    (
        "contextlib.suppress in a helper",
        "import contextlib, json\n"
        "def _load_record(path):\n"
        "    with contextlib.suppress(OSError, ValueError):\n"
        "        return json.loads(path.read_text(encoding='utf-8'))\n"
        "    return None\n"
        "def discover(stage_dir):\n"
        "    return _load_record(stage_dir / 'stage_config.json')\n",
        False,
        True,
    ),
    (
        "bare suppress",
        "import json\nfrom contextlib import suppress\n"
        "def _read(path):\n"
        "    with suppress(json.JSONDecodeError):\n"
        "        return json.load(open(path, encoding='utf-8'))\n",
        False,
        True,
    ),
    (
        "f-string path behind is_file",
        "import json\nfrom pathlib import Path\n"
        "def _support(ancestor):\n"
        "    path = Path(f'{ancestor.stage_dir}/stage_config.json')\n"
        "    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None\n",
        True,
        False,
    ),
    (
        "an IOError handler",
        "import json\n"
        "def _read(path):\n"
        "    try:\n"
        "        return json.loads(path.read_text(encoding='utf-8'))\n"
        "    except IOError:\n"
        "        return None\n",
        False,
        True,
    ),
    (
        "a UnicodeDecodeError handler",
        "import json\n"
        "def _read(path):\n"
        "    try:\n"
        "        return json.loads(path.read_text(encoding='utf-8'))\n"
        "    except UnicodeDecodeError:\n"
        "        return None\n",
        False,
        True,
    ),
    (
        "a FileNotFoundError handler only (documented limit)",
        "import json\n"
        "def _read(path):\n"
        "    try:\n"
        "        return json.loads(path.read_text(encoding='utf-8'))\n"
        "    except FileNotFoundError:\n"
        "        return None\n",
        False,
        False,
    ),
    (
        "a handler that raises",
        "import json\n"
        "def _read(stage_dir):\n"
        "    try:\n"
        "        return json.loads((stage_dir / 'stage_config.json').read_text(encoding='utf-8'))\n"
        "    except (OSError, ValueError) as exc:\n"
        "        raise RuntimeError(stage_dir) from exc\n",
        True,
        False,
    ),
    (
        "a strict helper wrapped and swallowed (documented limit)",
        "def discover(stage_dir):\n"
        "    try:\n"
        "        return _read_json_object(stage_dir / 'stage_config.json', what='stage_config.json')\n"
        "    except (OSError, ResultBundleError):\n"
        "        return None\n",
        False,
        False,
    ),
    (
        "suppress of an unrelated error",
        "import json\nfrom contextlib import suppress\n"
        "def _read(path):\n"
        "    with suppress(KeyError):\n"
        "        return json.loads(path.read_text(encoding='utf-8'))['x']\n",
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    ("form", "source", "self_parser", "lenient"), TRIPWIRE_FORMS, ids=[form[0] for form in TRIPWIRE_FORMS]
)
def test_the_reader_tripwires_see_each_form_they_claim(
    form: str, source: str, self_parser: bool, lenient: bool
) -> None:
    sources: list[tuple[str, list[ast.AST]]] = [("environments/shared/probe_module.py", [ast.parse(source)])]
    assert bool(_stage_config_self_parsers(sources)) is self_parser
    assert bool(_lenient_json_readers(sources)) is lenient


@pytest.mark.parametrize(
    ("source", "utf8"),
    [
        ("p = stage_dir / 'stage_config.json'\nreturn json.loads(p.read_text(encoding='utf-8'))", [True]),
        ("p = stage_dir / 'stage_config.json'\nreturn json.loads(p.read_text())", [False]),
        ("p = stage_dir / 'stage_config.json'\nwith open(p) as f:\n    return json.load(f)", [False]),
        ("p = stage_dir / 'stage_config.json'\nwith p.open(encoding='utf-8') as f:\n    return json.load(f)", [True]),
        ("for p in [d / 'stage_config.json' for d in dirs]:\n    text = p.read_text()\n    json.loads(text)", [False]),
        ("p = stage_dir / 'metrics.json'\nreturn json.loads(p.read_text())", []),
    ],
)
def test_the_utf8_pin_follows_the_record_to_its_read(source: str, utf8: list[bool]) -> None:
    function = "def f(stage_dir, dirs):\n" + "".join(f"    {line}\n" for line in source.splitlines())
    sources: list[tuple[str, list[ast.AST]]] = [("environments/m.py", [ast.parse("import json\n" + function)])]
    assert _record_reads_by_function(sources, {"environments/m.py::f"}) == {"environments/m.py::f": utf8}
