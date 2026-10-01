"""The one sha256 digest pattern and the one set of field validators (cleanup CU-8c).

``environments/shared/record_fields.py`` holds ``SHA256_DIGEST_PATTERN``,
``is_sha256_digest`` and the six validators ``result_schema`` and
``species_catalog`` used to define twice.  Both modules bind their six private
validator names (``_require_*`` and ``_optional_*``) to them with
``functools.partial`` and their own error class; the result-bundle readers ask
``is_sha256_digest`` and keep their own messages, and the three whose checks
no other test drives through the table (the ancestor record, the audit's load
lineage and the evidence rows) are held to it here.
``plant_contract/digests.py`` keeps its character-loop check (a hasher file
the cleanup plan holds at a zero-line diff); it is held to the same
accept/reject table here.
"""

from __future__ import annotations

import ast
import csv
import functools
import math
import random
import re
import string
from pathlib import Path
from typing import Any, Iterator

import pytest

from environments.shared import record_fields, result_schema, species_catalog
from environments.shared.plant_contract import PlantContractError, digests
from environments.shared.result_bundle import ResultBundleError, audit, evidence
from environments.shared.result_bundle import ancestors as bundle_ancestors
from environments.shared.result_schema import ResultSchemaError
from environments.shared.species_catalog import CatalogError

from .notebook_cells import code_cells, strip_magics

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = REPO_ROOT / "environments"
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))

_LOWER_HEX = frozenset(string.hexdigits[:16])
_DIGEST = "sha256:" + "a" * 64


class _Text(str):
    """A ``str`` subclass: still a string."""


def _oracle(value: object) -> bool:
    """The rule written out without a regex: ``sha256:`` then exactly 64 characters of ``0-9a-f``."""
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71 and set(value[7:]) <= _LOWER_HEX


#: Hand-picked edges: case, other digit sets, lengths, prefixes, surrounding whitespace and junk.
EDGE_CASES: list[tuple[object, bool]] = [
    (_DIGEST, True),
    ("sha256:" + string.hexdigits[:16] * 4, True),
    (_Text("sha256:" + "b" * 64), True),
    ("sha256:" + "A" * 64, False),
    ("sha256:" + "a" * 63, False),
    ("sha256:" + "a" * 65, False),
    (_DIGEST + "\n", False),
    (_DIGEST + " ", False),
    (_DIGEST + "x", False),
    (" " + _DIGEST, False),
    ("SHA256:" + "a" * 64, False),
    ("sha1:" + "a" * 64, False),
    ("sha256:" + "١" * 64, False),  # Arabic-Indic digits: \d would accept them
    ("sha256:" + "ａ" * 64, False),  # full-width a
    ("sha256:" + "g" * 64, False),
    ("sha256:", False),
    ("", False),
    ("a" * 64, False),
    (None, False),
    (5, False),
    (5.0, False),
    (b"sha256:" + b"a" * 64, False),
    (["x"], False),
    ({"sha256": "a" * 64}, False),
]


def _fuzz_cases() -> Iterator[str]:
    """The CU-8c fuzz generator (20,000 strings, seed 0)."""
    rnd = random.Random(0)
    lower = string.hexdigits[:16]
    for _ in range(20000):
        alphabet = rnd.choice([lower, lower + "ABCDEF", lower + "٠", string.printable])
        yield (
            rnd.choice(["sha256:", "sha256", "sha1:", ""])
            + "".join(rnd.choice(alphabet) for _ in range(rnd.choice([63, 64, 65])))
            + rnd.choice(["", "", "\n", " "])
        )


def _accepts(check: Any, value: object) -> bool:
    try:
        check(value)
    except (ResultSchemaError, CatalogError, PlantContractError):
        return False
    return True


@pytest.mark.parametrize(("value", "accepted"), EDGE_CASES, ids=[repr(value)[:40] for value, _ in EDGE_CASES])
def test_is_sha256_digest_accepts_exactly_a_whole_lowercase_digest(value: object, accepted: bool) -> None:
    assert _oracle(value) is accepted
    assert record_fields.is_sha256_digest(value) is accepted
    assert _accepts(lambda v: result_schema._require_sha256(v, field="f"), value) is accepted
    assert _accepts(lambda v: species_catalog._require_sha256(v, field="f"), value) is accepted
    if isinstance(value, str):
        # The hasher's own check is reached only with strings (identity.py coerces every digest).
        assert _accepts(lambda v: digests._validate_digest(v, field="f"), value) is accepted


def test_every_digest_check_agrees_with_the_rule_on_the_fuzz_table():
    cases = list(_fuzz_cases())
    accepted = [case for case in cases if _oracle(case)]
    assert 0 < len(accepted) < len(cases)
    checks = {
        "is_sha256_digest": record_fields.is_sha256_digest,
        "result_schema._require_sha256": lambda v: _accepts(lambda x: result_schema._require_sha256(x, field="f"), v),
        "species_catalog._require_sha256": lambda v: _accepts(
            lambda x: species_catalog._require_sha256(x, field="f"), v
        ),
        "plant_contract.digests._validate_digest": lambda v: _accepts(
            lambda x: digests._validate_digest(x, field="f"), v
        ),
    }
    disagreements = {
        name: [case for case in cases if check(case) != _oracle(case)][:5] for name, check in checks.items()
    }
    assert disagreements == dict.fromkeys(checks, [])


# --- The bundle readers' own digest checks, each now ``is_sha256_digest`` (gate_verdict's is
# held to the rule by test_gate_verdict's digest-key refusals).


def _ancestor_record_accepts(value: object) -> bool:
    try:
        bundle_ancestors._digest_or_none(value, field="f", path=Path("p"))
    except ResultBundleError as exc:
        assert str(exc) == "p: f must be sha256:<64 lowercase hex> or null"
        return False
    return True


def _load_lineage_accepts(value: object) -> bool:
    verdicts = set()
    for key in ("parent_task_sha256", "parent_checkpoint_sha256"):
        _, problems = audit._audit_load_lineage({key: value}, stage=1, run_path=Path("run"), declared_hashes={})
        assert problems in ([], [f"stage 1 config run.{key} must be sha256:<64 lowercase hex>"]), problems
        verdicts.add(not problems)
    assert len(verdicts) == 1, value
    return verdicts.pop()


@pytest.mark.parametrize(
    ("site", "check", "null_accepted"),
    [
        ("result_bundle.ancestors._digest_or_none", _ancestor_record_accepts, True),
        ("result_bundle.audit._audit_load_lineage", _load_lineage_accepts, False),
    ],
)
def test_the_bundle_readers_digest_checks_accept_exactly_a_digest(site: str, check: Any, null_accepted: bool) -> None:
    """The ancestor record's digest-or-null check and the audit's check of a stage's recorded
    ``parent_task_sha256``/``parent_checkpoint_sha256`` accept exactly what ``is_sha256_digest``
    does on every edge case (the ancestor record also takes null, its "or null")."""
    disagreements = [
        value for value, accepted in EDGE_CASES if check(value) is not (accepted or (null_accepted and value is None))
    ]
    assert disagreements == []


def test_the_evidence_rows_digest_check_accepts_exactly_a_digest(tmp_path: Path) -> None:
    """``_evaluation_evidence_aggregates`` checks each row's ``checkpoint_sha256`` and
    ``normalization_sha256`` cell: it accepts exactly what ``is_sha256_digest`` accepts once the
    cell is stripped, and reads a blank cell as no digest.  CSV cells are strings, so only the
    string edge cases apply."""
    disagreements = []
    for value, _ in EDGE_CASES:
        if not isinstance(value, str):
            continue
        path = tmp_path / "evaluation_selected.csv"
        row = {
            "episode": "1",
            "evaluation_seed": "101",
            "checkpoint": "selected",
            "reward": "1.0",
            "length": "10",
            "mean_forward_velocity": "0.5",
            "distance_traveled": "2.0",
            "task_success": "true",
            "checkpoint_sha256": value,
            "normalization_sha256": value,
        }
        with path.open("w", newline="", encoding="utf-8") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        try:
            aggregates = evidence._evaluation_evidence_aggregates(
                path, checkpoint_label="selected", expected_episodes=1, evaluation_seeds=[101], stage=1
            )
        except ResultBundleError as exc:
            assert "selected evaluation evidence for stage 1 records a malformed checkpoint_sha256" in str(exc)
            accepted = False
        else:
            recorded = value.strip() or None
            assert aggregates["checkpoint_sha256"] == aggregates["normalization_sha256"] == recorded
            accepted = True
        if accepted is not (value.strip() == "" or record_fields.is_sha256_digest(value.strip())):
            disagreements.append(value)
    assert disagreements == []


# --- The two bindings.

VALIDATORS = (
    "require_mapping",
    "require_nonempty_string",
    "optional_nonempty_string",
    "require_positive_int",
    "optional_number",
    "require_sha256",
)
BINDINGS = [(result_schema, ResultSchemaError), (species_catalog, CatalogError)]


@pytest.mark.parametrize(("module", "error"), BINDINGS, ids=["result_schema", "species_catalog"])
def test_each_private_validator_is_the_shared_one_bound_to_the_modules_error(module: Any, error: type) -> None:
    for name in VALIDATORS:
        bound = getattr(module, f"_{name}")
        assert isinstance(bound, functools.partial), name
        assert bound.func is getattr(record_fields, name), name
        assert bound.args == () and bound.keywords == {"error": error}, name


#: ``(validator, bad value, message)``: every rule's one message, with ``field="f"``.
REFUSALS = [
    ("require_mapping", [], "f must be an object"),
    ("require_nonempty_string", "  ", "f must be a non-empty string"),
    ("require_nonempty_string", 3, "f must be a non-empty string"),
    ("optional_nonempty_string", "", "f must be a non-empty string"),
    ("require_positive_int", 0, "f must be a positive integer"),
    ("require_positive_int", True, "f must be a positive integer"),
    ("require_positive_int", 1.0, "f must be a positive integer"),
    ("optional_number", "1", "f must be a number or null"),
    ("optional_number", False, "f must be a number or null"),
    ("optional_number", math.nan, "f must be finite or null"),
    ("optional_number", -math.inf, "f must be finite or null"),
    ("require_sha256", "sha256:" + "A" * 64, "f must be sha256:<64 lowercase hex>"),
    ("require_sha256", None, "f must be sha256:<64 lowercase hex>"),
]


@pytest.mark.parametrize(("module", "error"), BINDINGS, ids=["result_schema", "species_catalog"])
@pytest.mark.parametrize(("name", "value", "message"), REFUSALS, ids=[f"{n}-{v!r}" for n, v, _ in REFUSALS])
def test_each_binding_refuses_with_its_own_error_class_and_the_one_message(
    module: Any, error: type, name: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError) as caught:
        getattr(module, f"_{name}")(value, field="f")
    assert type(caught.value) is error
    assert str(caught.value) == message


@pytest.mark.parametrize(("module", "error"), BINDINGS, ids=["result_schema", "species_catalog"])
def test_each_binding_returns_what_it_accepts(module: Any, error: type) -> None:
    mapping: dict[str, Any] = {}
    assert module._require_mapping(mapping, field="f") is mapping
    assert module._require_nonempty_string(" x ", field="f") == "x"
    assert module._optional_nonempty_string(None, field="f") is None
    assert module._optional_nonempty_string(" x", field="f") == "x"
    assert module._require_positive_int(3, field="f") == 3
    assert module._optional_number(None, field="f") is None
    assert module._optional_number(1.5, field="f") == 1.5
    assert module._optional_number(2, field="f") == 2
    assert module._require_sha256(_DIGEST, field="f") == _DIGEST


def test_the_pattern_is_the_documented_one():
    assert record_fields.SHA256_DIGEST_PATTERN.pattern == "sha256:[0-9a-f]{64}"
    assert record_fields.SHA256_DIGEST_PATTERN.flags == re.UNICODE


# --- Tripwires: one copy of the pattern.


def _string_constants() -> Iterator[tuple[str, str]]:
    """``(where, text)`` for every string constant in non-test code and every notebook code cell."""
    sources: list[tuple[str, str]] = []
    for path in sorted(ENVIRONMENTS.rglob("*.py")):
        if "tests" not in path.relative_to(ENVIRONMENTS).parts:
            sources.append((path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8")))
    for notebook in NOTEBOOKS:
        for index, source in code_cells(notebook):
            sources.append((f"{notebook.relative_to(REPO_ROOT).as_posix()} cell {index}", strip_magics(source)))
    for where, source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield where, node.value


def test_the_64_hex_pattern_is_written_only_in_record_fields():
    """Every string constant holding a 64-length regex quantifier (any text containing ``{64``:
    ``{64}``, ``{64,64}``) is ``SHA256_DIGEST_PATTERN``'s: a second regex copy fails here,
    however its character class is spelled (``[\\da-f]``, ``[0-9a-fA-F]``, a listed alphabet).
    The limit: a digest check written without a regex quantifier (a length test plus a character
    set) is not caught; the next test catches one only when it spells the hex alphabet."""
    found = {where for where, text in _string_constants() if "{64" in text}
    assert found == {"environments/shared/record_fields.py"}


def test_the_hex_alphabet_literal_is_only_the_hasher_check():
    """A character-loop copy of the rule names the alphabet; only the §4.2 hasher file's may."""
    found = {where for where, text in _string_constants() if string.hexdigits[:16] in text}
    assert found == {"environments/shared/plant_contract/digests.py"}
