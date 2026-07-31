"""Portable canonicalisation and semantic digests.

Encodes payload values so that architecture-dependent numerical noise in the
MuJoCo compiler does not change a digest, and digests Python source tokens so
that a reformat or comment edit does not read as a semantic change."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import io
import json
import math
import textwrap
import token
import tokenize
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import PORTABLE_FLOAT_SIGNIFICANT_DIGITS
from .errors import PlantContractError


def _validate_digest(value: str, *, field: str) -> None:
    prefix = "sha256:"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PlantContractError(f"{field} must be sha256:<64 lowercase hex>, got {value!r}")


def _canonical_float(value: float) -> str:
    """Encode floats portably while preserving meaningful model changes.

    MuJoCo's compiler can differ by a few ULPs across CPU architectures for
    derived inertias, quaternions, and geometry sizes.  Twelve significant
    decimal digits leave ample headroom for that numerical noise while exact
    MJCF and referenced-asset bytes remain protected by the source closure.
    """
    value = float(value)
    if math.isnan(value):
        raise PlantContractError("plant fingerprint cannot encode NaN")
    if math.isinf(value):
        # Gymnasium spaces deliberately use infinite observation bounds.
        # Encode them explicitly instead of relying on non-standard JSON.
        return "+inf" if value > 0 else "-inf"
    if value == 0.0:
        return "0"
    return format(value, f".{PORTABLE_FLOAT_SIGNIFICANT_DIGITS}g")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise PlantContractError(f"unsupported fingerprint value: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    normalized = _canonical_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _semantic_digest(schema: str, payload: Any) -> str:
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json(payload))
    return f"sha256:{digest.hexdigest()}"


def _normalized_python_tokens(source: str) -> list[list[str]]:
    """Return a Python-version-stable semantic token stream.

    Comments, blank lines, and indentation width are ignored. Structural
    INDENT/DEDENT/NEWLINE markers are retained, so block structure remains
    part of the interface. Unlike ``ast.dump``, this representation does not
    change when Python adds fields to its AST nodes.
    """
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    normalized: list[list[str]] = []
    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        raise PlantContractError(f"cannot parse policy-interface implementation: {exc}") from exc
    if any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree)):
        # Python 3.12 changed f-string tokenization (PEP 701).  Rejecting them
        # keeps the manifest portable across supported Python versions until
        # the contract defines a dedicated cross-version representation.
        raise PlantContractError("f-strings are not supported in fingerprinted policy-interface callables")
    docstring_ranges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            value = body[0].value
            docstring_ranges.add(
                ((value.lineno, value.col_offset), (value.end_lineno or value.lineno, value.end_col_offset or 0))
            )

    reader = io.StringIO(dedented).readline
    skip_docstring_newline = False
    try:
        tokens = tokenize.generate_tokens(reader)
        for item in tokens:
            if item.type in ignored:
                continue
            if item.type == tokenize.STRING and (item.start, item.end) in docstring_ranges:
                skip_docstring_newline = True
                continue
            if skip_docstring_newline and item.type == tokenize.NEWLINE:
                skip_docstring_newline = False
                continue
            token_name = token.tok_name[item.type]
            if item.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}:
                normalized.append([token_name, ""])
            elif item.type == tokenize.STRING:
                try:
                    literal = ast.literal_eval(item.string)
                except (SyntaxError, ValueError):
                    literal = item.string
                normalized.append([token_name, repr(literal)])
            else:
                normalized.append([token_name, item.string])
    except (IndentationError, tokenize.TokenError) as exc:
        raise PlantContractError(f"cannot tokenize policy-interface implementation: {exc}") from exc
    return normalized


def _callable_semantics(callable_object: Any) -> dict[str, str]:
    """Fingerprint executable interface code without importing optional backends."""
    try:
        source = inspect.getsource(callable_object)
    except (OSError, TypeError) as exc:
        raise PlantContractError(f"cannot inspect policy-interface callable {callable_object!r}: {exc}") from exc
    return {
        "qualname": str(getattr(callable_object, "__qualname__", type(callable_object).__qualname__)),
        "tokens_sha256": _semantic_digest(
            "mesozoic.python-interface-tokens/v1",
            _normalized_python_tokens(source),
        ),
    }


def _module_function_semantics(module_name: str, function_names: Sequence[str]) -> dict[str, str]:
    """Fingerprint selected functions from an optional-backend-safe module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PlantContractError(f"cannot import policy-interface module {module_name}: {exc}") from exc
    functions = {name: getattr(module, name, None) for name in function_names}
    missing = sorted(name for name, value in functions.items() if not callable(value))
    if missing:
        raise PlantContractError(f"{module_name} is missing policy-interface functions: {missing}")
    return {name: _callable_semantics(functions[name])["tokens_sha256"] for name in function_names}


def _array_digest(schema: str, array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "sha256": _semantic_digest(schema, {"shape": list(array.shape), "values": array}),
    }
