"""Tests for environments.shared.plant_contract.digests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from environments.shared.plant_contract import (
    PlantContractError,
)
from environments.shared.plant_contract.digests import (
    _canonical_float,
    _normalized_python_tokens,
    _semantic_digest,
)


@pytest.mark.parametrize("value", [0.1, -0.1, 1.234567890123, 1e-9, 1e9, -math.pi])
def test_portable_float_canonicalization_collapses_compiler_ulp_noise(value):
    expected = _canonical_float(value)

    for direction in (-math.inf, math.inf):
        perturbed = value
        for _ in range(4):
            perturbed = math.nextafter(perturbed, direction)
        assert _canonical_float(perturbed) == expected


def test_portable_float_canonicalization_preserves_meaningful_changes():
    assert _canonical_float(1.0) != _canonical_float(1.000000001)

    base = np.array([0.1, -math.pi, 1e-9, 1e9], dtype=np.float64)
    perturbed = base.copy()
    directions = np.where(perturbed < 0.0, -np.inf, np.inf)
    for _ in range(4):
        perturbed = np.nextafter(perturbed, directions)

    assert _semantic_digest("portable-array-test/v1", base) == _semantic_digest("portable-array-test/v1", perturbed)

    changed = base.copy()
    changed[0] *= 1.0 + 1e-9
    assert _semantic_digest("portable-array-test/v1", base) != _semantic_digest("portable-array-test/v1", changed)


def test_portable_float_canonicalization_handles_special_values():
    assert _canonical_float(0.0) == _canonical_float(-0.0) == "0"
    assert _canonical_float(math.inf) == "+inf"
    assert _canonical_float(-math.inf) == "-inf"
    with pytest.raises(PlantContractError, match="cannot encode NaN"):
        _canonical_float(math.nan)


def test_interface_code_tokens_ignore_formatting_but_detect_semantic_changes():
    original = """\
def normalize(distance):
    # Formatting and comments are not interface semantics.
    return distance / (distance + 1e-8)
"""
    reformatted = """\
def normalize(distance):
    return distance / (distance + 1e-8)  # same calculation
"""
    changed = original.replace("1e-8", "1e-6")

    assert _normalized_python_tokens(original) == _normalized_python_tokens(reformatted)
    assert _normalized_python_tokens(original) != _normalized_python_tokens(changed)

    documented = '''\
def label(value):
    """Documentation does not execute."""
    return {"kind": "distance"}["kind"], value
'''
    requoted = """\
def label(value):
    '''Different documentation is also ignored.'''
    return {'kind': 'distance'}['kind'], value
"""
    assert _normalized_python_tokens(documented) == _normalized_python_tokens(requoted)


def test_interface_code_tokens_reject_cross_version_f_strings():
    with pytest.raises(PlantContractError, match="f-strings are not supported"):
        _normalized_python_tokens('def label(value):\n    return f"value={value}"\n')
