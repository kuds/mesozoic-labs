"""The per-step floor-contact reduction the gait recorder and the T. rex's opt-in CoP band share.

Scratch (the stance4 copband design, never merged).  The functions below
were :class:`environments.shared.gait.recorder.SubstepContactRecorder`'s
own reduction steps: the per-contact normal-force decode, the floor-contact
classification, the per-substep force sums and the sole and whole-foot
centres of pressure.  The recorder reduces a whole episode with them, and
:class:`.step_cop.StepFootCop` reads one control step with the same rules
and arithmetic in one fused pass, so the band's reading is the judge's K2b
reading for that step bit for bit.  They live here, outside ``gait/``, so
that a species env module never imports the gait package (the layering rule
``test_no_species_env_module_imports_the_gait_package`` pins, with no
exception), and this package imports nothing from ``gait/``.

Pure: the standard library and numpy only at import time (the decode
settings read mujoco's cone enum in their body), because two of gait's pure
modules import from here: ``gait/constants.py`` re-exports
:data:`CONTACT_THRESHOLD_N`, which is still a floor-truth measurement
constant (``measurement_constants`` records it, and changing it needs a new
``MEASUREMENT_VERSION``), and ``gait/stance_metrics.py`` normalizes the
whole-foot CoP with :func:`normalized_foot_cop`.

A reward that reads this code depends on measurement code no task
fingerprint hashes, so ``tests/test_anatomical_stance_cop_band.py`` pins a
short band-on rollout's per-step readings (the golden), and its byte-for-byte
comparisons of the band's reader with the recorder's trace and with these
functions (on synthetic contacts that straddle their thresholds) pin the two
paths together.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .morphology import Morphology

#: A contact, a limb or a sole bears load above this normal force, in N (``gait/constants.py`` documents the value).
CONTACT_THRESHOLD_N = 0.1
#: A centre of pressure is read only where its contacts' normal forces sum to more than this, in N (else NaN).
COP_LOAD_N = 1e-9


def decode_settings(model: Any) -> tuple[bool, np.ndarray | None]:
    """``(elliptic, edge_offsets)``: the normal-force decode for ``model``.

    ``edge_offsets`` is the one gather every contact shares when every geom
    has one condim and the model declares no explicit ``<pair>``, else
    ``None`` (the per-contact decode).  Shared by
    :class:`~environments.shared.gait.recorder.SubstepContactRecorder` and
    :class:`.step_cop.StepFootCop`.
    """
    import mujoco  # in the body: this module is pure at import time (module docstring)

    elliptic = int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    condims = {int(value) for value in model.geom_condim}
    edge_offsets = None
    if len(condims) == 1 and int(model.npair) == 0:
        condim = condims.pop()
        edges = 1 if (elliptic or condim == 1) else 2 * (condim - 1)
        edge_offsets = np.arange(edges, dtype=np.int64)[None, :]
    return elliptic, edge_offsets


def decode_normal_forces(
    contact: Any, efc_force: np.ndarray, *, elliptic: bool, edge_offsets: np.ndarray | None
) -> np.ndarray:
    """Each contact's normal force, exactly as ``mj_contactForce`` returns ``force[0]`` (``gait/recorder.py``)."""
    address = np.asarray(contact.efc_address)
    valid = address >= 0
    if efc_force.size == 0:
        return np.zeros(address.shape[0], dtype=np.float64)
    base = np.where(valid, address, 0)[:, None]
    last = efc_force.size - 1
    if edge_offsets is not None:
        gathered = efc_force[np.minimum(base + edge_offsets, last)]
        return np.where(valid, gathered.sum(axis=1), 0.0)
    dim = np.asarray(contact.dim)
    edges = np.ones_like(dim) if elliptic else np.where(dim > 1, 2 * (dim - 1), 1)
    offsets = np.arange(int(edges.max()), dtype=np.int64)[None, :]
    gathered = efc_force[np.minimum(base + offsets, last)]
    keep = (offsets < edges[:, None]) & valid[:, None]
    normal: np.ndarray = np.where(keep, gathered, 0.0).sum(axis=1)
    return normal


def classify_floor_contacts(
    morphology: Morphology, geoms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(other, on_floor, foot, leg_on_floor)`` for contact geom pairs ``geoms`` (N, 2).

    ``other`` is the non-floor side of a floor contact; ``on_floor`` marks a
    floor-against-animal contact; ``foot`` is its leg index (-1 off the legs);
    ``leg_on_floor`` a floor contact of some leg.
    """
    geom1, geom2 = geoms[:, 0], geoms[:, 1]
    floor1, floor2 = morphology.is_floor[geom1], morphology.is_floor[geom2]
    other = np.where(floor1, geom2, geom1)  # the non-floor side of a floor contact
    on_floor = (floor1 ^ floor2) & morphology.is_animal[other]
    foot = np.where(on_floor, morphology.geom_foot[other], -1)
    leg_on_floor = on_floor & (foot >= 0)
    return other, on_floor, foot, leg_on_floor


def per_substep_force(
    substep_of: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray,
    index: np.ndarray,
    width: int,
    steps: int,
    substeps: int,
) -> np.ndarray:
    """The normal force of the ``mask`` contacts summed per (step, substep, ``index``): (steps, substeps, width)."""
    flat = np.bincount(substep_of[mask] * width + index[mask], weights=normal[mask], minlength=steps * substeps * width)
    return flat.reshape(steps, substeps, width)


def sole_cop_steps(
    feet: Sequence[Any],
    half_size: Sequence[np.ndarray | None],
    pose: np.ndarray | None,
    substep_of: np.ndarray,
    foot: np.ndarray,
    on_sole: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    steps: int,
    substeps: int,
) -> np.ndarray:
    """Per step and foot, the sole CoP ``(fore, outer)`` in sole half-lengths (``EpisodeTrace.sole_cop``).

    Every sole contact in the sole frame of its own substep (``pose``: per
    substep and foot, the sole geom's xpos then xmat), weighted by its normal
    force; NaN on a step whose sole contacts carry no force.
    """
    n_feet = len(feet)
    cop = np.full((steps, n_feet, 2), np.nan)
    if on_sole.any():
        assert pose is not None
        for index in np.unique(foot[on_sole]).tolist():
            spec = feet[index]
            half = half_size[index]
            assert half is not None  # a foot with sole contacts has a sole
            chosen = np.flatnonzero(on_sole & (foot == index))
            at = substep_of[chosen]
            rotation = pose[at, index, 3:].reshape(-1, 3, 3)
            local = np.einsum("nji,nj->ni", rotation, points[chosen] - pose[at, index, :3])
            fore = spec.sole_forward_sign * local[:, spec.sole_forward_axis] / half[spec.sole_forward_axis]
            left = spec.sole_lateral_sign * local[:, spec.sole_lateral_axis] / half[spec.sole_lateral_axis]
            outer = -left if spec.side == "R" else left
            weight = normal[chosen]
            step_of = at // substeps
            weight_sum = np.bincount(step_of, weights=weight, minlength=steps)
            has_load = weight_sum > COP_LOAD_N
            for column, coordinate in enumerate((fore, outer)):
                weighted = np.bincount(step_of, weights=weight * coordinate, minlength=steps)
                cop[has_load, index, column] = weighted[has_load] / weight_sum[has_load]
    return cop


def whole_foot_cop_steps(
    feet: Sequence[Any],
    ankle_forward: np.ndarray,
    pose: np.ndarray | None,
    substep_of: np.ndarray,
    foot: np.ndarray,
    leg_on_floor: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    steps: int,
    substeps: int,
) -> np.ndarray:
    """Per step and foot, the whole-foot CoP in m ahead of the ankle axis (``EpisodeTrace.foot_cop``).

    Every floor contact of the limb, in the sole frame of its own substep,
    along the sole's signed forward axis less the ankle's coordinate there,
    weighted by its normal force; NaN off the feet that declare an ankle and
    on a step whose limb contacts carry no force.
    """
    foot_cop = np.full((steps, len(feet)), np.nan)
    if leg_on_floor.any():
        assert pose is not None
        for index in np.unique(foot[leg_on_floor]).tolist():
            if not np.isfinite(ankle_forward[index]):
                continue
            spec = feet[index]
            chosen = np.flatnonzero(leg_on_floor & (foot == index))
            at = substep_of[chosen]
            rotation = pose[at, index, 3:].reshape(-1, 3, 3)
            local = np.einsum("nji,nj->ni", rotation, points[chosen] - pose[at, index, :3])
            ahead = spec.sole_forward_sign * local[:, spec.sole_forward_axis] - ankle_forward[index]
            weight = normal[chosen]
            step_of = at // substeps
            weight_sum = np.bincount(step_of, weights=weight, minlength=steps)
            has_load = weight_sum > COP_LOAD_N
            weighted = np.bincount(step_of, weights=weight * ahead, minlength=steps)
            foot_cop[has_load, index] = weighted[has_load] / weight_sum[has_load]
    return foot_cop


def support_span_scale(support_span: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(spanned, middle, half)`` per foot of ``support_span`` (F,2): the span :func:`normalized_foot_cop` reads in."""
    span = np.asarray(support_span, dtype=np.float64).reshape(-1, 2)
    spanned = np.asarray(np.isfinite(span).all(axis=1), dtype=bool)
    middle = np.where(spanned, span.mean(axis=1), np.nan)
    half = np.where(spanned, (span[:, 1] - span[:, 0]) / 2.0, np.nan)
    return spanned, middle, half


def normalized_foot_cop(
    foot_cop: np.ndarray,
    leg_floor_max: np.ndarray,
    support_span: np.ndarray,
    *,
    scale: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``whole_foot_cop`` on bare arrays: ``foot_cop`` and ``leg_floor_max`` (T,F), ``support_span`` (F,2).

    The one normalization and loaded rule: the gait package's
    :func:`~environments.shared.gait.stance_metrics.whole_foot_cop` reads an
    episode through it, and the T. rex's opt-in CoP band
    (:class:`.step_cop.StepFootCop`) a single step, with ``scale``, its
    span's :func:`support_span_scale` computed once.
    """
    spanned, middle, half = support_span_scale(support_span) if scale is None else scale
    cop = (np.asarray(foot_cop, dtype=np.float64) - middle) / half
    loaded = np.isfinite(cop) & (np.asarray(leg_floor_max) > CONTACT_THRESHOLD_N)
    return spanned, cop, loaded
