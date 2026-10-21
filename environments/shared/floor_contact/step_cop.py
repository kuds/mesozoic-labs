"""One control step's per-foot centre of pressure, for the T. rex's opt-in CoP band (screen-only, never merged).

The stance judge's K2b statistic is each foot's whole-foot centre of
pressure (``<label>_foot_cop_fore_aft``): every floor contact of the limb, on
every physics substep, in the sole frame of its own substep, force-weighted,
in span half-lengths (``gait/pad_telemetry.py``).  Only
:class:`environments.shared.gait.recorder.SubstepContactRecorder` computes
it, per episode, after the fact.  :class:`StepFootCop` computes the same
reading for ONE control step, inside the env, so a reward can price it.  It
runs the recorder's arithmetic in one fused pass per step (the recorder's
reduction functions, run per step and per foot, cost +0.62 ms a step in
small numpy calls; the stance4 copband design's throughput item):

* the hot path copies the rows the recorder's hook copies (the contact geom
  pairs and points, and the geoms' poses), once per substep, from the env's
  ``_accumulate_substep`` slot (``BaseDinoEnv.step`` calls it right after
  the probe hook the recorder owns, so both read the same ``data``).  The
  normal forces are :func:`.reduction.decode_normal_forces`'s: when every
  contact shares one gather (:func:`.reduction.decode_settings`) its gather
  runs per substep and its mask once per step, else the function itself
  runs per substep, as the recorder runs it;
* :meth:`StepFootCop.reading` reduces the step for both feet at once.  Each
  contact's foot, sole and support column come from tables that the
  recorder's classification (:func:`.reduction.classify_floor_contacts`)
  filled for every geom pair when the reader was built.  Each contact's
  coordinate in the sole frame of its own substep is the recorder's
  expression (the same ``einsum``, the same operations in the same order),
  and ``np.bincount`` sums each foot's contacts in the recorder's order
  (substep, then contact), so every sum is the recorder's sequential sum.
  The load rule (:data:`.reduction.COP_LOAD_N`), the contact threshold and
  the normalization and loaded rule (:func:`.reduction.normalized_foot_cop`)
  are the shared ones.  So each foot's whole-foot CoP and pad CoP equal the
  recorder's trace for that step byte for byte
  (``tests/test_anatomical_stance_cop_band.py`` pins it on recorded
  rollouts, and against the shared reduction on synthetic contacts that
  straddle its thresholds).

It reads ``model`` and ``data`` and never writes them: a trajectory with it
is bit-identical to one without.

Layering: this module and everything it imports sit in
:mod:`environments.shared.floor_contact`, outside the gait package, so the
T. rex env imports it without breaking the rule that no species env module
imports ``gait``; the env builds the reader's morphology from its own
``FOOT_SUPPORT``, not through ``species_registry``, so it loads no ``gait``
module at run time either.  The env imports it only inside the method that
builds the band's reader, and only when the band is on, so an env with the
band off never loads it.  A reward that reads this code depends on
measurement code that no task fingerprint hashes: a later change to it, or
to the shared reduction, would change the reward, so the band's tests pin a
band-on rollout's readings (the golden).  See the stance4 copband design
note (screen-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .morphology import Morphology
from .reduction import (
    CONTACT_THRESHOLD_N,
    COP_LOAD_N,
    classify_floor_contacts,
    decode_normal_forces,
    decode_settings,
    normalized_foot_cop,
    support_span_scale,
)


@dataclass(frozen=True)
class StepCopReading:
    """One control step, per foot in ``Morphology.feet`` order (all arrays (F,)).

    ``foot_cop`` is the whole-foot CoP in span half-lengths (-1 heel line, +1
    toe line), NaN when the limb carries no force; ``foot_loaded`` is the
    gate's loaded rule (a finite ``foot_cop`` and more than
    ``CONTACT_THRESHOLD_N`` on some substep).  ``pad_cop`` is the sole
    (pad) CoP along its forward axis in pad half-lengths
    (``EpisodeTrace.sole_cop[..., 0]``), NaN when the sole carries no force;
    ``pad_loaded`` the pad rail's rule (a finite ``pad_cop`` and a sole
    contact above ``CONTACT_THRESHOLD_N`` on some substep).
    ``leg_force_mean`` / ``leg_force_max`` are the limb's floor force over
    the step's substeps (``EpisodeTrace.leg_floor_mean`` / ``leg_floor_max``),
    ``digit_share`` the step's digit floor force over the limb's (NaN when the
    limb carries nothing), and ``substeps`` how many substeps were reduced
    (``frame_skip`` in a step, 1 for a state read outside one).
    """

    foot_cop: np.ndarray
    foot_loaded: np.ndarray
    pad_cop: np.ndarray
    pad_loaded: np.ndarray
    leg_force_mean: np.ndarray
    leg_force_max: np.ndarray
    digit_share: np.ndarray
    substeps: int


class StepFootCop:
    """Per-substep copies of the floor contacts of one control step, reduced as the recorder reduces them.

    Usage (the T. rex env): ``record(model, data, step, substep)`` after
    every physics substep, with ``step`` the control step the substep belongs
    to (the env's ``_step_count + 1`` during ``step()``); ``reading(step,
    model, data)`` in the reward.  A reading asked for any other step (a
    state scored outside ``step()``: after ``reset``, or a hand-posed state)
    reduces the current ``data`` as one substep.
    """

    def __init__(self, env: Any, morphology: Morphology) -> None:
        unwrapped = getattr(env, "unwrapped", env)
        self._morphology = morphology
        feet = morphology.feet
        n_feet = morphology.n_feet
        n_support = len(morphology.support_geom_ids)
        model = unwrapped.model
        self._n_feet = n_feet
        self._n_support = n_support
        # Every geom pair, classified once by the recorder's rule: a leg's floor contact maps to its foot, a sole's
        # to its foot again in the sole table, a support geom's to its column; anything else to the last bin of
        # each table (n_feet, n_support), which the reduction drops.
        n_geom = len(morphology.is_floor)
        pairs = np.stack(np.divmod(np.arange(n_geom * n_geom, dtype=np.int64), n_geom), axis=1)
        other, _on_floor, foot, leg_on_floor = classify_floor_contacts(morphology, pairs)
        sole_of_foot = np.array([spec.sole_geom for spec in feet] + [-2])  # as the recorder: foot -1 -> -2
        on_sole = leg_on_floor & (other == sole_of_foot[foot])
        support_index = np.where(leg_on_floor, morphology.geom_support[other], -1)
        self._n_geom = n_geom
        self._pair_foot = np.where(leg_on_floor, foot, n_feet)
        self._pair_sole = np.where(on_sole, foot, n_feet)
        self._pair_support = np.where(support_index >= 0, support_index, n_support)
        # Per foot, then the dropped bin: the geom whose substep pose frames its contacts (as the recorder, a foot
        # without a sole samples geom 0), its signed forward axis, its ankle and its pad's half length on that axis.
        self._frame_geom = np.array([max(spec.sole_geom, 0) for spec in feet] + [0], dtype=np.int64)
        self._forward_axis = np.array([spec.sole_forward_axis for spec in feet] + [0], dtype=np.int64)
        self._forward_sign = np.array([spec.sole_forward_sign for spec in feet] + [1.0], dtype=np.float64)
        forward = {(int(spec.sole_forward_axis), float(spec.sole_forward_sign)) for spec in feet}
        self._forward = forward.pop() if len(forward) == 1 else None
        self._ankle_forward = np.array([spec.ankle_forward_m for spec in feet] + [0.0], dtype=np.float64)
        self._has_ankle = np.isfinite(self._ankle_forward[:n_feet])
        self._half_forward = np.array(
            [
                np.asarray(model.geom_size[spec.sole_geom], dtype=np.float64)[spec.sole_forward_axis]
                if spec.sole_geom >= 0
                else 1.0
                for spec in feet
            ]
            + [1.0]
        )
        spans = [[spec.heel_line_m, spec.toe_line_m] for spec in feet]
        self._support_span = np.array(spans, dtype=np.float64).reshape(-1, 2)
        self._span_scale = support_span_scale(self._support_span)
        # Each foot's digits: its registered support geoms other than its sole (pad_telemetry's digit share).
        support_foot = np.asarray(morphology.support_geom_foot, dtype=np.int64)
        support_ids = np.asarray(morphology.support_geom_ids, dtype=np.int64)
        digit_mask = np.zeros((n_feet, len(support_ids)), dtype=bool)
        for index, spec in enumerate(feet):
            digit_mask[index] = (support_foot == index) & (support_ids != spec.sole_geom)
        self._digit_weights = digit_mask.astype(np.float64)
        # Every geom's pose at each substep of the step (one copy each, cheaper than gathering the soles).
        substeps = max(int(getattr(unwrapped, "frame_skip", 1)), 1)
        self._xpos = np.zeros((substeps, n_geom, 3))
        self._xmat = np.zeros((substeps, n_geom, 9))
        self._substep_index = np.arange(substeps)
        self._decoded_model: Any = None
        self._elliptic = False
        self._edge_offsets: np.ndarray | None = None
        self._row_offsets: np.ndarray | None = None
        self._step = -1
        self._reset_rows()

    @property
    def morphology(self) -> Morphology:
        return self._morphology

    def _reset_rows(self) -> None:
        self._geoms: list[np.ndarray] = []
        self._points: list[np.ndarray] = []
        self._normal: list[np.ndarray] = []
        self._address: list[np.ndarray] = []
        self._efc_rows: list[np.ndarray] = []
        self._counts: list[int] = []

    def _copy(self, model: Any, data: Any) -> None:
        """One substep's rows: what ``SubstepContactRecorder._hook`` copies, and the normal-force decode's gather."""
        if model is not self._decoded_model:
            self._decoded_model = model
            self._elliptic, self._edge_offsets = decode_settings(model)
            # One constraint row per contact (an elliptic cone, condim 1): no offsets to add, no rows to sum.
            single = self._edge_offsets is not None and self._edge_offsets.size == 1
            self._row_offsets = None if single else self._edge_offsets
        substep = len(self._counts)
        if substep == len(self._xpos):
            raise ValueError(f"substep {substep} exceeds the reader's frame_skip ({len(self._xpos)})")
        ncon = int(data.ncon)
        if ncon:
            contact = data.contact
            self._geoms.append(contact.geom.copy())
            self._points.append(contact.pos.copy())
            if self._edge_offsets is None:  # the per-contact decode: substep by substep, as the recorder runs it
                self._normal.append(
                    decode_normal_forces(contact, data.efc_force, elliptic=self._elliptic, edge_offsets=None)
                )
            else:  # one gather for every contact: the decode's rows now, its validity mask at the step's end
                address = contact.efc_address.copy()
                efc_force = data.efc_force
                rows = address if self._row_offsets is None else address[:, None] + self._row_offsets
                self._address.append(address)
                self._efc_rows.append(efc_force.take(rows, mode="clip") if efc_force.size else np.zeros(rows.shape))
        self._counts.append(ncon)
        self._xpos[substep] = data.geom_xpos
        self._xmat[substep] = data.geom_xmat

    def record(self, model: Any, data: Any, step: int, substep: int) -> None:
        """Copy physics substep ``substep`` (0 first) of control step ``step``; substep 0 starts a new record."""
        if substep == 0:
            self._reset_rows()
            self._step = int(step)
        elif step != self._step:
            raise ValueError(f"substep {substep} of step {step} arrived inside the record of step {self._step}")
        self._copy(model, data)

    def forget(self) -> None:
        """Drop the record (the env's state was changed out of band): the next reading reads the current data."""
        self._reset_rows()
        self._step = -1

    def reading(self, step: int, model: Any, data: Any) -> StepCopReading:
        """The reading of control step ``step`` when it was recorded; else the current ``data`` as one substep."""
        if step == self._step and self._counts:
            return self._reduce()
        self._reset_rows()
        self._step = -1
        self._copy(model, data)
        try:
            return self._reduce()
        finally:
            self._reset_rows()

    def _reduce(self) -> StepCopReading:
        """The recorder's reduction (``SubstepContactRecorder._reduce``) of the recorded substeps, as one step."""
        n_feet = self._n_feet
        n_support = self._n_support
        substeps = len(self._counts)
        if self._geoms:
            counts = self._counts
            geoms = np.concatenate(self._geoms)
            points = np.concatenate(self._points)
            if self._address:
                # decode_normal_forces's gather and mask: a contact without constraint rows carries 0 N.  Its clamp
                # never binds on a contact that has rows (it holds 2 (condim - 1) of them, or condim when elliptic).
                rows = np.concatenate(self._efc_rows)
                valid = np.concatenate(self._address) >= 0
                normal = np.where(valid, rows if rows.ndim == 1 else rows.sum(axis=1), 0.0)
            else:
                normal = np.concatenate(self._normal)
            substep_of = np.repeat(self._substep_index[:substeps], counts)
        else:
            geoms = np.zeros((0, 2), dtype=np.int64)
            points = np.zeros((0, 3))
            normal = np.zeros(0)
            substep_of = np.zeros(0, dtype=np.int64)
        pair = geoms[:, 0] * self._n_geom + geoms[:, 1]
        foot = self._pair_foot[pair]
        sole = self._pair_sole[pair]
        # Each contact in the sole frame of its own substep, along its foot's signed forward axis (the recorder's
        # sole_cop_steps and whole_foot_cop_steps, for both feet at once).
        frame = self._frame_geom[foot]
        rotation = self._xmat[substep_of, frame].reshape(-1, 3, 3)
        local = np.einsum("nji,nj->ni", rotation, points - self._xpos[substep_of, frame])
        if self._forward is not None:  # every foot reads the same signed axis
            axis, sign = self._forward
            forward = sign * local[:, axis]
        else:
            forward = self._forward_sign[foot] * local[np.arange(len(foot)), self._forward_axis[foot]]
        ahead = forward - self._ankle_forward[foot]  # m ahead of the ankle axis (EpisodeTrace.foot_cop)
        fore = forward / self._half_forward[foot]  # pad half-lengths (EpisodeTrace.sole_cop[..., 0])
        bins = n_feet + 1
        weight_sum = np.bincount(foot, weights=normal, minlength=bins)[:n_feet]
        weighted = np.bincount(foot, weights=normal * ahead, minlength=bins)[:n_feet]
        pad_sum = np.bincount(sole, weights=normal, minlength=bins)[:n_feet]
        pad_weighted = np.bincount(sole, weights=normal * fore, minlength=bins)[:n_feet]
        # NaN where nothing is read: the whole-foot CoP in m, the pad CoP, the digit share.
        foot_cop_m, pad_cop, digit_share = np.full((3, n_feet), np.nan)
        np.divide(weighted, weight_sum, out=foot_cop_m, where=self._has_ankle & (weight_sum > COP_LOAD_N))
        np.divide(pad_weighted, pad_sum, out=pad_cop, where=pad_sum > COP_LOAD_N)
        # The limb's floor force at each substep (EpisodeTrace.leg_floor_max and _mean, the mean as np.mean takes it).
        leg = np.bincount(substep_of * bins + foot, weights=normal, minlength=substeps * bins)
        leg = leg.reshape(substeps, bins)[:, :n_feet]
        leg_force_max = leg.max(axis=0)
        leg_force_mean = np.add.reduce(leg, axis=0) / substeps
        _spanned, foot_cop, foot_loaded = normalized_foot_cop(
            foot_cop_m, leg_force_max, self._support_span, scale=self._span_scale
        )
        # pad_telemetry._gate_cop_per_foot's loaded rule: a finite CoP and a sole contact above the threshold.
        sole_contact = np.bincount(sole, weights=normal > CONTACT_THRESHOLD_N, minlength=bins)[:n_feet] > 0
        pad_loaded = np.isfinite(pad_cop) & sole_contact
        if n_support:
            # The digits' floor force (EpisodeTrace.support_floor_mean, summed over each foot's digits).
            columns = n_support + 1
            support_index = self._pair_support[pair]
            support = np.bincount(substep_of * columns + support_index, weights=normal, minlength=substeps * columns)
            support_mean = np.add.reduce(support.reshape(substeps, columns)[:, :n_support], axis=0) / substeps
            digits = self._digit_weights @ support_mean
        else:
            digits = np.zeros(n_feet)
        np.divide(digits, leg_force_mean, out=digit_share, where=leg_force_mean > 0)
        return StepCopReading(
            foot_cop=foot_cop,
            foot_loaded=foot_loaded,
            pad_cop=pad_cop,
            pad_loaded=pad_loaded,
            leg_force_mean=leg_force_mean,
            leg_force_max=leg_force_max,
            digit_share=digit_share,
            substeps=substeps,
        )
