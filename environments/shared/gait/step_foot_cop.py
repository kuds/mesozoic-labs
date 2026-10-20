"""One control step's per-foot centre of pressure, for the T. rex's opt-in CoP band (screen-only, never merged).

The stance judge's K2b statistic is each foot's whole-foot centre of
pressure (``<label>_foot_cop_fore_aft``): every floor contact of the limb, on
every physics substep, in the sole frame of its own substep, force-weighted,
in span half-lengths (:mod:`.pad_telemetry`).  Only
:class:`.recorder.SubstepContactRecorder` computes it, per episode, after the
fact.  :class:`StepFootCop` computes the same reading for ONE control step,
inside the env, so a reward can price it.  It does not reimplement anything:

* the hot path copies the rows the recorder's hook copies (the contact geom
  pairs, their normal forces decoded by :func:`.recorder.decode_normal_forces`,
  their points, and the sole geoms' poses), once per substep, from the env's
  ``_accumulate_substep`` slot (``BaseDinoEnv.step`` calls it right after the
  probe hook the recorder owns, so both read the same ``data``);
* :meth:`StepFootCop.reading` reduces them with the recorder's own
  functions (:func:`.recorder.classify_floor_contacts`,
  :func:`.recorder.per_substep_force`, :func:`.recorder.sole_cop_steps`,
  :func:`.recorder.whole_foot_cop_steps`) and the gate's normalization and
  loaded rule (:func:`.stance_metrics.normalized_foot_cop`), with one step
  of ``frame_skip`` substeps.  ``np.bincount`` sums each step's contacts in
  the same order either way, so the reading equals the recorder's trace for
  that step bit for bit (``tests/test_anatomical_stance_cop_band.py`` pins
  it on recorded rollouts).

It reads ``model`` and ``data`` and never writes them: a trajectory with it
is bit-identical to one without.

Layering: :mod:`environments.shared.gait` is a measurement library, and no
species env module imported it before this.  The T. rex env imports this
module only inside the method that builds the band's reader, and only when
the band is on, so an env with the band off never loads it.  That is still
an exception to the rule ``test_no_species_env_module_imports_the_gait_package``
pins; the test names it.  A reward that reads this code depends on
measurement code that no task fingerprint hashes: a later change to the
recorder's reduction would change the reward.  See the stance4 copband
design note (screen-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import CONTACT_THRESHOLD_N
from .morphology import Morphology
from .recorder import (
    classify_floor_contacts,
    decode_normal_forces,
    decode_settings,
    per_substep_force,
    sole_cop_steps,
    whole_foot_cop_steps,
)
from .stance_metrics import normalized_foot_cop


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
        self._feet = morphology.feet
        n_feet = morphology.n_feet
        model = unwrapped.model
        # As the recorder: a foot without a sole samples geom 0 in its slot, and no contact is ever on it.
        self._soles = np.array([max(foot.sole_geom, 0) for foot in morphology.feet], dtype=np.int64)
        self._half_size = [
            np.asarray(model.geom_size[foot.sole_geom], dtype=np.float64) if foot.sole_geom >= 0 else None
            for foot in morphology.feet
        ]
        self._ankle_forward = np.array([foot.ankle_forward_m for foot in morphology.feet], dtype=np.float64)
        self._support_span = np.array(
            [[foot.heel_line_m, foot.toe_line_m] for foot in morphology.feet], dtype=np.float64
        ).reshape(-1, 2)
        self._sole_of_foot = np.array([foot.sole_geom for foot in morphology.feet] + [-2])
        # Each foot's digits: its registered support geoms other than its sole (pad_telemetry's digit share).
        support_foot = np.asarray(morphology.support_geom_foot, dtype=np.int64)
        support_ids = np.asarray(morphology.support_geom_ids, dtype=np.int64)
        self._digit_mask = np.zeros((n_feet, len(support_ids)), dtype=bool)
        for index, foot in enumerate(morphology.feet):
            self._digit_mask[index] = (support_foot == index) & (support_ids != foot.sole_geom)
        self._decoded_model: Any = None
        self._elliptic = False
        self._edge_offsets: np.ndarray | None = None
        self._step = -1
        self._reset_rows()

    @property
    def morphology(self) -> Morphology:
        return self._morphology

    def _reset_rows(self) -> None:
        self._geoms: list[np.ndarray] = []
        self._normal: list[np.ndarray] = []
        self._points: list[np.ndarray] = []
        self._counts: list[int] = []
        self._pose: list[np.ndarray] = []

    def _copy(self, model: Any, data: Any) -> None:
        """One substep's rows, copied exactly as ``SubstepContactRecorder._hook`` copies them."""
        if model is not self._decoded_model:
            self._decoded_model = model
            self._elliptic, self._edge_offsets = decode_settings(model)
        ncon = int(data.ncon)
        if ncon:
            contact = data.contact
            self._normal.append(
                decode_normal_forces(contact, data.efc_force, elliptic=self._elliptic, edge_offsets=self._edge_offsets)
            )
            self._geoms.append(contact.geom.copy())
            self._points.append(contact.pos.copy())
        self._counts.append(ncon)
        self._pose.append(np.concatenate([data.geom_xpos[self._soles], data.geom_xmat[self._soles]], axis=1))

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
        morphology = self._morphology
        feet = self._feet
        n_feet = morphology.n_feet
        n_support = len(morphology.support_geom_ids)
        substeps = len(self._counts)
        counts = np.asarray(self._counts, dtype=np.int64)
        substep_of = np.repeat(np.arange(substeps), counts)
        if counts.sum():
            geoms = np.concatenate(self._geoms).astype(np.int64)
            normal = np.concatenate(self._normal)
            points = np.concatenate(self._points)
        else:
            geoms = np.zeros((0, 2), dtype=np.int64)
            normal = np.zeros(0)
            points = np.zeros((0, 3))
        other, _on_floor, foot, leg_on_floor = classify_floor_contacts(morphology, geoms)
        leg = per_substep_force(substep_of, normal, leg_on_floor, foot, n_feet, 1, substeps)
        support_index = np.where(leg_on_floor, morphology.geom_support[other], -1)
        on_support = support_index >= 0
        support = (
            per_substep_force(substep_of, normal, on_support, support_index, n_support, 1, substeps)
            if n_support
            else np.zeros((1, substeps, 0))
        )
        on_sole = leg_on_floor & (other == self._sole_of_foot[foot])
        loaded = on_sole & (normal > CONTACT_THRESHOLD_N)
        sole_contacts = np.bincount(substep_of[loaded] * n_feet + foot[loaded], minlength=substeps * n_feet).reshape(
            1, substeps, n_feet
        )
        pose = np.stack(self._pose) if on_sole.any() or leg_on_floor.any() else None
        cop = sole_cop_steps(feet, self._half_size, pose, substep_of, foot, on_sole, points, normal, 1, substeps)
        foot_cop_m = whole_foot_cop_steps(
            feet, self._ankle_forward, pose, substep_of, foot, leg_on_floor, points, normal, 1, substeps
        )
        leg_force_mean = leg.mean(axis=1)  # (1, F): EpisodeTrace.leg_floor_mean
        leg_force_max = leg.max(axis=1)  # (1, F): EpisodeTrace.leg_floor_max
        _spanned, foot_cop, foot_loaded = normalized_foot_cop(foot_cop_m, leg_force_max, self._support_span)
        pad_cop = cop[0, :, 0]
        # pad_telemetry._gate_cop_per_foot's loaded rule: a finite CoP and sole_contacts_mean > 0.
        pad_loaded = np.isfinite(pad_cop) & (sole_contacts.mean(axis=1)[0] > 0)
        support_mean = support.mean(axis=1)[0]  # (G,): EpisodeTrace.support_floor_mean
        digits = self._digit_mask.astype(np.float64) @ support_mean if n_support else np.zeros(n_feet)
        limb = leg_force_mean[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            digit_share = np.where(limb > 0, digits / np.where(limb > 0, limb, 1.0), np.nan)
        return StepCopReading(
            foot_cop=foot_cop[0],
            foot_loaded=foot_loaded[0],
            pad_cop=pad_cop,
            pad_loaded=pad_loaded,
            leg_force_mean=limb,
            leg_force_max=leg_force_max[0],
            digit_share=digit_share,
            substeps=substeps,
        )
