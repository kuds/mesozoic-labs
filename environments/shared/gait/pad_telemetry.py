"""Per-foot pad telemetry on a floor-truth :class:`.recorder.EpisodeTrace`: report-only, pure numpy.

``stance_quality/v2``'s ``max_sole_cop_fore_aft`` keeps one number per
episode, the worse pad's, unsigned.  A run that fails it says neither which
foot nor which edge, nor when in training the posture formed (the T. rex
stance run of 2026-10-11 failed it on 39 of 40 episodes with its right pad
on the toe edge, and its artifacts could not say where the left pad
stood).  This module reduces the same trace, with the same normalization,
per foot:

* ``<label>_sole_cop_fore_aft`` -- the SIGNED fore-aft centre of pressure on
  the pad, in pad half-lengths, + toward the toe (``EpisodeTrace.sole_cop``,
  the recorder's convention), averaged over the window steps the gate counts
  as loaded (a finite CoP and at least one sole contact above
  ``CONTACT_THRESHOLD_N``, as ``stance_metrics._sole_metrics`` reads them);
  NaN when no window step is loaded.  Signed, so it says which edge.  It
  cannot see a pad rocking heel to toe (that averages to its centre), which
  is why the next one is kept beside it;
* ``<label>_sole_cop_fore_aft_gate`` -- the gate's own per-foot statistic:
  the window mean of ``|CoP|`` with an unloaded step counting 1.  The max of
  it over the feet IS ``StanceEpisodeMetrics.max_sole_cop_fore_aft``, bit
  for bit (``tests/test_gait_pad_telemetry.py`` pins it on every pad model);
* ``<label>_digit_share`` -- the foot's registered support geoms other than
  its sole (the digits: ``FOOT_SUPPORT``'s ``support`` less its ``sole``),
  their summed window floor force over the whole limb's
  (``leg_floor_mean``, the denominator of ``support_geom_share``); NaN on a
  window with no floor load.

The window is the gate's, ``[settle_steps, T)``.  One rule says which
fields exist: a field is kept for a foot only when the foot has what it
measures.  Only feet with a BOX sole are pad feet, the feet the gate's
fore-aft CoP reads, and only a pad foot with digits has a digit share (the
compsognathus robot's soles and the dibothrosuchus's pads have none).  The
velociraptor's capsule feet and the brachiosaurus's ellipsoids have no pad,
so :meth:`PadLayout.from_morphology` gives them an empty layout and a caller
records no pad field for them at all.  Absent therefore means "the model has
no such thing", and NaN in a field that exists means "not measured this
time".

Nothing here is a gate criterion, a reward or an observation, and no
digest reads it: it is telemetry, for the periodic evaluations
(``eval_diagnostics.StageAwareEvalCallback``, ``gate_progress.npz``) and the
stance replay CSV (``evaluation.record_stage_video``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .morphology import Morphology
    from .recorder import EpisodeTrace

NAN = float("nan")

#: The per-episode field that is the gate's own per-foot statistic: its maximum over the feet is the
#: episode's ``max_sole_cop_fore_aft``.
GATE_COP_FIELD = "sole_cop_fore_aft_gate"
#: The per-episode pad fields, as suffixes of ``<label>_``.
PAD_EPISODE_FIELDS: tuple[str, ...] = ("sole_cop_fore_aft", GATE_COP_FIELD, "digit_share")
#: The per-step pad fields (the replay CSV): the signed CoP and the digit share of one step.
PAD_STEP_FIELDS: tuple[str, ...] = ("sole_cop_fore_aft", "digit_share")


@dataclass(frozen=True)
class PadFoot:
    """One foot with a box sole: its index in the trace's feet, its label, and its digits' support-geom indices."""

    index: int
    label: str
    #: Indices into ``EpisodeTrace.support_geom_names`` of the foot's support geoms other than its sole.
    digit_support: tuple[int, ...]


@dataclass(frozen=True)
class PadLayout:
    """The pad feet of one model, in ``Morphology.feet`` order (empty when no foot has a box sole)."""

    feet: tuple[PadFoot, ...] = ()

    @classmethod
    def from_morphology(cls, morphology: "Morphology") -> "PadLayout":
        """The feet whose registered sole is a box, with their digits (every other support geom of the foot)."""
        import mujoco  # in the body: this module stays importable without mujoco

        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        feet = []
        for index, foot in enumerate(morphology.feet):
            if foot.sole_geom < 0 or foot.sole_type != box:
                continue
            digits = tuple(
                support
                for support, (geom, owner) in enumerate(zip(morphology.support_geom_ids, morphology.support_geom_foot))
                if owner == index and geom != foot.sole_geom
            )
            feet.append(PadFoot(index=index, label=foot.label, digit_support=digits))
        return cls(feet=tuple(feet))

    def episode_keys(self) -> tuple[str, ...]:
        """``<label>_<field>`` for every pad foot and :data:`PAD_EPISODE_FIELDS`, foot-major (a digit share only with digits)."""
        return tuple(_keys(self.feet, PAD_EPISODE_FIELDS))

    def step_keys(self) -> tuple[str, ...]:
        """``<label>_<field>`` for every pad foot and :data:`PAD_STEP_FIELDS`, foot-major (a digit share only with digits)."""
        return tuple(_keys(self.feet, PAD_STEP_FIELDS))


def _keys(feet: tuple[PadFoot, ...], names: tuple[str, ...]) -> list[str]:
    return [f"{foot.label}_{name}" for foot in feet for name in names if name != "digit_share" or foot.digit_support]


def _gate_cop_per_foot(trace: "EpisodeTrace", window: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(box, loaded, per_foot)``: the gate's fore-aft CoP statistic per foot over ``window``.

    The same arrays and the same reduction as ``stance_metrics._sole_metrics``
    (a foot is a box sole when its tilt and keyframe tilt are finite and its
    corner lift is finite on every window step; a step counts as loaded when
    its CoP is finite and a sole contact carries more than
    ``CONTACT_THRESHOLD_N``; ``|CoP|`` on a loaded step, 1 on any other,
    averaged over the window), so ``per_foot[box].max()`` is
    ``max_sole_cop_fore_aft`` bit for bit.  ``box`` is (F,), ``loaded`` is
    (W, F) (False off the box feet) and ``per_foot`` is (F,), NaN off them.
    """
    n_feet = trace.n_feet
    tilt = trace.sole_tilt_deg[window]
    has_sole = np.isfinite(tilt).all(axis=0) & np.isfinite(trace.home_sole_tilt_deg)
    box = has_sole & np.isfinite(trace.sole_corner_lift[window]).all(axis=0)
    loaded_all = np.zeros((tilt.shape[0], n_feet), dtype=bool)
    per_foot = np.full(n_feet, np.nan)
    if box.any():
        fore = np.abs(trace.sole_cop[window][:, box, 0])
        loaded = np.isfinite(fore) & (trace.sole_contacts_mean[window][:, box] > 0)
        per_foot[box] = np.where(loaded, fore, 1.0).mean(axis=0)
        loaded_all[:, box] = loaded
    return box, loaded_all, per_foot


def episode_pad_telemetry(trace: "EpisodeTrace", layout: PadLayout, *, settle_steps: int) -> dict[str, float]:
    """The pad fields of one recorded episode, keyed as :meth:`PadLayout.episode_keys` (all NaN for an empty window)."""
    if isinstance(settle_steps, bool) or int(settle_steps) != settle_steps or settle_steps < 0:
        raise ValueError(f"settle_steps must be a non-negative whole number of steps, got {settle_steps!r}")
    steps = trace.length
    window = slice(int(settle_steps), steps)
    out = {key: NAN for key in layout.episode_keys()}
    if steps <= settle_steps or not layout.feet:
        return out
    box, loaded, per_foot = _gate_cop_per_foot(trace, window)
    signed = trace.sole_cop[window][:, :, 0]
    for foot in layout.feet:
        index, label = foot.index, foot.label
        if box[index]:
            mine = loaded[:, index]
            if mine.any():
                out[f"{label}_sole_cop_fore_aft"] = float(signed[mine, index].mean())
            out[f"{label}_{GATE_COP_FIELD}"] = float(per_foot[index])
        leg_total = float(trace.leg_floor_mean[window, index].sum())
        if foot.digit_support and leg_total > 0:
            digits = float(trace.support_floor_mean[window][:, list(foot.digit_support)].sum())
            out[f"{label}_digit_share"] = digits / leg_total
    return out


def pad_step_columns(trace: "EpisodeTrace", layout: PadLayout) -> dict[str, np.ndarray]:
    """Per step (the whole episode, no window): the pad fields of one step, keyed as :meth:`PadLayout.step_keys`.

    ``<label>_sole_cop_fore_aft`` is the step's signed fore-aft CoP, NaN on a
    step the gate counts as unloaded (so the gate's per-foot statistic over
    any window is the mean of ``|value|`` with NaN read as 1);
    ``<label>_digit_share`` (a foot with digits only) is the step's digit
    floor force over the whole limb's, NaN on a step the limb carries nothing.
    """
    steps = trace.length
    out: dict[str, np.ndarray] = {}
    if not layout.feet:
        return out
    box, loaded, _per_foot = _gate_cop_per_foot(trace, slice(0, steps))
    for foot in layout.feet:
        cop = np.full(steps, np.nan)
        if box[foot.index]:
            mine = loaded[:, foot.index]
            cop[mine] = trace.sole_cop[mine, foot.index, 0]
        out[f"{foot.label}_sole_cop_fore_aft"] = cop
        if foot.digit_support:
            share = np.full(steps, np.nan)
            leg = np.asarray(trace.leg_floor_mean[:, foot.index], dtype=np.float64)
            digits = trace.support_floor_mean[:, list(foot.digit_support)].sum(axis=1)
            carried = leg > 0
            share[carried] = digits[carried] / leg[carried]
            out[f"{foot.label}_digit_share"] = share
    return out
