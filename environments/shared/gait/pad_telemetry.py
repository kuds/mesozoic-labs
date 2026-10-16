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

The pad CoP reads one contact geom, and on a foot whose digits carry load
ahead of the pad it is not the toppling quantity.  A pad foot whose
``FOOT_SUPPORT`` entry declares its ankle and a support span (the T. rex:
its sole box's heel edge and its held-offset toe line, 0.060 to 0.213 m
ahead of the ankle since physics r10) also gets the
WHOLE-FOOT fields, from ``EpisodeTrace.foot_cop`` (every floor contact of
the limb, m ahead of the ankle axis along the sole's forward axis):

* ``<label>_foot_cop_fore_aft`` -- signed, in span half-lengths (-1 on the
  heel line, +1 on the toe line), averaged over the window steps the limb is
  loaded on (:func:`.stance_metrics.whole_foot_cop`);
* ``<label>_foot_cop_fore_aft_gate`` -- the window mean of its ``|.|`` with
  an unloaded step counting 1: the max over the feet IS
  ``StanceEpisodeMetrics.max_foot_cop_fore_aft``, bit for bit;
* ``<label>_foot_cop_std_m`` and ``<label>_foot_cop_dominant_hz`` -- the
  standard deviation (m) of the per-step whole-foot CoP over the loaded
  window steps, and the strongest non-zero frequency of that series (its
  de-meaned real FFT, unloaded steps dropped): the window mean cannot see a
  foot rocking heel to toe, these can;
* ``<label>_toe_margin_nm`` and ``<label>_heel_margin_nm`` -- the window
  mean over loaded steps of the floor-load moment left before the CoP
  reaches the toe line, ``F * (toe_line - x)``, and the heel line,
  ``F * (x - heel_line)``, in N m (``F`` the limb's floor force,
  ``leg_floor_mean``; negative past the line).

and per digit (a registered support geom other than the sole, named by its
geom name less the foot label and ``_geom``: ``r_toe_d2_geom`` is the digit
``toe_d2``, keyed ``r_toe_d2_load_share`` and ``r_toe_d2_load_pos``):

* ``<label>_<digit>_load_share`` -- the digit's window floor force over the
  whole limb's, on every pad foot with digits;
* ``<label>_<digit>_load_pos`` -- where along the digit its window floor
  load sits, force-weighted, 0 at the MTP joint's axis and 1 at the tip
  (``EpisodeTrace.support_load_pos``), on a hinged digit only (one with an
  MTP joint, ``Morphology.support_geom_mtp``); NaN when it carries nothing.

The window is the gate's, ``[settle_steps, T)``.  One rule says which
fields exist: a field is kept for a foot only when the foot has what it
measures.  Only feet with a BOX sole are pad feet, the feet the gate's
fore-aft CoP reads, and only a pad foot with digits has a digit share (the
compsognathus robot's soles and the dibothrosuchus's pads have none).  The
velociraptor's capsule feet and the brachiosaurus's ellipsoids have no pad,
so :meth:`PadLayout.from_morphology` gives them an empty layout and a caller
records no pad field for them at all.  Only the T. rex declares a support
span, so only its feet have the whole-foot fields, and only its digits are
hinged (the anatomical compsognathus's digits share the pad's body), so
only they have a load position.  Absent therefore means "the model has no
such thing", and NaN in a field that exists means "not measured this time".

Nothing here is a gate criterion, a reward or an observation, and no
digest reads it: it is telemetry, for the periodic evaluations
(``eval_diagnostics.StageAwareEvalCallback``, ``gate_progress.npz``) and the
stance replay CSV (``evaluation.record_stage_video``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .stance_metrics import whole_foot_cop

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
#: The whole-foot statistic's per-foot field: its maximum over the feet is ``max_foot_cop_fore_aft``.
FOOT_GATE_COP_FIELD = "foot_cop_fore_aft_gate"
#: The per-episode whole-foot fields (a foot that declares a support span), as suffixes of ``<label>_``.
FOOT_COP_EPISODE_FIELDS: tuple[str, ...] = (
    "foot_cop_fore_aft",
    FOOT_GATE_COP_FIELD,
    "foot_cop_std_m",
    "foot_cop_dominant_hz",
    "toe_margin_nm",
    "heel_margin_nm",
)
#: The per-step whole-foot fields (the replay CSV).
FOOT_COP_STEP_FIELDS: tuple[str, ...] = ("foot_cop_fore_aft", "toe_margin_nm", "heel_margin_nm")
#: The per-digit fields, as suffixes of ``<label>_<digit>_`` (the load position on a hinged digit only),
#: per episode and per step alike.
DIGIT_FIELDS: tuple[str, ...] = ("load_share", "load_pos")


@dataclass(frozen=True)
class PadFoot:
    """One foot with a box sole: its index in the trace's feet, its label, and its digits' support-geom indices."""

    index: int
    label: str
    #: Indices into ``EpisodeTrace.support_geom_names`` of the foot's support geoms other than its sole.
    digit_support: tuple[int, ...]
    #: Each digit's key name (its geom name less ``<label>_`` and ``_geom``), and whether it is hinged (has an
    #: MTP joint, so its load position is measured); aligned with ``digit_support``.  Empty: no per-digit field.
    digit_names: tuple[str, ...] = ()
    digit_hinged: tuple[bool, ...] = ()
    #: Whether the foot declares an ankle and a support span (``FOOT_SUPPORT``): its whole-foot fields exist.
    whole_foot: bool = False


@dataclass(frozen=True)
class PadLayout:
    """The pad feet of one model, in ``Morphology.feet`` order (empty when no foot has a box sole)."""

    feet: tuple[PadFoot, ...] = ()

    @classmethod
    def from_morphology(cls, morphology: "Morphology") -> "PadLayout":
        """The feet whose registered sole is a box, with their digits (every other support geom of the foot)."""
        import mujoco  # in the body: this module stays importable without mujoco

        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        names = tuple(morphology.support_geom_names)
        mtp = tuple(morphology.support_geom_mtp)
        feet = []
        for index, foot in enumerate(morphology.feet):
            if foot.sole_geom < 0 or foot.sole_type != box:
                continue
            digits = tuple(
                support
                for support, (geom, owner) in enumerate(zip(morphology.support_geom_ids, morphology.support_geom_foot))
                if owner == index and geom != foot.sole_geom
            )
            digit_names = tuple(_digit_name(names[support], foot.label) for support in digits) if names else ()
            hinged = tuple(bool(mtp and mtp[support] >= 0) for support in digits) if names else ()
            feet.append(
                PadFoot(
                    index=index,
                    label=foot.label,
                    digit_support=digits,
                    digit_names=digit_names,
                    digit_hinged=hinged,
                    whole_foot=foot.ankle_joint >= 0,
                )
            )
        return cls(feet=tuple(feet))

    def episode_keys(self) -> tuple[str, ...]:
        """Every pad foot's per-episode keys, foot-major: :data:`PAD_EPISODE_FIELDS` (a digit share only with
        digits), then :data:`FOOT_COP_EPISODE_FIELDS` (a foot with a support span), then each digit's
        :data:`DIGIT_FIELDS`."""
        return tuple(_keys(self.feet, PAD_EPISODE_FIELDS, FOOT_COP_EPISODE_FIELDS))

    def step_keys(self) -> tuple[str, ...]:
        """Every pad foot's per-step keys, foot-major, as :meth:`episode_keys` with the step fields."""
        return tuple(_keys(self.feet, PAD_STEP_FIELDS, FOOT_COP_STEP_FIELDS))


def _digit_name(geom_name: str, label: str) -> str:
    """A digit's key name: its geom name less the ``<label>_`` prefix and the ``_geom`` suffix."""
    name = geom_name[len(label) + 1 :] if geom_name.startswith(f"{label}_") else geom_name
    return name[: -len("_geom")] if name.endswith("_geom") and len(name) > len("_geom") else name


def _keys(feet: tuple[PadFoot, ...], names: tuple[str, ...], whole_foot: tuple[str, ...]) -> list[str]:
    keys = []
    for foot in feet:
        keys += [f"{foot.label}_{name}" for name in names if name != "digit_share" or foot.digit_support]
        if foot.whole_foot:
            keys += [f"{foot.label}_{name}" for name in whole_foot]
        for name, hinged in zip(foot.digit_names, foot.digit_hinged):
            keys += [f"{foot.label}_{name}_{field}" for field in DIGIT_FIELDS if field != "load_pos" or hinged]
    return keys


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
    spanned, foot_cop, foot_loaded = whole_foot_cop(trace)
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
        if foot.whole_foot and spanned[index]:
            out.update(_whole_foot_episode(trace, index, label, window, foot_cop, foot_loaded))
        for support, name, hinged in zip(foot.digit_support, foot.digit_names, foot.digit_hinged):
            load = np.asarray(trace.support_floor_mean[window, support], dtype=np.float64)
            if leg_total > 0:
                out[f"{label}_{name}_load_share"] = float(load.sum()) / leg_total
            if hinged:
                out[f"{label}_{name}_load_pos"] = _weighted_mean(trace.support_load_pos[window, support], load)
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """The *weights*-weighted mean of the finite *values* with a positive weight; NaN when there is none."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & (weights > 0)
    total = float(weights[keep].sum())
    return float((values[keep] * weights[keep]).sum() / total) if total > 0 else NAN


def _whole_foot_episode(
    trace: "EpisodeTrace", index: int, label: str, window: slice, cop: np.ndarray, loaded: np.ndarray
) -> dict[str, float]:
    """One spanned foot's whole-foot episode fields over *window* (module docstring)."""
    out: dict[str, float] = {}
    mine = loaded[window, index]
    gate = np.where(mine, np.abs(cop[window, index]), 1.0)
    out[f"{label}_{FOOT_GATE_COP_FIELD}"] = float(gate.mean())
    if not mine.any():
        return out
    out[f"{label}_foot_cop_fore_aft"] = float(cop[window, index][mine].mean())
    ahead = np.asarray(trace.foot_cop[window, index], dtype=np.float64)[mine]
    out[f"{label}_foot_cop_std_m"] = float(ahead.std())
    out[f"{label}_foot_cop_dominant_hz"] = _dominant_frequency(ahead, float(trace.dt))
    toe, heel = _margins(trace, index)
    out[f"{label}_toe_margin_nm"] = float(toe[window][mine].mean())
    out[f"{label}_heel_margin_nm"] = float(heel[window][mine].mean())
    return out


#: A whole-foot CoP series whose range is at most this (m) is still: it has no dominant frequency.
_STILL_M = 1e-9


def _dominant_frequency(series: np.ndarray, dt: float) -> float:
    """The strongest non-zero frequency (Hz) of the de-meaned *series* sampled every *dt*.

    NaN under 3 samples, and for a still series (a range of at most a
    nanometre), whose spectrum is rounding noise.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.size < 3 or float(np.ptp(series)) <= _STILL_M:
        return NAN
    power = np.abs(np.fft.rfft(series - series.mean())) ** 2
    return float(np.fft.rfftfreq(series.size, d=dt)[1 + int(np.argmax(power[1:]))])


def _margins(trace: "EpisodeTrace", index: int) -> tuple[np.ndarray, np.ndarray]:
    """Per step, the toe and heel moment margins of foot *index* in N m: ``F * (toe - x)``, ``F * (x - heel)``."""
    heel_line, toe_line = (float(value) for value in np.asarray(trace.foot_support_span, dtype=np.float64)[index])
    force = np.asarray(trace.leg_floor_mean[:, index], dtype=np.float64)
    ahead = np.asarray(trace.foot_cop[:, index], dtype=np.float64)
    return force * (toe_line - ahead), force * (ahead - heel_line)


def pad_step_columns(trace: "EpisodeTrace", layout: PadLayout) -> dict[str, np.ndarray]:
    """Per step (the whole episode, no window): the pad fields of one step, keyed as :meth:`PadLayout.step_keys`.

    ``<label>_sole_cop_fore_aft`` is the step's signed fore-aft CoP, NaN on a
    step the gate counts as unloaded (so the gate's per-foot statistic over
    any window is the mean of ``|value|`` with NaN read as 1);
    ``<label>_digit_share`` (a foot with digits only) is the step's digit
    floor force over the whole limb's, NaN on a step the limb carries nothing.
    On a foot with a support span, ``<label>_foot_cop_fore_aft`` is the
    step's signed whole-foot CoP in span half-lengths and
    ``<label>_toe_margin_nm`` / ``<label>_heel_margin_nm`` its moment margins,
    NaN on a step the limb is unloaded (so the per-foot whole-foot statistic
    over any window is again the mean of ``|value|`` with NaN read as 1); per digit,
    ``<label>_<digit>_load_share`` (NaN when the limb carries nothing) and,
    on a hinged digit, ``<label>_<digit>_load_pos`` (NaN when it carries
    nothing).
    """
    steps = trace.length
    out: dict[str, np.ndarray] = {}
    if not layout.feet:
        return out
    box, loaded, _per_foot = _gate_cop_per_foot(trace, slice(0, steps))
    spanned, foot_cop, foot_loaded = whole_foot_cop(trace)
    for foot in layout.feet:
        cop = np.full(steps, np.nan)
        if box[foot.index]:
            mine = loaded[:, foot.index]
            cop[mine] = trace.sole_cop[mine, foot.index, 0]
        out[f"{foot.label}_sole_cop_fore_aft"] = cop
        leg = np.asarray(trace.leg_floor_mean[:, foot.index], dtype=np.float64)
        carried = leg > 0
        if foot.digit_support:
            share = np.full(steps, np.nan)
            digits = trace.support_floor_mean[:, list(foot.digit_support)].sum(axis=1)
            share[carried] = digits[carried] / leg[carried]
            out[f"{foot.label}_digit_share"] = share
        if foot.whole_foot and spanned[foot.index]:
            mine = foot_loaded[:, foot.index]
            toe, heel = _margins(trace, foot.index)
            for name, values in (
                ("foot_cop_fore_aft", foot_cop[:, foot.index]),
                ("toe_margin_nm", toe),
                ("heel_margin_nm", heel),
            ):
                column = np.full(steps, np.nan)
                column[mine] = values[mine]
                out[f"{foot.label}_{name}"] = column
        for support, name, hinged in zip(foot.digit_support, foot.digit_names, foot.digit_hinged):
            share = np.full(steps, np.nan)
            share[carried] = np.asarray(trace.support_floor_mean[carried, support], dtype=np.float64) / leg[carried]
            out[f"{foot.label}_{name}_load_share"] = share
            if hinged:
                out[f"{foot.label}_{name}_load_pos"] = np.asarray(trace.support_load_pos[:, support], dtype=np.float64)
    return out
