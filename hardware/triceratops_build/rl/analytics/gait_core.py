"""Pure-numpy gait-quality measures for the 60 cm Triceratops (simulation analytics). No mujoco, no torch.

Everything here works on arrays sampled every physics step (1 ms) and is unit-tested on synthetic data
(``tests/test_gait_core.py``). The episode runner (``gait_report.py``) produces those arrays.

Definitions shared with the rest of the repository, reused rather than re-invented:

* contact: a foot is loaded when its touch sensor reads above the env's ``contact_threshold_n`` (2 N), as in
  ``rl/analysis/gait_pipe1.py`` and ``rl/gait_metrics.py``;
* debounce: the env's rule (``TrikeEnv.contact_debounce_s`` = 10 ms; ``gait_metrics.debounced_touchdowns``): a foot
  counts as lifted only after it has been unloaded for ``n`` consecutive samples, and only after its first landing
  since reset; a touchdown is the first loaded sample after a lift. So unloaded gaps shorter than ``n`` inside a stance
  are filled, a short loaded blip after a real lift is kept (it is a touchdown for the env too), and the first landing
  is not a touchdown. The shared T. rex library (``environments/shared/gait/events.debounce``) also drops short loaded
  blips and works per control step; this module deliberately keeps the Triceratops env's rule so touchdown counts
  equal the training evaluation's;
* steps: on top of the env's 10 ms rule, the shared library's ``events.debounce`` with a 50 ms window (unloaded
  gaps shorter than 50 ms filled, then loaded blips shorter than 50 ms dropped; runs touching either end kept). At
  0.25 m/s the env-debounced contact of the pipe1 policy still breaks for 10-50 ms inside a stance with the sole
  lifting a median under 1.5 mm (a contact-solver rebound), while 95 % of its longer gaps (swings) lift 7 mm or more
  (README: the split is less clean at 0.15 m/s); strides, phases, swing clearance and touchdown impacts are measured
  on steps, touchdown counts also on the env's rule;
* support polygon and margin: ``environments/shared/gait/stance_track.convex_hull`` and ``signed_margin`` (imported:
  + inside, the distance to the nearest edge; outside, minus the distance past the most violated edge line, which is
  the Euclidean distance except beyond a corner, where it is a lower bound).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))
from environments.shared.gait.events import debounce as shared_debounce  # noqa: E402  (pure numpy)
from environments.shared.gait.stance_track import convex_hull, signed_margin  # noqa: E402  (pure numpy)

CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
SHORT = {"left_fore": "LF", "right_fore": "RF", "left_hind": "LH", "right_hind": "RH"}

# Ideal phase offsets (fraction of a stride) of RF, LH and RH touchdowns after the LF touchdown.
GAIT_TEMPLATES = {
    "trot": {"right_fore": 0.5, "left_hind": 0.5, "right_hind": 0.0},
    "pace": {"right_fore": 0.5, "left_hind": 0.0, "right_hind": 0.5},
    "bound": {"right_fore": 0.0, "left_hind": 0.5, "right_hind": 0.5},
    "walk (lateral sequence)": {"right_fore": 0.5, "left_hind": 0.75, "right_hind": 0.25},
    "walk (diagonal sequence)": {"right_fore": 0.5, "left_hind": 0.25, "right_hind": 0.75},
    "pronk": {"right_fore": 0.0, "left_hind": 0.0, "right_hind": 0.0},
}
GAIT_TOLERANCE = 0.125  # cycles: half the 0.25 spacing between neighbouring templates, so the classes cannot overlap
MIN_PHASE_LOCK = 0.5  # resultant length below which a foot's phase is not locked to LF
OUTSIDE_TOL_M = 1e-6  # a margin below -1 um counts as outside (a CoP on a corner reads -1e-16 m)


# ---------------------------------------------------------------- runs and debounced contact
def runs(flags) -> list[tuple[bool, int, int]]:
    """Run-length encoding of a 1-D boolean series: ``[(value, start, length), ...]``."""
    x = np.asarray(flags, dtype=bool)
    if x.size == 0:
        return []
    change = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, x.size]
    return [(bool(x[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


def debounce_contact(raw, n: int) -> np.ndarray:
    """The env's debounced contact state for one foot (1-D) or several (samples x feet).

    An unloaded run shorter than ``n`` samples that follows a loaded sample is filled (the foot never counted as
    lifted); the unloaded run before the first landing stays unloaded; loaded runs are kept whatever their length."""
    x = np.asarray(raw, dtype=bool)
    if x.ndim == 2:
        return np.stack([debounce_contact(x[:, i], n) for i in range(x.shape[1])], axis=1)
    out = x.copy()
    for value, start, length in runs(x):
        if not value and start > 0 and length < n:
            out[start:start + length] = True
    return out


def step_contact(raw, n_debounce: int, n_step: int) -> np.ndarray:
    """Step-level contact (1-D or samples x feet): the env-debounced state (``debounce_contact`` with ``n_debounce``),
    then the shared library's ``events.debounce`` with ``n_step`` (gaps, then blips, shorter than ``n_step`` removed)."""
    deb = debounce_contact(raw, n_debounce)
    if deb.ndim == 2:
        return np.stack([shared_debounce(deb[:, i], n_step) for i in range(deb.shape[1])], axis=1)
    return shared_debounce(deb, n_step)


def touchdowns(deb) -> np.ndarray:
    """Touchdown sample indices of a debounced series (1-D): rising edges, the first landing excluded."""
    x = np.asarray(deb, dtype=np.int8)
    rise = np.flatnonzero(np.diff(x) == 1) + 1
    if rise.size and not x[: rise[0]].any():  # the first rising edge with nothing loaded before it: first landing
        rise = rise[1:]
    return rise


def touchdown_count(raw, n: int, start: int = 0) -> np.ndarray:
    """Debounced touchdowns per foot at samples >= ``start`` (``raw`` samples x feet): the quantity
    ``gait_metrics.debounced_touchdowns`` counts, from this module's debounced state."""
    deb = debounce_contact(raw, n)
    return np.array([int((touchdowns(deb[:, i]) >= start).sum()) for i in range(deb.shape[1])])


def liftoffs(deb) -> np.ndarray:
    """Lift-off sample indices of a debounced series (1-D): the first unloaded sample of each unloaded run."""
    x = np.asarray(deb, dtype=np.int8)
    return np.flatnonzero(np.diff(x) == -1) + 1


def segments(deb, start: int = 0, end: int | None = None, value: bool = True) -> list[tuple[int, int]]:
    """``(first, stop)`` index pairs of the complete runs of ``value`` lying wholly inside ``[start, end)``; a run
    touching either end of the series (or of the range) is left out, since its length is not known."""
    x = np.asarray(deb, dtype=bool)
    end = x.size if end is None else end
    out = []
    for v, s, n in runs(x):
        if v == value and s > 0 and s + n < x.size and s >= start and s + n <= end:
            out.append((s, s + n))
    return out


# ---------------------------------------------------------------- rhythm, phase and gait class
def circular_mean(phases) -> tuple[float, float]:
    """Circular mean (cycles, in [0, 1)) and resultant length R (0-1) of phases given in cycles."""
    p = np.asarray(phases, dtype=np.float64)
    if p.size == 0:
        return float("nan"), 0.0
    z = np.exp(2j * np.pi * p).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(abs(z))


def circ_dist(a: float, b: float) -> float:
    """Distance between two phases in cycles, 0-0.5."""
    d = (a - b) % 1.0
    return float(min(d, 1.0 - d))


def phase_offsets(td_ref, td_other) -> np.ndarray:
    """For each reference stride [td_ref[k], td_ref[k+1]), the first touchdown of the other foot inside it, as a
    fraction of that stride (strides with no touchdown of the other foot are skipped)."""
    r = np.asarray(td_ref, dtype=np.float64)
    o = np.asarray(td_other, dtype=np.float64)
    out = []
    for a, b in zip(r[:-1], r[1:]):
        inside = o[(o >= a) & (o < b)]
        if inside.size:
            out.append((inside[0] - a) / (b - a))
    return np.asarray(out)


def classify_gait(offsets: dict, locks: dict | None = None, tol: float = GAIT_TOLERANCE,
                  min_lock: float = MIN_PHASE_LOCK) -> dict:
    """The symmetric-gait class from the mean phase offsets of RF, LH and RH after LF (cycles).

    Nearest template by the largest circular distance over the three feet; that distance must be <= ``tol``,
    else 'other'. A foot whose phase is not locked (resultant length < ``min_lock``) makes the class 'other'.
    'pronk' is reported as 'other (pronk)'; the walks keep their sequence in ``detail``."""
    feet = ("right_fore", "left_hind", "right_hind")
    if any(offsets.get(f) is None or not np.isfinite(offsets[f]) for f in feet):
        return {"class": "other", "detail": "a foot has no phase offset", "distance": None, "nearest": None}
    dist = {name: max(circ_dist(offsets[f], tpl[f]) for f in feet) for name, tpl in GAIT_TEMPLATES.items()}
    nearest = min(dist, key=dist.get)
    if locks is not None and any(locks.get(f, 0.0) < min_lock for f in feet):
        weak = [SHORT[f] for f in feet if locks.get(f, 0.0) < min_lock]
        return {"class": "other", "detail": f"phase not locked ({', '.join(weak)})", "distance": dist[nearest],
                "nearest": nearest}
    if dist[nearest] > tol:
        return {"class": "other", "detail": f"nearest {nearest}, {dist[nearest]:.2f} cycles off", "distance": dist[nearest],
                "nearest": nearest}
    if nearest == "pronk":
        return {"class": "other", "detail": "pronk", "distance": dist[nearest], "nearest": nearest}
    cls = "walk" if nearest.startswith("walk") else nearest
    return {"class": cls, "detail": nearest, "distance": dist[nearest], "nearest": nearest}


def wrap_half(x: float) -> float:
    """A phase difference wrapped to [-0.5, 0.5) cycles."""
    return float((x + 0.5) % 1.0 - 0.5)


def diagonal_timing(offsets: dict) -> dict:
    """Timing figures of a (near-)symmetric gait from the phase offsets of RF, LH and RH after LF (cycles):
    Hildebrand's limb phase on each side (the lag of the fore touchdown after the same side's hind touchdown,
    0.5 for a trot, 0.25 for a lateral-sequence walk, 0 for a pace); the lag of each fore foot after its diagonal
    hind partner (0 for a perfect trot, + the hind foot lands first); and the interval from the LF+RH pair to the
    RF+LH pair (0.5 when the two diagonal pairs are evenly spaced)."""
    rf, lh, rh = (offsets.get(k) for k in ("right_fore", "left_hind", "right_hind"))
    if any(v is None or not np.isfinite(v) for v in (rf, lh, rh)):
        return {}
    pair1 = circular_mean([0.0, rh])[0]
    pair2 = circular_mean([rf, lh])[0]
    return {"limb_phase_left": float((0.0 - lh) % 1.0), "limb_phase_right": float((rf - rh) % 1.0),
            "fore_after_diagonal_hind_LF_RH": wrap_half(0.0 - rh), "fore_after_diagonal_hind_RF_LH": wrap_half(rf - lh),
            "pair_interval_LFRH_to_RFLH": float((pair2 - pair1) % 1.0)}


def symmetry_index(left: float, right: float) -> float | None:
    """Robinson's symmetry index, % : 100 x |L - R| / ((L + R) / 2); 0 is symmetric. None if undefined."""
    if left is None or right is None:
        return None
    s = 0.5 * (left + right)
    return None if s == 0 else float(100.0 * abs(left - right) / s)


def stride_resample(signal, td, n: int = 101) -> np.ndarray:
    """Each complete stride [td[k], td[k+1]) of ``signal`` (samples x channels) resampled onto ``n`` points of
    stride phase 0-1 (linear interpolation): strides x n x channels."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    out = []
    ph = np.linspace(0.0, 1.0, n)
    for a, b in zip(td[:-1], td[1:]):
        if b - a < 4:
            continue
        seg = x[a:b + 1]
        src = np.linspace(0.0, 1.0, seg.shape[0])
        out.append(np.stack([np.interp(ph, src, seg[:, c]) for c in range(x.shape[1])], axis=1))
    return np.asarray(out).reshape(-1, n, x.shape[1])


# ---------------------------------------------------------------- geometry and balance
def sole_bottom_corners(xpos, xmat, size) -> np.ndarray:
    """World positions of the four bottom-face corners of box soles.

    ``xpos`` (..., 3) geom centres, ``xmat`` (..., 9) geom rotation matrices (row-major), ``size`` (3,) half sizes
    with the box's -z face as the sole. Returns (..., 4, 3)."""
    xpos = np.asarray(xpos, dtype=np.float64)
    R = np.asarray(xmat, dtype=np.float64).reshape(xpos.shape[:-1] + (3, 3))
    sx, sy, sz = (float(v) for v in size)
    local = np.array([[sx, sy, -sz], [sx, -sy, -sz], [-sx, -sy, -sz], [-sx, sy, -sz]])
    return xpos[..., None, :] + np.einsum("...ij,kj->...ki", R, local)


def support_margin(points_xy, p_xy) -> float:
    """Signed distance of ``p_xy`` inside the convex hull of ``points_xy`` (shared ``signed_margin``; NaN when the
    points span no area)."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return float("nan")
    return signed_margin(convex_hull(pts), np.asarray(p_xy, dtype=np.float64))


def margins_over_time(corners_xy, down, *points) -> list[np.ndarray]:
    """Per sample, the signed margin of each ``points`` series (samples x 2) inside the support polygon: the hull of
    the sole corners of the feet down. ``corners_xy`` samples x feet x 4 x 2; ``down`` samples x feet (bool).
    NaN where no foot is down or the point is not finite."""
    n = len(down)
    out = [np.full(n, np.nan) for _ in points]
    for i in range(n):
        if not down[i].any():
            continue
        hull = convex_hull(corners_xy[i][down[i]].reshape(-1, 2))
        if hull.shape[0] < 3:
            continue
        for k, p in enumerate(points):
            if np.all(np.isfinite(p[i])):
                out[k][i] = signed_margin(hull, p[i])
    return out


# ---------------------------------------------------------------- signals
def lowpass_fft(x, dt: float, fc: float) -> np.ndarray:
    """Zero-phase brick-wall low-pass along axis 0 (FFT; the series' mean is kept)."""
    x = np.asarray(x, dtype=np.float64)
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], dt)
    X[f > fc] = 0.0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def moving_average(x, n: int) -> np.ndarray:
    """Centred moving average over ``n`` samples along axis 0 (edges use the samples available)."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 1:
        return x.copy()
    k = np.ones(n)
    if x.ndim == 1:
        return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")
    return np.stack([moving_average(x[:, i], n) for i in range(x.shape[1])], axis=1)


def dominant_frequency(x, dt: float, lo: float = 0.3, hi: float = 25.0) -> float:
    """Frequency (Hz) of the largest spectral peak of ``x`` (mean removed) between ``lo`` and ``hi``."""
    x = np.asarray(x, dtype=np.float64)
    f = np.fft.rfftfreq(x.size, dt)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    band = (f > lo) & (f < hi)
    return float(f[band][np.argmax(P[band])]) if band.any() else float("nan")


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def percentile_or_none(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else None


def stats(x) -> dict:
    """Median, 10th / 90th / 95th percentile, mean and count of a sample (None when empty)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "p95": None, "max": None}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def finite_mean(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else None
