"""Count complete swings from one foot's measured contact and terrain clearance."""

import numpy as np


def swing_events(loaded, clearance, *, minimum_clearance=0.005):
    """Return touchdown indices, peak swing gaps in metres, and shallow reload count.

    ``loaded`` is a Boolean time series after the caller's load threshold. A swing
    must start and end inside the window. Measure ``clearance`` from the lowest
    part of the foot to the local ground, including any raised surfaces.
    """
    loaded = np.asarray(loaded, dtype=bool)
    clearance = np.asarray(clearance, dtype=float)
    if loaded.ndim != 1 or clearance.shape != loaded.shape:
        raise ValueError("Contact and clearance must be matching 1D series")
    if not np.isfinite(clearance).all() or not np.isfinite(minimum_clearance) or minimum_clearance < 0:
        raise ValueError("Clearance values must be finite and the threshold nonnegative")
    edges = np.flatnonzero(np.diff(loaded)) + 1
    landings, peaks = [], []
    shallow = 0
    for start, end in zip(edges[:-1], edges[1:]):
        if not loaded[start]:
            peak = clearance[start:end].max()
            if peak > minimum_clearance:
                landings.append(end)
                peaks.append(peak)
            else:
                shallow += 1
    return np.asarray(landings, dtype=int), np.asarray(peaks, dtype=float), shallow
