"""Stage 1 leg-to-leg proximity on simulated traces (``STAGE1_PLAN.md`` section 4, "no leg interference").

Only the foot pads collide in the plant, as in the screened model, and the plant's joint ranges no longer stop a leg
where a single-joint sweep from home found it touching the other leg (``plant.usable_ranges``).  The legs'
proximity is therefore measured on the traces, with the clearance screen's geometry and its verified distance oracle
(``clearance._Screen`` on the screened MJCF: the ``hw_*`` hardware of groups 0 and 2 without cosmetic parts, and the
``cl_*`` shell proxies), over every pair of a screened geom on a left-leg body and one on a right-leg body.

``LegGap.trace`` reports, over a sequence of poses, the smallest signed distance (negative: interference) of

* all such pairs, and
* the pairs that are not within ``clearance.REPORT_CAP`` (25 mm) of each other at the home pose, the ones that only
  come close in motion (feet and shanks passing).  On the 60 cm bodies the two hip brackets sit 4 mm apart at home
  and stay so at every pose, which would otherwise mask them;

with the pair, the pose index, and how many poses had interference (d < 0) or a tight gap (0 <= d <
``clearance.TIGHT``, 3 mm).  Distances beyond the cap are reported as the cap.  The poses are samples (the replays
take one every 20 ms), so a brief touch between samples can be missed.
"""

from __future__ import annotations

import numpy as np

from .. import clearance
from . import plant


class LegGap:
    """Left-leg against right-leg distances of one body's screened geometry (module docstring)."""

    def __init__(self, body: str, cap: float = clearance.REPORT_CAP):
        xml, _ = plant.model_files(body)
        S = clearance._Screen(xml)
        self.S, self.cap = S, float(cap)
        side = np.zeros(S.m.ngeom, int)
        for g in S.screened:
            name = S.body_of(int(g))
            side[g] = 1 if name.startswith("left_") else 2 if name.startswith("right_") else 0
        k = np.flatnonzero(side[S.I] * side[S.J] == 2)  # one geom on each leg
        self.I, self.J = S.I[k], S.J[k]
        self.home = self.at(S.m.key_qpos[0])
        self.moving = self.home >= self.cap

    def _distances(self) -> np.ndarray:
        """Signed distance of every leg pair at the current pose, capped: an AABB broad phase, then the oracle."""
        lo, hi = self.S.aabb()
        a, b, cap = self.I, self.J, self.cap
        near = np.all((lo[a] - cap <= hi[b]) & (lo[b] - cap <= hi[a]), axis=1)
        out = np.full(len(a), cap)
        for k in np.flatnonzero(near):
            out[k] = min(self.S.dist(int(a[k]), int(b[k]), cap), cap)
        return out

    def at(self, qpos) -> np.ndarray:
        """Distances of every leg pair at the pose ``qpos`` (the plant's and the screened model's qpos agree)."""
        self.S.set_q(qpos)
        return self._distances()

    def pair(self, k: int) -> list[str]:
        return [self.S.gname[int(self.I[k])], self.S.gname[int(self.J[k])]]

    def per_pose(self, frames) -> np.ndarray:
        """Per pose of ``frames``: (smallest gap, its pair, smallest gap of the moving pairs, its pair); pair -1 when
        nothing is within the cap."""
        mov = np.flatnonzero(self.moving)
        out = np.empty((len(frames), 4))
        for i, q in enumerate(frames):
            d = self.at(q)
            k = int(np.argmin(d))
            km = int(mov[np.argmin(d[mov])]) if len(mov) else -1
            out[i] = (d[k], k if d[k] < self.cap else -1, d[km] if km >= 0 else self.cap,
                      km if km >= 0 and d[km] < self.cap else -1)
        return out

    def summary(self, rows: np.ndarray) -> dict:
        """Smallest gaps (mm) over the rows of ``per_pose`` (module docstring)."""
        out = dict(poses=len(rows), cap_mm=1e3 * self.cap,
                   interference_poses=int(np.sum(rows[:, 0] < 0.0)) if len(rows) else 0,
                   tight_poses=int(np.sum((rows[:, 0] >= 0.0) & (rows[:, 0] < clearance.TIGHT))) if len(rows) else 0)
        out["interference_free"] = out["interference_poses"] == 0
        for key, col in (("all", 0), ("moving", 2)):
            i = int(np.argmin(rows[:, col])) if len(rows) else None
            k = -1 if i is None else int(rows[i, col + 1])
            out[f"min_gap_{key}_mm"] = 1e3 * (self.cap if i is None else float(rows[i, col]))
            out[f"min_gap_{key}_pair"] = None if k < 0 else self.pair(k)
            out[f"min_gap_{key}_pose"] = None if k < 0 else i
        out["home_pairs_within_cap"] = [dict(pair=self.pair(k), home_gap_mm=1e3 * float(self.home[k]))
                                        for k in np.flatnonzero(~self.moving)]
        return out

    def trace(self, frames) -> dict:
        """Smallest gaps over the poses ``frames`` (module docstring), in mm."""
        return self.summary(self.per_pose(frames))
