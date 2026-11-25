"""Straight–arc–straight commands and horizontal path measurements for any moving body."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TurnCourse:
    """Signed radius in meters (left positive), and durations in simulation seconds."""

    radius_m: float
    entry_s: float
    turn_s: float
    exit_s: float

    def __post_init__(self):
        if not np.isfinite([self.radius_m, self.entry_s, self.turn_s, self.exit_s]).all():
            raise ValueError("Turn radius and durations must be finite")
        if self.radius_m == 0 or self.turn_s <= 0 or min(self.entry_s, self.exit_s) < 0:
            raise ValueError("Provide a nonzero radius, positive turn duration and nonnegative straight durations")

    @property
    def duration_s(self):
        return self.entry_s + self.turn_s + self.exit_s

    def curvature_at(self, time_s):
        """Command signed inverse meters. Walking speed belongs to the tested controller."""
        return 1.0 / self.radius_m if self.entry_s <= time_s < self.entry_s + self.turn_s else 0.0


def circular_path_metrics(xy, yaw_rad, radius_m):
    """Compare samples with the circle tangent to the first position and heading.

    Headings use radians counterclockwise from +X toward +Y. The radius is signed.
    The command defines the center before the trial. Fitting a circle to the result
    would conceal outward drift. Forward travel integrates displacement along the
    measured trunk heading to discount lateral body sway.
    """
    xy = np.asarray(xy, dtype=float)
    yaw = np.asarray(yaw_rad, dtype=float)
    if yaw.ndim != 1 or xy.shape != (len(yaw), 2) or len(yaw) < 2:
        raise ValueError("Provide at least two paired XY positions and headings")
    if not (np.isfinite(xy).all() and np.isfinite(yaw).all() and np.isfinite(radius_m) and radius_m != 0):
        raise ValueError("Positions, headings and the nonzero radius must be finite")
    yaw = np.unwrap(yaw)
    center = xy[0] + radius_m * np.array([-np.sin(yaw[0]), np.cos(yaw[0])])
    relative = xy - center
    radial_error = np.linalg.norm(relative, axis=1) - abs(radius_m)
    orbit = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    angle = orbit[-1] - orbit[0]
    turn = yaw[-1] - yaw[0]
    midpoint_yaw = (yaw[1:] + yaw[:-1]) / 2
    tangent = np.column_stack((np.cos(midpoint_yaw), np.sin(midpoint_yaw)))
    forward = (np.diff(xy, axis=0) * tangent).sum()
    return {
        "heading_change_deg": float(np.degrees(turn)),
        "swept_angle_deg": float(np.degrees(angle)),
        "projected_arc_distance_m": float(angle * radius_m),
        "forward_distance_m": float(forward),
        "measured_curvature_per_m": float(turn / forward) if abs(forward) > 1e-9 else None,
        "rms_radial_error_m": float(np.sqrt(np.mean(radial_error**2))),
        "max_radial_error_m": float(np.abs(radial_error).max()),
    }
