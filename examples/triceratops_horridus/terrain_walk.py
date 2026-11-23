"""Ground-aware foot placement for the anatomical Triceratops reference walk."""

import numpy as np

from .reference_walk import CONTROL_DT, FEET


class TerrainFootPlacement:
    """Use a known height map to modify targets for the sixteen existing leg motors."""

    def __init__(self, course):
        self.course = course
        self.reset()

    def reset(self):
        self.lateral_error = 0.0
        self.unloading = np.zeros(len(FEET))

    @staticmethod
    def support_shares(shares, support):
        # Keep a planned stance foot involved in weight transfer on a bank.
        # These are feedforward shares before normalization, not contact forces.
        shares[support] = np.maximum(shares[support], 0.08)

    def adapt(self, walk, data, targets, sole_pitch, swing, amplitude, body_sway):
        trunk_rotation = data.xmat[walk.trunk].reshape(3, 3)
        origin = data.xpos[walk.trunk]
        foot_xy = (targets @ trunk_rotation.T + origin)[:, :2]
        forward_xy = trunk_rotation[:2, 0] / np.linalg.norm(trunk_rotation[:2, 0])
        # Sample ahead of and behind each planned sole near the load-bearing pads.
        half_length = 0.14
        offset = half_length * forward_xy
        sample_xy = np.stack([foot_xy, foot_xy + offset, foot_xy - offset])
        height, toe, heel = self.course.height(sample_xy[:, :, 0], sample_xy[:, :, 1])

        # The trunk can follow the overall grade. The legs take up differences
        # between individual footholds and the plane through the planned feet.
        design = np.column_stack((targets[:, :2], np.ones(len(targets))))
        grade = np.linalg.lstsq(design, height, rcond=None)[0]
        fore_aft = targets[:, 0] - walk.home[:, 0]
        # Steer the next step toward the intended track. Filtering keeps the
        # normal lateral transfer from being mistaken for sideways drift.
        error = origin[1]
        if walk.path_xy is not None:
            normal = np.array([-np.sin(walk.path_heading), np.cos(walk.path_heading)])
            error = (origin[:2] - walk.path_xy) @ normal
        self.lateral_error += (CONTROL_DT / 0.7) * (error - self.lateral_error)
        targets[:, 1] -= 0.6 * self.lateral_error * fore_aft

        targets[:, 2] += 0.8 * (height - design @ grade)
        targets[:, 2] += 0.010 * amplitude * np.sin(np.pi * swing) ** 2
        local_slope = (toe - heel) / (2 * half_length)
        sole_pitch += 0.6 * np.clip(grade[0] - local_slope, -0.14, 0.14)

        # Move the support base aft on climbs and slightly forward on steep descents.
        targets[:, 0] -= 0.55 * max(grade[0], 0.0)
        targets[:, 0] -= 0.2 * min(0.0, grade[0] + np.tan(np.radians(6.0)))

        # A strong bank tends to unload the uphill forefoot. Allow the trunk
        # to follow mild banks, then shorten the uphill legs and extend the
        # downhill ones to keep the body closer to level.
        bank_excess = max(abs(grade[1]) - np.tan(np.radians(3.5)), 0.0)
        targets[:, 2] += 1.5 * np.sign(grade[1]) * bank_excess * walk.home[:, 1]
        targets[:, 1] += (0.035 / 0.145) * body_sway

        # Re-seat a stance pad that has unloaded on the uneven surface. The
        # 16 mm limit and 80 ms response avoid a discontinuity at toe-off.
        load = np.array([data.sensor(f"{foot}_sole_force").data[0] for foot in FEET]) / walk.weight
        deficit = np.clip(1 - load / 0.07, 0, 1) * (swing == 0) * amplitude
        self.unloading += (CONTROL_DT / 0.08) * (deficit - self.unloading)
        targets[:, 2] -= 0.016 * self.unloading
