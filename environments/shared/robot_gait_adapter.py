"""Apply calibrated phase corrections to the 57-sensor, 14-action walking interface.

Coefficients are fit on training seeds, then frozen in the controller bundle.
The scale applies to the learned correction carried by the PPO actor itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def adapt(obs: np.ndarray, action: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    """Apply six coefficients (roll, yaw, ankle, lift, slow ankle, slow lift)."""
    if np.shape(obs) != (57,) or np.shape(action) != (14,) or len(coefficients) != 6:
        raise ValueError("Walking correction requires one 57-value sensor packet, 14 actions and six coefficients")
    if not np.isfinite(coefficients).all():
        raise ValueError("Walking correction coefficients must be finite")
    roll, yaw, ankle, lift, slow_ankle, slow_lift = coefficients
    wave, cosine = obs[36:38]
    forward, _, turn = obs[-3:]
    activity = np.clip(max(forward / 0.08, abs(turn) / 0.20), 0, 1)
    relief = 1.0 - np.clip(float(forward) / 0.12, 0.0, 1.0) ** 4
    ankle += relief * slow_ankle
    lift += relief * slow_lift
    result = np.array(action).copy()
    for offset, side in ((0, 1), (6, -1)):
        result[offset] += activity * yaw * (turn / 0.5) * (-cosine * side)
        result[offset + 1] += activity * roll * wave
        result[offset + 5] -= activity * roll * wave
    # The fit evaluated the saturated phase correction before the slow-speed
    # correction. Preserving both saturations reproduces that controller.
    np.clip(result, -1, 1, out=result)
    for offset, side in ((0, 1), (6, -1)):
        result[offset + 5] += activity * ankle * wave
        knee = activity * lift * max(0.0, side * wave)
        result[offset + 3] += knee
        result[offset + 4] -= 1.25 * knee
    return np.clip(result, -1, 1)
