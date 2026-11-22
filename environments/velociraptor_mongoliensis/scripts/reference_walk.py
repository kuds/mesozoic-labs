"""Straight, slow reference walk for the anatomical Velociraptor on level ground.

Construct ReferenceWalk after env.reset(). Pass action(env.data, env.dt) to
env.step(). Only a private kinematics workspace receives prescribed positions.
The live root, passive digits and all floor contacts evolve in MuJoCo.
Run the species' audit_motion module with --mode walk to replay and measure it.
"""

from environments.shared.gait.biped_walk import BipedReferenceWalk

PERIOD = 0.70
STRIDE = 0.18
STANCE_FRACTION = 0.65
SWING_HEIGHT = 0.032
LATERAL_SWAY = 0.028
WALKING_CROUCH = 0.012
MEDIAL_STEP = 0.020
START = 0.80
SETTLE = 1.0


class ReferenceWalk(BipedReferenceWalk):
    """Straight gait for the raptor's two weight-bearing digits."""

    period = PERIOD
    stride = STRIDE
    stance_fraction = STANCE_FRACTION
    swing_height = SWING_HEIGHT
    lateral_sway = LATERAL_SWAY
    walking_crouch = WALKING_CROUCH
    medial_step = MEDIAL_STEP
    start = START
    settle = SETTLE
    pad_suffix = "mtp_pad"
    cop_min = (-0.010, -0.010)
    cop_max = (0.105, 0.010)
