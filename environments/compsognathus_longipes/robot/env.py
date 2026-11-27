"""The Compsognathus longipes robot (decision D-D33).

The twelve-servo robot on the species base
:class:`~environments.compsognathus_longipes.base_env.CompsognathusLongipesEnv`,
whose linear home-keyframe residual and heading-free observation it reads
(``bipedal-heading-free/v1``, 41 values since policy interface r4; the
world-frame ``bipedal-target/v1`` layout through r3), with its two foot
channels on floor-contact force.  ``environments/__init__.py`` registers it with
Gymnasium from the species manifest.
"""

from __future__ import annotations

from environments.compsognathus_longipes.base_env import CompsognathusLongipesEnv
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT


class CompsognathusLongipesRobotEnv(CompsognathusLongipesEnv):
    """The 12-servo robot, with the original fixed head and unpowered tail."""

    embodiment = "robot"
    MODEL_PATH = ModelId("compsognathus_longipes", "robot").asset_path(ENVIRONMENTS_ROOT)
    nominal_pelvis_height = 0.218
    _camera_distance = 0.9
    #: The floor-truth foot registry entry (``gait/morphology.py``): the geoms
    #: a flat, loaded foot stands on, the sole and the foot's reference site,
    #: as name templates (``{side}`` is ``right`` / ``left``).  It drops the
    #: two ``{side}_roll_cheek_*`` geoms that share the rolling-foot body: a
    #: cheek on the floor is a rolled foot, not a supported one.
    FOOT_SUPPORT = {
        "support": ("{side}_sole",),
        "sole": "{side}_sole",
        "foot_site": "{side}_foot_touch_volume",
    }
    #: The observation's foot channels read each sole's normal force against
    #: the floor, not the touch sensors (policy interface r4): the 10 x 9 cm
    #: soles collide with each other, and a touch sensor counts a sole resting
    #: on the other foot as support (docs/KNOWN_ISSUES.md).  The soles are
    #: FOOT_SUPPORT's support geoms and their touch volumes enclose them, so
    #: the two readings differ only by force between the robot's own geoms:
    #: equal on every step of the zero-action statue (seeds 3042-3061, 1,000
    #: steps each) and on every step without such a contact under action
    #: noise.  The reward's support flag stays on the touch sensors
    #: (CompsognathusLongipesEnv._get_reward_info).
    FOOT_OBSERVATION_FLOOR_GEOMS = (("right_sole",), ("left_sole",))
    #: The digest snapshot's reward probe: the roll's action-noise amplitude.
    #: Amplitude only: the species declares no success key, so no success
    #: probe runs.
    REWARD_PROBE = (0.01,)
