"""The Compsognathus longipes robot (decision D-D33).

The twelve-servo robot on the species base
:class:`~environments.compsognathus_longipes.base_env.CompsognathusLongipesEnv`,
whose linear home-keyframe residual and world-frame observation it keeps.
``environments/__init__.py`` registers it with Gymnasium from the species
manifest.
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
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.01, "body", "pelvis")
