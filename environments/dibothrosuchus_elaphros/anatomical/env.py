"""The anatomical Dibothrosuchus elaphros (decision D-D33).

Everything the species' embodiments share is
:class:`~environments.dibothrosuchus_elaphros.base_env.DibothrosuchusElaphrosEnv`;
this class adds what is this model's own.  ``environments/__init__.py``
registers it with Gymnasium from the species manifest.
"""

from __future__ import annotations

from environments.dibothrosuchus_elaphros.base_env import DibothrosuchusElaphrosEnv
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT


class DibothrosuchusElaphrosAnatomicalEnv(DibothrosuchusElaphrosEnv):
    """Dibothrosuchus elaphros, the anatomical model."""

    embodiment = "anatomical"
    MODEL_PATH = ModelId("dibothrosuchus_elaphros", "anatomical").asset_path(ENVIRONMENTS_ROOT)
    #: The floor-truth foot registry entry (``gait/morphology.py``): the geoms
    #: a flat, loaded foot stands on, the sole and the foot's reference site,
    #: as name templates. The four distal digital pads share a single toe hinge.
    FOOT_SUPPORT = {
        "support": ("{s}_foot_geom", "{s}_digit_1", "{s}_digit_2", "{s}_digit_3", "{s}_digit_4"),
        "sole": "{s}_foot_geom",
        "foot_site": "{s}_foot_contact",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.2, "geom", "snout_snap")
