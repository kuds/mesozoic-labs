"""The anatomical Brachiosaurus altithorax (decision D-D33).

Everything the species' embodiments share is
:class:`~environments.brachiosaurus_altithorax.base_env.BrachiosaurusAltithoraxEnv`;
this class adds what is this model's own.  ``environments/__init__.py``
registers it with Gymnasium from the species manifest, and the frozen MJX
registration (D-D17) is ``mjx_config.py`` beside it.
"""

from __future__ import annotations

from environments.brachiosaurus_altithorax.base_env import BrachiosaurusAltithoraxEnv
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT


class BrachiosaurusAltithoraxAnatomicalEnv(BrachiosaurusAltithoraxEnv):
    """Brachiosaurus Altithorax, the anatomical model."""

    embodiment = "anatomical"
    MODEL_PATH = ModelId("brachiosaurus_altithorax", "anatomical").asset_path(ENVIRONMENTS_ROOT)
    #: The floor-truth foot registry entry (``gait/morphology.py``): the geoms
    #: a flat, loaded foot stands on, the sole and the foot's reference site,
    #: as name templates.  It drops ``{s}_meta_geom``, the metapodial capsule
    #: in the touch group, which carries 0 load in the statue (at most 3.2e-4
    #: of the floor load, grazing; 40-episode means, seeds 3042-3081).  Its
    #: ellipsoid sole is authored tilted 19.2 degrees (fore) and 15.8 degrees
    #: (hind), so its flatness keys must be the ``*_excess_*`` forms measured
    #: against that keyframe tilt.
    FOOT_SUPPORT = {
        "support": ("{s}_foot_geom",),
        "sole": "{s}_foot_geom",
        "foot_site": "{s}_foot_contact",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.2, "site", "head_tip")
