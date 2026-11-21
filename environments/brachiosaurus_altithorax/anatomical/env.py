"""The anatomical Brachiosaurus altithorax (decision D-D33).

Everything the species' embodiments share is
:class:`~environments.brachiosaurus_altithorax.base_env.BrachiosaurusAltithoraxEnv`;
this class adds what is this model's own.  ``environments/__init__.py``
registers it with Gymnasium from the species manifest.
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
    #: as name templates. The metapodial capsule can touch obstacles. On level
    #: ground the flat manus and pes are supported by their sole ellipsoids.
    FOOT_SUPPORT = {
        "support": ("{s}_foot_geom",),
        "sole": "{s}_foot_geom",
        "foot_site": "{s}_foot_contact",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.2, "site", "head_tip")
