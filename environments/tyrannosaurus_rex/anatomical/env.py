"""The anatomical Tyrannosaurus rex (decision D-D33).

Everything the species' embodiments share is
:class:`~environments.tyrannosaurus_rex.base_env.TyrannosaurusRexEnv`; this
class adds what is this model's own.  ``environments/__init__.py`` registers
it with Gymnasium from the species manifest.
"""

from __future__ import annotations

from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.tyrannosaurus_rex.base_env import TyrannosaurusRexEnv


class TyrannosaurusRexAnatomicalEnv(TyrannosaurusRexEnv):
    """Tyrannosaurus Rex, the anatomical model."""

    embodiment = "anatomical"
    MODEL_PATH = ModelId("tyrannosaurus_rex", "anatomical").asset_path(ENVIRONMENTS_ROOT)
    #: The floor-truth foot registry entry (``gait/morphology.py``): the geoms
    #: a flat, loaded foot stands on, the sole and the foot's reference site,
    #: as name templates.  It drops ``{s}_metatarsus_geom``, the capsule above
    #: the plantar box: it carries 0 load in the statue (40-episode means,
    #: seeds 3042-3081), and as a REQUIRED support geom it would fail every
    #: statue's coverage.
    FOOT_SUPPORT = {
        "support": ("{s}_plantar_geom", "{s}_toe_d2_geom", "{s}_toe_d3_geom", "{s}_toe_d4_geom"),
        "sole": "{s}_plantar_geom",
        "foot_site": "{s}_foot",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.1, "geom", "head_bite")
