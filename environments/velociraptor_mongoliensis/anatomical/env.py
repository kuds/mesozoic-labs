"""The anatomical Velociraptor mongoliensis (decision D-D33).

Everything the species' embodiments share is
:class:`~environments.velociraptor_mongoliensis.base_env.VelociraptorMongoliensisEnv`;
this class adds what is this model's own.  ``environments/__init__.py``
registers it with Gymnasium from the species manifest.
"""

from __future__ import annotations

from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.velociraptor_mongoliensis.base_env import VelociraptorMongoliensisEnv


class VelociraptorMongoliensisAnatomicalEnv(VelociraptorMongoliensisEnv):
    """Velociraptor Mongoliensis, the anatomical model."""

    embodiment = "anatomical"
    MODEL_PATH = ModelId("velociraptor_mongoliensis", "anatomical").asset_path(ENVIRONMENTS_ROOT)
    #: The floor-truth foot registry entry (``gait/morphology.py``): the geoms
    #: a flat, loaded foot stands on, the sole and the foot's reference site,
    #: as name templates.  It drops ``{s}_claw_geom``, digit II's sickle claw.
    #: Since physics r3 the metatarsus, digit III and digit IV each carry a
    #: touch site, so the generic rule finds all three support geoms, and with
    #: them the claw that hangs off the metatarsus: it is contype/conaffinity
    #: 2/2 against the floor's 1/1, so it never touches the floor and is not
    #: support.  The statue stands on d3 / d4 / metatarsus at 0.540 / 0.176 /
    #: 0.284 of each foot's load (40-episode means, seeds 3042-3081).  It has
    #: no sole (capsules only).  The reference site is pinned to ``{s}_foot``,
    #: digit III's, which leads the touch group.  The entry predates the r3
    #: sensors, when digit III's site was the foot's only one and the generic
    #: rule missed digit IV and the metatarsus; pinning the support set and the
    #: site then is why the r3 sensors moved neither.
    FOOT_SUPPORT = {
        "support": ("{s}_toe_d3_geom", "{s}_toe_d4_geom", "{s}_metatarsus_geom"),
        "sole": None,
        "foot_site": "{s}_foot",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    REWARD_PROBE = (0.1, "geom", "r_claw_geom")
