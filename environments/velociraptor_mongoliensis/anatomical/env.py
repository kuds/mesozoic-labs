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
    #: as name templates.  Since physics r5 each foot stands on its MTP pad
    #: (on the metatarsus) and its passive digits III and IV, each with its
    #: own touch site, so the generic rule finds all three, and with them the
    #: metatarsus shaft capsule and digit II's sickle claw, which hang off the
    #: touch-site bodies and are dropped here: the shaft ends 8 mm short of
    #: the MTP condyle and sits 14.4 mm clear of the floor at home, and the
    #: claw is contype/conaffinity 2/2 against the floor's 1/1, so neither
    #: ever touches the floor (digit II's phalanges are visual, contype 0).
    #: The statue stands on pad / d3 / d4 at 0.300 / 0.357 / 0.343 of each
    #: foot's load (40-episode means over steps 100-999, the stance config,
    #: seeds 3042-3081; r4 stood on d3 / d4 / metatarsus at 0.540 / 0.176 /
    #: 0.284), every site loaded on every one of those steps (at least
    #: 3.1 N).  No sole: the pad is a rounded
    #: capsule, as the animal's ("a large, rounded metatarsophalangeal pad",
    #: Xing et al. 2013), not a box, so the raptor keeps the padless paths
    #: (support-geom duty and coverage, no sole flatness or box centre of
    #: pressure).  The reference site is ``{s}_foot``, the pad's, which leads
    #: the touch group.
    FOOT_SUPPORT = {
        "support": ("{s}_mtp_pad", "{s}_toe_d3_geom", "{s}_toe_d4_geom"),
        "sole": None,
        "foot_site": "{s}_foot",
    }
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: and the MJCF element (kind, name) the success probe moves the target onto.
    #: The amplitude is half the largest that keeps every stage up for the
    #: roll's 300 steps (reset seed 1042, action seed 7; no stage pushes):
    #: 0.26 on the physics-r5 plant, where 0.28 nosedives at step 154 on all
    #: three stages (amplitudes 0.02-0.60 by 0.02), so 0.13 (r4: 0.1).
    REWARD_PROBE = (0.13, "geom", "r_claw_geom")
