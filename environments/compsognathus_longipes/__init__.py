"""Compsognathus longipes: the species base environment, model API, data, scripts and tests.

Each embodiment (decision D-D33) is a subpackage holding its environment
module and MJCF: ``anatomical/`` and ``robot/``.  :data:`MODEL_PATHS` maps
each embodiment to its MJCF, built by ``ModelId.asset_path``, the one rule
the environment classes and the species manifest share (no hand-written
third copy).
"""

from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT

MODEL_PATHS = {
    embodiment: ModelId("compsognathus_longipes", embodiment).asset_path(ENVIRONMENTS_ROOT)
    for embodiment in ("anatomical", "robot")
}
