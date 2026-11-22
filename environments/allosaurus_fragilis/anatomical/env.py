"""Registered Gymnasium environment for the anatomical Allosaurus fragilis."""

from environments.allosaurus_fragilis.base_env import AllosaurusFragilisEnv
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT


class AllosaurusFragilisAnatomicalEnv(AllosaurusFragilisEnv):
    embodiment = "anatomical"
    MODEL_PATH = ModelId("allosaurus_fragilis", embodiment).asset_path(ENVIRONMENTS_ROOT)
    FOOT_SUPPORT = {
        "support": ("{s}_pad", "{s}_pad_d2", "{s}_pad_d3", "{s}_pad_d4"),
        "sole": "{s}_pad",
        "foot_site": "{s}_foot",
    }
    REWARD_PROBE = (0.02, "body", "pelvis")
