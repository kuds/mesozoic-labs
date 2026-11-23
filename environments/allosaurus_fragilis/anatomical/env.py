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
    #: The digest snapshot's reward probe: the roll's action-noise amplitude,
    #: then the success probe's effector.  The success probe moves the free
    #: root COURSE_LENGTH_M along the spawn heading and takes one zero-action
    #: step, which completes the course on the locomotion stage only (until
    #: policy interface r2 it moved the mocap target onto the pelvis).
    REWARD_PROBE = (0.02, "root_displacement", "COURSE_LENGTH_M")
