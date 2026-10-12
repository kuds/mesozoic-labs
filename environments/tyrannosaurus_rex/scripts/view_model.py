#!/usr/bin/env python3
"""
Passive viewer for T-Rex MJCF model.

Usage (from the repository root):
    python -m environments.tyrannosaurus_rex.scripts.view_model --embodiment anatomical

Controls:
    - Mouse drag: rotate view
    - Scroll: zoom
    - Double-click: track body
    - Ctrl+drag: pan
    - Space: pause/unpause
    - Backspace: reset
    - Tab: toggle UI panels
"""

import argparse

from environments.shared.harnesses.viewer import ViewerConfig, view_model
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.shared.species_names import embodiments_of

SPECIES = "tyrannosaurus_rex"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embodiment", required=True, choices=embodiments_of(SPECIES), help="The model to view")
    model_id = ModelId(SPECIES, parser.parse_args().embodiment)
    view_model(
        ViewerConfig(
            model_path=model_id.asset_path(ENVIRONMENTS_ROOT),
            species_name=str(model_id),
            height_label="Pelvis height",
            camera_distance=5.0,
            camera_lookat_z=0.5,
        )
    )
