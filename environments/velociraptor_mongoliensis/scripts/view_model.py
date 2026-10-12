#!/usr/bin/env python3
"""
Passive viewer for velociraptor MJCF iteration.

Usage (from the repository root):
    python -m environments.velociraptor_mongoliensis.scripts.view_model

Controls:
    - Mouse drag: rotate view
    - Scroll: zoom
    - Double-click: track body
    - Ctrl+drag: pan
    - Space: pause/unpause
    - Backspace: reset
    - Tab: toggle UI panels
"""

from pathlib import Path

from environments.shared.harnesses.viewer import ViewerConfig, view_model

if __name__ == "__main__":
    view_model(
        ViewerConfig(
            model_path=Path(__file__).parent.parent / "assets" / "velociraptor_mongoliensis_anatomical.xml",
            species_name="velociraptor",
            height_label="Pelvis height",
            camera_distance=2.0,
            camera_lookat_z=0.3,
        )
    )
