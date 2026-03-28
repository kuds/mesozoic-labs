#!/usr/bin/env python3
"""
Passive viewer for T-Rex MJCF model.

Usage:
    python view_model.py

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

from environments.shared.view_model_base import ViewerConfig, view_model

if __name__ == "__main__":
    view_model(
        ViewerConfig(
            model_path=Path(__file__).parent.parent / "assets" / "trex.xml",
            species_name="tyrannosaurus_rex",
            height_label="Pelvis height",
            camera_distance=5.0,
            camera_lookat_z=0.5,
        )
    )
