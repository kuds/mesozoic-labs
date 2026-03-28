#!/usr/bin/env python3
"""
Passive viewer for Brachiosaurus MJCF iteration.

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
            model_path=Path(__file__).parent.parent / "assets" / "brachiosaurus.xml",
            species_name="brachiosaurus",
            height_label="Torso height",
            camera_distance=8.0,
            camera_lookat_z=1.5,
        )
    )
