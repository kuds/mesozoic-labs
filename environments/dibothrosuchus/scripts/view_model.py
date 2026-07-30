#!/usr/bin/env python3
"""
Passive viewer for Dibothrosuchus MJCF iteration.

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

from environments.shared.harnesses.viewer import ViewerConfig, view_model

if __name__ == "__main__":
    view_model(
        ViewerConfig(
            model_path=Path(__file__).parent.parent / "assets" / "dibothrosuchus.xml",
            species_name="dibothrosuchus",
            height_label="Trunk height",
            camera_distance=1.8,
            camera_lookat_z=0.25,
        )
    )
