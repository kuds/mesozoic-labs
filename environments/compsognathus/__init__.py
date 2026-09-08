"""Experimental Compsognathus MJCF assets; not a registered training environment."""

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATHS = {
    "biological": ASSET_DIR / "compsognathus.xml",
    "robot": ASSET_DIR / "compsognathus_robot.xml",
}
