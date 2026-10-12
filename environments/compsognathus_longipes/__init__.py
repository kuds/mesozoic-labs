"""Compsognathus anatomical/robot models and SB3 training environments."""

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATHS = {
    "anatomical": ASSET_DIR / "compsognathus_longipes_anatomical.xml",
    "robot": ASSET_DIR / "compsognathus_longipes_robot.xml",
}
