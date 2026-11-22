"""The interactive viewer should start in the stance used for the balance measurements."""

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import mujoco.viewer
import numpy as np


def test_viewer_reports_animal_and_uses_the_standing_controls(monkeypatch, capsys):
    def capture(model, data):
        np.testing.assert_array_equal(data.ctrl, model.key("home").ctrl)
        np.testing.assert_array_equal(data.qpos, model.key("home").qpos)
        console = capsys.readouterr().out
        assert "Animal mass: 15.00 kg" in console
        x, y, z = data.body("pelvis").subtree_com
        assert f"Initial CoM: x={x:.3f}, y={y:.3f}, z={z:.3f}" in console
        handle = MagicMock()
        handle.__enter__.return_value.is_running.return_value = False
        return handle

    script = Path(__file__).resolve().parents[1] / "scripts" / "view_model.py"
    monkeypatch.setattr(mujoco.viewer, "launch_passive", capture)
    monkeypatch.setattr(sys, "argv", [str(script), "--embodiment", "anatomical"])
    runpy.run_path(str(script), run_name="__main__")
