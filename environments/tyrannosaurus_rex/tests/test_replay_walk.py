"""``scripts/replay_walk.py`` replays an archived Pilot4 walking stage (physics r10, policy interface r15, 63
observations), whose recorded config passes the bite reward's and the prey's arguments.  The environment dropped
those arguments (``e3cccca`` and ``configs/plant_versions.toml`` note 24) and observes 59 values since note 24, so on
this checkout the script must stop at once, before it reads a stage, and name the last commit it ran at."""

from __future__ import annotations

from pathlib import Path

import pytest

from environments.tyrannosaurus_rex.scripts import replay_walk

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_replay_refuses_this_checkout_and_names_its_last_commit(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        replay_walk.main([str(tmp_path / "no_stage"), str(tmp_path / "no_plant.xml")])
    assert stopped.value.code == 2
    message = capsys.readouterr().err
    assert "refuses ['bite_bonus', 'prey_distance_range'] and observes 59" in message
    assert "It last ran at commit 3368abe; run it in a checkout of 3368abe or earlier" in message
    with pytest.raises(RuntimeError, match="last ran at commit 3368abe"):
        replay_walk.replay(tmp_path / "no_stage", tmp_path / "no_plant.xml", [42])


def test_the_readme_names_the_same_commit():
    assert f"runs in a checkout of `{replay_walk.LAST_RUNNABLE_COMMIT}` or earlier" in README.read_text(
        encoding="utf-8"
    )
