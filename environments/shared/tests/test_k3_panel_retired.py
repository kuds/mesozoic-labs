"""The in-run K3 noise panel is retired (the maintainer's decision of 2026-11-26): no module, launch option or reader.

Decision D-D48 added the panel (``environments/shared/k3_panel.py``) as a report-only readout for the first r10
T. rex stance round; that purpose was met, and the offline judges' K3 stays the K3 of record
(``docs/TRAINING_ARTIFACT_PRESENTATION.md``, "The in-run K3 noise panel (retired)").  These pins fail if the
module, ``train()``'s ``k3_panel`` keyword, the command line's ``--k3-panel`` flags, the notebook's ``K3_PANEL``
settings or the training history's reader of ``k3_panel.json`` come back.  Needs no SB3.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.model_id import ModelId

from .notebook_cells import cell_sources

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
#: The retired names, and only these: a later K3-style readout under another name is not the panel.
RETIRED_TRAINER_NAMES = ("k3_panel", "K3Panel", "roll_final", "hold_restored_normalization")
NOTEBOOK_PATH = Path(__file__).resolve().parents[3] / "notebooks" / "sb3_training.ipynb"
#: The record a panel run wrote (``mesozoic.k3-panel/v1``), as the pre-retirement r10 screen runs keep it.
PANEL_RECORD = {
    "schema": "mesozoic.k3-panel/v1",
    "spec": {"seeds": list(range(3042, 3062)), "horizon": 1000, "every": 1_048_576, "joint_sigma": {}},
    "panels": [
        {
            "panel": 0,
            "timesteps": 2048,
            "kind": "grid",
            "failure": None,
            "tests": {"own": {"n": 20, "falls": 0}, "joint": {"n": 20, "falls": 1}},
        }
    ],
}


def test_the_panel_module_is_gone():
    assert importlib.util.find_spec("environments.shared.k3_panel") is None


def test_the_trainer_takes_no_k3_panel_option():
    """``train()``, the stage body and the callback builder take no panel parameter; nothing rolls a final panel or
    holds a restored normalization for one."""
    from environments.shared import train_base

    def retired(name: str) -> bool:
        return any(part in name for part in RETIRED_TRAINER_NAMES)

    for function in (train_base.train, train_base._train_stage_body, train_base._build_core_callbacks):
        assert not [name for name in inspect.signature(function).parameters if retired(name)], function.__name__
    assert not [name for name in vars(train_base) if retired(name)]
    source = inspect.getsource(train_base)
    assert "roll_final_panel" not in source and "hold_restored_normalization" not in source


@pytest.mark.parametrize("flag", ["--k3-panel", "--k3-panel-sigma-file=sigma.json", "--k3-panel-sigma-key=80"])
def test_the_command_line_refuses_the_retired_flags(capsys, flag):
    """argparse refuses each flag before ``train()`` is reached, as it refuses the retired GCS flags."""
    from environments.shared.cli import main

    species_cfg = MagicMock()
    species_cfg.model_id = TYRANNOSAURUS_REX
    mock_load = MagicMock(return_value={n: {"env_kwargs": {}, "ppo_kwargs": {}} for n in (1, 2, 3)})
    with (
        patch("environments.shared.config.load_all_stages", mock_load),
        patch("environments.shared.train_base.train") as mock_train,
        patch("sys.argv", ["prog", "train", "--stage", "1", "--timesteps", "1000", flag]),
        pytest.raises(SystemExit) as excinfo,
    ):
        main(species_cfg)
    assert excinfo.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err
    mock_train.assert_not_called()


def test_the_notebook_has_no_k3_panel_setting():
    """No cell, code or markdown, sets, reads or describes the panel."""
    for index, source in enumerate(cell_sources(NOTEBOOK_PATH)):
        assert "K3_PANEL" not in source and "k3_panel" not in source, f"cell {index}"
        assert "in-run K3" not in source, f"cell {index}"


def test_the_training_history_does_not_read_a_pre_retirement_panel_file(tmp_path):
    """A stage directory that still holds ``k3_panel.json`` reads as one without it: no ``k3_panel`` block, no
    absent or error line, no panel column in the checkpoint table and nothing on page 17 but the judge's."""
    pytest.importorskip("matplotlib")
    from environments.shared.reporting import training_history as th

    from .test_training_history import _stage

    stage = _stage(tmp_path)
    (stage / "k3_panel.json").write_text(json.dumps(PANEL_RECORD), encoding="utf-8")
    run = th.load_run_history(stage, read_std=False)
    record = th.run_record(run, [2048])
    assert "k3_panel" not in record and record["judge_k3"] is None
    # The retired reader's lines were "in-run K3 panel: ..." (absent) and "K3 panel at ...: ..." (errors).
    assert not any("K3 panel" in entry or "k3_panel" in entry for entry in [*record["absent"], *record["errors"]])
    rows = record["checkpoint_table"]
    assert rows and not any(key.startswith("panel_") for row in rows for key in row)  # panel_joint/own_falls
    assert record["exploration"].get("leg_std_at_k3", []) == []
    falls = th._exploration_figure(th._make_context([run], marks=[], smooth=5), title="t", dpi=30, width=12.0).axes[1]
    assert not [line for line in falls.get_lines() if not line.get_label().startswith("_")]  # the marks' only
    assert [text.get_text() for text in falls.texts] == ["no K3 source: no judge files named (--judged)"]
    assert not [name for name in vars(th) if name.startswith("K3_PANEL") or name == "read_k3_panel"]
