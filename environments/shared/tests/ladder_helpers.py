"""A three-stage advancing ladder for the curriculum-mechanics tests.

Since decision D-D59 removed the hunt nodes, every committed manifest ends its
advancing ladder at locomotion (legacy 2), so no committed chain has a node
between the root and an advancing target, and no committed target has an
advancing descendant.  The tests that pin how the command-line curriculum
walks a ladder (reuse of every ancestor but the target, ``--retrain-from``
covering a node and its descendants, a handoff two nodes deep, the manager
advanced once per node) run on a copy of the committed configs with one more
advancing node appended: ``behavior``, legacy number 3 (the only id that
number may carry, ``stage_manifest.LEGACY_STAGE_IDS``), warm-started from
locomotion, labelled :data:`LADDER_LEAF_LABEL` and configured by a copy of
the model's ``locomotion.toml``.  It is a stand-in for a third advancing
stage, not the removed hunt: no hunt reward, label or gate is restored.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

#: The recipe label of the appended node (``hunt`` is retired, decision D-D59).
LADDER_LEAF_LABEL = "sprint"
#: The appended node's id and legacy number.
LADDER_LEAF_ID = "behavior"
LADDER_LEAF_NUMBER = 3

_LEAF_ENTRY = (
    "\n[[stages]]                       # test fixture: a third advancing stage (ladder_helpers)\n"
    f'id = "{LADDER_LEAF_ID}"\n'
    f'config = "{LADDER_LEAF_ID}.toml"\n'
    f"legacy_number = {LADDER_LEAF_NUMBER}\n"
    'warm_start_from = "locomotion"\n'
    f'recipe = "{LADDER_LEAF_LABEL}"\n'
    "deliverable = true\n"
)


def install_three_stage_ladder(configs_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the committed ``configs/`` to *configs_root*, append the third advancing node to every model's manifest,
    and point both readers of the configs root (the stage manifest and the stage config loader) at the copy.

    Returns *configs_root*.  Every model gets the node, so a test may use any of them.
    """
    from environments.shared import config as config_module
    from environments.shared import stage_manifest as stage_manifest_module

    source = stage_manifest_module._CONFIGS_DIR
    shutil.copytree(source, configs_root, dirs_exist_ok=True)
    for manifest in sorted(configs_root.glob("*/*/stages.toml")):
        model_dir = manifest.parent
        shutil.copyfile(model_dir / "locomotion.toml", model_dir / f"{LADDER_LEAF_ID}.toml")
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(_LEAF_ENTRY)
    monkeypatch.setattr(stage_manifest_module, "_CONFIGS_DIR", configs_root)
    monkeypatch.setattr(config_module, "_CONFIGS_DIR", configs_root)
    return configs_root
