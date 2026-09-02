"""VecNormalize sidecar resolution in the offline diagnostic report scripts.

All three scripts guessed the sidecar as ``<stem>_vecnorm.pkl``, which can
never match ``CheckpointCallback(save_vecnormalize=True)``'s
``<prefix>_vecnormalize_<steps>_steps.pkl``, so every periodic checkpoint was
silently evaluated on raw observations (review OP7).  ``stance_gate_report``
treats a missing sidecar as fatal; these pin the same posture on its siblings,
behind one explicit ``--allow-unnormalized`` escape hatch.
"""

from __future__ import annotations

import inspect

import pytest

from environments.shared.scripts import (
    action_bound_report,
    joint_excursion_report,
    observation_ablation_report,
)

SCRIPTS = pytest.mark.parametrize(
    "script",
    [joint_excursion_report, action_bound_report, observation_ablation_report],
    ids=lambda module: module.__name__.rsplit(".", 1)[-1],
)


@SCRIPTS
def test_periodic_checkpoint_resolves_to_the_vecnormalize_sidecar(script, tmp_path):
    sidecar = tmp_path / "stage1_vecnormalize_500000_steps.pkl"
    sidecar.touch()
    model = tmp_path / "stage1_500000_steps.zip"
    assert script.resolve_vecnorm_path(str(model), None, False) == str(sidecar)


@SCRIPTS
def test_curated_checkpoint_still_resolves(script, tmp_path):
    sidecar = tmp_path / "robust_best_model_vecnorm.pkl"
    sidecar.touch()
    model = tmp_path / "robust_best_model.zip"
    assert script.resolve_vecnorm_path(str(model), None, False) == str(sidecar)


@SCRIPTS
def test_an_explicit_vecnorm_argument_wins(script, tmp_path):
    (tmp_path / "stage1_vecnormalize_500000_steps.pkl").touch()
    model = tmp_path / "stage1_500000_steps.zip"
    assert script.resolve_vecnorm_path(str(model), "/elsewhere/stats.pkl", False) == "/elsewhere/stats.pkl"


@SCRIPTS
def test_missing_sidecar_is_fatal_and_names_the_escape_hatch(script, tmp_path):
    model = tmp_path / "stage1_500000_steps.zip"
    with pytest.raises(SystemExit) as excinfo:
        script.resolve_vecnorm_path(str(model), None, False)
    message = str(excinfo.value)
    assert str(model) in message
    assert "--allow-unnormalized" in message
    assert "different policy" in message


@SCRIPTS
def test_allow_unnormalized_returns_none(script, tmp_path):
    model = tmp_path / "stage1_500000_steps.zip"
    assert script.resolve_vecnorm_path(str(model), None, True) is None


@SCRIPTS
def test_the_flag_and_banner_are_wired_into_main(script):
    """The helper is only protection if main() calls it and prints the banner."""
    source = inspect.getsource(script.main)
    assert "--allow-unnormalized" in source
    assert "resolve_vecnorm_path(args.model, args.vecnorm, args.allow_unnormalized)" in source
    assert "UNNORMALIZED_BANNER" in source
    assert "UNNORMALIZED EVAL" in script.UNNORMALIZED_BANNER
