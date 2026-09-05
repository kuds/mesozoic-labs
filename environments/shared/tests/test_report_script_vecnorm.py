"""VecNormalize sidecar resolution in the offline diagnostic report scripts.

All three scripts guessed the sidecar as ``<stem>_vecnorm.pkl``, which can
never match ``CheckpointCallback(save_vecnormalize=True)``'s
``<prefix>_vecnormalize_<steps>_steps.pkl``, so every periodic checkpoint was
silently evaluated on raw observations (review OP7).  ``stance_gate_report``
treats a missing sidecar as fatal; these pin the same posture on its siblings,
behind one explicit ``--allow-unnormalized`` escape hatch.

The resolution and the checkpoint load now live once, in
:mod:`environments.shared.policy_loading`; the wiring tests pin that each
script still calls the loader fail-closed and turns its refusal into a clean
exit rather than a traceback.
"""

from __future__ import annotations

import inspect

import pytest

from environments.shared import policy_loading
from environments.shared.policy_loading import PolicyLoadError, load_sb3_checkpoint, resolve_vecnorm_path
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


def test_periodic_checkpoint_resolves_to_the_vecnormalize_sidecar(tmp_path):
    sidecar = tmp_path / "stage1_vecnormalize_500000_steps.pkl"
    sidecar.touch()
    model = tmp_path / "stage1_500000_steps.zip"
    assert resolve_vecnorm_path(str(model), None, False) == str(sidecar)


def test_curated_checkpoint_still_resolves(tmp_path):
    sidecar = tmp_path / "robust_best_model_vecnorm.pkl"
    sidecar.touch()
    model = tmp_path / "robust_best_model.zip"
    assert resolve_vecnorm_path(str(model), None, False) == str(sidecar)


def test_an_explicit_vecnorm_argument_wins(tmp_path):
    (tmp_path / "stage1_vecnormalize_500000_steps.pkl").touch()
    model = tmp_path / "stage1_500000_steps.zip"
    assert resolve_vecnorm_path(str(model), "/elsewhere/stats.pkl", False) == "/elsewhere/stats.pkl"


def test_missing_sidecar_is_fatal_and_names_the_escape_hatch(tmp_path):
    model = tmp_path / "stage1_500000_steps.zip"
    with pytest.raises(PolicyLoadError) as excinfo:
        resolve_vecnorm_path(str(model), None, False)
    message = str(excinfo.value)
    assert str(model) in message
    assert "--allow-unnormalized" in message
    assert "different policy" in message


def test_allow_unnormalized_returns_none(tmp_path):
    model = tmp_path / "stage1_500000_steps.zip"
    assert resolve_vecnorm_path(str(model), None, True) is None


class _StubModel:
    """Stands in for a loaded PPO so the loader's sidecar branches run without a checkpoint."""

    @classmethod
    def load(cls, path, device):  # noqa: ARG003 - signature matches PPO.load
        return cls()


@pytest.fixture
def stub_ppo(monkeypatch):
    pytest.importorskip("stable_baselines3")
    import stable_baselines3

    monkeypatch.setattr(stable_baselines3, "PPO", _StubModel)
    return _StubModel


def test_the_loader_is_fail_closed_by_default(stub_ppo, tmp_path):
    model = tmp_path / "stage1_500000_steps.zip"
    with pytest.raises(PolicyLoadError):
        load_sb3_checkpoint(str(model), None, lambda: None)
    with pytest.raises(PolicyLoadError):
        load_sb3_checkpoint(str(model), None, lambda: None, guess_sidecar=False)


def test_the_loader_runs_unnormalised_only_when_allowed(stub_ppo, tmp_path):
    model = tmp_path / "stage1_500000_steps.zip"
    loaded, normalizer, resolved = load_sb3_checkpoint(str(model), None, lambda: None, allow_unnormalized=True)
    assert isinstance(loaded, _StubModel)
    assert normalizer is None
    assert resolved is None


def test_the_loader_freezes_the_statistics_it_finds(stub_ppo, monkeypatch, tmp_path):
    from stable_baselines3.common import vec_env as vec_env_module

    class _Normalizer:
        training = True
        norm_reward = True

    sidecar = tmp_path / "stage1_vecnormalize_500000_steps.pkl"
    sidecar.touch()
    monkeypatch.setattr(vec_env_module.VecNormalize, "load", staticmethod(lambda path, venv: _Normalizer()))
    monkeypatch.setattr(vec_env_module, "DummyVecEnv", lambda fns: None)

    _, normalizer, resolved = load_sb3_checkpoint(str(tmp_path / "stage1_500000_steps.zip"), None, lambda: None)
    assert resolved == str(sidecar)
    assert normalizer.training is False
    assert normalizer.norm_reward is False


@SCRIPTS
def test_the_flag_and_banner_are_wired_into_main(script):
    """The loader is only protection if main() calls it fail-closed and prints the banner."""
    source = inspect.getsource(script.main)
    assert "--allow-unnormalized" in source
    assert "load_sb3_checkpoint(" in source
    assert "allow_unnormalized=args.allow_unnormalized" in source
    # A refusal is a message and an exit status at the CLI, not a traceback.
    assert "except PolicyLoadError" in source
    assert "raise SystemExit(str(exc))" in source
    assert "UNNORMALIZED_BANNER" in source
    assert script.UNNORMALIZED_BANNER is policy_loading.UNNORMALIZED_BANNER
    assert "UNNORMALIZED EVAL" in script.UNNORMALIZED_BANNER
