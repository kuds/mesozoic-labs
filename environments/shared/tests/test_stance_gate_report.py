"""Error paths of the stance-gate checkpoint report.

Only the hand-written failure handling is covered here. The gate arithmetic
itself is tested in ``test_stance_gate.py``, and the happy path needs a real
MuJoCo rollout, which belongs in a manual run rather than the suite.

Both cases below share a rationale: normalising observations with the wrong
statistics, or not at all, silently evaluates a *different policy*. A report
that did so would blame the gate for a loading mistake, so each must fail
loudly rather than degrade.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from environments.shared.scripts import stance_gate_report


class _StubModel:
    """Stands in for a loaded PPO so the tests exercise the vecnorm branch."""

    @classmethod
    def load(cls, path, device):  # noqa: ARG003 - signature matches PPO.load
        return cls()

    def predict(self, obs, deterministic=True):  # noqa: ARG002
        return obs, None


@pytest.fixture
def stub_ppo(monkeypatch):
    pytest.importorskip("stable_baselines3")
    import stable_baselines3

    monkeypatch.setattr(stable_baselines3, "PPO", _StubModel)
    return _StubModel


def test_corrupt_vecnorm_is_fatal_with_a_diagnosable_message(stub_ppo, tmp_path: Path) -> None:
    truncated = tmp_path / "vecnorm.pkl"
    truncated.write_bytes(pickle.dumps({"obs_rms": None})[:12])

    with pytest.raises(SystemExit) as excinfo:
        stance_gate_report._load_policy("model.zip", str(truncated), lambda: None)

    message = str(excinfo.value)
    # Names the file, says why it is fatal, and points at the likely cause.
    assert str(truncated) in message
    assert "binary mode" in message
    assert "different policy" in message


def test_no_vecnorm_is_allowed_but_labelled(stub_ppo) -> None:
    """Running unnormalised is permitted only when explicitly not supplied."""
    _predict, description = stance_gate_report._load_policy("model.zip", None, lambda: None)
    assert "no obs normalisation" in description


class _FakeObsRms:
    mean = 0.0
    var = 1.0


class _FakeNormalizer:
    epsilon = 1e-8
    clip_obs = 10.0

    def __init__(self, obs_rms):
        self.obs_rms = obs_rms


def test_dict_observation_statistics_are_rejected(stub_ppo, monkeypatch) -> None:
    """Per-key stats mean the checkpoint used a different observation contract."""
    from stable_baselines3.common import vec_env as vec_env_module

    monkeypatch.setattr(
        vec_env_module.VecNormalize,
        "load",
        staticmethod(lambda path, venv: _FakeNormalizer({"a": _FakeObsRms(), "b": _FakeObsRms()})),
    )
    monkeypatch.setattr(vec_env_module, "DummyVecEnv", lambda fns: None)

    with pytest.raises(SystemExit) as excinfo:
        stance_gate_report._load_policy("model.zip", "v.pkl", lambda: None)

    message = str(excinfo.value)
    assert "Dict observation space" in message
    assert "'a'" in message


def test_flat_statistics_are_applied(stub_ppo, monkeypatch) -> None:
    from stable_baselines3.common import vec_env as vec_env_module

    monkeypatch.setattr(
        vec_env_module.VecNormalize,
        "load",
        staticmethod(lambda path, venv: _FakeNormalizer(_FakeObsRms())),
    )
    monkeypatch.setattr(vec_env_module, "DummyVecEnv", lambda fns: None)

    _predict, description = stance_gate_report._load_policy("model.zip", "v.pkl", lambda: None)
    assert "VecNormalize stats applied" in description
