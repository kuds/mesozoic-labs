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


class TestTrainingPipelineHook:
    """The artifact writer emits the report automatically, and safely.

    Three properties matter more than the report's content, which is covered
    above and in test_stance_gate.py: it fires only for stages the criteria
    govern, it prefers the same checkpoint the trainer reloads, and it can
    never cost a finished run its artifacts.
    """

    def _stage_config(self, gate_kind: str) -> dict:
        return {
            "curriculum_kwargs": {
                "gate_schema_version": 1,
                "gate_kind": gate_kind,
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_eval_episodes": 40,
            },
            "env_kwargs": {"max_episode_steps": 1000},
        }

    def test_skipped_for_a_stage_that_does_not_gate_on_stance(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True
            raise AssertionError("must not roll a panel for a non-stance stage")

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _boom)
        stage_artifacts._write_stance_gate_report(
            species="trex",
            stage=2,
            stage_config=self._stage_config("reward_and_length/v1"),
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
        )
        assert not called
        assert not (tmp_path / "stance_gate_report.json").exists()

    def test_missing_checkpoint_is_logged_not_raised(self, tmp_path, caplog):
        from environments.shared.reporting import stage_artifacts

        with caplog.at_level("WARNING"):
            stage_artifacts._write_stance_gate_report(
                species="trex",
                stage=1,
                stage_config=self._stage_config("stance_quality/v1"),
                stage_dir=tmp_path,
                model_dir=tmp_path / "models",
            )
        assert "no checkpoint" in caplog.text
        assert not (tmp_path / "stance_gate_report.json").exists()

    def test_a_failing_report_does_not_propagate(self, tmp_path, monkeypatch, caplog):
        """A diagnostic must never sink a completed training run."""
        from environments.shared.reporting import stage_artifacts

        models = tmp_path / "models"
        models.mkdir()
        (models / "robust_best_model.zip").write_bytes(b"not really a checkpoint")

        monkeypatch.setattr(
            stance_gate_report,
            "build_stance_gate_report",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rollout exploded")),
        )
        with caplog.at_level("WARNING"):
            stage_artifacts._write_stance_gate_report(
                species="trex",
                stage=1,
                stage_config=self._stage_config("stance_quality/v1"),
                stage_dir=tmp_path,
                model_dir=models,
            )
        assert "Stance gate report failed" in caplog.text

    def test_prefers_robust_best_model_and_writes_both_forms(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        models = tmp_path / "models"
        models.mkdir()
        (models / "best_model.zip").write_bytes(b"x")
        (models / "robust_best_model.zip").write_bytes(b"x")
        seen: dict = {}

        def _fake(species, stage, *, stage_config, model_path, vecnorm_path=None, **kwargs):
            seen["model_path"] = model_path
            seen["vecnorm_path"] = vecnorm_path
            return {
                "schema": "mesozoic.stance-gate-report/v1",
                "species": species,
                "stage": stage,
                "gate_kind": "stance_quality/v1",
                "policy": "stub",
                "episodes": 40,
                "seed": 3042,
                "settle_steps": 200,
                "horizon": 1000,
                "passed": False,
                "failures": ["stub"],
                "thresholds": {
                    "min_full_horizon_fraction": 0.95,
                    "max_unsupported_duty": 0.02,
                    "max_unsupported_duty_ucb": 0.02,
                    "min_avg_reward": 1950.0,
                    "min_eval_episodes": 40,
                },
                "metrics": {
                    "reward_mean": 2313.7,
                    "reward_std": 25.2,
                    "episode_length_mean": 1000.0,
                    "full_horizon_fraction": 1.0,
                    "mean_unsupported_duty": 0.1877,
                    "unsupported_duty_ucb": 0.1904,
                    "n_duty_episodes": 40,
                    "bilateral_support_duty": 0.664,
                    "single_support_duty": 0.148,
                },
                "terminations": {"truncated": 40},
                "reward_components": {"reward_alive": 832.23},
            }

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _fake)
        stage_artifacts._write_stance_gate_report(
            species="trex",
            stage=1,
            stage_config=self._stage_config("stance_quality/v1"),
            stage_dir=tmp_path,
            model_dir=models,
        )

        # robust_best_model wins over SB3's mean-reward best_model, matching
        # the order the trainer itself reloads in.
        assert seen["model_path"].endswith("robust_best_model.zip")
        # Absent vecnorm is passed as None rather than a path that is not there.
        assert seen["vecnorm_path"] is None

        import json as _json

        text = (tmp_path / "stance_gate_report.txt").read_text()
        assert "GATE: FAIL" in text
        assert "bilateral support" in text
        payload = _json.loads((tmp_path / "stance_gate_report.json").read_text())
        assert payload["metrics"]["mean_unsupported_duty"] == pytest.approx(0.1877)
