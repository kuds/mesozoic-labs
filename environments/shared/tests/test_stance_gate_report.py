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

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from environments.shared.scripts import stance_gate_report
from environments.shared.scripts.stance_gate_report import (
    StanceGateReportError,
    write_stance_gate_report,
)


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

    with pytest.raises(StanceGateReportError) as excinfo:
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

    with pytest.raises(StanceGateReportError) as excinfo:
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
        (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")

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

    def test_skips_when_the_selected_checkpoint_has_no_vecnorm(self, tmp_path, monkeypatch, caplog):
        """No statistics means no verdict, rather than a verdict for another policy.

        This used to pass ``vecnorm_path=None`` and score the checkpoint on
        unnormalised observations -- a different policy -- then write the
        result as this one's stance-gate verdict.
        """
        from environments.shared.reporting import stage_artifacts

        models = tmp_path / "models"
        models.mkdir()
        (models / "robust_best_model.zip").write_bytes(b"x")
        (models / "best_model.zip").write_bytes(b"x")

        called: list = []
        monkeypatch.setattr(
            stance_gate_report,
            "build_stance_gate_report",
            lambda *a, **k: called.append(k) or {},
        )
        with caplog.at_level("WARNING"):
            stage_artifacts._write_stance_gate_report(
                species="trex",
                stage=1,
                stage_config=self._stage_config("stance_quality/v1"),
                stage_dir=tmp_path,
                model_dir=models,
            )

        assert called == []
        assert "matched _vecnorm.pkl" in caplog.text
        assert not (tmp_path / "stance_gate_report.json").exists()

    def test_falls_back_when_only_best_model_is_complete(self, tmp_path, monkeypatch):
        """A robust checkpoint without its statistics is skipped, not paired."""
        from environments.shared.reporting import stage_artifacts

        models = tmp_path / "models"
        models.mkdir()
        (models / "robust_best_model.zip").write_bytes(b"x")
        (models / "best_model.zip").write_bytes(b"x")
        (models / "best_model_vecnorm.pkl").write_bytes(b"stats")

        seen: dict = {}
        monkeypatch.setattr(
            stance_gate_report,
            "build_stance_gate_report",
            lambda *a, **k: seen.update(k) or (_ for _ in ()).throw(RuntimeError("stop here")),
        )
        stage_artifacts._write_stance_gate_report(
            species="trex",
            stage=1,
            stage_config=self._stage_config("stance_quality/v1"),
            stage_dir=tmp_path,
            model_dir=models,
        )

        assert seen["model_path"].endswith("best_model.zip")
        assert seen["vecnorm_path"].endswith("best_model_vecnorm.pkl")

    def test_prefers_robust_best_model_and_writes_both_forms(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        models = tmp_path / "models"
        models.mkdir()
        for name in ("best_model", "robust_best_model"):
            (models / f"{name}.zip").write_bytes(b"x")
            (models / f"{name}_vecnorm.pkl").write_bytes(b"stats")
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
        # the order the trainer itself reloads in — the same
        # _select_handoff_checkpoint call the replay and the next-stage
        # handoff make, so all three describe one policy.
        assert seen["model_path"].endswith("robust_best_model.zip")
        # Its MATCHED statistics, never None: scoring on unnormalised
        # observations would report a verdict for a different policy.
        assert seen["vecnorm_path"].endswith("robust_best_model_vecnorm.pkl")

        import json as _json

        text = (tmp_path / "stance_gate_report.txt").read_text()
        assert "GATE: FAIL" in text
        assert "bilateral support" in text
        payload = _json.loads((tmp_path / "stance_gate_report.json").read_text())
        assert payload["metrics"]["mean_unsupported_duty"] == pytest.approx(0.1877)


def test_loader_failures_are_ordinary_exceptions_not_systemexit() -> None:
    """They used to be ``SystemExit``, which is a ``BaseException``.

    ``reporting.stage_artifacts._write_stance_gate_report`` guards the report
    with ``except Exception`` so a diagnostic cannot sink a finished run.
    ``SystemExit`` sailed straight through it, so a truncated VecNormalize
    ``.pkl`` aborted artifact generation before the graphs and videos.
    """
    assert issubclass(StanceGateReportError, Exception)
    assert not issubclass(StanceGateReportError, SystemExit)


class TestJsonIsParseable:
    """The JSON form must be readable by tooling that is not Python."""

    def _report(self, **metric_overrides):
        metrics = {
            "reward_mean": 210.0,
            "reward_std": 30.0,
            "episode_length_mean": 140.0,
            "full_horizon_fraction": 0.0,
            "mean_unsupported_duty": float("inf"),
            "unsupported_duty_ucb": float("inf"),
            "n_duty_episodes": 0,
            "bilateral_support_duty": float("nan"),
            "single_support_duty": float("nan"),
        }
        metrics.update(metric_overrides)
        return {
            "schema": "mesozoic.stance-gate-report/v1",
            "species": "trex",
            "stage": 1,
            "gate_kind": "stance_quality/v1",
            "policy": "robust_best_model.zip",
            "episodes": 40,
            "seed": 3042,
            "settle_steps": 200,
            "horizon": 1000,
            "passed": False,
            "failures": ["no full-horizon episode supplied a measurable unsupported duty"],
            "thresholds": {
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_avg_reward": 1950.0,
                "min_eval_episodes": 40,
            },
            "metrics": metrics,
            "terminations": {"fell": 40},
            "reward_components": {"reward_alive": 140.0},
        }

    def test_a_failed_panel_still_writes_strict_json(self, tmp_path):
        # The unmeasurable panel is the one worth inspecting, and it is the
        # one that used to emit bare NaN / Infinity tokens.
        import json

        stance_gate_report.write_stance_gate_report(tmp_path, self._report())
        text = (tmp_path / "stance_gate_report.json").read_text()

        assert "NaN" not in text
        assert "Infinity" not in text
        payload = json.loads(
            text,
            parse_constant=lambda name: pytest.fail(f"bare {name} is not valid JSON"),
        )
        assert payload["metrics"]["mean_unsupported_duty"] is None
        assert payload["metrics"]["bilateral_support_duty"] is None
        # The count is a real zero, not a sentinel, and must survive as one.
        assert payload["metrics"]["n_duty_episodes"] == 0

    def test_the_text_form_still_shows_the_sentinel(self, tmp_path):
        stance_gate_report.write_stance_gate_report(tmp_path, self._report())
        text = (tmp_path / "stance_gate_report.txt").read_text()
        assert "inf" in text

    def test_finite_values_are_untouched(self, tmp_path):
        import json

        stance_gate_report.write_stance_gate_report(
            tmp_path,
            self._report(mean_unsupported_duty=0.0187, unsupported_duty_ucb=0.0191),
        )
        payload = json.loads((tmp_path / "stance_gate_report.json").read_text())
        assert payload["metrics"]["mean_unsupported_duty"] == 0.0187
        assert payload["metrics"]["unsupported_duty_ucb"] == 0.0191


class TestStanceSharesSumToOne:
    """The three shares must be measured over the same episodes.

    The report prints bilateral and single support *because* the three sum to
    1 -- that identity is the stated reason a falling unsupported duty does
    not by itself mean the feet are being planted. They used to be averaged
    over every episode that measured anything while the gated duty uses
    full-horizon episodes only, so the identity broke whenever an episode
    failed early, and it broke in the direction that hides the problem: the
    flailing episodes were folded into bilateral and single but excluded from
    unsupported.
    """

    HORIZON = 1000
    SETTLE = 200

    def _run(self, monkeypatch, n_short: int) -> dict:
        import types

        import numpy as np

        horizon, settle = self.HORIZON, self.SETTLE

        class FakeEnv:
            action_space = types.SimpleNamespace(shape=(6,))

            def __init__(self, **kwargs):
                self.index = -1

            def reset(self, seed=None):
                self.index += 1
                self.step_count = 0
                self.short = self.index < n_short
                return np.zeros(4), {}

            def step(self, action):
                self.step_count += 1
                if self.short:
                    # Flailing: airborne 4 steps in 5.
                    unsupported, bilateral = self.step_count % 5 != 0, self.step_count % 5 == 0
                else:
                    unsupported, bilateral = False, True
                info = {
                    "r_foot_contact": 0.0 if unsupported else 50.0,
                    "l_foot_contact": 0.0 if (unsupported or not bilateral) else 50.0,
                    "reward_alive": 1.0,
                }
                done = self.step_count >= (400 if self.short else horizon)
                return np.zeros(4), 3.0, done and self.short, done and not self.short, info

            def close(self):
                pass

        # monkeypatch rather than assign-and-restore: it restores on failure
        # too, and it keeps mypy from checking a SimpleNamespace stub against
        # the registry's declared Callable[[], SpeciesConfig].
        monkeypatch.setattr(
            stance_gate_report, "SPECIES_FACTORIES", {"trex": lambda: types.SimpleNamespace(env_class=FakeEnv)}
        )
        return stance_gate_report.run_panel(
            "trex",
            predict=lambda obs: np.zeros(6),
            episodes=40,
            seed=0,
            settle_steps=settle,
            horizon=horizon,
            env_kwargs={},
        )

    def test_the_shares_sum_to_one_when_episodes_fail_early(self, monkeypatch):
        result = self._run(monkeypatch, n_short=5)
        panel = result["panel"]
        assert panel.n_duty_episodes == 35, "duty is measured on full-horizon episodes only"
        total = panel.mean_unsupported_duty + result["bilateral_duty"] + result["single_duty"]
        assert total == pytest.approx(1.0)

    def test_the_shares_sum_to_one_on_a_clean_panel(self, monkeypatch):
        result = self._run(monkeypatch, n_short=0)
        panel = result["panel"]
        total = panel.mean_unsupported_duty + result["bilateral_duty"] + result["single_duty"]
        assert total == pytest.approx(1.0)

    def test_shares_are_nan_when_no_episode_reaches_the_horizon(self, monkeypatch):
        import math

        result = self._run(monkeypatch, n_short=40)
        assert math.isnan(result["bilateral_duty"])
        assert math.isnan(result["single_duty"])


class TestPanelHygiene:
    """Lower-severity defects in the rollout harness."""

    def _fake_env_class(self, recorder):
        import types

        import numpy as np

        class FakeEnv:
            action_space = types.SimpleNamespace(shape=(6,))

            def __init__(self, **kwargs):
                self.steps = 0
                recorder["opened"] = recorder.get("opened", 0) + 1

            def reset(self, seed=None):
                self.steps = 0
                return np.zeros(4), {}

            def step(self, action):
                self.steps += 1
                info = {"r_foot_contact": 50.0, "l_foot_contact": 50.0}
                return np.zeros(4), 1.0, False, self.steps >= 10, info

            def close(self):
                recorder["closed"] = recorder.get("closed", 0) + 1

        return FakeEnv

    def _run_panel(self, monkeypatch, recorder, **overrides):
        import types

        import numpy as np

        monkeypatch.setattr(
            stance_gate_report,
            "SPECIES_FACTORIES",
            {"trex": lambda: types.SimpleNamespace(env_class=self._fake_env_class(recorder))},
        )
        kwargs = dict(
            predict=lambda obs: np.zeros(6),
            episodes=3,
            seed=0,
            settle_steps=0,
            horizon=10,
            env_kwargs={},
        )
        kwargs.update(overrides)
        return stance_gate_report.run_panel("trex", **kwargs)

    def test_the_environment_is_closed(self, monkeypatch):
        # A MuJoCo env holds native handles and the artifact path builds one
        # per stage; leaking them across a sweep is how a worker runs out.
        recorder: dict = {}
        self._run_panel(monkeypatch, recorder)
        assert recorder["closed"] == recorder["opened"]

    def test_the_environment_is_closed_when_the_rollout_raises(self, monkeypatch):
        recorder: dict = {}
        with pytest.raises(RuntimeError):
            self._run_panel(
                monkeypatch, recorder, predict=lambda obs: (_ for _ in ()).throw(RuntimeError("policy blew up"))
            )
        assert recorder["closed"] == recorder["opened"]

    def test_per_episode_evidence_is_returned_not_discarded(self, monkeypatch):
        result = self._run_panel(monkeypatch, {})
        evidence = result["episodes"]
        assert len(evidence) == 3
        first = evidence[0]
        assert set(first) == {
            "episode",
            "seed",
            "length",
            "reward",
            "reached_horizon",
            "unsupported_duty",
            "bilateral_support_duty",
            "single_support_duty",
        }
        # Aligned with the panel it was reduced into.
        assert [row["length"] for row in evidence] == list(result["lengths"])
        assert all(row["reached_horizon"] for row in evidence)


class TestBuildReportArguments:
    def _stage_config(self):
        return {
            "curriculum_kwargs": {
                "gate_schema_version": 1,
                "gate_kind": "stance_quality/v1",
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_eval_episodes": 40,
            },
            "env_kwargs": {"max_episode_steps": 1000},
        }

    def test_zero_episodes_is_rejected_rather_than_treated_as_absent(self):
        # `episodes or min_eval_episodes` silently produced a full 40-episode
        # panel for `--episodes 0`, and skipped the under-powered warning too,
        # so the flag read as an override that did nothing.
        with pytest.raises(ValueError, match="at least 1"):
            stance_gate_report.build_stance_gate_report(
                "trex", 1, stage_config=self._stage_config(), zero_action=True, episodes=0
            )

    def test_negative_episodes_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            stance_gate_report.build_stance_gate_report(
                "trex", 1, stage_config=self._stage_config(), zero_action=True, episodes=-5
            )


class TestStanceReportEpisodesKnob:
    """`stance_report_episodes` gives the artifact path a cost control.

    The report costs a 40-episode rollout per stage AND per sweep trial,
    which a fifty-trial sweep pays fifty times for a verdict nobody reads
    until a trial is shortlisted.
    """

    def _stage_config(self, **curriculum):
        base = {
            "gate_schema_version": 1,
            "gate_kind": "stance_quality/v1",
            "min_full_horizon_fraction": 0.95,
            "max_unsupported_duty": 0.02,
            "max_unsupported_duty_ucb": 0.02,
            "min_eval_episodes": 40,
        }
        base.update(curriculum)
        return {"curriculum_kwargs": base, "env_kwargs": {"max_episode_steps": 1000}}

    def _invoke(self, tmp_path, monkeypatch, stage_config):
        from environments.shared.reporting import stage_artifacts

        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        (models / "robust_best_model.zip").write_bytes(b"x")
        (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")

        seen: dict = {}
        monkeypatch.setattr(
            stance_gate_report,
            "build_stance_gate_report",
            lambda *a, **k: seen.update(k) or (_ for _ in ()).throw(RuntimeError("stop after the call")),
        )
        stage_artifacts._write_stance_gate_report(
            species="trex",
            stage=1,
            stage_config=stage_config,
            stage_dir=tmp_path,
            model_dir=models,
        )
        return seen

    def test_defaults_to_the_stages_panel_size(self, tmp_path, monkeypatch):
        seen = self._invoke(tmp_path, monkeypatch, self._stage_config())
        assert seen["episodes"] == 40

    def test_zero_skips_the_report_entirely(self, tmp_path, monkeypatch):
        seen = self._invoke(tmp_path, monkeypatch, self._stage_config(stance_report_episodes=0))
        assert seen == {}, "no rollout should be attempted"
        assert not (tmp_path / "stance_gate_report.json").exists()

    def test_an_override_is_honoured_and_warned_about(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            seen = self._invoke(tmp_path, monkeypatch, self._stage_config(stance_report_episodes=8))
        assert seen["episodes"] == 8
        # The bound's power is specified at min_eval_episodes; a smaller panel
        # does not certify what the gate claims, and the log has to say so.
        assert "does not certify what the gate claims" in caplog.text

    def test_the_schema_accepts_the_key(self):
        from environments.shared.curriculum.gate_schema import validate_gate_config

        assert validate_gate_config(1, self._stage_config(stance_report_episodes=8)["curriculum_kwargs"])


class TestPlantValidation:
    """A verdict measured on a different plant is not a verdict about this stage.

    Every other artifact path already refuses that pairing; this script did
    not, while its own docstring advertises scoring already-finished runs --
    exactly the checkpoints most likely to predate the current plant.
    """

    def _stage_config(self):
        return {
            "curriculum_kwargs": {
                "gate_schema_version": 1,
                "gate_kind": "stance_quality/v1",
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_eval_episodes": 2,
            },
            "env_kwargs": {"max_episode_steps": 1000},
        }

    def test_a_checkpoint_without_a_plant_identity_is_refused(self, stub_ppo):
        from environments.shared.plant_contract import PlantCompatibilityError, current_plant_identity

        identity = current_plant_identity("trex", verify_generated=False)
        with pytest.raises(PlantCompatibilityError):
            stance_gate_report._load_policy("model.zip", None, lambda: None, plant_identity=identity)

    def test_allow_legacy_plant_permits_it(self, stub_ppo):
        from environments.shared.plant_contract import current_plant_identity

        identity = current_plant_identity("trex", verify_generated=False)
        predict, _ = stance_gate_report._load_policy(
            "model.zip", None, lambda: None, plant_identity=identity, allow_legacy_plant=True
        )
        assert callable(predict)

    def test_no_identity_supplied_skips_validation(self, stub_ppo):
        predict, _ = stance_gate_report._load_policy("model.zip", None, lambda: None)
        assert callable(predict)

    def test_the_report_records_whether_the_plant_was_validated(self, tmp_path):
        # The flag travels with the number rather than being reconstructable
        # only from whoever happened to read the console.
        report = {
            "schema": "mesozoic.stance-gate-report/v1",
            "species": "trex",
            "stage": 1,
            "gate_kind": "stance_quality/v1",
            "policy": "p",
            "episodes": 40,
            "seed": 3042,
            "settle_steps": 200,
            "horizon": 1000,
            "passed": True,
            "failures": [],
            "checkpoint_plant_validated": False,
            "thresholds": {
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_avg_reward": 1950.0,
                "min_eval_episodes": 40,
            },
            "metrics": {
                "reward_mean": 1.0,
                "reward_std": 0.0,
                "episode_length_mean": 1000.0,
                "full_horizon_fraction": 1.0,
                "mean_unsupported_duty": 0.0,
                "unsupported_duty_ucb": 0.0,
                "n_duty_episodes": 40,
                "bilateral_support_duty": 1.0,
                "single_support_duty": 0.0,
            },
            "terminations": {},
            "reward_components": {},
            "episode_evidence": [],
        }
        import json

        stance_gate_report.write_stance_gate_report(tmp_path, report)
        payload = json.loads((tmp_path / "stance_gate_report.json").read_text())
        assert payload["checkpoint_plant_validated"] is False


class TestCheckpointPlantProvenance:
    """The recorded flag must describe what was actually checked.

    It was ``plant_validated = not allow_legacy_plant``, but the rollout
    environment is validated unconditionally and ``allow_legacy_plant`` only
    relaxes the *checkpoint* check -- so for ``--zero-action``, which has no
    checkpoint at all, a ``False`` here claimed something untrue.
    """

    def _stage_config(self):
        return {
            "curriculum_kwargs": {
                "gate_schema_version": 1,
                "gate_kind": "stance_quality/v1",
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_eval_episodes": 1,
            },
            "env_kwargs": {"max_episode_steps": 4},
        }

    def _report(self, monkeypatch, **kwargs):
        import types

        import numpy as np

        class FakeEnv:
            action_space = types.SimpleNamespace(shape=(6,))

            def __init__(self, **kw):
                self.steps = 0

            def reset(self, seed=None):
                self.steps = 0
                return np.zeros(4), {}

            def step(self, action):
                self.steps += 1
                return np.zeros(4), 1.0, False, self.steps >= 4, {"r_foot_contact": 50.0, "l_foot_contact": 50.0}

            def close(self):
                pass

        monkeypatch.setattr(
            stance_gate_report, "SPECIES_FACTORIES", {"trex": lambda: types.SimpleNamespace(env_class=FakeEnv)}
        )
        monkeypatch.setattr(
            "environments.shared.plant_contract.validate_environment_plant", lambda *a, **k: None, raising=False
        )
        return stance_gate_report.build_stance_gate_report(
            "trex", 1, stage_config=self._stage_config(), episodes=1, **kwargs
        )

    def test_zero_action_records_none_because_there_is_no_checkpoint(self, monkeypatch):
        report = self._report(monkeypatch, zero_action=True)
        assert report["checkpoint_plant_validated"] is None

    def test_zero_action_with_allow_legacy_still_records_none(self, monkeypatch):
        # The flag is about a checkpoint; there isn't one either way.
        report = self._report(monkeypatch, zero_action=True, allow_legacy_plant=True)
        assert report["checkpoint_plant_validated"] is None


class TestStancePanelEvidenceFile:
    """The per-episode duty record the result bundle certifies a stage from.

    `result_bundle.evidence` used to refuse stance-gated bundles outright,
    because `evaluation_selected.csv` carries reward and length but no duty
    and certifying on the reward rail alone would pass the statue. The refusal
    was right and it also made the stage-1 milestone unreachable. This file is
    what lifts it, so its shape is part of the contract.
    """

    @staticmethod
    def _report(episodes):
        return {
            "schema": "mesozoic.stance-gate-report/v1",
            "species": "trex",
            "stage": 1,
            "gate_kind": "stance_quality/v1",
            "policy": "robust_best_model.zip",
            "episodes": len(episodes),
            "seed": 3042,
            "settle_steps": 200,
            "horizon": 1000,
            "passed": False,
            "failures": ["mean_unsupported_duty 0.2120 > 0.0200"],
            "thresholds": {
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_avg_reward": 1950.0,
                "min_eval_episodes": 40,
            },
            "metrics": {
                "reward_mean": 2295.5,
                "reward_std": 36.9,
                "episode_length_mean": 1000.0,
                "full_horizon_fraction": 1.0,
                "mean_unsupported_duty": 0.212,
                "unsupported_duty_ucb": 0.2153,
                "n_duty_episodes": len(episodes),
                "bilateral_support_duty": 0.667,
                "single_support_duty": 0.121,
            },
            "checkpoint_plant_validated": True,
            "terminations": {"truncated": len(episodes)},
            "reward_components": {},
            "episode_evidence": episodes,
        }

    @staticmethod
    def _episode(index, *, length, reward, duty):
        return {
            "episode": index,
            "seed": 3042 + index,
            "length": length,
            "reward": reward,
            "reached_horizon": length >= 1000,
            "unsupported_duty": duty,
            "bilateral_support_duty": None if duty is None else 1.0 - duty,
            "single_support_duty": None if duty is None else 0.0,
        }

    def test_the_panel_csv_is_written_beside_the_report(self, tmp_path):
        written = write_stance_gate_report(
            tmp_path,
            self._report([self._episode(0, length=1000, reward=2295.5, duty=0.212)]),
        )
        assert written["stance_panel_csv"] == tmp_path / "stance_panel_selected.csv"
        assert written["stance_panel_csv"].is_file()

    def test_an_unmeasurable_duty_is_blank_rather_than_zero(self, tmp_path):
        """Zero duty is the statue's score and the best attainable one.

        Writing 0.0 for an episode too short to measure would turn missing
        evidence into a perfect result, which is the fail-open shape this
        gate exists to refuse.
        """
        written = write_stance_gate_report(
            tmp_path,
            self._report([self._episode(0, length=300, reward=600.0, duty=None)]),
        )
        with written["stance_panel_csv"].open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        assert rows[0]["unsupported_duty"] == ""
        assert rows[0]["bilateral_support_duty"] == ""

    def test_the_evidence_the_auditor_reads_round_trips(self, tmp_path):
        """Writer and auditor must agree on the columns, not merely coexist."""
        from environments.shared.result_bundle.evidence import _validate_stance_panel_evidence

        episodes = [self._episode(index, length=1000, reward=2400.0, duty=0.0) for index in range(40)]
        written = write_stance_gate_report(tmp_path, self._report(episodes))
        curriculum = {
            "gate_kind": "stance_quality/v1",
            "min_full_horizon_fraction": 0.95,
            "max_unsupported_duty": 0.02,
            "max_unsupported_duty_ucb": 0.02,
            "min_eval_episodes": 40,
            "settle_steps": 200,
        }
        # Passing panel: no refusal.
        _validate_stance_panel_evidence(
            written["stance_panel_csv"],
            curriculum,
            env_kwargs={"max_episode_steps": 1000},
            stage=1,
        )

    def test_a_report_without_episode_evidence_writes_no_panel(self, tmp_path):
        """Older reports predate the field; the auditor's own refusal covers it."""
        report = self._report([])
        report.pop("episode_evidence")
        written = write_stance_gate_report(tmp_path, report)
        assert "stance_panel_csv" not in written
        assert not (tmp_path / "stance_panel_selected.csv").exists()


def _minimal_stance_report() -> dict:
    """A report dict complete enough for the renderer."""
    return {
        "schema": "mesozoic.stance-gate-report/v1",
        "species": "trex",
        "stage": 1,
        "gate_kind": "stance_quality/v1",
        "policy": "robust_best_model.zip",
        "episodes": 40,
        "seed": 3042,
        "settle_steps": 200,
        "horizon": 1000,
        "passed": True,
        "failures": [],
        "thresholds": {
            "min_full_horizon_fraction": 0.95,
            "max_unsupported_duty": 0.02,
            "max_unsupported_duty_ucb": 0.02,
            "min_avg_reward": 2550.0,
            "min_eval_episodes": 40,
        },
        "metrics": {
            "reward_mean": 3004.3,
            "reward_std": 17.1,
            "episode_length_mean": 1000.0,
            "full_horizon_fraction": 1.0,
            "mean_unsupported_duty": 0.0,
            "unsupported_duty_ucb": 0.0,
            "n_duty_episodes": 40,
            "bilateral_support_duty": 1.0,
            "single_support_duty": 0.0,
        },
        "terminations": {"truncated": 40},
        "reward_components": {"reward_alive": 995.2},
    }


class TestActionFilterProbe:
    """`--filter-actions` scores a MODIFIED policy, and must say so.

    It exists to answer whether a policy's high-frequency action content is
    load-bearing or waste (issue #489): if a filtered policy still stands, the
    tremor was waste and the fix belongs on the action path; if it falls, the
    tremor is closed-loop stabilisation and the fix belongs elsewhere.
    """

    @staticmethod
    def _filtered(cutoff_hz, control_dt=0.01, actions=None):
        from environments.shared.scripts.stance_gate_report import _low_pass_predict

        seq = iter(actions or [])
        base = lambda _obs: np.asarray(next(seq), dtype=np.float64)  # noqa: E731
        return _low_pass_predict(base, cutoff_hz, control_dt)

    def test_a_constant_action_passes_through_unchanged(self):
        """A low-pass must be a no-op on DC, or it would move the pose itself."""
        predict = self._filtered(5.0, actions=[[0.5]] * 20)
        out = [float(predict(None)[0]) for _ in range(20)]
        assert out[0] == pytest.approx(0.5)
        assert out[-1] == pytest.approx(0.5)

    def test_it_attenuates_a_fast_alternation(self):
        predict = self._filtered(5.0, actions=[[1.0], [-1.0]] * 30)
        out = [float(predict(None)[0]) for _ in range(60)]
        # Seeded with the first action, so settle before measuring.
        tail = out[20:]
        assert max(abs(v) for v in tail) < 0.4, "22.6 Hz-class content should be cut hard at 5 Hz"

    def test_it_seeds_on_the_first_action_rather_than_zero(self):
        """Starting from zero injects a step transient that is a probe artefact."""
        predict = self._filtered(5.0, actions=[[0.9]])
        assert float(predict(None)[0]) == pytest.approx(0.9)

    def test_reset_prevents_one_episode_leaking_into_the_next(self):
        predict = self._filtered(5.0, actions=[[1.0], [-1.0]])
        assert float(predict(None)[0]) == pytest.approx(1.0)
        predict.reset()
        assert float(predict(None)[0]) == pytest.approx(-1.0)

    def test_the_report_records_the_probe_so_a_pass_cannot_be_mistaken(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        report = _minimal_stance_report()
        report["filter_actions_hz"] = 5.0
        text = render_stance_gate_report(report)
        assert "PROBE" in text
        assert "not a gate result" in text
        # The warning must precede the verdict a reader would otherwise act on.
        assert text.index("PROBE") < text.index("GATE:")

    def test_an_unfiltered_report_says_nothing_about_the_probe(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        assert "PROBE" not in render_stance_gate_report(_minimal_stance_report())


class TestTheProbeCannotBeMistakenForTheVerdict:
    """A filtered rollout scored a policy that was never actually run.

    Two ways it could corrupt the record: overwriting the real report, or
    supplying `stance_panel_selected.csv`, which is the per-episode evidence
    `result_bundle.evidence` certifies a stance-gated stage from. Both are
    made structurally impossible rather than left to the caller.
    """

    def test_a_probe_writes_its_own_filenames(self, tmp_path):
        report = _minimal_stance_report()
        report["filter_actions_hz"] = 5.0
        written = write_stance_gate_report(tmp_path, report)
        assert (tmp_path / "stance_gate_probe_filtered.txt").exists()
        assert (tmp_path / "stance_gate_probe_filtered.json").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()
        assert "stance_panel_csv" not in written

    def test_a_probe_never_writes_the_certification_evidence(self, tmp_path):
        real = _minimal_stance_report()
        real["episode_evidence"] = [
            {
                "episode": 0,
                "seed": 3042,
                "length": 1000,
                "reward": 3004.3,
                "reached_horizon": True,
                "unsupported_duty": 0.0,
                "bilateral_support_duty": 1.0,
                "single_support_duty": 0.0,
            }
        ]
        write_stance_gate_report(tmp_path, real)
        before = (tmp_path / "stance_panel_selected.csv").read_text()

        probe = dict(real)
        probe["filter_actions_hz"] = 5.0
        probe["metrics"] = dict(real["metrics"], reward_mean=-99.0)
        write_stance_gate_report(tmp_path, probe)

        # The real panel must be untouched: a filtered panel here would
        # certify a policy that was never run.
        assert (tmp_path / "stance_panel_selected.csv").read_text() == before

    def test_the_real_report_is_unaffected(self, tmp_path):
        written = write_stance_gate_report(tmp_path, _minimal_stance_report())
        assert (tmp_path / "stance_gate_report.txt").exists()
        assert not (tmp_path / "stance_gate_probe_filtered.txt").exists()
        # No episode_evidence in this fixture, so no panel CSV either; the
        # evidence path is covered by the test above.
        assert "stance_panel_csv" not in written


class TestTheProbeIsWiredIntoTrainingArtifacts:
    def test_it_is_skipped_when_unconfigured(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _boom)
        stage_artifacts._write_filtered_action_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            episodes=40,
        )
        assert not called

    def test_it_passes_the_configured_cutoff_through(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        seen = {}

        def _capture(species, stage, **kwargs):
            seen.update(kwargs)
            report = _minimal_stance_report()
            report["filter_actions_hz"] = kwargs["filter_actions_hz"]
            return report

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _capture)
        stage_artifacts._write_filtered_action_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {"stance_probe_filter_hz": 5.0}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            episodes=40,
        )
        assert seen["filter_actions_hz"] == 5.0
        assert (tmp_path / "stance_gate_probe_filtered.txt").exists()

    def test_a_failing_probe_does_not_sink_the_run(self, tmp_path, monkeypatch, caplog):
        from environments.shared.reporting import stage_artifacts

        monkeypatch.setattr(
            stance_gate_report,
            "build_stance_gate_report",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe exploded")),
        )
        with caplog.at_level("WARNING"):
            stage_artifacts._write_filtered_action_probe(
                species="trex",
                stage=1,
                stage_config={"curriculum_kwargs": {"stance_probe_filter_hz": 5.0}},
                stage_dir=tmp_path,
                model_path="m.zip",
                vecnorm_path="v.pkl",
                episodes=40,
            )
        assert "Filtered action probe failed" in caplog.text

    def test_the_config_key_is_accepted_by_the_gate_schema(self):
        """An unregistered key is unreachable under the fail-closed check."""
        from environments.shared.config import load_stage_config
        from environments.shared.curriculum.gate_schema import validate_gate_config

        curriculum = load_stage_config("trex", 1)["curriculum_kwargs"]
        assert curriculum["stance_probe_filter_hz"] == 5.0
        validate_gate_config(1, curriculum)
