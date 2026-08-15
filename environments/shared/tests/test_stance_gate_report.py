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
import math
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
                "schema": "mesozoic.stance-gate-report/v2",
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
            "schema": "mesozoic.stance-gate-report/v2",
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
            "schema": "mesozoic.stance-gate-report/v2",
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
        assert report["schema"] == "mesozoic.stance-gate-report/v2"
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
            "schema": "mesozoic.stance-gate-report/v2",
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
        "schema": "mesozoic.stance-gate-report/v2",
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

    def test_a_scalar_cutoff_still_works(self):
        from environments.shared.reporting.stage_artifacts import _probe_cutoffs

        assert _probe_cutoffs(5.0) == [5.0]

    def test_a_list_becomes_a_sorted_deduplicated_sweep(self):
        from environments.shared.reporting.stage_artifacts import _probe_cutoffs

        assert _probe_cutoffs([20, 5.0, 10, 5.0]) == [5.0, 10.0, 20.0]

    def test_unusable_entries_are_dropped_not_fatal(self):
        """One bad entry must not cost the cutoffs that are fine."""
        from environments.shared.reporting.stage_artifacts import _probe_cutoffs

        assert _probe_cutoffs([5.0, "nonsense", -3.0, 0.0, 10.0]) == [5.0, 10.0]

    def test_the_sweep_writes_one_curve_and_no_panel_evidence(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_action_filter_sweep

        reports = []
        for hz, length in ((20.0, 289.2), (5.0, 96.2), (10.0, 189.1)):
            r = _minimal_stance_report()
            r["filter_actions_hz"] = hz
            r["metrics"] = dict(r["metrics"], episode_length_mean=length, full_horizon_fraction=0.0)
            r["terminations"] = {"tail_contact": 10}
            reports.append(r)
        written = write_action_filter_sweep(tmp_path, reports, probe_episodes=10)
        assert set(written) == {"action_filter_sweep_txt", "action_filter_sweep_json"}
        text = (tmp_path / "stance_gate_probe_filtered.txt").read_text()
        # Sorted by cutoff so the curve reads in order, whatever order they ran.
        assert text.index("5 Hz") < text.index("10 Hz") < text.index("20 Hz")
        assert "MODIFIED policy" in text
        assert not (tmp_path / "stance_panel_selected.csv").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()
        import json

        payload = json.loads((tmp_path / "stance_gate_probe_filtered.json").read_text())
        assert [row["filter_actions_hz"] for row in payload["sweep"]] == [5.0, 10.0, 20.0]
        assert payload["episodes_per_cutoff"] == 10

    def test_a_partial_sweep_is_still_written(self, tmp_path, monkeypatch):
        """A failure at one cutoff must not discard the ones that succeeded."""
        from environments.shared.reporting import stage_artifacts

        calls = {"n": 0}

        def _flaky(species, stage, **kwargs):
            calls["n"] += 1
            if kwargs["filter_actions_hz"] == 10.0:
                raise RuntimeError("this cutoff exploded")
            r = _minimal_stance_report()
            r["filter_actions_hz"] = kwargs["filter_actions_hz"]
            return r

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _flaky)
        stage_artifacts._write_filtered_action_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {"stance_probe_filter_hz": [5.0, 10.0, 20.0]}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            episodes=40,
        )
        import json

        payload = json.loads((tmp_path / "stance_gate_probe_filtered.json").read_text())
        assert [row["filter_actions_hz"] for row in payload["sweep"]] == [5.0]

    def test_the_probe_rolls_fewer_episodes_than_the_gate(self, tmp_path, monkeypatch):
        """Each cutoff costs a panel; the probe certifies nothing."""
        from environments.shared.reporting import stage_artifacts

        seen = {}

        def _capture(species, stage, **kwargs):
            seen["episodes"] = kwargs["episodes"]
            r = _minimal_stance_report()
            r["filter_actions_hz"] = kwargs["filter_actions_hz"]
            return r

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _capture)
        stage_artifacts._write_filtered_action_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {"stance_probe_filter_hz": [5.0]}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            episodes=40,
        )
        assert seen["episodes"] == 10

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
        assert not (tmp_path / "stance_panel_selected.csv").exists()

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
        assert curriculum["stance_probe_filter_hz"] == [1.0, 2.5, 5.0, 8.0, 10.0]
        validate_gate_config(1, curriculum)


class TestPerActuatorActionReporting:
    """Which actuators carry the offset decides whether a pose weight reaches it.

    `leg_home_pose` covers only the leg joints, so a displacement living in the
    tail or neck is invisible to it and unfixable by it (issue #489). The
    scalar summaries cannot answer that; this is what makes one five-minute
    run on an existing checkpoint test it.
    """

    @staticmethod
    def _stats(sequence):
        from environments.shared.scripts.stance_gate_report import (
            _accumulate_action,
            _new_action_stats,
        )

        stats = _new_action_stats()
        for action in sequence:
            _accumulate_action(stats, np.asarray(action, dtype=np.float64))
        return stats

    def test_dc_and_ac_are_reported_per_actuator(self):
        from environments.shared.scripts.stance_gate_report import _reduce_action_stats

        # Actuator 0 holds 0.5 and shakes +/-0.1; actuator 1 is still at 0.
        stats = self._stats([[0.6, 0.0], [0.4, 0.0]] * 10)
        out = _reduce_action_stats(stats, ["r_hip_pitch", "tail_1"], 0.01)
        by_joint = {entry["joint"]: entry for entry in out["per_actuator"]}
        assert by_joint["r_hip_pitch"]["dc"] == pytest.approx(0.5)
        assert by_joint["r_hip_pitch"]["ac_rms"] == pytest.approx(0.1)
        assert by_joint["tail_1"]["dc"] == pytest.approx(0.0)
        assert by_joint["tail_1"]["ac_rms"] == pytest.approx(0.0)

    def test_the_effective_frequency_matches_the_inversion(self):
        from environments.shared.scripts.stance_gate_report import _reduce_action_stats

        # Alternating every step is the Nyquist rate: 50 Hz at dt = 0.01.
        stats = self._stats([[1.0], [-1.0]] * 20)
        out = _reduce_action_stats(stats, ["j"], 0.01)
        assert out["effective_freq_hz"] == pytest.approx(50.0, abs=0.5)

    def test_a_constant_action_has_no_frequency(self):
        from environments.shared.scripts.stance_gate_report import _reduce_action_stats

        out = _reduce_action_stats(self._stats([[0.5]] * 10), ["j"], 0.01)
        # No first difference leaves the ratio undefined. NaN is honest;
        # 0 Hz would read as a measured slow drift.
        assert math.isnan(out["effective_freq_hz"])
        assert out["dc_rms"] == pytest.approx(0.5)

    def test_no_measured_steps_yields_an_empty_block(self):
        from environments.shared.scripts.stance_gate_report import (
            _new_action_stats,
            _reduce_action_stats,
        )

        assert _reduce_action_stats(_new_action_stats(), [], 0.01) == {}

    def test_unnamed_actuators_fall_back_to_an_index(self):
        from environments.shared.scripts.stance_gate_report import _reduce_action_stats

        out = _reduce_action_stats(self._stats([[0.1, 0.2]] * 4), [], 0.01)
        assert [entry["joint"] for entry in out["per_actuator"]] == ["actuator_0", "actuator_1"]

    def test_the_text_form_lists_the_largest_offsets_first(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        report = _minimal_stance_report()
        report["action"] = {
            "dc_rms": 0.8,
            "ac_rms": 0.277,
            "delta_per_step": 2.73,
            "jerk_per_step": 4.64,
            "effective_freq_hz": 22.6,
            "per_actuator": [
                {"index": 0, "joint": "tail_1", "dc": 0.9, "ac_rms": 0.1},
                {"index": 1, "joint": "r_knee", "dc": -0.2, "ac_rms": 0.3},
            ],
        }
        text = render_stance_gate_report(report)
        assert "22.6 Hz" in text
        assert text.index("tail_1") < text.index("r_knee")

    def test_a_report_without_the_block_still_renders(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        assert "commanded action" not in render_stance_gate_report(_minimal_stance_report())


def _held_report(label, *, holds=True, handoff=200, ramp=0, length=1000.0, full=1.0, reward=3004.3, source=None):
    """A probe report for one constant-hold variant."""
    report = _minimal_stance_report()
    report["episodes"] = 10
    report["hold_constant"] = {
        "label": label,
        "source": source or ("measured" if holds else "unmodified"),
        "holds": holds,
        "handoff_steps": handoff,
        "ramp_steps": ramp,
        "actions": [0.5, -0.25],
    }
    report["metrics"] = dict(
        report["metrics"],
        episode_length_mean=length,
        full_horizon_fraction=full,
        reward_mean=reward,
    )
    report["terminations"] = {"truncated": 10} if full > 0 else {"tail_contact": 10}
    return report


class TestConstantHoldWrapper:
    """`--hold-constant` cuts the feedback, where the low-pass only slows it.

    The distinction is the whole point: a filtered policy still responds to
    what it sees, so a fall under the filter proves the policy needs bandwidth
    without saying whether it needs feedback at all. Holding a constant says.
    """

    @staticmethod
    def _held(hold, actions=None):
        from environments.shared.scripts.stance_gate_report import _hold_constant_predict

        seq = iter(actions or [])
        base = lambda _obs: np.asarray(next(seq), dtype=np.float64)  # noqa: E731
        return _hold_constant_predict(base, hold)

    def test_the_policy_runs_until_handoff_then_stops_being_consulted(self):
        from environments.shared.scripts.stance_gate_report import ConstantHold

        hold = ConstantHold((0.5,), handoff_steps=2)
        # Only two policy actions are supplied: a third call would raise
        # StopIteration if the wrapper kept consulting it.
        predict = self._held(hold, actions=[[0.1], [0.2]])
        assert [float(predict(None)[0]) for _ in range(5)] == pytest.approx([0.1, 0.2, 0.5, 0.5, 0.5])

    def test_a_handoff_at_the_horizon_never_substitutes(self):
        """This is how the unmodified control variant is expressed."""
        from environments.shared.scripts.stance_gate_report import ConstantHold

        hold = ConstantHold((9.9,), handoff_steps=1000, holds=False)
        predict = self._held(hold, actions=[[0.1], [0.2], [0.3]])
        assert [float(predict(None)[0]) for _ in range(3)] == pytest.approx([0.1, 0.2, 0.3])

    def test_the_ramp_blends_from_the_policys_last_action(self):
        from environments.shared.scripts.stance_gate_report import ConstantHold

        hold = ConstantHold((1.0,), handoff_steps=1, ramp_steps=4)
        predict = self._held(hold, actions=[[0.0]])
        out = [float(predict(None)[0]) for _ in range(6)]
        assert out[0] == pytest.approx(0.0)
        # Linear from the last commanded action to the constant, then held.
        assert out[1:5] == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert out[5] == pytest.approx(1.0)

    def test_a_ramp_from_reset_starts_at_the_home_keyframe(self):
        """There is no previous action at step 0, and `action = 0` IS home.

        Starting the ramp anywhere else would jump the pose on the first step,
        which is the transient the ramp exists to avoid.
        """
        from environments.shared.scripts.stance_gate_report import ConstantHold

        hold = ConstantHold((1.0,), handoff_steps=0, ramp_steps=2)
        predict = self._held(hold)
        assert [float(predict(None)[0]) for _ in range(3)] == pytest.approx([0.5, 1.0, 1.0])

    def test_reset_prevents_one_episode_starting_past_handoff(self):
        from environments.shared.scripts.stance_gate_report import ConstantHold

        hold = ConstantHold((0.5,), handoff_steps=1)
        predict = self._held(hold, actions=[[0.1], [0.2]])
        assert float(predict(None)[0]) == pytest.approx(0.1)
        assert float(predict(None)[0]) == pytest.approx(0.5)
        predict.reset()
        # Without the reset the second episode would open already held, and the
        # panel would average two different experiments.
        assert float(predict(None)[0]) == pytest.approx(0.2)


class TestConstantHoldExtraction:
    def test_it_reads_the_per_actuator_dc_in_index_order(self):
        from environments.shared.scripts.stance_gate_report import constant_hold_actions

        report = {
            "action": {
                "per_actuator": [
                    {"index": 2, "joint": "c", "dc": 0.3, "ac_rms": 0.0},
                    {"index": 0, "joint": "a", "dc": 0.1, "ac_rms": 0.0},
                    {"index": 1, "joint": "b", "dc": 0.2, "ac_rms": 0.0},
                ]
            }
        }
        assert constant_hold_actions(report) == pytest.approx((0.1, 0.2, 0.3))

    def test_a_report_without_the_block_fails_loudly(self):
        """Silently holding zeros would score the statue and call it the policy."""
        from environments.shared.scripts.stance_gate_report import constant_hold_actions

        with pytest.raises(StanceGateReportError, match="no per-actuator action statistics"):
            constant_hold_actions(_minimal_stance_report())


class TestConstantHoldVariants:
    def test_the_set_brackets_the_held_rows_with_two_controls(self):
        from environments.shared.scripts.stance_gate_report import constant_hold_variants

        variants = constant_hold_variants((0.5, -0.5), settle_steps=200, horizon=1000)
        by_label = {v.label: v for v in variants}
        assert not by_label["policy (control)"].holds
        assert by_label["policy (control)"].handoff_steps == 1000, "the control must never substitute"
        assert by_label["hold_zero (statue control)"].actions == (0.0, 0.0)
        assert by_label["hold_after_settle"].handoff_steps == 200
        assert by_label["hold_after_settle"].ramp_steps == 0
        assert by_label["hold_after_settle_ramped"].ramp_steps > 0
        assert by_label["hold_from_reset"].handoff_steps == 0

    def test_the_zero_control_matches_the_hold_width(self):
        from environments.shared.scripts.stance_gate_report import constant_hold_variants

        variants = constant_hold_variants(tuple([0.5] * 21), settle_steps=200, horizon=1000)
        assert all(len(v.actions) == 21 for v in variants)


class TestConstantHoldProbeCannotBeMistakenForTheVerdict:
    def test_it_writes_its_own_filenames(self, tmp_path):
        report = _held_report("hold_after_settle")
        written = write_stance_gate_report(tmp_path, report)
        assert (tmp_path / "stance_gate_probe_constant.txt").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()
        assert "stance_panel_csv" not in written

    def test_even_the_unmodified_control_is_marked_a_probe(self, tmp_path):
        """The control rolls the real policy, so nothing else would flag it.

        Ten episodes at the panel seeds is not the gate's 40, and a report on
        those filenames would be certified as if it were.
        """
        report = _held_report("policy (control)", holds=False, handoff=1000)
        write_stance_gate_report(tmp_path, report)
        assert (tmp_path / "stance_gate_probe_constant.txt").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()

    def test_the_banner_precedes_the_verdict(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        text = render_stance_gate_report(_held_report("hold_after_settle"))
        assert "PROBE" in text
        assert "not a gate result" in text
        assert text.index("PROBE") < text.index("GATE:")

    def test_the_control_row_says_it_was_not_held(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        text = render_stance_gate_report(_held_report("policy (control)", holds=False, handoff=1000))
        assert "unmodified control" in text

    def test_every_probe_marker_maps_to_a_distinct_stem(self):
        """Two probes sharing a stem would overwrite each other's artifact."""
        from environments.shared.scripts.stance_gate_report import _PROBE_MARKERS

        assert len(set(_PROBE_MARKERS.values())) == len(_PROBE_MARKERS)
        assert "stance_gate_report" not in set(_PROBE_MARKERS.values())

    def test_an_ordinary_report_is_still_not_a_probe(self):
        from environments.shared.scripts.stance_gate_report import probe_stem

        assert probe_stem(_minimal_stance_report()) is None


class TestConstantHoldProbeTable:
    @staticmethod
    def _standard(full_when_held):
        """The five standard variants, with the three measured holds set to *full_when_held*."""
        length = 1000.0 if full_when_held > 0 else 231.0
        return [
            _held_report("policy (control)", holds=False, handoff=1000),
            _held_report("hold_after_settle", full=full_when_held, length=length),
            _held_report("hold_after_settle_ramped", ramp=50, full=full_when_held, length=length),
            _held_report("hold_from_reset", handoff=0, ramp=50, full=full_when_held, length=length),
            _held_report("hold_zero (statue control)", handoff=0, source="zeros"),
        ]

    def test_the_statue_control_is_not_read_as_a_measurement(self, tmp_path):
        """It holds a constant too, but its standing is a known fact about the plant.

        Counting it among the measured holds would make an all-falling result
        read as mixed, which is the one reading that stops the probe concluding.
        """
        from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

        write_constant_hold_probe(tmp_path, self._standard(0.0), probe_episodes=10)
        assert "REQUIRES" in (tmp_path / "stance_gate_probe_constant.txt").read_text()

    def test_all_held_rows_standing_reads_as_an_optimisation_failure(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

        write_constant_hold_probe(tmp_path, self._standard(1.0), probe_episodes=10)
        text = (tmp_path / "stance_gate_probe_constant.txt").read_text()
        assert "does NOT" in text and "optimisation" in text
        assert "MODIFIED policy" in text

    def test_all_held_rows_falling_reads_as_load_bearing_feedback(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

        write_constant_hold_probe(tmp_path, self._standard(0.0), probe_episodes=10)
        text = (tmp_path / "stance_gate_probe_constant.txt").read_text()
        assert "REQUIRES" in text
        assert "penalising" in text, "the conclusion must warn against the obvious wrong fix"

    def test_a_mixed_result_refuses_to_conclude(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

        reports = self._standard(1.0)
        reports[1]["metrics"] = dict(reports[1]["metrics"], full_horizon_fraction=0.0, episode_length_mean=231.0)
        write_constant_hold_probe(tmp_path, reports, probe_episodes=10)
        text = (tmp_path / "stance_gate_probe_constant.txt").read_text()
        assert "Mixed" in text
        assert "step transient" in text, "the ramped variant is what tells the two apart"

    def test_it_never_writes_the_certification_evidence(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

        written = write_constant_hold_probe(tmp_path, self._standard(1.0), probe_episodes=10)
        assert set(written) == {"constant_hold_probe_txt", "constant_hold_probe_json"}
        assert not (tmp_path / "stance_panel_selected.csv").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()

    def test_the_json_carries_the_held_vector_not_the_controls_zeros(self, tmp_path):
        import json

        from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

        write_constant_hold_probe(tmp_path, self._standard(1.0), probe_episodes=10)
        payload = json.loads((tmp_path / "stance_gate_probe_constant.json").read_text())
        assert payload["hold_actions"] == [0.5, -0.25]
        assert payload["episodes_per_variant"] == 10
        assert [row["label"] for row in payload["variants"]][0] == "policy (control)"


class TestTheConstantHoldProbeIsWiredIntoTrainingArtifacts:
    def test_it_is_skipped_when_unconfigured(self, tmp_path, monkeypatch):
        from environments.shared.reporting import stage_artifacts

        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _boom)
        stage_artifacts._write_constant_hold_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            episodes=40,
            measured=_minimal_stance_report(),
        )
        assert not called

    def test_a_measured_report_without_dc_does_not_sink_the_run(self, tmp_path):
        """The gate report is the caller's return value; a probe must not raise."""
        from environments.shared.reporting import stage_artifacts

        stage_artifacts._write_constant_hold_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {"stance_probe_hold_constant": True}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            episodes=40,
            measured=_minimal_stance_report(),
        )
        assert not (tmp_path / "stance_gate_probe_constant.txt").exists()

    def test_the_config_key_is_reachable_from_a_toml(self):
        """Unregistered keys are rejected fail-closed, which makes them unreachable."""
        from environments.shared.curriculum.gate_schema import _DIAGNOSTIC_KEYS

        assert "stance_probe_hold_constant" in _DIAGNOSTIC_KEYS


def _report_with_actuators(entries):
    """A report carrying a per-actuator block, for the group selectors."""
    report = _minimal_stance_report()
    report["action"] = {
        "dc_rms": 0.7,
        "ac_rms": 0.36,
        "delta_per_step": 2.7,
        "jerk_per_step": 4.6,
        "effective_freq_hz": 22.6,
        "per_actuator": [
            {"index": index, "joint": joint, "dc": dc, "ac_rms": 0.0} for index, (joint, dc) in enumerate(entries)
        ],
    }
    return report


_TREX_ACTUATORS = [
    ("neck_pitch", -0.433),
    ("neck_yaw", -0.208),
    ("head_pitch", -1.0),
    ("r_hip_pitch", 0.052),
    ("r_hip_roll", 1.0),
    ("r_knee", 0.147),
    ("r_ankle", -0.133),
    ("r_toe_d2_joint", 1.0),
    ("l_hip_roll", -1.0),
    ("tail_1_yaw", 1.0),
    # Appended (not inserted) so the index assertions above stay valid: the
    # per-side groups must resolve a left knee/ankle, not just l_hip_roll,
    # or `left_leg` degenerates into a synonym for the left hip.
    ("l_knee", 0.31),
    ("l_ankle", -0.24),
]


class TestActuatorGroupSelection:
    def test_saturation_is_measured_not_named(self):
        from environments.shared.scripts.stance_gate_report import saturated_actuator_indices

        report = _report_with_actuators(_TREX_ACTUATORS)
        assert saturated_actuator_indices(report) == (2, 4, 7, 8, 9)

    def test_the_threshold_is_a_magnitude_so_both_limits_count(self):
        """`l_hip_roll` sits at -1.0 and is exactly as saturated as +1.0."""
        from environments.shared.scripts.stance_gate_report import saturated_actuator_indices

        report = _report_with_actuators([("a", -1.0), ("b", 0.5)])
        assert saturated_actuator_indices(report) == (0,)

    def test_groups_match_by_joint_name(self):
        from environments.shared.scripts.stance_gate_report import actuator_indices_matching

        report = _report_with_actuators(_TREX_ACTUATORS)
        assert actuator_indices_matching(report, ("tail",)) == (9,)
        assert actuator_indices_matching(report, ("toe",)) == (7,)
        assert actuator_indices_matching(report, ("hip_roll",)) == (4, 8)
        # head and neck are one region: three joints, two prefixes.
        assert actuator_indices_matching(report, ("head", "neck")) == (0, 1, 2)

    def test_release_returns_the_named_actuators_to_home(self):
        from environments.shared.scripts.stance_gate_report import constant_hold_released

        assert constant_hold_released((0.5, -0.5, 1.0), (0, 2)) == pytest.approx((0.0, -0.5, 0.0))

    def test_releasing_everything_is_the_statue(self):
        """The ablation is a path between two measured endpoints; this is one."""
        from environments.shared.scripts.stance_gate_report import constant_hold_released

        assert constant_hold_released((0.5, -0.5), (0, 1)) == pytest.approx((0.0, 0.0))

    def test_releasing_nothing_leaves_the_pose_alone(self):
        from environments.shared.scripts.stance_gate_report import constant_hold_released

        assert constant_hold_released((0.5, -0.5), ()) == pytest.approx((0.5, -0.5))


class TestReleaseAblationVariants:
    @staticmethod
    def _variants():
        from environments.shared.scripts.stance_gate_report import (
            constant_hold_actions,
            constant_hold_release_variants,
        )

        report = _report_with_actuators(_TREX_ACTUATORS)
        hold = constant_hold_actions(report)
        return {v.label: v for v in constant_hold_release_variants(hold, report, horizon=1000)}

    def test_every_group_gets_both_directions(self):
        """One side alone is not evidence: necessity and sufficiency differ.

        The tuple must track `_ACTUATOR_GROUPS` in stance_gate_report.py: the
        r7 additions (knees_ankles and the per-side legs) are the groups the
        passive-toes run's knee-localization conclusion rests on, and they
        went untested for a week because this list wasn't extended with them.
        """
        variants = self._variants()
        for group in (
            "saturated",
            "tail",
            "toes",
            "head_neck",
            "hip_rolls",
            "knees_ankles",
            "left_leg",
            "right_leg",
        ):
            assert f"release_{group}" in variants
            assert f"only_{group}" in variants

    def test_release_and_only_are_exact_complements(self):
        variants = self._variants()
        release = variants["release_tail"].actions
        only = variants["only_tail"].actions
        held = variants["hold_all"].actions
        for index in range(len(held)):
            # Exactly one side holds each actuator, and together they
            # reconstruct the full pose. A gap or an overlap would make the
            # necessity and sufficiency answers incomparable.
            assert (release[index] == 0.0) != (only[index] == 0.0) or held[index] == 0.0
            assert release[index] + only[index] == pytest.approx(held[index])

    def test_the_endpoints_are_included_as_controls(self):
        variants = self._variants()
        assert not variants["policy (control)"].holds
        assert all(value == 0.0 for value in variants["hold_zero (statue control)"].actions)
        assert variants["hold_all"].actions == variants["policy (control)"].actions

    def test_every_ablated_variant_is_commanded_from_reset(self):
        """A handoff partway through makes the statue endpoint stop being the statue.

        Measured, when this was wrong: handing off at settle_steps snaps some
        subset of joints from +/-1.000 to 0 in one step, the transient swamps
        the ablation, all thirteen variants land within 311-349 steps of each
        other, and the all-released row -- which is supposed to BE the statue
        and stand for 1000 -- falls at 323. Both endpoints on the same side of
        the answer leaves the path with no signal to locate.
        """
        for label, variant in self._variants().items():
            if label == "policy (control)":
                continue
            assert variant.handoff_steps == 0, f"{label} must be commanded from reset, not after a handoff"

    def test_the_statue_endpoint_commands_home_for_the_whole_episode(self):
        """Which is what makes it a control the other rows can be read against."""
        statue = self._variants()["hold_zero (statue control)"]
        assert statue.handoff_steps == 0
        assert all(value == 0.0 for value in statue.actions)

    def test_a_group_covering_every_actuator_is_not_an_ablation(self):
        """Releasing all of them is the statue, which is already a row."""
        from environments.shared.scripts.stance_gate_report import constant_hold_release_variants

        report = _report_with_actuators([("tail_1_pitch", 1.0), ("tail_2_pitch", 1.0)])
        labels = {v.label for v in constant_hold_release_variants((1.0, 1.0), report, horizon=1000)}
        assert "release_tail" not in labels
        assert "release_saturated" not in labels


def _ablation_row(label, full, *, holds=True, length=None, released=0):
    report = _held_report(
        label, holds=holds, full=full, length=length if length is not None else (1000.0 if full else 300.0)
    )
    report["hold_constant"]["actions"] = [0.0] * released + [0.5] * (21 - released)
    return report


class TestAblationVerdicts:
    @staticmethod
    def _render(pairs, **extra):
        from environments.shared.scripts.stance_gate_report import render_constant_hold_ablation

        reports = [_ablation_row("policy (control)", 1.0, holds=False), _ablation_row("hold_all", 0.0)]
        for group, (release_full, only_full) in pairs.items():
            reports.append(_ablation_row(f"release_{group}", release_full))
            reports.append(_ablation_row(f"only_{group}", only_full))
        reports.append(_ablation_row("hold_zero (statue control)", 1.0, released=21))
        return render_constant_hold_ablation(reports, probe_episodes=8, **extra)

    def test_necessary_and_sufficient_reads_as_the_cause(self):
        text, payload = self._render({"tail": (1.0, 0.0)})
        assert "CAUSE" in text
        verdict = next(v for v in payload["verdicts"] if v["group"] == "tail")
        assert verdict["necessary"] and verdict["sufficient"]

    def test_necessary_alone_only_contributes(self):
        """Removing it rescues the pose, but it cannot break the statue by itself."""
        text, payload = self._render({"tail": (1.0, 1.0)})
        assert "contributes" in text
        verdict = next(v for v in payload["verdicts"] if v["group"] == "tail")
        assert verdict["necessary"] and not verdict["sufficient"]

    def test_neither_reads_as_a_bystander(self):
        text, payload = self._render({"saturated": (0.0, 1.0)})
        assert "bystander" in text
        verdict = next(v for v in payload["verdicts"] if v["group"] == "saturated")
        assert not verdict["necessary"] and not verdict["sufficient"]

    def test_a_middling_result_refuses_to_round(
        self,
    ):
        """The two conclusions are opposite, so the gap must not be split."""
        text, payload = self._render({"tail": (0.5, 0.0)})
        assert "inconclusive" in text
        assert next(v for v in payload["verdicts"] if v["group"] == "tail")["verdict"].startswith("inconclusive")

    def test_each_group_is_judged_independently(self):
        _, payload = self._render({"tail": (1.0, 0.0), "toes": (0.0, 1.0)})
        by_group = {v["group"]: v["verdict"] for v in payload["verdicts"]}
        assert by_group["tail"].startswith("CAUSE")
        assert by_group["toes"].startswith("bystander")

    def test_the_table_warns_against_reading_one_side(self):
        text, _ = self._render({"tail": (1.0, 0.0)})
        assert "never one side" in text
        assert "bystander" in text, "the failure mode must be named, not just the success"

    def test_it_writes_its_own_filenames_and_no_evidence(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_constant_hold_ablation

        reports = [_ablation_row("policy (control)", 1.0, holds=False), _ablation_row("hold_all", 0.0)]
        written = write_constant_hold_ablation(tmp_path, reports, probe_episodes=8)
        assert set(written) == {"constant_hold_ablation_txt", "constant_hold_ablation_json"}
        assert (tmp_path / "stance_gate_probe_release.txt").exists()
        assert not (tmp_path / "stance_panel_selected.csv").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()
        # The other two probes' artifacts must not be collided with either.
        assert not (tmp_path / "stance_gate_probe_constant.txt").exists()
        assert not (tmp_path / "stance_gate_probe_filtered.txt").exists()

    def test_nothing_necessary_but_something_sufficient_reads_as_over_determined(self):
        """Measured on the T-Rex checkpoint: toes break it alone, yet removing
        them does not rescue it, because the rest of the pose breaks it too."""
        text, _ = self._render({"toes": (0.0, 0.0), "tail": (0.0, 1.0)})
        assert "OVER-DETERMINED" in text
        assert "toes" in text.split("Overall:")[1]
        assert "wrong question" in text

    def test_nothing_necessary_and_nothing_sufficient_says_regroup(self):
        text, _ = self._render({"tail": (0.0, 1.0), "toes": (0.0, 1.0)})
        assert "Regroup" in text

    def test_a_necessary_group_suppresses_the_over_determined_reading(self):
        text, _ = self._render({"tail": (1.0, 0.0)})
        assert "OVER-DETERMINED" not in text


class TestControlTargetsInDegrees:
    """`|action| = 1` is not one physical event, and the report must not imply it is.

    On the T-Rex a saturated tail target moves 12deg from its origin and a toe
    target moves 37.5deg (to an absolute +50deg endpoint). Ranking the
    per-actuator table by normalised action put twelve
    joints at exactly +/-1.000 at the top and gave no way to tell them apart --
    and the most extreme-looking of them, the tail, was later measured to be
    mechanically inert while the toes carried the whole effect.
    """

    @staticmethod
    def _stats(means, mapping, names=None):
        from environments.shared.scripts.stance_gate_report import (
            _commanded_angle,
            _new_action_stats,
            _reduce_action_stats,
        )

        stats = _new_action_stats()
        stats["sum"] = np.asarray(means, dtype=float)
        stats["sq_sum"] = np.asarray(means, dtype=float) ** 2
        stats["count"] = 1
        if len(mapping) == len(means):
            controls = np.asarray([_commanded_angle(entry, float(action)) for entry, action in zip(mapping, means)])
            stats["ctrl_sum"] = controls
            stats["ctrl_sq_sum"] = controls**2
            stats["ctrl_count"] = 1
        return _reduce_action_stats(stats, names or [f"j{i}" for i in range(len(means))], 0.01, mapping)

    #: tail_1_pitch (+/-12deg, home-centred) and r_toe_d2 (-25..+50deg,
    #: home +12.5deg), the two real T-Rex actuators the ablation separated.
    _TAIL = {
        "ctrl_min": -0.2094,
        "ctrl_max": 0.2094,
        "action_negative_ctrl": -0.2094,
        "action_zero_ctrl": 0.0,
        "action_positive_ctrl": 0.2094,
        "home_ctrl": 0.0,
        "home_qpos": 0.0,
    }
    _TOE = {
        "ctrl_min": -0.4363,
        "ctrl_max": 0.8727,
        "action_negative_ctrl": -0.4363,
        "action_zero_ctrl": 0.2182,
        "action_positive_ctrl": 0.8727,
        "home_ctrl": 0.2182,
        "home_qpos": 0.2182,
    }

    def test_the_same_saturated_action_is_a_different_angle_per_joint(self):
        action = self._stats([1.0, 1.0], [self._TAIL, self._TOE], names=["tail_1_pitch", "r_toe_d2_joint"])
        by_joint = {entry["joint"]: entry for entry in action["per_actuator"]}
        assert by_joint["tail_1_pitch"]["dc"] == pytest.approx(1.0)
        assert by_joint["r_toe_d2_joint"]["dc"] == pytest.approx(1.0)
        # Identical in normalised units, a factor of ~3 apart in the world.
        assert by_joint["tail_1_pitch"]["dc_deg"] == pytest.approx(12.0, abs=0.1)
        assert by_joint["r_toe_d2_joint"]["dc_deg"] == pytest.approx(37.5, abs=0.1)

    def test_the_table_is_ordered_by_degrees_not_by_normalised_action(self):
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        report = _minimal_stance_report()
        report["action"] = self._stats([1.0, 0.6], [self._TAIL, self._TOE], names=["tail_1_pitch", "r_toe_d2_joint"])
        text = render_stance_gate_report(report)
        # The tail saturates and the toe does not, yet the toe moves further.
        assert text.index("r_toe_d2_joint") < text.index("tail_1_pitch")
        assert "Ordered by DEGREES" in text

    def test_a_centred_mapping_reports_applied_control_rms(self):
        """The physical AC field comes from controls, even in the easy symmetric case."""
        from environments.shared.scripts.stance_gate_report import (
            _accumulate_action,
            _new_action_stats,
            _reduce_action_stats,
        )

        stats = _new_action_stats()
        half_span = 0.5 * (self._TOE["ctrl_max"] - self._TOE["action_zero_ctrl"])
        for action, control in (
            (-0.5, self._TOE["action_zero_ctrl"] - half_span),
            (0.5, self._TOE["action_zero_ctrl"] + half_span),
        ):
            _accumulate_action(stats, np.asarray([action]), np.asarray([control]))
        action = _reduce_action_stats(stats, ["r_toe_d2_joint"], 0.01, [self._TOE])
        assert action["per_actuator"][0]["ac_rms"] == pytest.approx(0.5)
        # 0.5 x half of 75deg = 18.75deg
        assert action["per_actuator"][0]["ac_rms_deg"] == pytest.approx(18.75, abs=0.05)

    def test_asymmetric_mapping_uses_actual_control_moments(self):
        """E[f(a)] and rms(f(a)) cannot be recovered by scaling action moments."""
        from environments.shared.scripts.stance_gate_report import (
            _accumulate_action,
            _new_action_stats,
            _reduce_action_stats,
        )

        mapping = {
            "ctrl_min": -1.0,
            "ctrl_max": 3.0,
            "action_negative_ctrl": -1.0,
            "action_zero_ctrl": 0.0,
            "action_positive_ctrl": 3.0,
            "home_ctrl": 0.0,
            "home_qpos": 0.0,
        }
        stats = _new_action_stats()
        for action, control in ((-0.5, -0.5), (0.25, 0.75)):
            _accumulate_action(stats, np.asarray([action]), np.asarray([control]))

        entry = _reduce_action_stats(stats, ["joint"], 0.01, [mapping])["per_actuator"][0]
        assert entry["dc"] == pytest.approx(-0.125)
        assert entry["ac_rms"] == pytest.approx(0.375)
        assert entry["dc_deg"] == pytest.approx(math.degrees(0.125))
        assert entry["ac_rms_deg"] == pytest.approx(math.degrees(0.625))

    def test_missing_applied_controls_does_not_guess_physical_moments(self):
        from environments.shared.scripts.stance_gate_report import _new_action_stats, _reduce_action_stats

        stats = _new_action_stats()
        stats["sum"] = np.asarray([0.25])
        stats["sq_sum"] = np.asarray([0.25])
        stats["count"] = 1
        entry = _reduce_action_stats(stats, ["joint"], 0.01, [self._TOE])["per_actuator"][0]
        assert "dc_deg" not in entry
        assert "ac_rms_deg" not in entry
        assert entry["action_zero_ctrl"] == pytest.approx(self._TOE["action_zero_ctrl"])

    def test_trex_mapping_uses_home_control_and_keeps_preload_separate(self):
        """Action zero is home control; ankle control-vs-pose preload is intentional."""
        from environments.shared.scripts.stance_gate_report import (
            _actuator_joint_names,
            _actuator_pose_mapping,
            render_stance_gate_report,
        )
        from environments.trex.envs.trex_env import TRexEnv

        env = TRexEnv()
        try:
            mapping = _actuator_pose_mapping(env)
            names = _actuator_joint_names(env)
        finally:
            env.close()

        report = _minimal_stance_report()
        report["action"] = self._stats(np.zeros(len(mapping)), mapping, names=names)
        by_joint = {entry["joint"]: entry for entry in report["action"]["per_actuator"]}
        assert all(entry["zero_offset_deg"] == pytest.approx(0.0, abs=1e-9) for entry in by_joint.values())
        assert by_joint["r_ankle"]["home_preload_deg"] == pytest.approx(5.5, abs=0.1)
        assert by_joint["l_ankle"]["home_preload_deg"] == pytest.approx(5.5, abs=0.1)
        assert by_joint["r_ankle"]["action_zero_ctrl"] == pytest.approx(by_joint["r_ankle"]["home_ctrl"])
        assert by_joint["r_ankle"]["home_ctrl"] != pytest.approx(by_joint["r_ankle"]["home_qpos"])
        assert by_joint["neck_pitch"]["home_preload_deg"] == pytest.approx(0.0, abs=0.01)
        assert by_joint["head_pitch"]["home_preload_deg"] == pytest.approx(0.0, abs=0.01)

        text = render_stance_gate_report(report)
        assert "differs from the named home control" not in text
        assert "Nominal control preload" in text
        assert "this is not policy displacement" in text

    def test_trex_asymmetric_piecewise_mapping_matches_the_environment(self):
        from environments.shared.scripts.stance_gate_report import (
            _actuator_joint_names,
            _actuator_pose_mapping,
            _commanded_angle,
        )
        from environments.trex.envs.trex_env import TRexEnv

        env = TRexEnv()
        try:
            by_joint = dict(zip(_actuator_joint_names(env), _actuator_pose_mapping(env)))
            neck = by_joint["neck_pitch"]
            for action, expected_deg in ((-1.0, -30.0), (-0.5, -15.0), (0.0, 0.0), (0.5, 20.0), (1.0, 40.0)):
                assert math.degrees(_commanded_angle(neck, action)) == pytest.approx(expected_deg, abs=0.01)
        finally:
            env.close()

    def test_trex_cross_zero_controls_are_reduced_after_scaling(self):
        from environments.shared.scripts.stance_gate_report import (
            _accumulate_action,
            _actuator_joint_names,
            _actuator_pose_mapping,
            _new_action_stats,
            _reduce_action_stats,
        )
        from environments.trex.envs.trex_env import TRexEnv

        env = TRexEnv()
        try:
            names = _actuator_joint_names(env)
            mapping = _actuator_pose_mapping(env)
            stats = _new_action_stats()
            for value in (-0.5, 0.5):
                action = np.full(env.action_space.shape, value, dtype=np.float64)
                _accumulate_action(stats, action, env._scale_action(action))
            by_joint = {
                entry["joint"]: entry for entry in _reduce_action_stats(stats, names, 0.01, mapping)["per_actuator"]
            }
        finally:
            env.close()

        assert by_joint["neck_pitch"]["dc"] == pytest.approx(0.0)
        assert by_joint["neck_pitch"]["ac_rms"] == pytest.approx(0.5)
        assert by_joint["neck_pitch"]["dc_deg"] == pytest.approx(2.5, abs=0.01)
        assert by_joint["neck_pitch"]["ac_rms_deg"] == pytest.approx(17.5, abs=0.01)

    def test_real_panel_records_the_controls_applied_by_trex(self):
        from environments.shared.scripts.stance_gate_report import run_panel

        result = run_panel(
            "trex",
            predict=lambda obs: np.zeros(15, dtype=np.float32),
            episodes=1,
            seed=3042,
            settle_steps=0,
            horizon=2,
            env_kwargs={"max_episode_steps": 2},
            control_dt=0.01,
        )
        by_joint = {entry["joint"]: entry for entry in result["action"]["per_actuator"]}
        assert by_joint["neck_pitch"]["dc_deg"] == pytest.approx(0.0, abs=1e-12)
        assert by_joint["r_ankle"]["dc_deg"] == pytest.approx(0.0, abs=1e-12)
        assert by_joint["r_ankle"]["home_preload_deg"] == pytest.approx(5.5, abs=0.1)

    def test_motor_controls_are_not_mislabeled_as_degrees(self):
        from environments.shared.scripts.stance_gate_report import run_panel

        action = np.zeros(22, dtype=np.float32)
        action[[6, 13]] = 1.0  # r/l claw motors, each with gear=50
        result = run_panel(
            "velociraptor",
            predict=lambda obs: action,
            episodes=1,
            seed=3042,
            settle_steps=0,
            horizon=2,
            env_kwargs={"max_episode_steps": 2},
            control_dt=0.01,
        )
        by_joint = {entry["joint"]: entry for entry in result["action"]["per_actuator"]}
        for joint in ("r_claw", "l_claw"):
            entry = by_joint[joint]
            assert entry["dc"] == pytest.approx(1.0)
            assert entry["ctrl_mean"] == pytest.approx(1.0)
            assert entry["ctrl_ac_rms"] == pytest.approx(0.0)
            assert not entry["angular_position_ctrl"]
            assert not {"dc_deg", "ac_rms_deg", "range_deg", "home_qpos", "home_preload_deg"} & entry.keys()

        assert by_joint["r_hip_pitch"]["angular_position_ctrl"]
        assert "dc_deg" in by_joint["r_hip_pitch"]

    def test_degrees_are_added_not_substituted(self):
        """`dc` still inverts through the action penalties; the degrees do not."""
        action = self._stats([0.8], [self._TOE])
        entry = action["per_actuator"][0]
        assert entry["dc"] == pytest.approx(0.8)
        assert "dc_deg" in entry and "range_deg" in entry
        assert action["dc_rms"] == pytest.approx(0.8)
        assert "dc_rms_deg" in action and "max_abs_dc_deg" in action

    def test_a_report_without_a_mapping_still_renders(self):
        """The model is unreachable from some callers; degrees are a convenience."""
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        report = _minimal_stance_report()
        report["action"] = self._stats([1.0], [], names=["tail_1_pitch"])
        assert "dc_deg" not in report["action"]["per_actuator"][0]
        assert "dc_rms_deg" not in report["action"]
        text = render_stance_gate_report(report)
        assert "tail_1_pitch" in text
        assert "Ordered by DEGREES" not in text

    def test_the_hold_probe_still_reads_normalised_dc(self):
        """`constant_hold_actions` commands actions, not angles."""
        from environments.shared.scripts.stance_gate_report import constant_hold_actions

        report = _minimal_stance_report()
        report["action"] = self._stats([0.4, -0.7], [self._TAIL, self._TOE])
        assert constant_hold_actions(report) == pytest.approx((0.4, -0.7))


def _impulse_report(speed, axis, *, length, full, reward=3000.0, step=200):
    report = _minimal_stance_report()
    report["episodes"] = 8
    report["impulse"] = {
        "kind": "root_impulse/v1",
        "step": step,
        "delta_v": [0.0, speed if axis.endswith("+y") else -speed, 0.0],
        "speed": speed,
        "axis_label": axis,
    }
    report["metrics"] = dict(
        report["metrics"], episode_length_mean=length, full_horizon_fraction=full, reward_mean=reward
    )
    report["terminations"] = {"truncated": 8} if full >= 1.0 else {"fallen": 8}
    return report


class TestRootImpulse:
    def test_the_impulse_translates_the_whole_animal(self):
        """qvel[0:3] is the free joint's world velocity; children compose from it."""
        from environments.shared.scripts.stance_gate_report import RootImpulse, _apply_root_impulse

        class _Data:
            qvel = np.zeros(10)

        class _Model:
            body_mass = np.array([1000.0, 500.0, 500.0])

        class _Env:
            unwrapped = type("U", (), {"data": _Data(), "model": _Model()})()

        env = _Env()
        momentum = _apply_root_impulse(env, RootImpulse(step=200, delta_v=(0.0, 1.5, 0.0)))
        assert env.unwrapped.data.qvel[:3] == pytest.approx([0.0, 1.5, 0.0])
        # Joint velocities are untouched: this is a shove, not a pose change.
        assert env.unwrapped.data.qvel[3:] == pytest.approx(np.zeros(7))
        assert momentum == pytest.approx(2000.0 * 1.5)

    def test_it_adds_rather_than_overwriting(self):
        """The animal is already moving; overwriting would erase its own state."""
        from environments.shared.scripts.stance_gate_report import RootImpulse, _apply_root_impulse

        class _Env:
            unwrapped = type(
                "U",
                (),
                {
                    "data": type("D", (), {"qvel": np.array([0.1, -0.2, 0.3, 0.0])})(),
                    "model": type("M", (), {"body_mass": np.array([1.0])})(),
                },
            )()

        env = _Env()
        _apply_root_impulse(env, RootImpulse(step=0, delta_v=(0.0, 1.0, 0.0)))
        assert env.unwrapped.data.qvel[:3] == pytest.approx([0.1, 0.8, 0.3])

    def test_the_speed_is_the_vector_magnitude(self):
        from environments.shared.scripts.stance_gate_report import RootImpulse

        assert RootImpulse(step=0, delta_v=(3.0, 4.0, 0.0)).speed == pytest.approx(5.0)

    def test_both_directions_are_swept(self):
        """The policy is asymmetric, so a one-sided sweep measures one side."""
        from environments.shared.scripts.stance_gate_report import impulse_variants

        labels = [v.axis_label for v in impulse_variants([1.0], step=200)]
        assert labels.count("lateral +y") == 1
        assert labels.count("lateral -y") == 1
        assert "none (control)" in labels

    def test_the_zero_control_is_always_present(self):
        from environments.shared.scripts.stance_gate_report import impulse_variants

        control = impulse_variants([2.0], step=200)[0]
        assert control.speed == 0.0

    def test_nonpositive_speeds_are_dropped_not_mirrored(self):
        from environments.shared.scripts.stance_gate_report import impulse_variants

        speeds = {v.speed for v in impulse_variants([1.0, 0.0, -1.0], step=200)}
        assert speeds == {0.0, 1.0}


class TestImpulseProbeIsNotAVerdict:
    def test_it_writes_its_own_filenames(self, tmp_path):
        report = _impulse_report(1.0, "lateral +y", length=400.0, full=0.0)
        written = write_stance_gate_report(tmp_path, report)
        assert (tmp_path / "stance_gate_probe_impulse.txt").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()
        assert "stance_panel_csv" not in written

    def test_the_banner_says_the_TASK_was_modified_not_the_policy(self):
        """Opposite direction from the other two probes, same consequence."""
        from environments.shared.scripts.stance_gate_report import render_stance_gate_report

        text = render_stance_gate_report(_impulse_report(1.0, "lateral +y", length=400.0, full=0.0))
        assert "MODIFIED TASK" in text
        assert "stage 1 has no disturbance" in text
        assert text.index("PROBE") < text.index("GATE:")

    def test_all_three_probes_have_distinct_stems(self):
        from environments.shared.scripts.stance_gate_report import _PROBE_MARKERS

        assert len(set(_PROBE_MARKERS.values())) == len(_PROBE_MARKERS) == 3


class TestImpulseRecoveryReading:
    @staticmethod
    def _sweep(policy_lengths, statue_lengths):
        """Build matched policy/statue sweeps from (length, full) pairs per row."""
        keys = [(0.0, "none (control)"), (1.0, "lateral +y"), (1.0, "lateral -y")]
        policy = [
            _impulse_report(speed, axis, length=length, full=full)
            for (speed, axis), (length, full) in zip(keys, policy_lengths)
        ]
        statue = [
            _impulse_report(speed, axis, length=length, full=full)
            for (speed, axis), (length, full) in zip(keys, statue_lengths)
        ]
        return policy, statue

    def test_a_large_positive_margin_reads_as_active_correction(self):
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (1000.0, 1.0), (1000.0, 1.0)],
            [(1000.0, 1.0), (300.0, 0.0), (280.0, 0.0)],
        )
        text, payload = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "ACTIVE CORRECTION EXISTS" in text
        assert "metric gap" in text
        assert payload["rows"][1]["margin_steps"] == pytest.approx(700.0)
        # Keyed off full recovery, not off a step margin.
        assert payload["recovery_envelopes"]["lateral +y"]["policy"] == pytest.approx(1.0)
        assert payload["recovery_envelopes"]["lateral +y"]["statue"] == 0.0

    def test_no_margin_reads_as_the_task_gap(self):
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (310.0, 0.0), (295.0, 0.0)],
            [(1000.0, 1.0), (300.0, 0.0), (290.0, 0.0)],
        )
        text, _ = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "Nothing has learned to recover" in text
        assert "STAGE1_SPLIT_PLAN" in text
        assert "No reweighting fixes this" in text

    def test_a_negative_margin_is_reported_as_such(self):
        """Worse than standing still is a finding, not a rounding of 'no effect'."""
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (120.0, 0.0), (110.0, 0.0)],
            [(1000.0, 1.0), (400.0, 0.0), (390.0, 0.0)],
        )
        text, _ = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "WORSE" in text

    def test_the_zero_impulse_control_is_excluded_from_the_margin(self):
        """Both sides reach the horizon there, so including it dilutes the signal."""
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (1000.0, 1.0), (1000.0, 1.0)],
            [(1000.0, 1.0), (300.0, 0.0), (300.0, 0.0)],
        )
        _, payload = render_impulse_probe(policy, statue, probe_episodes=8)
        disturbed = [row for row in payload["rows"] if row["speed"] > 0]
        assert len(disturbed) == 2
        assert all(row["margin_steps"] == pytest.approx(700.0) for row in disturbed)

    def test_it_writes_no_certification_evidence(self, tmp_path):
        from environments.shared.scripts.stance_gate_report import write_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (900.0, 0.5), (880.0, 0.5)],
            [(1000.0, 1.0), (300.0, 0.0), (290.0, 0.0)],
        )
        written = write_impulse_probe(tmp_path, policy, statue, probe_episodes=8)
        assert set(written) == {"impulse_probe_txt", "impulse_probe_json"}
        assert not (tmp_path / "stance_panel_selected.csv").exists()
        assert not (tmp_path / "stance_gate_report.txt").exists()

    def test_an_asymmetric_envelope_is_named_and_the_weak_side_governs(self):
        """Measured on the T-Rex: full recovery at 0.5 m/s one way, none the other.

        Pooled over directions that averages to a healthy-looking +135 steps
        and describes neither row. The envelope has to be reported per side.
        """
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (424.0, 0.0), (1000.0, 1.0)],
            [(1000.0, 1.0), (337.0, 0.0), (337.0, 0.0)],
        )
        text, payload = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "ASYMMETRIC" in text
        assert "WEAKER side" in text
        envelopes = payload["recovery_envelopes"]
        assert envelopes["lateral +y"]["policy"] == 0.0
        assert envelopes["lateral -y"]["policy"] == pytest.approx(1.0)

    def test_the_pooled_margin_is_reported_second_with_its_caveat(self):
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (424.0, 0.0), (1000.0, 1.0)],
            [(1000.0, 1.0), (337.0, 0.0), (337.0, 0.0)],
        )
        text, _ = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "reported second" in text
        assert text.index("recovery envelope") < text.index("Mean step margin")

    def test_a_symmetric_full_recovery_is_not_flagged_asymmetric(self):
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (1000.0, 1.0), (1000.0, 1.0)],
            [(1000.0, 1.0), (300.0, 0.0), (300.0, 0.0)],
        )
        text, _ = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "ACTIVE CORRECTION EXISTS" in text
        assert "ASYMMETRIC" not in text

    def test_a_statue_that_recovers_too_does_not_count_as_active_control(self):
        """Passive robustness is not correction; only the margin over it is."""
        from environments.shared.scripts.stance_gate_report import render_impulse_probe

        policy, statue = self._sweep(
            [(1000.0, 1.0), (1000.0, 1.0), (1000.0, 1.0)],
            [(1000.0, 1.0), (1000.0, 1.0), (1000.0, 1.0)],
        )
        text, _ = render_impulse_probe(policy, statue, probe_episodes=8)
        assert "ACTIVE CORRECTION EXISTS" not in text


class TestAllProbesAreWiredIntoTrainingArtifacts:
    """Every probe must run from `generate_stage_artifacts`, not only the CLI.

    The notebook calls `generate_stage_artifacts` and nothing else, so a probe
    reachable only from the command line produces nothing on a real run -- and
    a diagnostic nobody runs is a diagnostic that does not exist.
    """

    @staticmethod
    def _artifacts():
        from environments.shared.reporting import stage_artifacts

        return stage_artifacts

    @staticmethod
    def _probe_models_dir(tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "robust_best_model.zip").write_bytes(b"x")
        (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
        return models

    @staticmethod
    def _probe_stage_config() -> dict:
        return {
            "curriculum_kwargs": {"gate_kind": "stance_quality/v1", "min_eval_episodes": 40},
            "env_kwargs": {"max_episode_steps": 1000},
        }

    def _record_probes(self, monkeypatch) -> list:
        """Replace the four probe helpers with recorders; return the record."""
        calls: list = []
        for name in (
            "_write_filtered_action_probe",
            "_write_constant_hold_probe",
            "_write_constant_hold_ablation",
            "_write_impulse_probe",
        ):
            monkeypatch.setattr(
                self._artifacts(),
                name,
                lambda _name=name, **kwargs: calls.append((_name, kwargs)),
            )
        return calls

    def test_the_probe_runner_calls_every_probe(self, tmp_path, monkeypatch):
        """A behaviour test through the real call sites, so kwarg drift fails it."""
        calls = self._record_probes(monkeypatch)
        report = _minimal_stance_report()

        self._artifacts()._run_stance_probes(
            species="trex",
            stage=1,
            stage_config=self._probe_stage_config(),
            stage_dir=tmp_path,
            model_dir=self._probe_models_dir(tmp_path),
            report=report,
        )

        assert [name for name, _ in calls] == [
            "_write_filtered_action_probe",
            "_write_constant_hold_probe",
            "_write_constant_hold_ablation",
            "_write_impulse_probe",
        ]
        by_name = dict(calls)
        # The load-bearing arguments: the probes must annotate THIS report and
        # score THIS checkpoint, at the panel size the report was rolled at.
        assert by_name["_write_constant_hold_probe"]["measured"] is report
        assert by_name["_write_constant_hold_ablation"]["measured"] is report
        assert by_name["_write_impulse_probe"]["settle_steps"] == 200
        assert by_name["_write_filtered_action_probe"]["episodes"] == 40
        for name, kwargs in calls:
            assert kwargs["model_path"].endswith("robust_best_model.zip"), name
            assert kwargs["vecnorm_path"].endswith("robust_best_model_vecnorm.pkl"), name

    def test_the_probe_runner_skips_without_a_measured_report(self, tmp_path, monkeypatch):
        """No panel means nothing to probe -- and no checkpoint loads at all."""
        calls = self._record_probes(monkeypatch)

        self._artifacts()._run_stance_probes(
            species="trex",
            stage=1,
            stage_config=self._probe_stage_config(),
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            report=None,
        )

        assert calls == []

    def test_a_probe_wiring_failure_costs_that_probe_alone(self, tmp_path, monkeypatch, caplog):
        """Caller-side failures too: the helpers' internal handlers cannot see a
        TypeError raised at their own call site, which is exactly how a signature
        drift presents. The runner's per-probe handler must contain it."""
        calls = self._record_probes(monkeypatch)

        def _drifted_signature(**kwargs):
            raise TypeError("unexpected keyword argument")

        monkeypatch.setattr(self._artifacts(), "_write_constant_hold_probe", _drifted_signature)

        with caplog.at_level("WARNING"):
            self._artifacts()._run_stance_probes(
                species="trex",
                stage=1,
                stage_config=self._probe_stage_config(),
                stage_dir=tmp_path,
                model_dir=self._probe_models_dir(tmp_path),
                report=_minimal_stance_report(),
            )

        assert [name for name, _ in calls] == [
            "_write_filtered_action_probe",
            "_write_constant_hold_ablation",
            "_write_impulse_probe",
        ]
        assert "Stance probe (constant hold) failed" in caplog.text

    def test_the_verdict_is_recorded_before_any_probe_runs(self):
        """Probes are pure diagnostics and run dead last: the recorded verdict,
        the summary, and the replays must never sit behind ~300 probe episodes,
        and a probe failure must have nothing left to sink."""
        import inspect

        source = inspect.getsource(self._artifacts().generate_stage_artifacts)
        for call in ("_write_stance_gate_report(", "_apply_stage_gate(", "_run_stance_probes("):
            assert call in source, f"generate_stage_artifacts no longer calls {call.rstrip('(')}"
        assert (
            source.index("_apply_stage_gate(")
            < source.index("write_stage_summary(")
            < source.index("_run_stance_probes(")
        ), "probes must run after the verdict is recorded and the summary written"

    def test_every_probe_config_key_is_reachable_from_a_toml(self):
        """Unregistered keys are rejected fail-closed, making them unsettable."""
        from environments.shared.curriculum.gate_schema import _DIAGNOSTIC_KEYS

        for key in (
            "stance_probe_filter_hz",
            "stance_probe_hold_constant",
            "stance_probe_release_ablation",
            "stance_probe_impulse_speeds",
        ):
            assert key in _DIAGNOSTIC_KEYS

    def test_trex_stage_1_switches_all_four_on(self):
        """The species this was built for must actually produce the artifacts."""
        from environments.shared.config import load_stage_config

        curriculum = load_stage_config("trex", 1)["curriculum_kwargs"]
        assert curriculum.get("stance_probe_filter_hz")
        assert curriculum.get("stance_probe_hold_constant") is True
        assert curriculum.get("stance_probe_release_ablation") is True
        assert curriculum.get("stance_probe_impulse_speeds")

    def test_the_ablation_is_skipped_when_unconfigured(self, tmp_path, monkeypatch):
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _boom)
        self._artifacts()._write_constant_hold_ablation(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            measured=_minimal_stance_report(),
        )
        assert not called

    def test_the_impulse_probe_is_skipped_when_unconfigured(self, tmp_path, monkeypatch):
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _boom)
        self._artifacts()._write_impulse_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            settle_steps=200,
        )
        assert not called

    def test_unusable_impulse_speeds_are_dropped_not_fatal(self, tmp_path, monkeypatch):
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", _boom)
        self._artifacts()._write_impulse_probe(
            species="trex",
            stage=1,
            stage_config={"curriculum_kwargs": {"stance_probe_impulse_speeds": ["x", -1.0, 0.0]}},
            stage_dir=tmp_path,
            model_path="m.zip",
            vecnorm_path="v.pkl",
            settle_steps=200,
        )
        assert not called, "no usable speed means no sweep, not a crash"

    def test_a_failing_probe_does_not_sink_the_gate_report(self, tmp_path):
        """The gate report is what certifies the stage; a probe must never cost it."""
        for name, kwargs in (
            ("_write_constant_hold_ablation", {"measured": _minimal_stance_report()}),
            ("_write_impulse_probe", {"settle_steps": 200}),
        ):
            getattr(self._artifacts(), name)(
                species="trex",
                stage=1,
                stage_config={
                    "curriculum_kwargs": {
                        "stance_probe_release_ablation": True,
                        "stance_probe_impulse_speeds": [1.0],
                    }
                },
                stage_dir=tmp_path,
                model_path="does-not-exist.zip",
                vecnorm_path=None,
                **kwargs,
            )
