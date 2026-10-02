"""The curriculum must judge the saved handoff on fresh physical evidence."""

from types import SimpleNamespace

import pytest

from environments.shared.gait import report
from environments.shared.reporting import gates
from environments.shared.train_base import _judge_selected_gait_handoff


@pytest.mark.parametrize("passed", [True, False])
def test_fresh_selected_handoff_is_authoritative(tmp_path, monkeypatch, passed):
    calls = []
    config = {
        "env_kwargs": {"reset_noise_scale": 0.03},
        "curriculum_kwargs": {"min_eval_episodes": 40, "gait_panel_seed_start": 3042},
    }

    def write(cfg, stage_config, model, normalization, output, **kwargs):
        calls.append((cfg, stage_config, model, normalization, output, kwargs))
        # This boolean is deliberately the opposite of the independent
        # reader's result. Neither live callback nor producer decides.
        return {"certified": not passed}

    statistics = {"passed": passed, "selected_gait_success_count": 40 if passed else 0}
    failures = [] if passed else ["gait lower confidence bound below target"]
    monkeypatch.setattr(report, "write_gait_report", write)
    monkeypatch.setattr(gates, "gait_statistics", lambda *args: (statistics, []))
    monkeypatch.setattr(gates, "evaluate_stage_gate", lambda *args, **kwargs: (passed, failures))
    cfg = SimpleNamespace(species="trex")
    result = _judge_selected_gait_handoff(
        cfg,
        config,
        tmp_path,
        stage=2,
        model_stem=str(tmp_path / "models" / "robust_best_model"),
        vecnorm_path=str(tmp_path / "models" / "robust_best_model_vecnorm.pkl"),
        algorithm="ppo",
    )
    assert result == (passed, failures, statistics)
    assert len(calls) == 1
    assert calls[0][1] == dict(config, _gait_stage=2)
    assert calls[0][2].endswith("robust_best_model.zip")
    assert calls[0][3].endswith("robust_best_model_vecnorm.pkl")
    assert calls[0][5] == {"episodes": 40, "seed": 3042, "algorithm": "ppo"}
    assert "_gait_stage" not in config


def test_replaced_pair_refuses_even_after_successful_roll(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "write_gait_report", lambda *args, **kwargs: {"certified": True})
    monkeypatch.setattr(gates, "gait_statistics", lambda *args: (None, ["selected checkpoint hash changed"]))

    def forbidden_judge(*args, **kwargs):
        pytest.fail("unbound gait statistics reached the judge")

    monkeypatch.setattr(gates, "evaluate_stage_gate", forbidden_judge)
    result = _judge_selected_gait_handoff(
        SimpleNamespace(species="trex"),
        {"curriculum_kwargs": {"min_eval_episodes": 40, "gait_panel_seed_start": 3042}},
        tmp_path,
        stage=2,
        model_stem="selected",
        vecnorm_path="selected_vecnorm.pkl",
        algorithm="ppo",
    )
    assert result == (False, ["selected checkpoint hash changed"], None)


def test_failed_fresh_panel_preserves_training_and_refuses(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("missing matched normalization")

    monkeypatch.setattr(report, "write_gait_report", fail)
    monkeypatch.setattr(gates, "gait_statistics", lambda *args: pytest.fail("a failed panel cannot be reused"))
    passed, failures, statistics = _judge_selected_gait_handoff(
        SimpleNamespace(species="trex"),
        {"curriculum_kwargs": {"min_eval_episodes": 40, "gait_panel_seed_start": 3042}},
        tmp_path,
        stage=2,
        model_stem="selected",
        vecnorm_path="missing.pkl",
        algorithm="ppo",
    )
    assert passed is False and statistics is None
    assert "missing matched normalization" in failures[0]
