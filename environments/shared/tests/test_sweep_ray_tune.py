"""Pure configuration tests for the Ray Tune training backend."""

from environments.shared.scripts.sweep.ray_tune import apply_collapse_overrides


def test_collapse_overrides_preserve_stage_settings_without_mutating_input():
    stage_config = {
        "env_kwargs": {"forward_vel_weight": 2.0},
        "curriculum_kwargs": {
            "collapse_min_evals": 20,
            "collapse_patience": 10,
            "collapse_drop_fraction": 0.5,
            "collapse_peak_floor": 100.0,
            "collapse_smoothing_window": 7,
        },
    }

    resolved = apply_collapse_overrides(stage_config, min_evals=30, patience=12)

    assert resolved["curriculum_kwargs"] == {
        "collapse_min_evals": 30,
        "collapse_patience": 12,
        "collapse_drop_fraction": 0.5,
        "collapse_peak_floor": 100.0,
        "collapse_smoothing_window": 7,
    }
    assert resolved["env_kwargs"] == stage_config["env_kwargs"]
    assert stage_config["curriculum_kwargs"]["collapse_min_evals"] == 20
    assert stage_config["curriculum_kwargs"]["collapse_patience"] == 10


def test_missing_overrides_inherit_stage_values():
    stage_config = {
        "curriculum_kwargs": {
            "collapse_min_evals": 20,
            "collapse_patience": 10,
        }
    }

    resolved = apply_collapse_overrides(stage_config)

    assert resolved == stage_config
    assert resolved is not stage_config
    assert resolved["curriculum_kwargs"] is not stage_config["curriculum_kwargs"]


def test_non_positive_overrides_are_ignored_independently():
    stage_config = {
        "curriculum_kwargs": {
            "collapse_min_evals": 20,
            "collapse_patience": 10,
        }
    }

    resolved = apply_collapse_overrides(stage_config, min_evals=0, patience=12)

    assert resolved["curriculum_kwargs"]["collapse_min_evals"] == 20
    assert resolved["curriculum_kwargs"]["collapse_patience"] == 12

    resolved = apply_collapse_overrides(stage_config, min_evals=24, patience=-1)

    assert resolved["curriculum_kwargs"]["collapse_min_evals"] == 24
    assert resolved["curriculum_kwargs"]["collapse_patience"] == 10


def test_positive_override_initializes_missing_curriculum_settings():
    stage_config = {"env_kwargs": {"forward_vel_weight": 2.0}}

    resolved = apply_collapse_overrides(stage_config, min_evals=24)

    assert resolved["curriculum_kwargs"] == {"collapse_min_evals": 24}
    assert "curriculum_kwargs" not in stage_config
