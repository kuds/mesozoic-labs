"""Tests for the JAX curriculum gate logic (jax_curriculum.py).

These tests run without JAX installed — check_stage_gate is pure Python.
"""

from __future__ import annotations

import pytest

from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
from environments.shared.jax_curriculum import check_stage_gate

#: The gate declaration every real stage config carries. Tests that exercise a
#: working gate must include it, because an undeclared gate is now fatal.
_GATE = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "reward_and_length/v1"}


class TestCheckStageGate:
    def test_passes_when_reward_meets_threshold(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_episode_return": 150.0}, stage_config) is True

    def test_fails_when_reward_below_threshold(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_episode_return": 50.0}, stage_config) is False

    def test_reads_curriculum_kwargs_key_from_real_config(self):
        """Regression: the gate must read the loader's 'curriculum_kwargs'
        key.  It previously read 'curriculum' (the raw TOML table name),
        which load_stage_config never emits — min_reward defaulted to inf
        and every curriculum run stopped after stage 1.
        """
        from environments.shared.config import load_stage_config

        stage_config = load_stage_config("velociraptor", 1)
        assert "curriculum_kwargs" in stage_config
        min_reward = stage_config["curriculum_kwargs"]["min_avg_reward"]
        # The shipped config also declares a length floor, which the gate now
        # enforces, so hold length comfortably clear to isolate the reward
        # threshold this test is about.
        passing_length = stage_config["curriculum_kwargs"]["min_avg_episode_length"] + 1.0
        assert (
            check_stage_gate(
                {"mean_episode_return": min_reward + 1.0, "mean_episode_length": passing_length},
                stage_config,
            )
            is True
        )
        assert (
            check_stage_gate(
                {"mean_episode_return": min_reward - 1.0, "mean_episode_length": passing_length},
                stage_config,
            )
            is False
        )

    def test_prefers_episode_return_over_per_step_reward(self):
        """Regression: JaxTrainer eval_metrics carry BOTH a per-step
        'mean_reward' (~0.5-2 for a good policy) and an episode-level
        'mean_episode_return'.  The TOML min_avg_reward thresholds are
        episode-level, so the gate must use the episode return — gating on
        the per-step value failed every well-trained policy.
        """
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        realistic_trainer_metrics = {"mean_reward": 1.2, "mean_episode_return": 240.0}
        assert check_stage_gate(realistic_trainer_metrics, stage_config) is True

    def test_a_missing_episode_return_is_fatal_not_a_per_step_substitute(self):
        """Substituting the per-step value compares different units.

        It used to fall back to ``mean_reward`` with a warning, which meant a
        gate verdict computed from numbers three orders of magnitude apart --
        right by accident when the per-step value happened to land on the
        correct side of the threshold, and wrong the rest of the time. Against
        trex stage 1's rail it gives 3.3 < 1950, so the stage never advances
        and the only clue is a rail failure that reads like a bad policy.
        """
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}

        with pytest.raises(GateSchemaError) as excinfo:
            check_stage_gate({"mean_reward": 150.0}, stage_config)
        message = str(excinfo.value)
        assert "mean_episode_return" in message
        # Names the number it refused to substitute and what it would have
        # been compared against, so the mismatch is diagnosable from the log.
        assert "150" in message
        assert "100" in message

        with pytest.raises(GateSchemaError):
            check_stage_gate({}, stage_config)

    def test_the_rail_is_optional_so_an_absent_return_is_not_fatal(self):
        """``stance_quality/v1`` may decline to set a reward rail.

        Nothing is compared then, so the missing metric cannot be wrong.
        """
        from environments.shared.jax_curriculum import episode_return_for_gate

        assert episode_return_for_gate({}, threshold=float("-inf")) == float("-inf")

    def test_the_episode_return_is_used_when_present(self):
        from environments.shared.jax_curriculum import episode_return_for_gate

        assert episode_return_for_gate({"mean_reward": 1.2, "mean_episode_return": 240.0}, threshold=100.0) == 240.0

    def test_missing_gate_declaration_is_fatal(self):
        """Fail closed: an undeclared gate must not advance by default.

        This previously logged a warning and returned ``True``, so a stage
        whose reward threshold had been removed advanced unconditionally.
        Combined with the SB3 path's permissive StageThreshold defaults, that
        made "no gate" indistinguishable from "gate satisfied" on both
        backends. See docs/STAGE1_SPLIT_PLAN.md section 5.2.
        """
        with pytest.raises(GateSchemaError, match="no gate_kind declared"):
            check_stage_gate({"mean_reward": 0.0}, {})
        with pytest.raises(GateSchemaError, match="no gate_kind declared"):
            check_stage_gate({"mean_reward": 0.0}, {"curriculum_kwargs": {}})

    def test_declared_gate_without_a_reward_threshold_is_fatal(self):
        """A declared gate with nothing to check must raise, not pass.

        The raise now comes from the schema itself rather than a JAX-side
        check, so the SB3 path rejects the same config instead of falling back
        to StageThreshold's permissive ``min_avg_reward = -inf`` default.
        """
        with pytest.raises(GateSchemaError, match="missing required threshold"):
            check_stage_gate({"mean_episode_return": 0.0}, {"curriculum_kwargs": dict(_GATE)})

    def test_non_advancing_pilot_cannot_be_used_to_advance(self):
        """gate_kind "none/v1" is the recorded way to run without a gate."""
        pilot = {"curriculum_kwargs": {"gate_schema_version": 1, "gate_kind": "none/v1"}}
        with pytest.raises(GateSchemaError, match="non-advancing pilot"):
            check_stage_gate({"mean_episode_return": 1e9}, pilot)

    def test_unknown_gate_kind_is_fatal(self):
        bogus = {"curriculum_kwargs": {"gate_schema_version": 1, "gate_kind": "made_up/v9"}}
        with pytest.raises(GateSchemaError, match="unknown gate_kind"):
            check_stage_gate({"mean_episode_return": 1e9}, bogus)

    def test_misspelled_threshold_is_fatal_rather_than_ignored(self):
        """A typo used to silently disable the threshold it meant to set."""
        typo = {"curriculum_kwargs": dict(_GATE, min_avg_rewrad=100.0)}
        with pytest.raises(GateSchemaError, match="unrecognised"):
            check_stage_gate({"mean_episode_return": 0.0}, typo)


class TestLengthThresholdIsEnforced:
    """The length half of ``reward_and_length/v1`` must bind on JAX too.

    ``check_stage_gate`` used to read only ``min_avg_reward`` while the SB3
    ``CurriculumManager`` enforced reward AND length, so a stage carrying
    ``min_avg_episode_length`` advanced on reward alone under JAX.  That is the
    "one backend silently ignores a gate the other enforces" divergence
    ``gate_schema`` exists to prevent, and it became load-bearing when stage 1
    started encoding its full-horizon floor in that field.
    """

    def test_short_episodes_fail_even_with_sufficient_reward(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        metrics = {"mean_episode_return": 5000.0, "mean_episode_length": 300.0}
        assert check_stage_gate(metrics, stage_config) is False

    def test_passes_when_both_thresholds_are_met(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        metrics = {"mean_episode_return": 5000.0, "mean_episode_length": 1000.0}
        assert check_stage_gate(metrics, stage_config) is True

    def test_reward_still_binds_independently(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        metrics = {"mean_episode_return": 50.0, "mean_episode_length": 1000.0}
        assert check_stage_gate(metrics, stage_config) is False

    def test_declared_length_gate_without_the_metric_is_fatal(self):
        """Half-enforcing must raise rather than pass on reward alone."""
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        with pytest.raises(GateSchemaError, match="no mean_episode_length"):
            check_stage_gate({"mean_episode_return": 5000.0}, stage_config)

    def test_reward_only_config_is_unaffected(self):
        """A stage that declares no length threshold keeps its old behaviour."""
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_episode_return": 150.0}, stage_config) is True

    def test_every_stage_one_config_binds_a_full_horizon_floor(self):
        """Every shipped stage-1 config enforces a full-horizon requirement.

        The two gate kinds express it differently -- ``reward_and_length/v1``
        as a mean episode-step floor, ``stance_quality/v1`` as a fraction of
        episodes reaching the horizon -- but neither may be satisfied by
        unbounded reward alone. That is the property worth pinning: the
        specific field is an implementation detail of the kind.
        """
        from environments.shared.config import load_stage_config

        for species in ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus"):
            cfg = load_stage_config(species, 1)
            curriculum = cfg["curriculum_kwargs"]
            if curriculum["gate_kind"] == STANCE_GATE_KIND:
                floor = curriculum["min_full_horizon_fraction"]
                generous = {
                    "mean_episode_return": 1e9,
                    "n_eval_episodes": curriculum["min_eval_episodes"],
                    "full_horizon_fraction": floor - 0.01,
                    "n_duty_episodes": curriculum["min_eval_episodes"],
                    "mean_unsupported_duty": 0.0,
                    "unsupported_duty_ucb": 0.0,
                }
            else:
                floor = curriculum["min_avg_episode_length"]
                generous = {"mean_episode_return": 1e9, "mean_episode_length": floor - 1}
            assert check_stage_gate(generous, cfg) is False, species


class TestStanceRailUnits:
    """The stance gate's reward rail is episode-level too.

    Its branch used to fall back to the per-step ``mean_reward`` *silently* --
    no warning at all, unlike the ``reward_and_length`` branch. Against trex
    stage 1's 1950 rail that is 3.3 < 1950: the stage never advances, and the
    only trace is an INFO-level rail failure indistinguishable from a policy
    that genuinely threw away its return.
    """

    STANCE = {
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "gate_kind": "stance_quality/v1",
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "min_eval_episodes": 40,
    }

    def _metrics(self, **overrides):
        metrics = {
            "n_eval_episodes": 40,
            "full_horizon_fraction": 1.0,
            "n_duty_episodes": 40,
            "mean_unsupported_duty": 0.001,
            "unsupported_duty_ucb": 0.002,
        }
        metrics.update(overrides)
        return metrics

    def test_a_declared_rail_without_the_episode_return_is_fatal(self):
        config = {"curriculum_kwargs": dict(self.STANCE, min_avg_reward=1950.0)}
        with pytest.raises(GateSchemaError) as excinfo:
            check_stage_gate(self._metrics(mean_reward=3.27), config)
        message = str(excinfo.value)
        assert "3.27" in message
        assert "1950" in message

    def test_no_rail_means_the_episode_return_is_not_required(self):
        config = {"curriculum_kwargs": dict(self.STANCE)}
        assert check_stage_gate(self._metrics(mean_reward=3.27), config) is True

    def test_the_rail_is_checked_against_the_episode_return(self):
        config = {"curriculum_kwargs": dict(self.STANCE, min_avg_reward=1950.0)}
        assert check_stage_gate(self._metrics(mean_reward=3.27, mean_episode_return=3271.8), config) is True
        assert check_stage_gate(self._metrics(mean_reward=3.27, mean_episode_return=900.0), config) is False


class TestJaxKwargsPlumbing:
    """[jax] keys must reach train_fn or fail loudly — never go silently inert.

    ``ramp_attr`` and ``obs_rms_decay_on_resume`` were set in eight TOMLs and
    honored only by the notebook: both library entry points dropped them with
    no warning, while the config comments asserted they worked. These tests
    run without JAX installed, like the rest of this file.
    """

    def test_every_shipped_jax_table_validates(self):
        from environments.shared.config import load_stage_config
        from environments.shared.jax_curriculum import validate_jax_kwargs

        for species in ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus"):
            for stage in (1, 2, 3):
                cfg = load_stage_config(species, stage)
                validate_jax_kwargs(cfg.get("jax_kwargs", {}), source=f"{species} stage {stage} [jax]")

    def test_an_unknown_key_is_rejected_by_name(self):
        from environments.shared.jax_curriculum import validate_jax_kwargs

        with pytest.raises(ValueError, match="obs_rms_decay_on_resmue"):
            validate_jax_kwargs({"obs_rms_decay_on_resmue": 0.01}, source="test [jax]")

    def test_the_key_map_only_names_train_jax_parameters(self):
        """A mapping to a parameter train_jax does not accept is a silent drop
        one level down; inspect the real signature so the two cannot drift."""
        import inspect

        from environments.shared.jax_curriculum import _JAX_KEY_MAP
        from environments.shared.jax_training import train_jax

        accepted = set(inspect.signature(train_jax).parameters)
        unmapped = {param for param in _JAX_KEY_MAP.values() if param not in accepted}
        assert not unmapped, f"_JAX_KEY_MAP targets parameters train_jax lacks: {sorted(unmapped)}"

    def test_run_curriculum_decays_carried_obs_stats_and_passes_ramp_attr(self, monkeypatch):
        """The stage-2 resume must decay the carried normalization count with
        the entered stage's configured factor, and the TOML ramp_attr must
        reach train_fn instead of being dropped."""
        import sys
        import types

        from environments.shared.config import load_stage_config
        from environments.shared.jax_curriculum import run_curriculum

        decay_calls: list = []
        stub = types.ModuleType("environments.shared.jax_normalization")

        def _stub_decay(stats, decay_factor):
            decay_calls.append((stats, decay_factor))
            return ("decayed", stats)

        stub.decay_running_stats = _stub_decay
        monkeypatch.setitem(sys.modules, "environments.shared.jax_normalization", stub)

        stage1_gate = load_stage_config("velociraptor", 1)["curriculum_kwargs"]
        passing_metrics = {
            "mean_episode_return": float(stage1_gate["min_avg_reward"]) + 1.0,
            "mean_episode_length": float(stage1_gate["min_avg_episode_length"]) + 1.0,
        }

        seen: dict[int, dict] = {}

        def fake_train(species, stage, **kwargs):
            seen[stage] = kwargs
            return (f"params-s{stage}", dict(passing_metrics), f"stats-s{stage}")

        run_curriculum("velociraptor", fake_train, stages=(1, 2))

        assert "init_obs_stats" not in seen[1], "stage 1 starts from fresh stats"
        stage2_jax = load_stage_config("velociraptor", 2)["jax_kwargs"]
        assert decay_calls == [("stats-s1", float(stage2_jax["obs_rms_decay_on_resume"]))]
        assert seen[2]["init_obs_stats"] == ("decayed", "stats-s1")
        assert seen[2]["init_params"] == "params-s1"
        assert seen[2]["ramp_attr"] == stage2_jax["ramp_attr"]
