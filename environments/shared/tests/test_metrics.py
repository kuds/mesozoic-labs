"""Tests for LocomotionMetrics."""

import numpy as np
import pytest

from environments.shared.metrics import FALLBACK_SUCCESS_KEYS, LocomotionMetrics
from environments.shared.model_id import ModelId


class TestLocomotionMetrics:
    """Test locomotion metrics computation."""

    @pytest.fixture
    def metrics(self):
        return LocomotionMetrics()

    def test_empty_compute_returns_nan_defaults(self, metrics):
        result = metrics.compute()
        assert result["episode_length"] == 0
        assert np.isnan(result["mean_forward_velocity"])
        assert np.isnan(result["total_distance"])
        assert "mean_forward_velocity" in result

    def test_forward_velocity_stats(self, metrics):
        for _ in range(10):
            metrics.record_step({"forward_vel": 2.0}, reward=1.0)
        result = metrics.compute()
        assert result["mean_forward_velocity"] == pytest.approx(2.0)
        assert result["std_forward_velocity"] == pytest.approx(0.0, abs=1e-6)

    def test_total_distance(self, metrics):
        metrics.dt = 0.1
        for _ in range(10):
            metrics.record_step({"forward_vel": 1.0}, reward=1.0)
        result = metrics.compute()
        assert result["total_distance"] == pytest.approx(1.0)  # 10 * 1.0 * 0.1

    def test_gait_symmetry_stride_frequency_and_cost_of_transport_are_not_reported(self, metrics):
        """They measured the wrong thing (in-phase hopping scored symmetry 1.0, a double-support walk 0 Hz, and
        the transport cost divided the action penalty by distance), and no gate or record read them."""
        for step in range(20):
            contact = float(step % 4 < 2)
            metrics.record_step(
                {
                    "forward_vel": 1.0,
                    "reward_energy": -0.5,
                    "l_foot_contact": contact,
                    "r_foot_contact": contact,
                    "rl_foot_contact": contact,
                    "rr_foot_contact": contact,
                }
            )
        removed = {"gait_symmetry", "quad_gait_symmetry", "stride_frequency", "cost_of_transport", "total_energy"}
        assert not removed & set(metrics.compute())
        assert not removed & set(LocomotionMetrics().compute())

    def test_pelvis_height_stats(self, metrics):
        for h in [0.5, 0.6, 0.5, 0.6]:
            metrics.record_step({"forward_vel": 0.0, "pelvis_height": h})
        result = metrics.compute()
        assert result["mean_pelvis_height"] == pytest.approx(0.55)
        assert result["pelvis_height_variance"] > 0

    def test_prey_distance_tracking(self, metrics):
        distances = [5.0, 4.0, 3.0, 2.0, 1.0, 0.3]
        for d in distances:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        assert result["initial_prey_distance"] == 5.0
        assert result["final_prey_distance"] == 0.3
        assert result["min_prey_distance"] == 0.3

    def test_time_to_target(self, metrics):
        metrics.dt = 0.5
        distances = [5.0, 3.0, 1.0, 0.4]  # reaches <0.5 at step 3
        for d in distances:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        assert result["time_to_target"] == pytest.approx(3 * 0.5)  # 1.5 seconds
        assert result["target_reached"] == 1.0

    def test_time_to_target_never_reached(self, metrics):
        import math

        for d in [5.0, 4.0, 3.0]:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        # NaN (not a -1 sentinel) so aggregate_episodes excludes it from
        # mean_time_to_target; target_reached carries the reach fraction.
        assert math.isnan(result["time_to_target"])
        assert result["target_reached"] == 0.0

    def test_reward_statistics(self, metrics):
        rewards = [1.0, 2.0, 3.0, 4.0]
        for r in rewards:
            metrics.record_step({"forward_vel": 1.0}, reward=r)
        result = metrics.compute()
        assert result["total_reward"] == pytest.approx(10.0)
        assert result["mean_step_reward"] == pytest.approx(2.5)
        assert result["episode_length"] == 4

    def test_reset_clears_data(self, metrics):
        metrics.record_step({"forward_vel": 1.0})
        metrics.reset()
        result = metrics.compute()
        assert result["episode_length"] == 0
        assert np.isnan(result["mean_forward_velocity"])

    def test_velocity_consistency_perfect(self, metrics):
        for _ in range(10):
            metrics.record_step({"forward_vel": 2.0})
        result = metrics.compute()
        assert result["velocity_consistency"] == pytest.approx(1.0, abs=0.01)

    def test_velocity_consistency_variable(self, metrics):
        for v in [0.0, 4.0, 0.0, 4.0, 0.0, 4.0]:
            metrics.record_step({"forward_vel": v})
        result = metrics.compute()
        assert result["velocity_consistency"] < 1.0

    def test_distance_traveled(self, metrics):
        """distance_traveled uses the last recorded value (cumulative from env)."""
        distances = [0.0, 0.5, 1.2, 2.0, 3.5]
        for d in distances:
            metrics.record_step({"forward_vel": 1.0, "distance_traveled": d})
        result = metrics.compute()
        assert result["distance_traveled"] == pytest.approx(3.5)

    def test_distance_traveled_missing(self, metrics):
        """When distance_traveled is not in info, it should not appear in result."""
        metrics.record_step({"forward_vel": 1.0})
        result = metrics.compute()
        assert "distance_traveled" not in result

    def test_distance_traveled_reset(self, metrics):
        metrics.record_step({"forward_vel": 1.0, "distance_traveled": 5.0})
        metrics.reset()
        result = metrics.compute()
        assert "distance_traveled" not in result

    def test_stance_metrics_include_raw_and_derived_signals(self, metrics):
        metrics.record_step(
            {
                "forward_vel": 0.0,
                "bite_success": 0.0,
                "r_foot_contact": 75.0,
                "l_foot_contact": 25.0,
                "head_tip_z": 0.65,
            }
        )
        result = metrics.compute()

        assert result["mean_bilateral_support_duty"] == pytest.approx(1.0)
        assert result["mean_r_foot_load_share"] == pytest.approx(0.75)
        assert result["mean_foot_load_imbalance"] == pytest.approx(0.5)
        assert result["mean_head_tip_z"] == pytest.approx(0.65)

    def test_stance_metrics_reset(self, metrics):
        metrics.record_step(
            {
                "bite_success": 0.0,
                "r_foot_contact": 1.0,
                "l_foot_contact": 1.0,
            }
        )
        metrics.reset()

        assert metrics._stance_metrics == {}

    def test_quadruped_contacts_do_not_create_biped_stance_metrics(self, metrics):
        metrics.record_step(
            {
                "r_foot_contact": 75.0,
                "l_foot_contact": 25.0,
                "rr_foot_contact": 50.0,
                "rl_foot_contact": 50.0,
            }
        )
        result = metrics.compute()

        assert "mean_bilateral_support_duty" not in result
        assert "mean_r_foot_load_share" not in result
        assert "mean_foot_load_imbalance" not in result


class TestAggregateEpisodes:
    """Test multi-episode aggregation."""

    def test_aggregate_two_episodes(self):
        reports = [
            {"mean_forward_velocity": 1.0, "total_reward": 50.0},
            {"mean_forward_velocity": 3.0, "total_reward": 150.0},
        ]
        agg = LocomotionMetrics.aggregate_episodes(reports)
        assert agg["mean_mean_forward_velocity"] == pytest.approx(2.0)
        assert agg["mean_total_reward"] == pytest.approx(100.0)
        assert agg["n_episodes"] == 2.0

    def test_aggregate_empty(self):
        assert LocomotionMetrics.aggregate_episodes([]) == {}


class TestSuccessRate:
    """Episode-level success semantics (success_rate is 0/1 per episode)."""

    def test_success_on_final_step_counts_as_full_success(self):
        metrics = LocomotionMetrics()
        for _ in range(99):
            metrics.record_step({"forward_vel": 1.0, "strike_success": 0.0})
        metrics.record_step({"forward_vel": 1.0, "strike_success": 1.0})
        result = metrics.compute()
        assert result["success_rate"] == 1.0
        assert result["total_successes"] == 1.0

    def test_no_success_is_zero(self):
        metrics = LocomotionMetrics()
        for _ in range(10):
            metrics.record_step({"forward_vel": 1.0, "bite_success": 0.0})
        result = metrics.compute()
        assert result["success_rate"] == 0.0

    def test_aggregated_success_rate_is_episode_fraction(self):
        reports = []
        for success in (1.0, 0.0, 1.0, 1.0):
            m = LocomotionMetrics()
            m.record_step({"forward_vel": 1.0, "food_reached": success})
            reports.append(m.compute())
        agg = LocomotionMetrics.aggregate_episodes(reports)
        assert agg["mean_success_rate"] == pytest.approx(0.75)


#: Models whose success keys the old hard-coded tuples missed: snap_success and course_complete everywhere,
#: target_success in the curriculum's supplementary and standalone evaluations.
SPECIES_SUCCESS_CASES = [
    (ModelId("dibothrosuchus_elaphros", "anatomical"), "snap_success"),
    (ModelId("allosaurus_fragilis", "anatomical"), "course_complete"),
    (ModelId("compsognathus_longipes", "anatomical"), "target_success"),
    (ModelId("compsognathus_longipes", "robot"), "target_success"),
]


class TestSpeciesSuccessKeys:
    """``LocomotionMetrics`` reads the species' ``success_keys``, and its fallback covers every species."""

    @pytest.mark.parametrize(("model_id", "key"), SPECIES_SUCCESS_CASES, ids=lambda case: str(case))
    def test_the_species_success_key_is_counted(self, model_id, key):
        from environments.shared.species_registry import get_species_config

        success_keys = get_species_config(model_id).success_keys
        assert key in success_keys
        reports = []
        for success in (0.0, 1.0, 1.0, 0.0):
            metrics = LocomotionMetrics(success_keys=success_keys)
            metrics.record_step({"forward_vel": 0.0, key: 0.0})
            metrics.record_step({"forward_vel": 0.0, key: success})
            reports.append(metrics.compute())
        assert [report["success_rate"] for report in reports] == [0.0, 1.0, 1.0, 0.0]
        assert LocomotionMetrics.aggregate_episodes(reports)["mean_success_rate"] == pytest.approx(0.5)

    def test_another_species_key_is_not_this_species_success(self):
        metrics = LocomotionMetrics(success_keys=["snap_success"])
        metrics.record_step({"forward_vel": 0.0, "snap_success": 0.0, "bite_success": 1.0})
        assert metrics.compute()["success_rate"] == 0.0

    def test_a_step_succeeds_when_any_of_the_species_keys_does(self):
        """The T-Rex lists bite_success then strike_success: the first present key alone used to decide."""
        metrics = LocomotionMetrics(success_keys=["bite_success", "strike_success"])
        metrics.record_step({"forward_vel": 0.0, "bite_success": 0.0, "strike_success": 1.0})
        assert metrics.compute()["success_rate"] == 1.0

    def test_the_fallback_names_every_species_success_key(self):
        """Read without a species, it must name every key a manifest model's env class lists, and no other."""
        from environments.shared.species_names import model_ids
        from environments.shared.species_registry import env_class

        declared = {key for model_id in model_ids() for key in env_class(model_id).SUCCESS_KEYS}
        assert declared == set(FALLBACK_SUCCESS_KEYS)

    @pytest.mark.parametrize("key", FALLBACK_SUCCESS_KEYS)
    def test_without_species_keys_the_fallback_is_read(self, key):
        metrics = LocomotionMetrics()
        metrics.record_step({"forward_vel": 0.0, key: 1.0})
        assert metrics.compute()["success_rate"] == 1.0


class TestDt:
    """Control-timestep handling."""

    def test_default_dt_matches_species_models(self):
        # All species XMLs use timestep=0.002 with frame_skip=5 -> dt=0.01
        assert LocomotionMetrics().dt == pytest.approx(0.01)

    def test_env_dt_reads_plain_attribute(self):
        from environments.shared.metrics import env_dt

        class FakeEnv:
            dt = 0.025

        assert env_dt(FakeEnv()) == pytest.approx(0.025)

    def test_env_dt_reads_vec_env_get_attr(self):
        from environments.shared.metrics import env_dt

        class FakeVecEnv:
            def get_attr(self, name):
                assert name == "dt"
                return [0.05]

        assert env_dt(FakeVecEnv()) == pytest.approx(0.05)

    def test_env_dt_falls_back_to_default(self):
        from environments.shared.metrics import env_dt

        assert env_dt(object()) == pytest.approx(0.01)
        assert env_dt(object(), default=0.02) == pytest.approx(0.02)
