"""Tests for LocomotionMetrics."""

import numpy as np
import pytest

from environments.shared.metrics import LocomotionMetrics
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

    def test_prey_distance_is_not_a_metric(self, metrics):
        """The prey-distance and time-to-target metrics left with the hunt (decision D-D59)."""
        for d in [5.0, 3.0, 1.0, 0.4]:
            metrics.record_step({"forward_vel": 1.0, "prey_distance": d})
        result = metrics.compute()
        hunt = {"initial_prey_distance", "final_prey_distance", "min_prey_distance", "target_reached"}
        assert not hunt & set(result) and "time_to_target" not in result

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
                "leg_home_pose_error": 0.0,
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
                "leg_home_pose_error": 0.0,
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
    """Episode-level success semantics (success_rate is 0/1 per episode), from the model's success keys only."""

    def test_success_on_final_step_counts_as_full_success(self):
        metrics = LocomotionMetrics(success_keys=("course_complete",))
        for _ in range(99):
            metrics.record_step({"forward_vel": 1.0})
        metrics.record_step({"forward_vel": 1.0, "course_complete": 1.0})
        result = metrics.compute()
        assert result["success_rate"] == 1.0
        assert result["total_successes"] == 1.0

    def test_an_episode_without_the_key_is_a_failure(self):
        """An episode whose info never carries a declared success key scores 0."""
        metrics = LocomotionMetrics(success_keys=("course_complete",))
        for _ in range(10):
            metrics.record_step({"forward_vel": 1.0})
        result = metrics.compute()
        assert result["success_rate"] == 0.0

    def test_any_declared_key_counts(self):
        metrics = LocomotionMetrics(success_keys=("a_success", "b_success"))
        metrics.record_step({"forward_vel": 1.0, "a_success": 0.0, "b_success": 1.0})
        assert metrics.compute()["success_rate"] == 1.0

    def test_no_success_keys_report_no_success(self):
        """A model with no success event reports none, never a zero, even if an info key looks like one."""
        metrics = LocomotionMetrics()
        for _ in range(10):
            metrics.record_step({"forward_vel": 1.0, "bite_success": 1.0, "target_success": 1.0})
        result = metrics.compute()
        assert "success_rate" not in result and "total_successes" not in result
        agg = LocomotionMetrics.aggregate_episodes([result])
        assert "mean_success_rate" not in agg

    def test_aggregated_success_rate_is_episode_fraction(self):
        reports = []
        for success in (1.0, 0.0, 1.0, 1.0):
            m = LocomotionMetrics(success_keys=("target_success",))
            m.record_step({"forward_vel": 1.0, "target_success": success})
            reports.append(m.compute())
        agg = LocomotionMetrics.aggregate_episodes(reports)
        assert agg["mean_success_rate"] == pytest.approx(0.75)


#: The models with a success event and its key: since the hunt left (decision D-D59) only the Allosaurus course.
SPECIES_SUCCESS_CASES = [
    (ModelId("allosaurus_fragilis", "anatomical"), "course_complete"),
]


class TestSpeciesSuccessKeys:
    """``LocomotionMetrics`` reads the species' ``success_keys``, and no other key."""

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
        metrics = LocomotionMetrics(success_keys=["course_complete"])
        metrics.record_step({"forward_vel": 0.0, "course_complete": 0.0, "bite_success": 1.0})
        assert metrics.compute()["success_rate"] == 0.0

    def test_the_cases_name_every_species_success_key(self):
        """Every key a manifest model's env class lists is a case above (the cases cannot go stale)."""
        from environments.shared.species_names import model_ids
        from environments.shared.species_registry import env_class

        declared = {key for model_id in model_ids() for key in env_class(model_id).SUCCESS_KEYS}
        assert declared == {key for _, key in SPECIES_SUCCESS_CASES}

    def test_a_species_list_is_accepted_as_well_as_a_tuple(self):
        """``SpeciesConfig.success_keys`` is a list; any sequence of keys reads the same."""
        for keys in (["course_complete"], ("course_complete",)):
            metrics = LocomotionMetrics(success_keys=keys)
            metrics.record_step({"forward_vel": 0.0, "course_complete": 1.0})
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
