"""Tests for environments.shared.reporting.text_summaries."""

from environments.shared.reporting import write_stage_summary, write_training_summary

from .reporting_helpers import make_stage_result


class TestWriteStageSummary:
    """Tests for write_stage_summary (per-stage text files)."""

    def test_creates_summary_file(self, tmp_path):
        result = make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        assert path.exists()
        assert path.name == "stage_summary.txt"

    def test_returns_path(self, tmp_path):
        result = make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        assert path == tmp_path / "stage_summary.txt"

    def test_contains_species(self, tmp_path):
        result = make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Velociraptor" in text

    def test_contains_algorithm(self, tmp_path):
        result = make_stage_result(1)
        path = write_stage_summary(tmp_path, result, "trex", "SAC")
        text = path.read_text()
        assert "SAC" in text

    def test_contains_stage_number(self, tmp_path):
        result = make_stage_result(2)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Stage 2" in text

    def test_contains_reward(self, tmp_path):
        result = make_stage_result(1, mean_reward=123.45, std_reward=6.78)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "123.45" in text
        assert "6.78" in text

    def test_contains_forward_velocity(self, tmp_path):
        result = make_stage_result(1, mean_forward_vel=1.23, std_forward_vel=0.45)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "1.23" in text

    def test_includes_best_eval_when_present(self, tmp_path):
        result = make_stage_result(1, best_eval_reward=99.5, best_eval_std=3.2, best_eval_timestep=50000)
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "99.5" in text
        assert "50,000 steps" in text

    def test_includes_best_model_section(self, tmp_path):
        result = make_stage_result(
            1,
            best_model_reward=88.0,
            best_model_std_reward=2.0,
            best_model_length=180.5,
            best_model_std_length=15.0,
            best_model_fwd_vel=1.1,
            best_model_std_fwd_vel=0.2,
            best_model_success_rate=0.75,
        )
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Best Model Evaluation" in text
        assert "88.0" in text
        assert "75%" in text

    def test_no_best_eval_when_empty(self, tmp_path):
        result = make_stage_result(1, best_eval_reward="", best_eval_std="")
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Best eval:" not in text

    def test_includes_best_eval_length(self, tmp_path):
        result = make_stage_result(
            1,
            best_eval_reward=80.0,
            best_eval_std=3.0,
            best_eval_timestep=40000,
            best_eval_length=250.0,
            best_eval_std_length=10.0,
        )
        path = write_stage_summary(tmp_path, result, "velociraptor", "PPO")
        text = path.read_text()
        assert "Best ep length:" in text


class TestWriteTrainingSummary:
    """Tests for write_training_summary (overall training text file)."""

    def test_creates_summary_file(self, tmp_path):
        results = [make_stage_result(1), make_stage_result(2)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        assert path.exists()
        assert path.name == "training_summary.txt"

    def test_returns_path(self, tmp_path):
        results = [make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        assert path == tmp_path / "training_summary.txt"

    def test_contains_species(self, tmp_path):
        results = [make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "trex", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Trex" in text

    def test_contains_algorithm(self, tmp_path):
        results = [make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "SAC", seed=42, n_envs=4)
        text = path.read_text()
        assert "SAC" in text

    def test_contains_seed(self, tmp_path):
        results = [make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=123, n_envs=4)
        text = path.read_text()
        assert "123" in text

    def test_contains_all_stages(self, tmp_path):
        results = [make_stage_result(1), make_stage_result(2), make_stage_result(3)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Stage 1" in text
        assert "Stage 2" in text
        assert "Stage 3" in text

    def test_contains_total_training_time(self, tmp_path):
        results = [make_stage_result(1, duration_seconds=60.0), make_stage_result(2, duration_seconds=120.0)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Total training time:" in text
        assert "3m" in text

    def test_contains_quick_test_flag(self, tmp_path):
        results = [make_stage_result(1)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4, quick_test=True)
        text = path.read_text()
        assert "True" in text

    def test_includes_best_eval_when_present(self, tmp_path):
        results = [make_stage_result(1, best_eval_reward=95.0, best_eval_std=2.5, best_eval_timestep=80000)]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Best eval:" in text
        assert "95.0" in text

    def test_no_best_eval_when_empty_string(self, tmp_path):
        results = [make_stage_result(1, best_eval_reward="")]
        path = write_training_summary(tmp_path, results, "velociraptor", "PPO", seed=42, n_envs=4)
        text = path.read_text()
        assert "Best eval:" not in text
