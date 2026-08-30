"""Tests for same-stage resume continuation (gap review Phase R).

The 2026-08 pipeline gap review found five compounding defects in the
resume path (``docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md`` §1): the SB3
step counter restarted at zero (TC1 — checkpoint renumbering let retention
delete every fresh checkpoint while keeping stale pre-crash ones), the
best-model trackers restarted at ``-inf`` (TC2 — the first post-resume eval
overwrote a better pre-crash best and truncated ``evaluations.npz``),
schedules snapped back (TC3), a warm-up marker pickled into a checkpoint
disabled entropy decay forever (EE3), and the CLI retrained the full budget
(TC4). These tests pin the fixes at their seams.
"""

import numpy as np
import pytest

from environments.shared.cli import _default_resume_timesteps
from environments.shared.curriculum.checkpoints import (
    RobustBestModelCallback,
    prune_periodic_checkpoints,
    seed_resume_eval_state,
)

sb3 = pytest.importorskip("stable_baselines3")


class _FakeEvalCallback:
    """The attribute surface seed_resume_eval_state touches on EvalCallback."""

    def __init__(self):
        self.best_mean_reward = -np.inf
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []


def _write_prior_npz(path, results, timesteps=None, successes=None):
    payload = {
        "timesteps": np.asarray(timesteps if timesteps is not None else list(range(1, len(results) + 1))),
        "results": np.asarray(results, dtype=float),
        "ep_lengths": np.full_like(np.asarray(results, dtype=float), 1000.0),
    }
    if successes is not None:
        payload["successes"] = np.asarray(successes, dtype=float)
    np.savez(path, **payload)


class TestSeedResumeEvalState:
    def test_seeds_best_trackers_and_history(self, tmp_path):
        npz = tmp_path / "evaluations.npz"
        # Two prior evals: mean 2300 (std 0), then mean 900 — the pre-crash
        # best is 2300 and must survive the resume.
        _write_prior_npz(npz, [[2300.0, 2300.0], [900.0, 900.0]])
        eval_cb = _FakeEvalCallback()
        robust = RobustBestModelCallback(eval_cb, model_dir=tmp_path)

        seeded = seed_resume_eval_state(eval_cb, [robust], npz)

        assert seeded == 2
        assert eval_cb.best_mean_reward == 2300.0
        assert len(eval_cb.evaluations_results) == 2
        assert len(eval_cb.evaluations_timesteps) == 2
        # Risk-adjusted score max over history: identical episodes ⇒ std 0.
        assert robust.best_score == 2300.0
        # The seeded history must not read as a fresh evaluation of the
        # loaded model — otherwise robust_best is overwritten at step one.
        assert robust._last_seen_n == 2

    def test_first_post_resume_eval_cannot_downgrade_best(self, tmp_path):
        npz = tmp_path / "evaluations.npz"
        _write_prior_npz(npz, [[2300.0, 2300.0]])
        eval_cb = _FakeEvalCallback()
        robust = RobustBestModelCallback(eval_cb, model_dir=tmp_path)
        seed_resume_eval_state(eval_cb, [robust], npz)

        # Simulate SB3 appending a worse post-resume evaluation.
        eval_cb.evaluations_results.append(np.asarray([900.0, 900.0]))
        # EvalCallback's own guard: 900 < seeded best ⇒ no best_model save.
        assert 900.0 < eval_cb.best_mean_reward
        # RobustBestModelCallback sees ONE new eval and its score loses too.
        latest = eval_cb.evaluations_results[-1]
        assert len(eval_cb.evaluations_results) > robust._last_seen_n
        assert float(latest.mean() - latest.std()) < robust.best_score

    def test_successes_are_seeded_when_present(self, tmp_path):
        npz = tmp_path / "evaluations.npz"
        _write_prior_npz(npz, [[10.0, 20.0]], successes=[[1.0, 0.0]])
        eval_cb = _FakeEvalCallback()
        seed_resume_eval_state(eval_cb, [], npz)
        assert len(eval_cb.evaluations_successes) == 1

    def test_missing_or_corrupt_record_resumes_cleanly_empty(self, tmp_path):
        eval_cb = _FakeEvalCallback()
        assert seed_resume_eval_state(eval_cb, [], tmp_path / "evaluations.npz") == 0
        assert eval_cb.best_mean_reward == -np.inf

        corrupt = tmp_path / "evaluations.npz"
        corrupt.write_bytes(b"not an npz")
        assert seed_resume_eval_state(eval_cb, [], corrupt) == 0
        assert eval_cb.best_mean_reward == -np.inf


class TestContinuedCheckpointNumberingSurvivesRetention:
    def test_resumed_checkpoints_outrank_stale_ones(self, tmp_path):
        """TC1's deletion scenario, inverted by the continued counter.

        Pre-crash retention kept stance_{1.5M..3.5M}. With the counter
        continued (reset_num_timesteps=False) the first post-resume
        checkpoint lands ABOVE the stale maximum, so pruning removes the
        oldest stale step-point — never the fresh one.
        """
        for step in (1_500_000, 2_000_000, 2_500_000, 3_000_000, 3_500_000):
            (tmp_path / f"stance_{step}_steps.zip").touch()
            (tmp_path / f"stance_vecnormalize_{step}_steps.pkl").touch()
        # First post-resume save, cumulative numbering.
        (tmp_path / "stance_3600000_steps.zip").touch()
        (tmp_path / "stance_vecnormalize_3600000_steps.pkl").touch()

        removed = prune_periodic_checkpoints(tmp_path, "stance", max_checkpoints=5)

        removed_names = {p.name for p in removed}
        assert "stance_1500000_steps.zip" in removed_names
        assert (tmp_path / "stance_3600000_steps.zip").exists()
        assert (tmp_path / "stance_vecnormalize_3600000_steps.pkl").exists()


class TestResumeLearnWiring:
    def test_resumed_learn_continues_the_step_counter(self, tmp_path):
        """End-to-end through real SB3: counter, filenames, schedule progress.

        This is the runtime probe from the review, pinned: save at N steps,
        reload, learn more with ``reset_num_timesteps=False`` — the counter
        and the checkpoint filenames must stay cumulative.
        """
        gym = pytest.importorskip("gymnasium")
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

        model = PPO("MlpPolicy", TinyEnv(), n_steps=32, batch_size=32, verbose=0)
        model.learn(total_timesteps=64)
        assert model.num_timesteps == 64
        model.save(tmp_path / "ckpt")

        resumed = PPO.load(tmp_path / "ckpt", env=TinyEnv())
        assert resumed.num_timesteps == 64  # restored from the checkpoint

        ckpt_cb = CheckpointCallback(save_freq=32, save_path=str(tmp_path), name_prefix="stage1")
        resumed.learn(total_timesteps=64, callback=ckpt_cb, reset_num_timesteps=False)

        # Counter continued: 64 + 64, not restarted at 0.
        assert resumed.num_timesteps == 128
        # Periodic filenames stay on the cumulative axis (>= 64), so
        # retention's keep-the-largest ordering keeps the FRESH ones.
        steps = sorted(int(p.name.split("_")[1]) for p in tmp_path.glob("stage1_*_steps.zip"))
        assert steps and min(steps) > 64


class TestDefaultResumeTimesteps:
    def test_periodic_same_stage_resume_gets_remaining_budget(self):
        assert _default_resume_timesteps("m/stage2_5000000_steps.zip", "resume_same_stage", 8_000_000) == 3_000_000

    def test_exhausted_budget_clamps_to_zero(self):
        assert _default_resume_timesteps("m/stage1_12000000_steps.zip", "resume_same_stage", 11_000_000) == 0

    def test_boundary_crossing_load_keeps_full_budget(self):
        assert _default_resume_timesteps("m/stage1_5000000_steps.zip", "initialize_next_stage", 8_000_000) is None

    def test_curated_checkpoint_keeps_full_budget(self):
        # best_model.zip carries no cumulative step count in its name; the
        # caller cannot know the remaining budget without loading it.
        assert _default_resume_timesteps("m/best_model.zip", "resume_same_stage", 8_000_000) is None

    def test_no_load_keeps_full_budget(self):
        assert _default_resume_timesteps(None, "resume_same_stage", 8_000_000) is None


class TestWarmupMarkerClearedOnResume:
    def test_marker_is_cleared_for_resume_same_stage(self):
        """EE3: a mid-warm-up checkpoint must not disable entropy decay forever."""
        from environments.shared.curriculum.schedules import ENT_COEF_WARMUP_MARKER
        from environments.shared.train_base import _create_or_load_model

        class FakeModel:
            pass

        loaded = FakeModel()
        setattr(loaded, ENT_COEF_WARMUP_MARKER, True)

        class FakeAlg:
            @staticmethod
            def load(path, env=None, **kwargs):
                return loaded

        model = _create_or_load_model(
            {"PPO": FakeAlg, "SAC": FakeAlg},
            "ppo",
            {},
            train_env=None,
            load_path="/fake/mid_warmup_ckpt.zip",
            task_load_mode="resume_same_stage",
        )
        assert getattr(model, ENT_COEF_WARMUP_MARKER) is False

    def test_marker_survives_boundary_crossing_load(self):
        """initialize_next_stage attaches a fresh warm-up, which owns the marker."""
        from environments.shared.curriculum.schedules import ENT_COEF_WARMUP_MARKER
        from environments.shared.train_base import _create_or_load_model

        class FakeModel:
            pass

        loaded = FakeModel()
        setattr(loaded, ENT_COEF_WARMUP_MARKER, True)

        class FakeAlg:
            @staticmethod
            def load(path, env=None, **kwargs):
                return loaded

        model = _create_or_load_model(
            {"PPO": FakeAlg, "SAC": FakeAlg},
            "ppo",
            {},
            train_env=None,
            load_path="/fake/mid_warmup_ckpt.zip",
            task_load_mode="initialize_next_stage",
        )
        assert getattr(model, ENT_COEF_WARMUP_MARKER) is True


class TestPublishCheckpointPairCallback:
    def test_publishes_pairs_atomically_and_clears_scratch(self, tmp_path):
        from environments.shared.curriculum.checkpoints import PublishCheckpointPairCallback

        local = tmp_path / "scratch"
        publish = tmp_path / "models"
        local.mkdir()
        publish.mkdir()
        (local / "stage1_100000_steps.zip").write_bytes(b"policy")
        (local / "stage1_vecnormalize_100000_steps.pkl").write_bytes(b"stats")
        (local / "best_model.zip").write_bytes(b"curated")  # not a periodic pair

        cb = PublishCheckpointPairCallback(local_dir=local, publish_dir=publish, name_prefix="stage1", save_freq=10)
        cb._publish()

        assert (publish / "stage1_100000_steps.zip").read_bytes() == b"policy"
        assert (publish / "stage1_vecnormalize_100000_steps.pkl").read_bytes() == b"stats"
        # Published files leave scratch; curated names are not this
        # callback's to move.
        assert not (local / "stage1_100000_steps.zip").exists()
        assert (local / "best_model.zip").exists()
        assert not (publish / "best_model.zip").exists()


class TestRemoteMountDetection:
    def test_drive_and_gcs_paths_are_remote(self):
        from environments.shared.train_base import _is_remote_mount_path

        assert _is_remote_mount_path("/content/drive/MyDrive/run/models")
        assert _is_remote_mount_path("/gcs/bucket/run/models")
        assert not _is_remote_mount_path("/home/user/run/models")
