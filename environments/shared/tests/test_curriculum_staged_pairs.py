"""The best, robust-best and final pairs are staged and published like the periodic ones (CU-3).

On a Drive/GCS mount ``train()`` used to write these three fixed-name pairs
straight into ``models/``: a reclaim during the save left a truncated zip
that ``select_handoff_checkpoint`` still returned and JUDGE then failed to
load (reproduced 2026-09-26), and a reclaim between a zip and its sidecar
could pair a new file with the previous pair's other half. They are now
saved to local scratch and published by ``publish_staged_pair``, which
removes the destination file it publishes last before publishing anything,
so a reclaim part-way leaves only half of the new pair, which that pair's
readers already treat as incomplete. Off a mount nothing changes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from environments.shared.curriculum.checkpoints import publish_staged_pair, select_handoff_checkpoint


def _pair(stem: Path, tag: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.zip").write_text(f"{tag} zip", encoding="utf-8")
    Path(f"{stem}_vecnorm.pkl").write_text(f"{tag} pkl", encoding="utf-8")


def _contents(stem: Path) -> dict[str, str]:
    found = {}
    for suffix in (".zip", "_vecnorm.pkl"):
        path = Path(f"{stem}{suffix}")
        if path.exists():
            found[suffix] = path.read_text(encoding="utf-8")
    return found


class TestPublishStagedPair:
    @pytest.mark.parametrize("zip_last", [True, False])
    def test_publishes_the_new_pair_and_empties_the_stage(self, tmp_path, zip_last):
        _pair(tmp_path / "models" / "best_model", "old")
        _pair(tmp_path / "stage" / "best_model", "new")

        publish_staged_pair(tmp_path / "stage" / "best_model", tmp_path / "models" / "best_model", zip_last=zip_last)

        assert _contents(tmp_path / "models" / "best_model") == {".zip": "new zip", "_vecnorm.pkl": "new pkl"}
        assert list((tmp_path / "stage").iterdir()) == []

    @pytest.mark.parametrize("zip_last", [True, False])
    @pytest.mark.parametrize("failing_copy", [1, 2])
    def test_no_failure_point_leaves_a_mixed_pair(self, tmp_path, monkeypatch, zip_last, failing_copy):
        from environments.shared import file_io

        real_copy = file_io.atomic_copy
        calls = []

        def copy(src, dst):
            calls.append(Path(dst).name)
            if len(calls) == failing_copy:
                raise OSError("runtime reclaimed")
            real_copy(src, dst)

        monkeypatch.setattr(file_io, "atomic_copy", copy)
        destination = tmp_path / "models" / "robust_best_model"
        _pair(destination, "old")
        _pair(tmp_path / "stage" / "robust_best_model", "new")

        with pytest.raises(OSError, match="reclaimed"):
            publish_staged_pair(tmp_path / "stage" / "robust_best_model", destination, zip_last=zip_last)

        left = _contents(destination)
        tags = {text.split()[0] for text in left.values()}
        assert len(tags) <= 1 or len(left) < 2, f"a mixed pair was left: {left}"
        assert len(left) < 2, "the pair the publish did not finish must not look complete"
        if zip_last:
            # The handoff readers need both files: a lone sidecar is skipped.
            assert ".zip" not in left
            assert select_handoff_checkpoint(destination.parent) is None
        else:
            # checkpoint_pair_problem reports a zip without its sidecar (D-D16).
            assert "_vecnorm.pkl" not in left

    def test_without_a_staged_sidecar_the_old_sidecar_goes(self, tmp_path):
        destination = tmp_path / "models" / "best_model"
        _pair(destination, "old")
        staged = tmp_path / "stage" / "best_model"
        staged.parent.mkdir()
        Path(f"{staged}.zip").write_text("new zip", encoding="utf-8")

        publish_staged_pair(staged, destination, zip_last=True)

        assert _contents(destination) == {".zip": "new zip"}


# ── With Stable-Baselines3 ──────────────────────────────────────────────────


class Reclaimed(BaseException):
    """A runtime reclaim: not an Exception, so nothing downstream swallows it."""


def _tiny_env_class():
    gym = pytest.importorskip("gymnasium")

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    return gym, TinyEnv


def _vec_env(training: bool = True):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    gym, tiny = _tiny_env_class()
    env = VecNormalize(DummyVecEnv([lambda: gym.wrappers.TimeLimit(tiny(), max_episode_steps=5)]))
    env.training = training
    return env


def _tiny_model():
    sb3 = pytest.importorskip("stable_baselines3")
    env = _vec_env()
    return sb3.PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, verbose=0, device="cpu"), env


def _half_then_reclaimed(real_save):
    def save(self, path, *args, **kwargs):
        target = str(path) if str(path).endswith(".zip") else f"{path}.zip"
        real_save(self, path, *args, **kwargs)
        data = Path(target).read_bytes()
        Path(target).write_bytes(data[: len(data) // 2])
        raise Reclaimed()

    return save


def _intact(stem: Path) -> bool:
    from environments.shared.curriculum import checkpoint_pair_problem

    return checkpoint_pair_problem(Path(f"{stem}.zip"), Path(f"{stem}_vecnorm.pkl")) is None


@pytest.fixture
def mount(tmp_path, monkeypatch):
    from environments.shared import train_base

    root = tmp_path / "content" / "drive"
    (root / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(train_base, "_REMOTE_MOUNT_ROOTS", (str(root),))
    return root


class TestHandoffCallbacks:
    def test_robust_best_saves_to_the_stage_then_publishes(self, tmp_path):
        from environments.shared.curriculum import RobustBestModelCallback

        model, _ = _tiny_model()
        models, staging = tmp_path / "models", tmp_path / "stage"
        models.mkdir()
        staging.mkdir()
        eval_cb = type("Eval", (), {"evaluations_results": [[1.0, 1.0]]})()
        callback = RobustBestModelCallback(eval_cb, model_dir=models, staging_dir=staging)
        callback.init_callback(model)

        callback.on_step()

        assert _intact(models / "robust_best_model")
        assert list(staging.iterdir()) == []

    def test_a_reclaim_mid_save_leaves_the_previous_robust_pair_intact(self, tmp_path, monkeypatch):
        sb3 = pytest.importorskip("stable_baselines3")
        from environments.shared.curriculum import RobustBestModelCallback

        model, env = _tiny_model()
        models, staging = tmp_path / "models", tmp_path / "stage"
        models.mkdir()
        staging.mkdir()
        model.save(str(models / "robust_best_model"))
        env.save(str(models / "robust_best_model_vecnorm.pkl"))
        before = (models / "robust_best_model.zip").read_bytes()
        eval_cb = type("Eval", (), {"evaluations_results": [[1.0, 1.0]]})()
        callback = RobustBestModelCallback(eval_cb, model_dir=models, staging_dir=staging)
        callback.init_callback(model)
        monkeypatch.setattr(sb3.PPO, "save", _half_then_reclaimed(sb3.PPO.save))

        with pytest.raises(Reclaimed):
            callback.on_step()

        assert (models / "robust_best_model.zip").read_bytes() == before
        assert _intact(models / "robust_best_model")
        assert select_handoff_checkpoint(models)[0] == "robust_best_model"

    def test_the_best_pair_is_published_after_its_sidecar_is_saved(self, tmp_path):
        from environments.shared.curriculum import SaveVecNormalizeCallback

        model, _ = _tiny_model()
        models, staging = tmp_path / "models", tmp_path / "stage"
        models.mkdir()
        staging.mkdir()
        model.save(str(staging / "best_model"))  # what SB3's EvalCallback does on a new best
        callback = SaveVecNormalizeCallback(str(staging / "best_model_vecnorm.pkl"), publish_dir=models)
        callback.init_callback(model)

        callback.on_step()

        assert _intact(models / "best_model")
        assert list(staging.iterdir()) == []

    def test_publishing_needs_a_sidecar_name(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from environments.shared.curriculum import SaveVecNormalizeCallback

        with pytest.raises(ValueError, match="_vecnorm.pkl"):
            SaveVecNormalizeCallback(str(tmp_path / "stats.pkl"), publish_dir=tmp_path)


class TestFinalPair:
    def test_on_a_mount_the_pair_is_staged_then_published(self, mount, monkeypatch, tmp_path):
        import tempfile

        from environments.shared import train_base

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        model, env = _tiny_model()
        models = mount / "MyDrive" / "run" / "01_stance" / "models"
        models.mkdir(parents=True)

        final_path = train_base._save_final_and_sync_tb(model, env, models, 1, None, mount / "tb")

        assert final_path == models / "stage1_final"
        assert _intact(final_path)
        assert list(scratch.iterdir()) == [], "the final pair's staging directory must be removed"

    def test_a_reclaim_mid_save_leaves_no_truncated_final_zip(self, mount, monkeypatch):
        sb3 = pytest.importorskip("stable_baselines3")
        from environments.shared import train_base

        model, env = _tiny_model()
        models = mount / "MyDrive" / "run" / "01_stance" / "models"
        models.mkdir(parents=True)
        monkeypatch.setattr(sb3.PPO, "save", _half_then_reclaimed(sb3.PPO.save))

        with pytest.raises(Reclaimed):
            train_base._save_final_and_sync_tb(model, env, models, 1, None, mount / "tb")

        assert not (models / "stage1_final.zip").exists()
        assert list(models.iterdir()) == []

    def test_a_hard_kill_mid_save_leaves_no_truncated_final_zip(self, tmp_path):
        """``os._exit`` mid-save: no ``finally`` or ``except`` runs, as on a real reclaim."""
        pytest.importorskip("stable_baselines3")
        root = tmp_path / "content" / "drive"
        models = root / "MyDrive" / "run" / "01_stance" / "models"
        models.mkdir(parents=True)
        script = textwrap.dedent(
            f"""
            import os, numpy as np, gymnasium as gym, stable_baselines3 as sb3
            from pathlib import Path
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from environments.shared import train_base
            train_base._REMOTE_MOUNT_ROOTS = ({str(root)!r},)
            class TinyEnv(gym.Env):
                observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
                action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
                def reset(self, *, seed=None, options=None):
                    super().reset(seed=seed)
                    return np.zeros(1, dtype=np.float32), {{}}
                def step(self, action):
                    return np.zeros(1, dtype=np.float32), 0.0, False, False, {{}}
            env = VecNormalize(DummyVecEnv([TinyEnv]))
            model = sb3.PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, verbose=0, device="cpu")
            real = sb3.PPO.save
            def save(self, path, *a, **k):
                target = str(path) if str(path).endswith(".zip") else f"{{path}}.zip"
                real(self, path, *a, **k)
                data = Path(target).read_bytes()
                Path(target).write_bytes(data[: len(data) // 2])
                os._exit(137)
            sb3.PPO.save = save
            train_base._save_final_and_sync_tb(
                model, env, Path({str(models)!r}), 1, None, Path({str(root / "tb")!r})
            )
            """
        )
        repository = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(filter(None, [str(repository), os.environ.get("PYTHONPATH")])),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 137, result.stderr
        assert not (models / "stage1_final.zip").exists(), "a reclaim mid-save published a truncated final zip"


class TestWiring:
    @staticmethod
    def _callbacks(stage_dir: Path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback

        from environments.shared.train_base import _build_core_callbacks

        eval_env = _vec_env(training=False)
        try:
            callbacks, _, _ = _build_core_callbacks(
                {"CheckpointCallback": CheckpointCallback},
                eval_env,
                stage_dir / "models",
                stage_dir,
                1,
                1,
                100,
                100,
                0,
                {"curriculum_kwargs": {"min_avg_reward": 1.0}},
                local_tb_dir=None,
                gcs_tb_path=None,
            )
        finally:
            eval_env.close()
        return callbacks

    @staticmethod
    def _handoff_callbacks(callbacks):
        from stable_baselines3.common.callbacks import EvalCallback

        from environments.shared.curriculum import RobustBestModelCallback

        evaluation = next(cb for cb in callbacks if isinstance(cb, EvalCallback))
        robust = next(cb for cb in callbacks if isinstance(cb, RobustBestModelCallback))
        return evaluation, evaluation.callback_on_new_best, robust

    def test_on_a_mount_both_handoff_pairs_are_saved_to_scratch(self, mount):
        stage_dir = mount / "MyDrive" / "run" / "01_stance"
        evaluation, on_new_best, robust = self._handoff_callbacks(self._callbacks(stage_dir))

        scratch = Path(evaluation.best_model_save_path)
        assert not str(scratch).startswith(str(mount)), "the best pair must be saved off the mount"
        assert Path(on_new_best.save_path) == scratch / "best_model_vecnorm.pkl"
        assert on_new_best.publish_dir == stage_dir / "models"
        assert robust.staging_dir == scratch
        assert robust.model_dir == stage_dir / "models"

    def test_off_a_mount_both_handoff_pairs_are_written_in_place(self, tmp_path):
        evaluation, on_new_best, robust = self._handoff_callbacks(self._callbacks(tmp_path))

        assert Path(evaluation.best_model_save_path) == tmp_path / "models"
        assert on_new_best.publish_dir is None
        assert robust.staging_dir is None

    def test_training_on_a_mount_publishes_intact_handoff_pairs(self, mount):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CallbackList

        stage_dir = mount / "MyDrive" / "run" / "01_stance"
        callbacks = self._callbacks(stage_dir)
        evaluation, _, _ = self._handoff_callbacks(callbacks)
        model, _ = _tiny_model()

        model.learn(total_timesteps=240, callback=CallbackList(callbacks))

        models = stage_dir / "models"
        assert _intact(models / "best_model")
        assert _intact(models / "robust_best_model")
        assert select_handoff_checkpoint(models)[0] == "robust_best_model"
        assert list(Path(evaluation.best_model_save_path).iterdir()) == []
