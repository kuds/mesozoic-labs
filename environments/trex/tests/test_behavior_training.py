"""Bounded recipe/CLI checks; optimizer and physics integration live in sibling suites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from environments.shared import train_behaviors  # noqa: E402
from environments.trex.envs.behavior_env import TRexBehaviorEnv  # noqa: E402

PRESETS = Path(__file__).parents[3] / "configs" / "trex" / "behavior_pilots"


@pytest.mark.parametrize(
    "name",
    [
        "trex_follow_direction",
        "trex_follow_direction_speed",
        "trex_terrain_contact",
        "trex_gentle_terrain",
        "trex_bumps_terrain",
        "trex_depressions_terrain",
        "trex_mixed_terrain",
        "trex_combined_terrain",
    ],
)
def test_each_committed_recipe_instantiates_and_steps(name):
    recipe, commands, terrain, kwargs = train_behaviors.read_recipe(PRESETS / f"{name}.toml")
    assert recipe["pilot"]["timesteps"] > 0
    env = TRexBehaviorEnv(commands=commands, terrain=terrain, run_seed=42, **kwargs)
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == (64,) and np.all(np.isfinite(obs))
        assert env.action_space.shape == (15,)
        next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(15))
        assert np.all(np.isfinite(next_obs)) and np.isfinite(reward)
        assert not terminated and not truncated
        assert step_info["command_event_id"] == 0
        assert step_info["heading_error_rad"] == pytest.approx(
            ((step_info["desired_heading"] - step_info["actual_heading"] + np.pi) % (2 * np.pi) - np.pi)
            if step_info["heading_active"]
            else 0.0
        )
        if env.terrain is None:
            assert np.isfinite(env.lowest_ground_clearance())
        else:
            expected_schema = (
                "mesozoic.gentle-terrain/v1" if terrain.template == "sloped" else "mesozoic.terrain-templates/v2"
            )
            assert info["terrain"]["schema"] == expected_schema
            with pytest.raises(NotImplementedError, match="heightfield"):
                env.lowest_ground_clearance()
    finally:
        env.close()


@pytest.mark.parametrize(
    "content",
    [
        "[unknown]\nx=1\n",
        "[commands]\nswitch_inteval_s=1.0\n",
        "[terrain]\nroughnes_amplitude=0.01\n",
        "[ppo]\nlearn_rate=0.0001\n",
        "[env]\nmax_episode_step=1000\n",
        "[terrain]\nenabled=1\n",
        "[pilot]\ntimesteps=1.5\n",
        "[pilot]\ntimesteps=-1\n",
        "[ppo]\nent_coef=nan\n",
        "[ppo]\nent_coef=-0.1\n",
        "[ppo]\ntarget_kl=-0.1\n",
        "[ppo]\nlearning_rate=0.0\n",
        "[ppo]\nwarmup_timesteps=1.5\n",
        "[ppo]\nwarmup_clip_range=0.5\n",
        "[terrain_sampler]\nflat=1\n",
        '[terrain]\nenabled=true\nmode="flat"\n[terrain_sampler]\nflat=1\n',
        '[terrain]\nenabled=true\nmode="gentle"\n[terrain_sampler]\nbumpps=1\n',
        '[terrain]\nenabled=true\nmode="gentle"\n[terrain_sampler]\nflat=-1\n',
        '[terrain]\nenabled=true\nmode="gentle"\n[terrain_sampler]\nflat=1\n[env]\nflat_probability=0.25\n',
    ],
)
def test_bad_recipe_refuses_before_environment_or_policy_loading(tmp_path, content):
    path = tmp_path / "bad.toml"
    path.write_text(content)
    with pytest.raises((ValueError, TypeError)):
        train_behaviors.read_recipe(path)


def _bundle(tmp_path):
    model, normalizer = tmp_path / "model.zip", tmp_path / "vecnormalize.pkl"
    model.write_bytes(b"selected-model")
    normalizer.write_bytes(b"paired-stats")
    manifest = {
        "schema": "mesozoic.behavior-bundle/v1",
        "model": model.name,
        "normalizer": normalizer.name,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "normalizer_sha256": hashlib.sha256(normalizer.read_bytes()).hexdigest(),
        "canonical_certification": False,
    }
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    return model, normalizer


@pytest.mark.parametrize("changed", ["model", "normalizer", "manifest"])
def test_resume_refuses_missing_or_altered_pair(tmp_path, changed):
    model, normalizer = _bundle(tmp_path)
    train_behaviors._verify_bundle(model, normalizer)
    if changed == "model":
        model.write_bytes(b"other-model")
    elif changed == "normalizer":
        normalizer.write_bytes(b"other-stats")
    else:
        (tmp_path / "bundle.json").unlink()
    with pytest.raises(ValueError, match="bundle|hash"):
        train_behaviors._verify_bundle(model, normalizer)


@pytest.fixture
def stubbed_cli(monkeypatch, tmp_path):
    """Exercise actual CLI orchestration with an explicit no-optimizer model."""
    from environments.shared import behavior_checkpoint

    calls = {"learn": [], "loads": [], "interrupt_after": None, "rollout_starts": []}

    class FakeModel:
        def __init__(self, state=None):
            self.num_timesteps = 8_000_000
            self.n_steps = 64
            self.batch_size = 16
            self.n_epochs = 1
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.device = "cpu"
            self.__dict__.update(state or {})

        def set_random_seed(self, seed):
            self.seed = seed

        def set_logger(self, logger):
            pass

        def learn(self, *, total_timesteps, reset_num_timesteps, callback):
            assert not reset_num_timesteps
            calls["learn"].append(total_timesteps)
            if calls["interrupt_after"] is not None:
                self.num_timesteps += calls["interrupt_after"]
                calls["interrupt_after"] = None
                raise KeyboardInterrupt
            start = self.num_timesteps
            for elapsed in calls["rollout_starts"]:
                self.num_timesteps = start + elapsed
                for current_callback in callback:
                    current_callback.model = self
                    current_callback._on_rollout_start()
            self.num_timesteps = start + total_timesteps
            return self

        def save(self, path):
            Path(path).write_text(json.dumps(self.__dict__))

    class FakeNormalizer:
        def __init__(self, env):
            self.env = env
            self.norm_reward = False

        def save(self, path):
            Path(path).write_text("paired-normalizer")

        def close(self):
            self.env.close()

    def loader(mode):
        def load(model_path, vecnorm_path, env, **kwargs):
            calls["loads"].append(mode)
            state = json.loads(Path(model_path).read_text())
            return FakeModel(state), FakeNormalizer(env), {"loader": mode}

        return load

    for mode, function in (
        ("prepare", "prepare_behavior_checkpoint"),
        ("resume", "load_behavior_checkpoint"),
        ("adapt", "adapt_behavior_checkpoint"),
    ):
        monkeypatch.setattr(behavior_checkpoint, function, loader(mode))
    parent, normalization = tmp_path / "parent.zip", tmp_path / "parent.pkl"
    parent.write_text(json.dumps({"num_timesteps": 8_000_000}))
    normalization.write_text("parent-normalizer")
    return calls, parent, normalization


def _args(config, checkpoint, normalization, output):
    return [
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--vecnormalize",
        str(normalization),
        "--output",
        str(output),
        "--seed",
        "42",
        "--eval-episodes",
        "0",
    ]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_cli_seed_refuses_before_loading_or_creating_output(tmp_path, capsys, seed):
    output = tmp_path / "output"
    args = _args(tmp_path / "missing.toml", tmp_path / "model.zip", tmp_path / "stats.pkl", output)
    with pytest.raises(SystemExit) as error:
        train_behaviors.main(args + ["--seed", str(seed)])
    assert error.value.code == 2
    assert "--seed must be between 0 and 2**32 - 1" in capsys.readouterr().err
    assert not output.exists()


def test_eval_only_never_learns_or_overwrites_parent(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    original = parent.read_bytes(), normalization.read_bytes()
    output = tmp_path / "evaluation"
    train_behaviors.main(_args(PRESETS / "trex_follow_direction.toml", parent, normalization, output) + ["--eval-only"])
    assert calls["learn"] == []
    assert (parent.read_bytes(), normalization.read_bytes()) == original
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete"
    assert run["training"]["actual_additional_steps"] == 0
    assert run["training"]["requested_additional_steps"] == 0
    train_behaviors._verify_bundle(output / "model.zip", output / "vecnormalize.pkl")


@pytest.mark.parametrize("problem", ["missing_encoder", "excessive_fps"])
def test_unavailable_requested_replay_fails_before_loading_or_learning(stubbed_cli, tmp_path, monkeypatch, problem):
    from environments.shared import behavior_replay

    calls, parent, normalization = stubbed_cli

    def preflight():
        if problem == "missing_encoder":
            raise behavior_replay.ReplayExportError("The requested video encoder is unavailable")

    monkeypatch.setattr(behavior_replay, "require_replay_dependencies", preflight)
    args = _args(PRESETS / "trex_mixed_terrain.toml", parent, normalization, tmp_path / "unavailable-replay")
    args[args.index("--eval-episodes") + 1] = "1"
    args += ["--record-video", "--video-fps", "101" if problem == "excessive_fps" else "25"]
    with pytest.raises((SystemExit, ValueError, behavior_replay.ReplayExportError)):
        train_behaviors.main(args)
    assert calls["loads"] == []
    assert calls["learn"] == []


def test_resume_refuses_changed_ppo_recipe_before_loading(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    source = tmp_path / "first"
    config = PRESETS / "trex_follow_direction.toml"
    train_behaviors.main(_args(config, parent, normalization, source) + ["--steps", "0"])
    changed = tmp_path / "changed.toml"
    changed.write_text(config.read_text().replace("ent_coef = 0.005", "ent_coef = 0.1"))
    previous_loads = list(calls["loads"])
    with pytest.raises((ValueError, SystemExit), match="recipe|PPO|resume|settings"):
        train_behaviors.main(
            _args(changed, source / "model.zip", source / "vecnormalize.pkl", tmp_path / "resumed") + ["--resume"]
        )
    assert calls["loads"] == previous_loads
    assert calls["learn"] == []


def test_resume_finishes_original_budget_without_repeating_full_training(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    source, resumed = tmp_path / "interrupted", tmp_path / "resumed"
    config = tmp_path / "short_recipe.toml"
    config.write_text(
        (PRESETS / "trex_follow_direction.toml").read_text().replace("timesteps = 3000000", "timesteps = 100")
    )
    calls["interrupt_after"] = 40
    train_behaviors.main(_args(config, parent, normalization, source))
    first = json.loads((source / "run.json").read_text())
    assert first["status"] == "interrupted"
    assert first["training"]["actual_additional_steps"] == 40
    train_behaviors.main(_args(config, source / "model.zip", source / "vecnormalize.pkl", resumed) + ["--resume"])
    assert calls["learn"] == [100, 60]
    run = json.loads((resumed / "run.json").read_text())
    assert run["training"]["actual_additional_steps"] == 60
    assert json.loads((resumed / "bundle.json").read_text())["num_timesteps"] == 8_000_100


def test_cli_rejects_conflicting_load_modes_and_nonempty_output(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    args = _args(PRESETS / "trex_follow_direction.toml", parent, normalization, tmp_path / "unused")
    with pytest.raises(SystemExit):
        train_behaviors.main(args + ["--resume", "--adapt"])
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep.txt"
    sentinel.write_text("preserved")
    with pytest.raises(SystemExit):
        train_behaviors.main(_args(PRESETS / "trex_follow_direction.toml", parent, normalization, occupied))
    assert sentinel.read_text() == "preserved"
    assert calls["loads"] == [] and calls["learn"] == []


def test_periodic_rollout_checkpoints_are_complete_matched_bundles(stubbed_cli, tmp_path):
    calls, parent, normalization = stubbed_cli
    calls["rollout_starts"] = [100_000, 200_000]
    output = tmp_path / "periodic"
    train_behaviors.main(
        _args(PRESETS / "trex_follow_direction.toml", parent, normalization, output) + ["--steps", "250000"]
    )
    checkpoints = sorted((output / "checkpoints").iterdir())
    assert [path.name for path in checkpoints] == ["step-8100000", "step-8200000"]
    for checkpoint in checkpoints:
        train_behaviors._verify_bundle(checkpoint / "model.zip", checkpoint / "vecnormalize.pkl")
        metadata = json.loads((checkpoint / "bundle.json").read_text())
        assert metadata["training_recipe"]["ppo"]["ent_coef"] == 0.005
    latest = json.loads((output / "latest_checkpoint.json").read_text())
    assert latest["num_timesteps"] == 8_200_000
    assert Path(latest["directory"]) == checkpoints[-1]


def test_interrupted_evaluation_keeps_trained_bundle_and_records_status(stubbed_cli, tmp_path, monkeypatch):
    from environments.shared import behavior_evaluation

    calls, parent, normalization = stubbed_cli
    output = tmp_path / "interrupted-evaluation"
    saved = {}

    def interrupt(*args, **kwargs):
        saved.update({name: (output / name).read_bytes() for name in ("model.zip", "vecnormalize.pkl", "bundle.json")})
        raise KeyboardInterrupt

    monkeypatch.setattr(behavior_evaluation, "evaluate_behavior", interrupt)
    args = _args(PRESETS / "trex_follow_direction.toml", parent, normalization, output)
    args[args.index("--eval-episodes") + 1] = "1"
    train_behaviors.main(args + ["--steps", "100"])
    assert calls["learn"] == [100]
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "interrupted"
    assert run["training"]["actual_additional_steps"] == 100
    assert "evaluation" not in run
    assert saved == {name: (output / name).read_bytes() for name in saved}
    train_behaviors._verify_bundle(output / "model.zip", output / "vecnormalize.pkl")


def test_interrupted_bundle_save_finishes_a_resumable_pair(stubbed_cli, tmp_path, monkeypatch):
    _, parent, normalization = stubbed_cli
    output = tmp_path / "interrupted-save"
    save = train_behaviors._save_bundle
    attempts = []

    def interrupt_once(model, normalizer, directory, identity, recipe):
        attempts.append(directory)
        if len(attempts) == 1:
            model.save(str(directory / "model.zip"))
            raise KeyboardInterrupt
        save(model, normalizer, directory, identity, recipe)

    monkeypatch.setattr(train_behaviors, "_save_bundle", interrupt_once)
    train_behaviors.main(
        _args(PRESETS / "trex_follow_direction.toml", parent, normalization, output) + ["--steps", "100"]
    )
    assert attempts == [output, output]
    assert json.loads((output / "run.json").read_text())["status"] == "interrupted"
    train_behaviors._verify_bundle(output / "model.zip", output / "vecnormalize.pkl")


def test_real_ppo_cli_resume_preserves_recipe_and_releases_stage_warmup(tmp_path, monkeypatch):
    """Actual PPO updates exercise the loader/runner boundary and persisted anchor."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE
    from environments.shared.tests.test_behavior_checkpoint import CommandEnv

    torch.set_num_threads(1)
    parent_normalizer = VecNormalize(DummyVecEnv([CommandEnv]), gamma=0.97)
    parent = PPO(
        "MlpPolicy",
        parent_normalizer,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        gamma=0.97,
        gae_lambda=0.91,
        policy_kwargs={"net_arch": [16, 16]},
        seed=7,
        device="cpu",
    )
    checkpoint, stats = tmp_path / "parent.zip", tmp_path / "parent.pkl"
    try:
        parent.learn(32)
        for artifact in (parent, parent_normalizer):
            attach_plant_identity(artifact, current_plant_identity("trex"))
        setattr(parent, MODEL_TASK_ATTRIBUTE, {"species": "trex", "stage": 2})
        parent.save(checkpoint)
        parent_normalizer.save(str(stats))
    finally:
        parent_normalizer.close()

    class PilotEnv(CommandEnv):
        def __init__(self, **kwargs):
            super().__init__(live=True)
            self.behavior_identity = {"schema": "runner-test/v1", "task": "live-commands"}

    monkeypatch.setattr(train_behaviors, "get_behavior_env_class", lambda species: PilotEnv)
    config = tmp_path / "short.toml"
    config.write_text("[pilot]\ntimesteps = 48\n[ppo]\nwarmup_timesteps = 24\n")
    updates = []
    train = PPO.train

    def record_update(model):
        updates.append((model.num_timesteps, model.clip_range(1.0), model.lr_schedule(1.0), model.ent_coef))
        return train(model)

    monkeypatch.setattr(PPO, "train", record_update)
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    train_behaviors.main(_args(config, checkpoint, stats, first) + ["--steps", "16"])
    train_behaviors.main(_args(config, first / "model.zip", first / "vecnormalize.pkl", resumed) + ["--resume"])
    assert updates == [(48, 0.02, 5e-5, 0.005), (64, 0.2, 5e-5, 0.005), (80, 0.2, 5e-5, 0.005)]
    saved = PPO.load(resumed / "model.zip", device="cpu")
    assert saved.num_timesteps == 80
    assert saved.mesozoic_behavior_stage_start == 32
    assert (saved.n_steps, saved.batch_size, saved.n_epochs) == (16, 8, 2)
    assert (saved.gamma, saved.gae_lambda) == (0.97, 0.91)
    assert saved.target_kl == 0.03
    assert all(torch.isfinite(value).all() for value in saved.policy.state_dict().values())
    normalizer = VecNormalize.load(str(resumed / "vecnormalize.pkl"), DummyVecEnv([PilotEnv]))
    try:
        raw = np.ones((1, 64), dtype=np.float32)
        raw[:, -3:] = [0.7, -0.2, 0.4]
        np.testing.assert_array_equal(normalizer.normalize_obs(raw)[:, -3:], raw[:, -3:])
        assert normalizer.norm_reward and normalizer.gamma == 0.97
    finally:
        normalizer.close()
    run = json.loads((resumed / "run.json").read_text())
    assert run["training"]["requested_additional_steps"] == 32
    assert run["training"]["actual_additional_steps"] == 32
    train_behaviors._verify_bundle(resumed / "model.zip", resumed / "vecnormalize.pkl")
