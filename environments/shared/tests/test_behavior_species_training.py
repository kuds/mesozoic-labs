"""Exercise real PPO preparation, continuation and terrain transfer for every plant."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared.behavior_checkpoint import (  # noqa: E402
    COMMAND_LAYERS,
    BehaviorCheckpointError,
    load_behavior_checkpoint,
    prepare_behavior_checkpoint,
)
from environments.shared.plant_contract import attach_plant_identity, current_plant_identity  # noqa: E402
from environments.shared.species_names import species_display_names  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE  # noqa: E402
from environments.shared.train_behaviors import (  # noqa: E402
    REPO_ROOT,
    _verify_bundle,
    create_behavior_env,
    main,
    read_recipe,
)


@pytest.fixture(params=list(species_display_names(backend="stable-baselines3")))
def walker(request, tmp_path):
    """A short optimizer smoke fixture, explicitly not a trained locomotion policy."""
    torch.set_num_threads(1)
    species = request.param
    identity = current_plant_identity(species)
    normalizer = VecNormalize(DummyVecEnv([get_species_config(species).env_class]))
    model = PPO(
        "MlpPolicy",
        normalizer,
        seed=11,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [16, 16]},
        device="cpu",
    )
    try:
        model.learn(16)
        for artifact in (model, normalizer):
            attach_plant_identity(artifact, identity)
        setattr(model, MODEL_TASK_ATTRIBUTE, {"species": species, "stage": "locomotion"})
        checkpoint, stats = tmp_path / "parent.zip", tmp_path / "parent.pkl"
        model.save(checkpoint)
        normalizer.save(str(stats))
    finally:
        normalizer.close()
    return species, checkpoint, stats


def _args(recipe, checkpoint, stats, output, species):
    return [
        "--species",
        species,
        "--recipe",
        str(recipe),
        "--checkpoint",
        str(checkpoint),
        "--vecnormalize",
        str(stats),
        "--output",
        str(output),
        "--seed",
        "17",
        "--steps",
        "8",
        "--eval-episodes",
        "0",
    ]


def test_real_species_prepare_resume_and_combined_terrain_adaptation(walker, tmp_path):
    species, checkpoint, stats = walker
    source_bytes = checkpoint.read_bytes(), stats.read_bytes()
    recipe = REPO_ROOT / "configs" / species / "behaviors" / "follow_direction.toml"
    prepared, resumed, adapted = (tmp_path / name for name in ("prepared", "resumed", "adapted"))
    main(_args(recipe, checkpoint, stats, prepared, species))
    main(_args(recipe, prepared / "model.zip", prepared / "vecnormalize.pkl", resumed, species) + ["--resume"])
    mixed = REPO_ROOT / "configs" / species / "behaviors" / "follow_direction_difficult_terrain.toml"
    # Short diagnostic horizon keeps this integration test bounded while using
    # the complete real heightfield, normalization, optimizer and CSV scorer.
    short_recipe = tmp_path / "combined.toml"
    short_recipe.write_text(re.sub(r"max_episode_steps = \d+", "max_episode_steps = 4", mixed.read_text()))
    arguments = _args(short_recipe, resumed / "model.zip", resumed / "vecnormalize.pkl", adapted, species)
    arguments[arguments.index("--eval-episodes") + 1] = "5"
    main(arguments + ["--adapt"])
    reports = [json.loads((output / "run.json").read_text()) for output in (prepared, resumed, adapted)]
    for report, output in zip(reports, (prepared, resumed, adapted), strict=True):
        assert report["status"] == "complete"
        assert report["species"] == species
        assert report["parent_stage"] == "locomotion"
        assert report["training"]["actual_additional_steps"] == 8
        assert not report["canonical_certification"]
        _verify_bundle(output / "model.zip", output / "vecnormalize.pkl")
    assert reports[0]["preparation"]["max_action_delta"] <= 1e-6
    assert reports[0]["preparation"]["max_value_delta"] <= 1e-6
    assert reports[1]["training"]["stage_start_timesteps"] == reports[0]["training"]["stage_start_timesteps"]
    transition = reports[2]["preparation"]["transitions"][-1]
    assert transition["command_weights_preserved"] and transition["optimizer_tensors_preserved"]
    assert reports[2]["evaluation"]
    scored = json.loads((adapted / "evaluation_summary.json").read_text())
    assert {episode["terrain_family"] for episode in scored["episodes"]} == {
        "flat",
        "sloped",
        "bumps",
        "depressions",
        "mixed",
    }
    assert source_bytes == (checkpoint.read_bytes(), stats.read_bytes())
    _, commands, terrain, kwargs = read_recipe(short_recipe, species)
    env = create_behavior_env(species, commands=commands, terrain=terrain, run_seed=17, **kwargs)
    loaded, normalizer, _ = load_behavior_checkpoint(
        adapted / "model.zip",
        adapted / "vecnormalize.pkl",
        env,
        behavior_identity=env.behavior_identity,
        species=species,
    )
    try:
        assert loaded.num_timesteps == 40
        assert all(torch.isfinite(value).all() for value in loaded.policy.state_dict().values())
        assert any(torch.count_nonzero(loaded.policy.get_submodule(name).weight[:, -3:]) for name in COMMAND_LAYERS)
        raw = normalizer.reset()
        assert np.isfinite(raw).all()
    finally:
        normalizer.close()


def test_explicit_species_cannot_disagree_with_behavior_parent(walker):
    species, checkpoint, stats = walker
    other = next(name for name in species_display_names() if name != species)
    # Check before loading a network: some different species share tensor shapes.
    with pytest.raises(BehaviorCheckpointError, match="identity and requested species disagree"):
        prepare_behavior_checkpoint(
            checkpoint,
            stats,
            None,
            species=other,
            behavior_identity={"parent_plant": {"species": species}},
        )
