"""Exercise real PPO preparation, continuation and terrain transfer for every plant."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared.behavior_certification import behavior_library_key, evaluate_saved_panel  # noqa: E402
from environments.shared.behavior_checkpoint import (  # noqa: E402
    COMMAND_LAYERS,
    BehaviorCheckpointError,
    load_behavior_checkpoint,
    prepare_behavior_checkpoint,
)
from environments.shared.certified_library import publish_candidate  # noqa: E402
from environments.shared.plant_contract import attach_plant_identity, current_plant_identity  # noqa: E402
from environments.shared.result_bundle import sha256_file  # noqa: E402
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
    resume_args = _args(recipe, prepared / "model.zip", prepared / "vecnormalize.pkl", resumed, species)
    resume_args[resume_args.index("--seed") + 1] = "29"
    main(resume_args + ["--resume"])
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
    assert reports[0]["certification_training_seed"] == reports[1]["certification_training_seed"] == 17
    assert reports[1]["run_seed"] == 29
    assert reports[0]["certification_training_parent_sha256"] == reports[1]["certification_training_parent_sha256"]
    assert reports[2]["certification_training_parent_sha256"] != reports[1]["certification_training_parent_sha256"]
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
    panel_hashes = sha256_file(adapted / "model.zip"), sha256_file(adapted / "vecnormalize.pkl")
    panel = evaluate_saved_panel(
        model_path=adapted / "model.zip",
        normalization_path=adapted / "vecnormalize.pkl",
        recipe_path=short_recipe,
        species=species,
        identity=reports[2]["behavior_identity"],
        output_dir=tmp_path / "saved-panel",
        run_seed=910001,
        seed_start=920001,
        episodes=5,
    )
    assert (panel["model_sha256"], panel["normalization_sha256"]) == panel_hashes
    assert (sha256_file(adapted / "model.zip"), sha256_file(adapted / "vecnormalize.pkl")) == panel_hashes
    assert panel["protocol"]["environment_run_seed"] == 910001
    assert panel["protocol"]["episode_seeds"] == list(range(920001, 920006))
    assert panel["training_lineage"]["seed"] == 17
    assert panel["training_lineage"]["parent_model_sha256"] == sha256_file(resumed / "model.zip")
    assert panel["training_lineage"]["parent_normalization_sha256"] == sha256_file(resumed / "vecnormalize.pkl")
    assert panel["training_lineage"]["stage_start_timesteps"] == 32
    assert panel["training_lineage"]["num_timesteps"] == 40
    for episode in panel["episodes"]:
        if episode["terrain_family"] == "flat":
            assert episode["terrain"]["family"] == "flat_plane"
        else:
            assert episode["terrain"]["template"] == episode["terrain_family"]
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


def test_real_species_auto_source_copies_full_bundle_and_preserves_original_seed(walker, tmp_path):
    """Exercise routing with a labeled storage fixture, not a claim of learned skill."""
    species, checkpoint, stats = walker
    recipe_path = REPO_ROOT / "configs" / species / "behaviors" / "follow_direction.toml"
    prepared = tmp_path / "prepared"
    main(_args(recipe_path, checkpoint, stats, prepared, species))
    report = json.loads((prepared / "run.json").read_text())
    library = tmp_path / "library"
    source_files = {path.relative_to(prepared).as_posix(): path for path in prepared.rglob("*") if path.is_file()}
    publication = publish_candidate(
        library,
        key=behavior_library_key(species, "follow_direction", report["behavior_identity"]),
        recipe_sha256=report["recipe_sha256"],
        training_seed=17,
        source_run_id="synthetic-storage-fixture",
        files=source_files,
        model_path="model.zip",
        normalization_path="vecnormalize.pkl",
        certificate={"passed": True, "fixture_only": True, "not_learned_skill_evidence": True},
        comparison={
            "protocol": {"version": "storage-integration-fixture/v1", "episode_seeds": [100, 101]},
            "metrics": [{"name": "fixture", "values": [1.0, 1.0], "direction": "higher", "margin": 0.0}],
        },
    )
    assert publication["recommended"]
    resumed = tmp_path / "auto-resumed"
    arguments = _args(recipe_path, prepared / "model.zip", prepared / "vecnormalize.pkl", resumed, species)
    for flag in ("--checkpoint", "--vecnormalize"):
        position = arguments.index(flag)
        del arguments[position : position + 2]
    arguments[arguments.index("--seed") + 1] = "29"
    main(arguments + ["--auto-source", "--resume", "--certified-library", str(library)])
    continuation = json.loads((resumed / "run.json").read_text())
    selection = json.loads((resumed / "certified_source.json").read_text())
    assert continuation["status"] == "complete"
    assert continuation["run_seed"] == 29
    assert continuation["certification_training_seed"] == 17
    assert continuation["certification_training_parent_sha256"] == sha256_file(checkpoint)
    assert continuation["certification_training_parent_normalization_sha256"] == sha256_file(stats)
    assert continuation["training"]["actual_additional_steps"] == 8
    assert continuation["training"]["stage_start_timesteps"] == 16
    assert selection["version"] == publication["version"]
    copied = Path(selection["directory"])
    assert copied.is_relative_to(resumed / "certified_inputs")
    assert (copied / "manifest.json").is_file()
    assert Path(selection["selection_record"]).is_relative_to(resumed)
    for name, source in source_files.items():
        assert (copied / name).read_bytes() == source.read_bytes()
    _verify_bundle(copied / "model.zip", copied / "vecnormalize.pkl")
    _verify_bundle(resumed / "model.zip", resumed / "vecnormalize.pkl")


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
