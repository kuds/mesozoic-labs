"""Real CPU training, checkpoint, evaluation and notebook integration smoke tests.

Tiny budgets exercise gradient updates and artifact contracts; they are not
learning-performance tests. Production curriculum thresholds remain intact.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sb3 = pytest.importorskip("stable_baselines3")

from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.curriculum import load_vecnorm_stats  # noqa: E402
from environments.shared.evaluation import eval_policy  # noqa: E402
from environments.shared.plant_contract import (  # noqa: E402
    PlantCompatibilityError,
    current_plant_identity,
    validate_model_plant,
)
from environments.shared.reporting import generate_stage_artifacts  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE  # noqa: E402
from environments.shared.train_base import create_vec_env, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SPECIES = ("compsognathus", "compsognathus_robot")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def smoke_configs(species):
    configs = deepcopy(load_all_stages(species))
    for config in configs.values():
        config["env_kwargs"].update(max_episode_steps=32)
        config["ppo_kwargs"].update(n_steps=32, batch_size=32, n_epochs=1)
        config["ppo_kwargs"]["policy_kwargs"] = {"net_arch": [16, 16], "log_std_init": -2}
        config["sac_kwargs"].update(learning_starts=16, batch_size=16, buffer_size=256, train_freq=1)
        config["sac_kwargs"]["policy_kwargs"] = {"net_arch": [16, 16]}
        # Turn off transition shaping for these 64-step integration runs.
        # Do not alter the actual advancement criteria.
        config["curriculum_kwargs"].update(warmup_timesteps=0, ramp_timesteps=0)
    return configs


def assert_checkpoint_round_trip(species, algorithm, configs, stage, checkpoint, stats):
    identity = current_plant_identity(species)
    model = getattr(sb3, algorithm.upper()).load(checkpoint, device="cpu")
    validate_model_plant(model, identity)
    assert getattr(model, MODEL_TASK_ATTRIBUTE)
    assert model._n_updates > 0
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    other = SPECIES[1] if species == SPECIES[0] else SPECIES[0]
    with pytest.raises(PlantCompatibilityError):
        validate_model_plant(model, current_plant_identity(other))
    eval_env = create_vec_env(
        get_species_config(species),
        configs,
        stage,
        1,
        1234,
        algorithm=algorithm,
        plant_identity=identity,
    )
    try:
        assert load_vecnorm_stats(stats, eval_env, current_plant=identity)
        eval_env.training = False
        eval_env.norm_reward = False
        obs = eval_env.reset()
        expected = model.predict(obs, deterministic=True)[0]
        reloaded = getattr(sb3, algorithm.upper()).load(checkpoint, device="cpu")
        np.testing.assert_array_equal(expected, reloaded.predict(obs, deterministic=True)[0])
        panels = eval_policy(model, eval_env, ["target_success"], n_episodes=2)
        assert all(len(panel) == 2 and np.all(np.isfinite(panel)) for panel in panels)
        assert all(0 < length <= 32 for length in panels[1])
    finally:
        eval_env.close()


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_shared_trainer_all_stages_and_checkpoint_handoffs(species, algorithm, tmp_path):
    config = get_species_config(species)
    stages = smoke_configs(species)
    previous = None
    for stage in (1, 2, 3):
        directory = tmp_path / f"stage{stage}"
        model = train(
            config,
            stages,
            stage,
            total_timesteps=64,
            n_envs=1,
            seed=42,
            load_path=previous,
            task_load_mode="initialize_next_stage" if previous else "resume_same_stage",
            eval_freq=64,
            save_freq=64,
            log_dir=str(directory),
            verbose=0,
            algorithm=algorithm,
            use_tensorboard=False,
            post_eval_episodes=2,
        )
        assert model.num_timesteps >= 64
        previous = str(directory / "models" / f"stage{stage}_final.zip")
        stats = str(directory / "models" / f"stage{stage}_final_vecnorm.pkl")
        assert_checkpoint_round_trip(species, algorithm, stages, stage, previous, stats)
        metadata = json.loads((directory / "stage_config.json").read_text())
        assert metadata["plant_identity"]["species"] == species
        assert (directory / "metrics.json").is_file()


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_actual_notebook_training_and_stance_report(species, algorithm, tmp_path):
    notebook = json.loads((ROOT / "notebooks/sb3_training.ipynb").read_text())

    def cell(marker):
        return next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code" and marker in "".join(c["source"])
        )

    namespace = {"IN_COLAB": False}
    exec(compile(cell("# Add repo root to path"), "sb3_setup", "exec"), namespace)
    selection = cell("# ===== SPECIES SELECTION =====")
    selection = selection.replace('SPECIES = "Velociraptor Mongoliensis"', f'SPECIES = "{species}"')
    exec(compile(selection, "sb3_selection", "exec"), namespace)
    exec(compile(cell("EnvClass = SPECIES_CFG.env_class"), "sb3_environment", "exec"), namespace)
    namespace.update(
        STAGE_CONFIGS=smoke_configs(species),
        ALGORITHM=algorithm,
        N_ENVS=1,
        PLANT_IDENTITY=current_plant_identity(species),
        CHECKPOINT_SELECTION_SEED=1042,
        EVALUATION_SEED=3042,
        USE_GOOGLE_DRIVE=False,
        AUTO_DISCONNECT=False,
    )
    exec(compile(cell("def train_stage("), "sb3_training_infrastructure", "exec"), namespace)
    model, selected, final, directory, stats, results = namespace["train_stage"](
        1, 64, run_dir=tmp_path, eval_freq=64, save_freq=64
    )
    assert model._n_updates > 0
    assert_checkpoint_round_trip(species, algorithm, namespace["STAGE_CONFIGS"], 1, selected + ".zip", stats)
    results = generate_stage_artifacts(
        namespace["SPECIES_CFG"],
        namespace["STAGE_CONFIGS"][1],
        1,
        algorithm,
        directory,
        seed=42,
        stage_results=results,
        timesteps=64,
        record_videos=False,
        generate_graphs=False,
    )
    # A tiny run must not certify the production stance recipe from reward alone.
    assert not results["publication_gate_passed"]
    report = json.loads((directory / "stance_gate_report.json").read_text())
    assert np.isfinite(report["metrics"]["mean_unsupported_duty"])
    assert results["gate_failures"]
