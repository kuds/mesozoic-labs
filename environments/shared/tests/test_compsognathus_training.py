"""Real CPU training, checkpoint, evaluation and notebook integration smoke tests.

Tiny budgets exercise gradient updates and artifact contracts; they are not
learning-performance tests. Production curriculum thresholds remain intact.
"""

import csv
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
from environments.shared.task_fingerprint import (  # noqa: E402
    MODEL_TASK_ATTRIBUTE,
    MODEL_TASK_LINEAGE_ATTRIBUTE,
)
from environments.shared.train_base import create_vec_env, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SPECIES = ("compsognathus", "compsognathus_robot")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def ppo_updates(monkeypatch):
    """Observe the settings consumed by real gradient updates, not just TOML."""
    updates = []
    original_train = sb3.PPO.train

    def record_update(model):
        settings = {
            "step": model.num_timesteps,
            "ent_coef": model.ent_coef,
            "clip_range": model.clip_range(model._current_progress_remaining),
        }
        original_train(model)
        updates.append(settings)

    monkeypatch.setattr(sb3.PPO, "train", record_update)
    return updates


def smoke_configs(species):
    configs = deepcopy(load_all_stages(species))
    for config in configs.values():
        config["env_kwargs"].update(max_episode_steps=32)
        if config["env_kwargs"].get("perturbation_capture_velocity_multiple", 0.0) > 0:
            # Put real, non-overlapping pushes inside the tiny test horizon.
            # Preserve the dimensionless impulse and production gate thresholds.
            config["env_kwargs"].update(perturbation_interval=0.2, perturbation_jitter=0.0, perturbation_duration=0.02)
        config["ppo_kwargs"].update(n_steps=32, batch_size=32, n_epochs=1)
        config["ppo_kwargs"]["policy_kwargs"] = {"net_arch": [16, 16], "log_std_init": -2}
        config["sac_kwargs"].update(learning_starts=16, batch_size=16, buffer_size=256, train_freq=1)
        config["sac_kwargs"]["policy_kwargs"] = {"net_arch": [16, 16]}
        # Exercise one PPO update inside the stage-entry warm-up and one
        # after release. Keep the production clip/entropy settings and the
        # absolute entropy-decay anchor; only compress the warm-up duration.
        # Reward ramping is unrelated to this test; advancement stays intact.
        config["curriculum_kwargs"].update(warmup_timesteps=40, ramp_timesteps=0)
    return configs


def assert_optimizer_recipe(model, algorithm, config):
    """Check live and deserialized algorithms receive the executable recipe."""
    if algorithm == "ppo":
        ppo = config["ppo_kwargs"]
        assert callable(model.learning_rate)
        assert model.lr_schedule(1.0) == pytest.approx(ppo["learning_rate"])
        assert model.lr_schedule(0.0) == pytest.approx(ppo["learning_rate_end"])
        assert model.lr_schedule(0.5) == pytest.approx((ppo["learning_rate"] + ppo["learning_rate_end"]) / 2)
        assert model.target_kl == ppo["target_kl"]
        progress = min(1.0, model.num_timesteps / ppo["ent_coef_decay_timesteps"])
        expected_entropy = ppo["ent_coef"] + progress * (ppo["ent_coef_end"] - ppo["ent_coef"])
        # Evaluation checkpoints are saved before the current step's entropy
        # callback. Allow exactly that one-step lag, not a missing callback.
        entropy_step = abs(ppo["ent_coef_end"] - ppo["ent_coef"]) / ppo["ent_coef_decay_timesteps"]
        assert model.ent_coef == pytest.approx(expected_entropy, abs=entropy_step * 1.01, rel=0)
        assert model.ent_coef < ppo["ent_coef"]
        optimizers = [model.policy.optimizer]
    else:
        sac = config["sac_kwargs"]
        assert model.learning_rate == sac["learning_rate"]
        assert model.lr_schedule(1.0) == model.lr_schedule(0.0) == sac["learning_rate"]
        assert model.ent_coef == sac["ent_coef"] == "auto"
        assert torch.isfinite(model.log_ent_coef).all()
        optimizers = [model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer]
    expected_lr = model.lr_schedule(model._current_progress_remaining)
    for optimizer in optimizers:
        assert all(group["lr"] == pytest.approx(expected_lr) for group in optimizer.param_groups)


def assert_checkpoint_round_trip(species, algorithm, configs, stage, checkpoint, stats):
    identity = current_plant_identity(species)
    model = getattr(sb3, algorithm.upper()).load(checkpoint, device="cpu")
    validate_model_plant(model, identity)
    assert getattr(model, MODEL_TASK_ATTRIBUTE)
    assert model._n_updates > 0
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    assert_optimizer_recipe(model, algorithm, configs[stage])
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
def test_shared_trainer_all_stages_and_checkpoint_handoffs(species, algorithm, tmp_path, ppo_updates):
    config = get_species_config(species)
    stages = smoke_configs(species)
    previous = None
    for stage in (1, 2, 3):
        directory = tmp_path / f"stage{stage}"
        ppo_updates.clear()
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
        assert_optimizer_recipe(model, algorithm, stages[stage])
        if algorithm == "ppo":
            ppo = stages[stage]["ppo_kwargs"]
            assert [update["step"] for update in ppo_updates] == [32, 64]
            if stage > 1:
                curriculum = stages[stage]["curriculum_kwargs"]
                assert ppo_updates[0]["clip_range"] == curriculum["warmup_clip_range"]
                assert ppo_updates[0]["ent_coef"] == curriculum["warmup_ent_coef"]
                assert ppo_updates[0]["ent_coef"] <= ppo["ent_coef"]
            assert ppo_updates[-1]["clip_range"] == ppo["clip_range"]
            assert ppo_updates[-1]["ent_coef"] == model.ent_coef
        previous = str(directory / "models" / f"stage{stage}_final.zip")
        stats = str(directory / "models" / f"stage{stage}_final_vecnorm.pkl")
        assert_checkpoint_round_trip(species, algorithm, stages, stage, previous, stats)
        metadata = json.loads((directory / "stage_config.json").read_text())
        assert metadata["plant_identity"]["species"] == species
        assert (directory / "metrics.json").is_file()


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_recovery_warm_start_updates_and_checkpoint_handoff(species, algorithm, tmp_path, ppo_updates):
    species_config = get_species_config(species)
    stages = smoke_configs(species)
    stance_dir = tmp_path / "stance"
    common = {
        "total_timesteps": 64,
        "n_envs": 1,
        "seed": 42,
        "eval_freq": 64,
        "save_freq": 64,
        "verbose": 0,
        "algorithm": algorithm,
        "use_tensorboard": False,
        "post_eval_episodes": 2,
    }
    parent = train(species_config, stages, 1, log_dir=str(stance_dir), **common)
    parent_parameters = [parameter.detach().clone() for parameter in parent.policy.parameters()]
    parent_fingerprint = getattr(parent, MODEL_TASK_ATTRIBUTE)
    stance_checkpoint = stance_dir / "models/stage1_final.zip"
    recovery_dir = tmp_path / "recovery"
    ppo_updates.clear()
    child = train(
        species_config,
        stages,
        "recovery",
        load_path=str(stance_checkpoint),
        task_load_mode="initialize_next_stage",
        log_dir=str(recovery_dir),
        **common,
    )
    child_fingerprint = getattr(child, MODEL_TASK_ATTRIBUTE)
    lineage = getattr(child, MODEL_TASK_LINEAGE_ATTRIBUTE)
    assert lineage["mode"] == "initialize_next_stage"
    assert lineage["parent_task_sha256"] == parent_fingerprint["task_sha256"]
    assert lineage["child_task_sha256"] == child_fingerprint["task_sha256"]
    assert lineage["parent_task_sha256"] != lineage["child_task_sha256"]
    assert child.num_timesteps == 64
    assert child._n_updates > parent._n_updates
    assert any(not torch.equal(before, after) for before, after in zip(parent_parameters, child.policy.parameters()))
    if algorithm == "ppo":
        assert [update["step"] for update in ppo_updates] == [32, 64]
        curriculum = stages["recovery"]["curriculum_kwargs"]
        assert ppo_updates[0]["clip_range"] == curriculum["warmup_clip_range"]
        assert ppo_updates[0]["ent_coef"] == curriculum["warmup_ent_coef"]
        assert ppo_updates[-1]["clip_range"] == stages["recovery"]["ppo_kwargs"]["clip_range"]
    checkpoint = str(recovery_dir / "models/recovery_final.zip")
    stats = str(recovery_dir / "models/recovery_final_vecnorm.pkl")
    assert_checkpoint_round_trip(species, algorithm, stages, "recovery", checkpoint, stats)
    restored = getattr(sb3, algorithm.upper()).load(checkpoint, device="cpu")
    assert getattr(restored, MODEL_TASK_LINEAGE_ATTRIBUTE) == lineage
    metadata = json.loads((recovery_dir / "stage_config.json").read_text())
    assert metadata["plant_identity"]["species"] == species
    assert metadata["reward_weights"]["perturbation_capture_velocity_multiple"] > 0


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_profile_backed_recovery_freeze_rehearsal_cannot_certify(species, algorithm, tmp_path):
    """Roll the committed physical task and judge without weakening its gate."""
    from environments.shared.curriculum.gate_resolver import GateResolutionError, require_gate_resolution
    from environments.shared.harnesses.freeze_recovery_gate import (
        freeze_recovery_gate,
        roll_policy_panel,
        stage_task_fingerprint,
    )
    from environments.shared.plant_contract import attach_plant_identity
    from environments.shared.recovery_calibration import load_recovery_calibration
    from environments.shared.task_fingerprint import attach_task_fingerprint

    calibration = load_recovery_calibration(species)
    identity = current_plant_identity(species)
    env = create_vec_env(
        get_species_config(species), load_all_stages(species), 1, 1, 42, algorithm=algorithm, plant_identity=identity
    )
    try:
        env.reset()
        kwargs = {"n_steps": 32, "batch_size": 32, "n_epochs": 1} if algorithm == "ppo" else {"buffer_size": 64}
        model = getattr(sb3, algorithm.upper())(
            "MlpPolicy", env, policy_kwargs={"net_arch": [16, 16]}, device="cpu", **kwargs
        )
        # A real SB3 checkpoint that deterministically commands the home pose:
        # validates the source contract and quiet brace without claiming that
        # this synthetic zero controller has learned stance or recovery.
        with torch.no_grad():
            for parameter in model.policy.parameters():
                parameter.zero_()
        attach_plant_identity(model, identity)
        attach_task_fingerprint(model, stage_task_fingerprint(species, 1))
        checkpoint = tmp_path / "stance.zip"
        stats = tmp_path / "stance_vecnorm.pkl"
        model.save(checkpoint)
        env.save(stats)
    finally:
        env.close()
    frozen = freeze_recovery_gate(
        tmp_path, species=species, episodes=1, policy_zip=checkpoint, vecnorm=stats, algorithm=algorithm
    )
    assert set(frozen.null_evidence) == {"zero_action", "brace"}
    evidence = frozen.null_evidence["zero_action"]
    assert len(evidence.episodes) == 1
    assert evidence.shoves
    assert all(np.isfinite(shove.force_n) and shove.force_n > 0 for shove in evidence.shoves)
    assert evidence.safe_set == calibration.safe_set
    loaded = require_gate_resolution(tmp_path, current_task_sha256=frozen.task_fingerprint["task_sha256"])
    assert loaded["evaluation_spec"] == calibration.evaluation_spec()
    assert loaded["capability_spec"] == calibration.profile["capability_spec"]
    assert loaded["capability_spec"]["min_eval_episodes"] == 40
    assert loaded["null_provenance"]["brace"]["algorithm"] == algorithm
    assert evidence.successes_by_seed() == frozen.null_evidence["brace"].successes_by_seed()
    # The real policy-panel path refuses before checkpoint loading: this is
    # plumbing evidence, never qualification from one lucky episode.
    with pytest.raises(GateResolutionError, match="rehearsal freeze"):
        roll_policy_panel(tmp_path, "unused_policy.zip", "unused_vecnorm.pkl", species=species)


def test_notebook_recovery_refuses_invalid_existing_resolution_before_training(tmp_path):
    from environments.shared.curriculum.gate_resolver import GateResolutionError
    from environments.shared.stage_manifest import stage_dirname

    notebook = json.loads((ROOT / "notebooks/sb3_training.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "RUN_RECOVERY_STAGE = False" in "".join(cell["source"])
    ).replace("RUN_RECOVERY_STAGE = False", "RUN_RECOVERY_STAGE = True", 1)
    directory = tmp_path / stage_dirname("compsognathus", "recovery")
    directory.mkdir()
    (directory / "gate_resolution.json").write_text("{}")

    def forbidden_training(**kwargs):
        pytest.fail("an existing but invalid recovery resolution must refuse before training")

    namespace = {
        "Path": Path,
        "RUN_DIR": tmp_path,
        "SPECIES": "compsognathus",
        "STAGE_CONFIGS": load_all_stages("compsognathus"),
        "QUICK_TEST": False,
        "path_1": "unused_stance",
        "vecnorm_1": "unused_vecnorm.pkl",
        "ALGORITHM": "ppo",
        "train_stage": forbidden_training,
    }
    with pytest.raises(GateResolutionError):
        exec(compile(source, "sb3_recovery_invalid_resolution", "exec"), namespace)


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_actual_notebook_training_stance_and_recovery_reports(species, algorithm, tmp_path, monkeypatch):
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
    assert_optimizer_recipe(model, algorithm, namespace["STAGE_CONFIGS"][1])
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

    # Execute the real opt-in cell and trainer. Replace only the expensive
    # registered panels: below, one genuine pushed episode checks artifact
    # persistence; the separate profile test exercises the real freeze path.
    from environments.shared.harnesses import freeze_recovery_gate as freeze
    from environments.shared.recovery_evaluation import roll_recovery_panel

    events = []
    trained = {}
    panels = {}
    original_train_stage = namespace["train_stage"]

    def record_freeze(stage_dir, **kwargs):
        assert kwargs["species"] == species
        assert kwargs["algorithm"] == algorithm
        assert kwargs["policy_zip"] == selected + ".zip"
        assert kwargs["vecnorm"] == stats
        events.append("freeze")

    def record_training(**kwargs):
        assert kwargs["load_path"] == selected
        assert kwargs["vecnorm_path"] == stats
        assert events == ["freeze", "validate"]
        events.append("train")
        trained["result"] = original_train_stage(**kwargs, eval_freq=64, save_freq=64)
        return trained["result"]

    def record_validation(stage_dir, **kwargs):
        assert kwargs == {
            "species": species,
            "policy_zip": selected + ".zip",
            "vecnorm": stats,
            "algorithm": algorithm,
        }
        assert events == ["freeze"]
        events.append("validate")

    def one_episode_panel(stage_dir, policy_zip, vecnorm_pkl, **kwargs):
        assert kwargs == {"species": species, "algorithm": algorithm}
        assert events == ["freeze", "validate", "train"]
        events.append("panel")
        recovery_model, recovery_selected, _, _, recovery_stats, _ = trained["result"]
        assert policy_zip == recovery_selected + ".zip"
        assert vecnorm_pkl == recovery_stats
        assert_optimizer_recipe(recovery_model, algorithm, namespace["STAGE_CONFIGS"]["recovery"])
        # The controller loads the actual selected policy and its matching
        # normalization sidecar, including the SAC inference path.
        env = namespace["EnvClass"](**namespace["STAGE_CONFIGS"]["recovery"]["env_kwargs"])
        try:
            predict = freeze.policy_controller(
                policy_zip, vecnorm_pkl, action_space=env.action_space, algorithm=algorithm, inference="sb3"
            )
            evidence = roll_recovery_panel(
                env, predict, controller_id="policy", episodes=1, seed=3042, t_recover_steps=4, dwell_steps=2
            )
        finally:
            env.close()
        assert evidence.shoves, "the short notebook panel must actually encounter a shove"
        panels["evidence"] = evidence
        return evidence

    def recovery_report(**kwargs):
        assert events == ["freeze", "validate", "train", "panel"]
        events.append("report")
        evidence = panels["evidence"]
        assert kwargs["recovery_successes_by_seed"] == evidence.successes_by_seed()
        for kind, expected in (("episodes", evidence.episodes), ("shoves", evidence.shoves)):
            with (kwargs["stage_dir"] / f"recovery_{kind}_policy.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            assert len(rows) == len(expected)
            assert {int(row["panel_seed"]) for row in rows} == {3042}
        # No synthetic gate resolution is made for the shortened task. The
        # real report must refuse certification from this rehearsal evidence.
        return generate_stage_artifacts(**kwargs, record_videos=False, generate_graphs=False)

    monkeypatch.setattr(freeze, "freeze_recovery_gate", record_freeze)
    monkeypatch.setattr(freeze, "validate_recovery_resolution", record_validation)
    monkeypatch.setattr(freeze, "roll_policy_panel", one_episode_panel)
    namespace["STAGE_CONFIGS"]["recovery"]["curriculum_kwargs"]["timesteps"] = 64
    namespace.update(
        RUN_DIR=tmp_path,
        QUICK_TEST=False,
        SEED=42,
        path_1=selected,
        vecnorm_1=stats,
        results_1=results,
        completed_stages=[],
        train_stage=record_training,
        generate_stage_artifacts=recovery_report,
        curriculum_results=lambda stance_results: [stance_results, namespace["results_r"]],
        write_training_summary=lambda *_args: None,
        save_run_bundle=lambda *_args, **_kwargs: [],
    )
    recovery_source = cell("RUN_RECOVERY_STAGE = False").replace(
        "RUN_RECOVERY_STAGE = False", "RUN_RECOVERY_STAGE = True", 1
    )
    exec(compile(recovery_source, "sb3_recovery", "exec"), namespace)
    assert events == ["freeze", "validate", "train", "panel", "report"]
    assert not namespace["results_r"]["publication_gate_passed"]
    assert namespace["results_r"]["gate_failures"]
    assert namespace["completed_stages"] == [("recovery", namespace["dir_r"])]
    assert namespace["path_1"] == selected  # Stage 2 still initializes from stance.
