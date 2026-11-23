"""Evaluate a published Pilot4 PPO actor on its training plant and the revised plant.

Supply the original stage directory (config, gate verdict and models/) and the r10
MJCF. The verdict identifies both artifacts. This command only produces measurements.

It runs only in a checkout of :data:`LAST_RUNNABLE_COMMIT` or earlier (back to
``8c54c58``, which added it): the Pilot4 stage's recorded config passes the bite
reward's and the prey's arguments, which the environment refuses since ``e3cccca``
(the bite) and the target-free revision (``configs/plant_versions.toml`` note 24,
the prey), and its policy reads 63 observations where the environment observes 59
since that revision.  On a later checkout it stops at once and says so
(:func:`require_a_runnable_checkout`).
"""

import argparse
import inspect
import json
from pathlib import Path

from environments.shared.config import read_recorded_stage_config
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.file_io import read_json_object
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import PlantIdentity, validate_environment_plant
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.result_bundle.hashing import sha256_file
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.tyrannosaurus_rex.scripts.audit_motion import run

# Training-only terms on the Pilot4 experimental branch. They affect the scalar
# reward, which deterministic inference never reads. Retain the actual command.
PILOT_REWARD_KEYS = (
    "walk_support_gate_bw",
    "walk_touchdown_weight",
    "walk_heading_price_weight",
    "walk_heading_price_scale_rad",
)
#: The last commit whose environment takes the Pilot4 config and observes what its policy reads.
LAST_RUNNABLE_COMMIT = "3368abe"
#: The Pilot4 policy's observation width (T. rex policy interface r15; r16 kept it, r17 observes 59).
PILOT4_OBSERVATION_DIM = 63
#: Arguments the Pilot4 stage's recorded config passes that the environment has since dropped.
PILOT4_ARGUMENTS = ("bite_bonus", "prey_distance_range")


def require_a_runnable_checkout() -> None:
    """Refuse, before reading a stage, on a checkout whose environment cannot replay Pilot4."""
    accepted = inspect.signature(TyrannosaurusRexAnatomicalEnv).parameters
    refused = [name for name in PILOT4_ARGUMENTS if name not in accepted]
    with TyrannosaurusRexAnatomicalEnv() as env:
        width = int(env.observation_space.shape[0])
    if refused or width != PILOT4_OBSERVATION_DIM:
        raise RuntimeError(
            f"replay_walk cannot run on this checkout: the Pilot4 stage's config passes {list(PILOT4_ARGUMENTS)} "
            f"and its policy reads {PILOT4_OBSERVATION_DIM} observations, but this environment refuses {refused} "
            f"and observes {width}. It last ran at commit {LAST_RUNNABLE_COMMIT}; run it in a checkout of "
            f"{LAST_RUNNABLE_COMMIT} or earlier (for example `git worktree add ../pilot4-replay "
            f"{LAST_RUNNABLE_COMMIT}`)."
        )


def replay(stage: Path, original_xml: Path, seeds: list[int]) -> dict:
    require_a_runnable_checkout()
    config = read_recorded_stage_config(stage)
    verdict = read_json_object(stage / "gate_verdict.json")
    if config is None or verdict is None:
        raise ValueError(f"Missing or unreadable stage config or gate verdict in {stage}")
    identity = PlantIdentity.from_mapping(config["plant_identity"])
    if (identity.model_id, identity.physics_revision, identity.policy_interface_revision) != (
        ModelId("tyrannosaurus_rex", "anatomical"),
        10,
        15,
    ) or verdict["stage_id"] != "locomotion":
        raise ValueError("Expected the anatomical T. rex r10 / interface r15 Pilot4 walking stage.")
    paths = {}
    for key in ("checkpoint", "normalization"):
        path = stage / verdict[key]
        if sha256_file(path) != verdict[key + "_sha256"]:
            raise ValueError(f"Digest mismatch for {key} in {stage}")
        paths[key] = str(path)
    kwargs = dict(config["reward_weights"])
    for name in PILOT_REWARD_KEYS:
        kwargs.pop(name)
    kwargs["command_config"] = DirectionCommandConfig(**kwargs["command_config"])

    class TrainingPlant(TyrannosaurusRexAnatomicalEnv):
        MODEL_PATH = original_xml.resolve()

    def make_original():
        return TrainingPlant(**kwargs)

    with make_original() as source:
        validate_environment_plant(source, identity, artifact="Pilot4 source MJCF")
    actor, norm, _ = load_sb3_checkpoint(
        paths["checkpoint"], paths["normalization"], make_original, plant_identity=identity
    )
    assert norm is not None
    try:

        def predict(observation):
            return actor.predict(norm.normalize_obs(observation), deterministic=True)[0]

        panels = {}
        for label, cls in [("original", TrainingPlant), ("revised", TyrannosaurusRexAnatomicalEnv)]:
            with cls(**kwargs) as env:
                panels[label] = [
                    run(
                        env=env,
                        predict=predict,
                        seed=seed,
                        noise=env.reset_noise_scale,
                        seconds=env.max_episode_steps * env.dt,
                        warmup=1.0,
                        mode="policy",
                    )
                    for seed in seeds
                ]
    finally:
        norm.close()
    return {
        "training_seed": config["run"]["seed"],
        "checkpoint_sha256": verdict["checkpoint_sha256"],
        "normalization_sha256": verdict["normalization_sha256"],
        "omitted_reward_keys": PILOT_REWARD_KEYS,
        "panels": panels,
    }


def main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("original_xml", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args(argv)
    try:
        require_a_runnable_checkout()
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(replay(args.stage, args.original_xml, args.seeds), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
