"""Evaluate a published Pilot4 PPO actor on its training plant and the revised plant.

Supply the original stage directory (config, gate verdict and models/) and the r10
MJCF. The verdict identifies both artifacts. This command only produces measurements.
"""

import argparse
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


def replay(stage: Path, original_xml: Path, seeds: list[int]) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("original_xml", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    print(json.dumps(replay(args.stage, args.original_xml, args.seeds), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
